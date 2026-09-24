# Agent57 (Badia et al., 2020) on JAXAtari.
#
# STAGE selects how much of Agent57 is enabled (an ablation ladder; each stage
# is a strict superset of the previous one):
#   r2d2     Recurrent replay DQN: LSTM core, stored-state sequence replay with
#            burn-in, h-rescaled n-step double-Q loss, prioritised replay.
#   ngu      + intrinsic reward (episodic k-NN x RND) and a population of
#            NUM_ARMS (beta, gamma) policies sharing one UVFA network, trained
#            on the mixed reward r_e + beta * r_i.
#   split_q  + separate Q_e / Q_i networks, acting on
#            h(h^-1(Q_e) + beta * h^-1(Q_i)).
#   agent57  + per-actor sliding-window UCB bandit choosing the arm per episode.
#
# File layout (single module, matching feat/split-q):
#   networks         Q-network (split heads), embedding net, RND
#   replay           TimeStep, prioritised sequence buffer, burn-in split
#   intrinsic        episodic + lifelong novelty
#   meta controller  arm schedule and UCB bandit
#   agent            Agent57: act_step / learner_update / train_iteration,
#                    all pure and jit-compatible; make_env; single_run.
# Evaluation lives in agent57_eval.py.
import os
import random
import time
from functools import partial
from typing import NamedTuple

import flashbax as fbx
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


# ============================================================================
# NETWORKS
# ============================================================================
# Agent57 networks.
#
#     Torso               observation -> features (DQN conv stack or MLP)
#     RecurrentQNetwork   torso -> LSTM -> dueling head, one R2D2 Q-network
#     SplitQNetwork       NUM_HEADS copies of RecurrentQNetwork with separate
#                         weights, evaluated in ONE call via nn.vmap.
#                         Head 0 = Q_e (extrinsic), head 1 = Q_i (intrinsic).
#     EmbeddingTrainer    NGU controllable-state embedding + inverse dynamics
#     RNDNetwork          NGU lifelong novelty (frozen target / trained predictor)
#
# Provenance: Torso, RecurrentQNetwork, EmbeddingTrainer and RNDNetwork come
# from feat/infra and feat/ngu. SplitQNetwork replaces feat/split-q's two
# independent network.apply calls (one per Q) with a single vmapped module.

class Torso(nn.Module):
    """Observation -> flat feature vector.

    pixel  [R2D2] Table 2: "the same 3-layer convolutional network as DQN":
           32/64/64, kernels 8/4/3, strides 4/2/1, VALID. Input (B, 4, 84, 84) uint8.
    oc     MLP, width 512 and depth 2 by default.
    """
    pixel_based: bool
    mlp_width: int = 512
    mlp_depth: int = 2

    @nn.compact
    def __call__(self, x):
        if self.pixel_based:
            # stored as (B, C, H, W); flax Conv expects (B, H, W, C)
            x = jnp.transpose(x, (0, 2, 3, 1)).astype(jnp.float32) / 255.0
            x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
            x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
            x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.astype(jnp.float32)
            for _ in range(self.mlp_depth):
                x = nn.Dense(self.mlp_width, kernel_init=orthogonal(np.sqrt(2.0)),
                             bias_init=constant(0.0))(x)
                x = nn.relu(x)
        return x


class RecurrentQNetwork(nn.Module):
    """[R2D2] torso -> LSTM -> dueling head.

    The LSTM input is [features, one_hot(prev_action), prev_reward] and, when
    num_arms > 0, [NGU] one_hot(arm) and the previous intrinsic reward (UVFA:
    one set of weights plays every (beta, gamma) arm).

    prev_action < 0 marks the first step of an episode; the carry is zeroed
    there, both while acting and while unrolling replayed sequences.
    """
    action_dim: int
    pixel_based: bool
    hidden_size: int = 512
    dueling_units: int = 512
    mlp_width: int = 512
    mlp_depth: int = 2
    num_arms: int = 0

    @nn.compact
    def __call__(self, carry, obs, prev_action, prev_reward, arm, prev_intrinsic):
        first = (prev_action < 0)[:, None]
        carry = jax.tree.map(lambda c: jnp.where(first, 0.0, c), carry)
        x = Torso(self.pixel_based, self.mlp_width, self.mlp_depth)(obs)
        inputs = [x, jax.nn.one_hot(prev_action, self.action_dim), prev_reward[:, None]]
        if self.num_arms > 0:
            inputs += [jax.nn.one_hot(arm, self.num_arms), prev_intrinsic[:, None]]
        carry, x = nn.OptimizedLSTMCell(self.hidden_size)(carry, jnp.concatenate(inputs, -1))

        # dueling head; layer order (and so parameter names) as in feat/infra
        v = nn.relu(nn.Dense(self.dueling_units)(x))
        v = nn.Dense(1)(v)
        a = nn.relu(nn.Dense(self.dueling_units)(x))
        a = nn.Dense(self.action_dim)(a)
        return carry, v + (a - a.mean(axis=-1, keepdims=True))


class SplitQNetwork(nn.Module):
    """[Agent57 sec 3.1] num_heads independent RecurrentQNetworks in one call.

    Params and carry get a leading head axis; every other input is shared.
      carry: (c, h) each (num_heads, B, hidden)
      returns carry of the same shape and q of shape (num_heads, B, A)
    num_heads = 1 for the r2d2/ngu stages, 2 (Q_e, Q_i) for split_q/agent57.
    """
    num_heads: int
    action_dim: int
    pixel_based: bool
    hidden_size: int = 512
    dueling_units: int = 512
    mlp_width: int = 512
    mlp_depth: int = 2
    num_arms: int = 0

    @nn.compact
    def __call__(self, carry, obs, prev_action, prev_reward, arm, prev_intrinsic):
        heads = nn.vmap(
            RecurrentQNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(0, None, None, None, None, None),
            out_axes=0,
            axis_size=self.num_heads,
        )
        return heads(
            action_dim=self.action_dim, pixel_based=self.pixel_based,
            hidden_size=self.hidden_size, dueling_units=self.dueling_units,
            mlp_width=self.mlp_width, mlp_depth=self.mlp_depth, num_arms=self.num_arms,
        )(carry, obs, prev_action, prev_reward, arm, prev_intrinsic)

    def initial_carry(self, batch_size):
        z = jnp.zeros((self.num_heads, batch_size, self.hidden_size), jnp.float32)
        return (z, z)


class EmbeddingTrainer(nn.Module):
    """[NGU] controllable-state embedding f(x) + inverse-dynamics classifier.

    embed(obs)               -> (B, embedding_dim), linear output (compared
                                with euclidean distances, so no ReLU)
    classify(emb_t, emb_tp1) -> (B, action_dim) logits, 128 hidden units
    __call__(obs, next_obs)  -> logits (used for init)
    """
    action_dim: int
    pixel_based: bool
    embedding_dim: int = 32
    hidden: int = 128
    mlp_width: int = 512
    mlp_depth: int = 2

    def setup(self):
        self.torso = Torso(self.pixel_based, self.mlp_width, self.mlp_depth)
        self.proj = nn.Dense(self.embedding_dim)
        self.cls_hidden = nn.Dense(self.hidden)
        self.cls_out = nn.Dense(self.action_dim)

    def embed(self, obs):
        return self.proj(self.torso(obs))

    def classify(self, emb_t, emb_tp1):
        x = nn.relu(self.cls_hidden(jnp.concatenate([emb_t, emb_tp1], axis=-1)))
        return self.cls_out(x)

    def __call__(self, obs, next_obs):
        return self.classify(self.embed(obs), self.embed(next_obs))


class RNDNetwork(nn.Module):
    """[NGU] observation -> feature vector; used as frozen target and trained predictor."""
    pixel_based: bool
    output_dim: int = 128
    mlp_width: int = 512
    mlp_depth: int = 2

    @nn.compact
    def __call__(self, obs):
        return nn.Dense(self.output_dim)(Torso(self.pixel_based, self.mlp_width, self.mlp_depth)(obs))


# ============================================================================
# REPLAY
# ============================================================================
# [R2D2] prioritised sequence replay with stored recurrent state and burn-in.
#
# Storage is flashbax's prioritised trajectory buffer (already a dependency on
# master): each env's steps are kept in time order, and fixed-length windows of
# BURN_IN_LENGTH + SEQUENCE_LENGTH steps starting every SEQUENCE_PERIOD steps
# are sampled on device. No per-item Python slicing anywhere.
#
# Every step stores the LSTM carry it was acted with ("stored state"), so a
# sampled window can be re-warmed from its real starting state and then burned
# in for BURN_IN_LENGTH steps before the loss is taken.

@flax.struct.dataclass
class TimeStep:
    obs: jnp.ndarray              # observation BEFORE acting. OC float32, pixel uint8 (4, 84, 84)
    action: jnp.ndarray           # int32
    reward: jnp.ndarray           # extrinsic reward (clipped during training), float32
    done: jnp.ndarray             # bool, episode ended after this step
    prev_action: jnp.ndarray      # int32, -1 on the first step of an episode
    prev_reward: jnp.ndarray      # float32
    carry: tuple                  # (c, h), each (num_heads, hidden): state the step was acted with
    arm: jnp.ndarray              # int32, policy index that acted
    intrinsic_reward: jnp.ndarray  # float32, r_i of the state reached by this step
    prev_intrinsic: jnp.ndarray   # float32, UVFA input


def dummy_timestep(obs, num_heads, hidden):
    z = jnp.zeros((), jnp.float32)
    i = jnp.zeros((), jnp.int32)
    c = jnp.zeros((num_heads, hidden), jnp.float32)
    return TimeStep(obs=obs, action=i, reward=z, done=jnp.zeros((), jnp.bool_), prev_action=i,
                    prev_reward=z, carry=(c, c), arm=i, intrinsic_reward=z, prev_intrinsic=z)


def make_replay_buffer(num_envs, batch_size, burn_in, seq_len, period, buffer_size,
                       learning_starts, priority_exponent):
    sample_len = burn_in + seq_len
    return fbx.make_prioritised_trajectory_buffer(
        add_batch_size=num_envs,
        sample_batch_size=batch_size,
        sample_sequence_length=sample_len,
        period=period,
        min_length_time_axis=max(sample_len, learning_starts // num_envs),
        max_length_time_axis=buffer_size // num_envs,
        priority_exponent=priority_exponent,
    )


def split_burn_in(experience: TimeStep, burn_in):
    """(B, L, ...) batch -> time-major (burn, learn) segments and the start carry.

    start carry: (c, h) each (num_heads, B, hidden), taken from step 0.
    """
    data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), experience)
    start = jax.tree.map(lambda c: jnp.swapaxes(c[0], 0, 1), data.carry)
    burn = jax.tree.map(lambda x: x[:burn_in], data)
    learn = jax.tree.map(lambda x: x[burn_in:], data)
    return start, burn, learn


def importance_weights(probabilities, exponent):
    """(1/p)^exponent, normalised by the batch max. float32."""
    w = (1.0 / (probabilities.astype(jnp.float32) + 1e-10)) ** exponent
    return w / jnp.max(w)


def sequence_priority(abs_td, eta):
    """[R2D2] eta * max_t |td| + (1 - eta) * mean_t |td|, over axis -2 (time)."""
    return eta * abs_td.max(axis=-2) + (1.0 - eta) * abs_td.mean(axis=-2)


# ============================================================================
# INTRINSIC REWARD (NGU)
# ============================================================================
# [NGU] intrinsic reward: episodic novelty x lifelong novelty.
#
#     r_i = r_episodic * clip(alpha, 1, L)
#     r   = r_e + beta_j * r_i        (single-network stages)
#
# r_episodic  k-NN over the controllable-state embeddings seen so far in the
#             current episode (NGU Algorithm 1). Batched over envs, no host code.
# alpha       RND prediction error, normalised by a running mean/std.
#
# Logic from feat/ngu. Changes: the k-NN distances use the matmul expansion
# |a-b|^2 = |a|^2 + |b|^2 - 2ab (no (E, M, D) intermediate); hyperparameters
# are a NamedTuple of Python floats instead of the whole config dict; the
# embedding/RND updates are fused into one function that embeds each frame once.

class NoveltyParams(NamedTuple):
    num_neighbours: int = 10       # [NGU] Table 6
    kernel_epsilon: float = 1e-4   # [NGU] Table 6
    cluster_distance: float = 8e-3  # [NGU] Table 6
    pseudo_count_c: float = 1e-3   # [NGU] Table 6
    max_similarity: float = 8.0    # [NGU] Table 6
    clip_l: float = 5.0            # [NGU] Table 6

    @classmethod
    def from_config(cls, cfg):
        d = cls()
        return cls(
            num_neighbours=int(cfg.get("NUM_NEIGHBOURS", d.num_neighbours)),
            kernel_epsilon=float(cfg.get("KERNEL_EPSILON", d.kernel_epsilon)),
            cluster_distance=float(cfg.get("CLUSTER_DISTANCE", d.cluster_distance)),
            pseudo_count_c=float(cfg.get("PSEUDO_COUNT_C", d.pseudo_count_c)),
            max_similarity=float(cfg.get("MAX_SIMILARITY", d.max_similarity)),
            clip_l=float(cfg.get("INTRINSIC_CLIP_L", d.clip_l)),
        )


# ---- episodic memory ---------------------------------------------------------
@flax.struct.dataclass
class EpisodicMemory:
    """One preallocated ring buffer of embeddings per env."""
    embeddings: jnp.ndarray   # (E, M, D)
    count: jnp.ndarray        # (E,) valid slots
    write: jnp.ndarray        # (E,) next slot
    dist_mean: jnp.ndarray    # () running mean of squared k-NN distances
    dist_n: jnp.ndarray       # () number of distances in that mean


def init_episodic_memory(num_envs, size, dim):
    return EpisodicMemory(
        embeddings=jnp.zeros((num_envs, size, dim), jnp.float32),
        count=jnp.zeros((num_envs,), jnp.int32),
        write=jnp.zeros((num_envs,), jnp.int32),
        dist_mean=jnp.array(1.0, jnp.float32),
        dist_n=jnp.array(0.0, jnp.float32),
    )


def episodic_reward(memory: EpisodicMemory, embedding, first, p: NoveltyParams):
    """[NGU] Algorithm 1 for every env at once, then store the embedding.

    embedding: (E, D). first: (E,) bool, wipes that env's memory before the query.
    Returns (reward (E,) >= 0, new memory).
    """
    size = memory.embeddings.shape[1]
    k = min(p.num_neighbours, size)
    count = jnp.where(first, 0, memory.count)
    write = jnp.where(first, 0, memory.write)

    # squared distances via |m|^2 + |x|^2 - 2 m.x : (E, M), no (E, M, D) temporary
    m2 = jnp.sum(jnp.square(memory.embeddings), -1)
    x2 = jnp.sum(jnp.square(embedding), -1)[:, None]
    cross = jnp.einsum("emd,ed->em", memory.embeddings, embedding)
    d2 = jnp.maximum(m2 + x2 - 2.0 * cross, 0.0)
    d2 = jnp.where(jnp.arange(size)[None, :] < count[:, None], d2, jnp.inf)

    neg, _ = jax.lax.top_k(-d2, k)
    nn_d2 = -neg                                    # (E, k); inf where fewer than k exist
    valid = jnp.isfinite(nn_d2)

    n_new = jnp.sum(valid)
    total = memory.dist_n + n_new
    dist_mean = jnp.where(
        n_new > 0,
        (memory.dist_mean * memory.dist_n + jnp.sum(jnp.where(valid, nn_d2, 0.0))) / jnp.maximum(total, 1.0),
        memory.dist_mean,
    )
    d_n = jnp.maximum(jnp.where(valid, nn_d2, 0.0) / jnp.maximum(dist_mean, 1e-8) - p.cluster_distance, 0.0)
    kernel = jnp.where(valid, p.kernel_epsilon / (d_n + p.kernel_epsilon), 0.0)
    s = jnp.sqrt(jnp.sum(kernel, -1)) + p.pseudo_count_c
    reward = jnp.where(s > p.max_similarity, 0.0, 1.0 / s)
    # empty memory would give 1/c = 1000 on every episode's first step (feat/ngu choice: 0)
    reward = jnp.where(count > 0, reward, 0.0)

    embeddings = jax.vmap(lambda m, w, x: m.at[w].set(x))(memory.embeddings, write, embedding)
    return reward, EpisodicMemory(
        embeddings=embeddings,
        count=jnp.minimum(count + 1, size),
        write=(write + 1) % size,
        dist_mean=dist_mean,
        dist_n=total,
    )


# ---- lifelong novelty (RND) --------------------------------------------------
@flax.struct.dataclass
class RunningStats:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def init_running_stats():
    # explicit float32: weak-typed scalars become strong after the first update,
    # which changes the scan carry's type and forces a full recompile
    f32 = lambda v: jnp.array(v, jnp.float32)
    return RunningStats(mean=f32(0.0), var=f32(1.0), count=f32(1e-4))


def update_running_stats(stats: RunningStats, x):
    """Chan et al. parallel mean/variance update with a batch x."""
    batch_mean, batch_var, n = jnp.mean(x), jnp.var(x), x.size
    delta = batch_mean - stats.mean
    total = stats.count + n
    m2 = stats.var * stats.count + batch_var * n + jnp.square(delta) * stats.count * n / total
    return RunningStats(mean=stats.mean + delta * n / total, var=m2 / total, count=total)


def rnd_error(rnd_net, predictor_params, target_params, obs):
    """Per-observation squared prediction error (B,). No gradient into the target."""
    pred = rnd_net.apply(predictor_params, obs)
    target = jax.lax.stop_gradient(rnd_net.apply(target_params, obs))
    return jnp.mean(jnp.square(pred - target), axis=-1)


def rnd_modulator(error, stats: RunningStats, clip_l):
    """[NGU] alpha = 1 + (err - mean)/std clipped to [1, L]: only ever amplifies."""
    alpha = 1.0 + (error - stats.mean) / jnp.sqrt(stats.var + 1e-8)
    return jnp.clip(alpha, 1.0, clip_l)


def intrinsic_reward(r_episodic, modulator):
    return r_episodic * modulator


def mixed_reward(r_extrinsic, r_intrinsic, beta):
    """[NGU] r = r_e + beta * r_i."""
    return r_extrinsic + beta * r_intrinsic


def compute_intrinsic(emb_model, rnd_net, emb_params, rnd_params, rnd_target_params,
                      memory, rnd_stats, obs, first, p: NoveltyParams, update_stats=True):
    """Full per-step intrinsic reward pipeline, batched over envs.

    Returns (r_int (E,), memory, rnd_stats). update_stats=False at evaluation.
    """
    emb = emb_model.apply(emb_params, obs, method="embed")
    r_episodic, memory = episodic_reward(memory, emb, first, p)
    err = rnd_error(rnd_net, rnd_params, rnd_target_params, obs)
    if update_stats:
        rnd_stats = update_running_stats(rnd_stats, err)
    return intrinsic_reward(r_episodic, rnd_modulator(err, rnd_stats, p.clip_l)), memory, rnd_stats


# ---- side-network losses -----------------------------------------------------
def embedding_loss(emb_params, emb_model, obs_window, actions, mask):
    """[NGU] inverse-dynamics cross-entropy on consecutive frames.

    obs_window: (B, k+1, *obs) — every frame is embedded ONCE, then consecutive
    embeddings are paired (feat/ngu embedded the k overlapping frames twice).
    actions, mask: (B, k). Returns (loss, accuracy).
    """
    b, k1 = obs_window.shape[:2]
    emb = emb_model.apply(emb_params, obs_window.reshape((b * k1,) + obs_window.shape[2:]),
                          method="embed").reshape(b, k1, -1)
    logits = emb_model.apply(emb_params, emb[:, :-1], emb[:, 1:], method="classify")
    log_probs = jax.nn.log_softmax(logits)
    chosen = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
    denom = jnp.maximum(mask.sum(), 1.0)
    loss = -jnp.sum(chosen * mask) / denom
    accuracy = jnp.sum((jnp.argmax(logits, -1) == actions) * mask) / denom
    return loss, accuracy


def rnd_loss(predictor_params, rnd_net, target_params, obs):
    return jnp.mean(rnd_error(rnd_net, predictor_params, target_params, obs))


# ============================================================================
# META-CONTROLLER (Agent57)
# ============================================================================
# [Agent57] policy population and sliding-window UCB meta-controller.
#
# Population: NUM_ARMS policies j = 0..N-1, each a (beta_j, gamma_j) pair. All
# arms share one set of network weights (UVFA: the arm is a network input), so
# the population dimension is handled by gathering arm_betas[arm] /
# arm_gammas[arm] per env and per replayed step — never a Python loop over arms.
#
# Meta-controller: one sliding-window UCB bandit per actor (env). At the start
# of every episode it picks the arm with the best
#     mean_return_k + UCB_BETA * sqrt(1 / N_k)
# over the last UCB_WINDOW episodes, or a random arm with probability
# UCB_EPSILON. Arms absent from the window score +inf, so every arm is tried
# first. All state is a fixed-shape pytree; updates are pure jnp.
#
# arm_schedule is from feat/ngu, the bandit from feat/split-q.

def arm_schedule(num_arms, beta_max, gamma_max, gamma_min):
    """[NGU] (beta_j, gamma_j) for every arm, two (num_arms,) float32 arrays.

    beta_0 = 0, beta_{N-1} = beta_max, sigmoid in between.
    1 - gamma_j interpolated log-linearly between 1 - gamma_max and 1 - gamma_min.
    num_arms = 1 gives the single exploitative arm (beta 0, gamma_max).
    """
    n = int(num_arms)
    if n == 1:
        return jnp.zeros((1,), jnp.float32), jnp.full((1,), gamma_max, jnp.float32)
    j = jnp.arange(n, dtype=jnp.float32)
    inner = beta_max * jax.nn.sigmoid(10.0 * (2.0 * j - (n - 2)) / max(n - 2, 1))
    betas = jnp.where(j == 0, 0.0, jnp.where(j == n - 1, beta_max, inner))
    log_1mg = ((n - 1 - j) * jnp.log(1.0 - gamma_max) + j * jnp.log(1.0 - gamma_min)) / (n - 1)
    return betas.astype(jnp.float32), (1.0 - jnp.exp(log_1mg)).astype(jnp.float32)


@flax.struct.dataclass
class BanditState:
    """Ring buffers over EPISODES (not steps), one row per env."""
    arms: jnp.ndarray      # (E, W) arm played in each remembered episode
    returns: jnp.ndarray   # (E, W) extrinsic return of that episode
    write: jnp.ndarray     # (E,) next slot
    count: jnp.ndarray     # (E,) filled slots, capped at W


def init_bandit(num_envs, window):
    return BanditState(
        arms=jnp.zeros((num_envs, window), jnp.int32),
        returns=jnp.zeros((num_envs, window), jnp.float32),
        write=jnp.zeros((num_envs,), jnp.int32),
        count=jnp.zeros((num_envs,), jnp.int32),
    )


def arm_statistics(bandit: BanditState, num_arms):
    """Per-env visit counts and mean returns inside the window, both (E, A)."""
    window = bandit.arms.shape[1]
    filled = jnp.arange(window)[None, :] < bandit.count[:, None]
    is_arm = (bandit.arms[:, :, None] == jnp.arange(num_arms)[None, None, :]) & filled[:, :, None]
    n_k = jnp.sum(is_arm, axis=1)
    mean_k = jnp.sum(bandit.returns[:, :, None] * is_arm, axis=1) / jnp.maximum(n_k, 1)
    return n_k, mean_k


def bandit_select(bandit: BanditState, num_arms, rng, ucb_beta=1.0, ucb_epsilon=0.5):
    """Arm for each env's next episode, (E,) int32 in [0, num_arms)."""
    n_k, mean_k = arm_statistics(bandit, num_arms)
    score = jnp.where(n_k == 0, jnp.inf, mean_k + ucb_beta * jnp.sqrt(1.0 / jnp.maximum(n_k, 1)))
    greedy = jnp.argmax(score, axis=-1)
    k1, k2 = jax.random.split(rng)
    random_arm = jax.random.randint(k1, greedy.shape, 0, num_arms)
    explore = jax.random.uniform(k2, greedy.shape) < ucb_epsilon
    return jnp.where(explore, random_arm, greedy).astype(jnp.int32)


def bandit_update(bandit: BanditState, arm, episode_return, done):
    """Record (arm, return) for every env whose episode just ended (done)."""
    window = bandit.arms.shape[1]
    idx = bandit.write
    put = lambda buf, i, v, d: buf.at[i].set(jnp.where(d, v, buf[i]))
    return BanditState(
        arms=jax.vmap(put)(bandit.arms, idx, arm.astype(jnp.int32), done),
        returns=jax.vmap(put)(bandit.returns, idx, episode_return.astype(jnp.float32), done),
        write=jnp.where(done, (idx + 1) % window, idx),
        count=jnp.where(done, jnp.minimum(bandit.count + 1, window), bandit.count),
    )


def greedy_arm(bandit: BanditState, num_arms):
    """Best arm by windowed mean return, pooled over envs (used for evaluation)."""
    n_k, mean_k = arm_statistics(bandit, num_arms)
    n = n_k.sum(0)
    mean = (mean_k * n_k).sum(0) / jnp.maximum(n, 1)
    return jnp.argmax(jnp.where(n > 0, mean, -jnp.inf)).astype(jnp.int32)


# ============================================================================
# AGENT
# ============================================================================
STAGES = ("r2d2", "ngu", "split_q", "agent57")

# Paper values for every key the NGU / Agent57 stages need, so a config that
# only lists R2D2 values (agent57_*_original.yaml) still runs every stage.
DEFAULTS = {
    "STAGE": "r2d2",
    "DOUBLE_Q": True,
    "TAU": 1.0,
    "GRADIENT_STEPS": 1,
    "MLP_WIDTH": 512,
    "MLP_DEPTH": 2,
    # [NGU] Table 6
    "EMBEDDING_DIM": 32,
    "EPISODIC_MEMORY_SIZE": 30000,
    "NUM_NEIGHBOURS": 10,
    "KERNEL_EPSILON": 1e-4,
    "CLUSTER_DISTANCE": 8e-3,
    "PSEUDO_COUNT_C": 1e-3,
    "MAX_SIMILARITY": 8.0,
    "INTRINSIC_CLIP_L": 5.0,
    "RND_OUTPUT_DIM": 128,
    "NGU_LEARNING_RATE": 5e-4,
    "NGU_ADAM_EPS": 1e-4,
    "EMBEDDING_L2": 1e-5,
    "NGU_TRAIN_FRAMES": 5,
    "NUM_ARMS": 32,
    "BETA_MAX": 0.3,
    "GAMMA_MAX": 0.997,
    "GAMMA_MIN": 0.99,
    # [Agent57] Table 3 / sec 4
    "UCB_WINDOW": 160,
    "UCB_BETA": 1.0,
    "UCB_EPSILON": 0.5,
    "EVAL_ARM": None,
}


def h(x, eps):
    """[R2D2] value rescaling h(x) = sign(x)(sqrt(|x|+1) - 1) + eps x."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + eps * x


def h_inv(x, eps):
    """[R2D2] exact inverse of h."""
    a = jnp.sqrt(1.0 + 4.0 * eps * (jnp.abs(x) + 1.0 + eps)) - 1.0
    return jnp.sign(x) * (jnp.square(a / (2.0 * eps)) - 1.0)


def combine_q(q_e, q_i, beta, eps):
    """[Agent57 sec 3.1] Q = h(h^-1(Q_e) + beta h^-1(Q_i)). beta broadcasts over (..., A)."""
    return h(h_inv(q_e, eps) + jnp.asarray(beta)[..., None] * h_inv(q_i, eps), eps)


def n_step_targets(rewards, discounts, bootstrap, n):
    """G_t = sum_{k<n} (prod_{j<k} d_{t+j}) r_{t+k} + (prod_{j<n} d_{t+j}) bootstrap[t+n-1].

    rewards, discounts, bootstrap: (T, ...) with bootstrap[t] = value of s_{t+1}.
    Vectorised over T; the loop is over the static, small n (unrolled at trace).
    Steps near the end use as many rewards as remain and bootstrap from the last value.
    """
    t, pad = rewards.shape[0], n - 1
    rewards = jnp.concatenate([rewards, jnp.zeros((pad,) + rewards.shape[1:], rewards.dtype)])
    discounts = jnp.concatenate([discounts, jnp.ones((pad,) + discounts.shape[1:], discounts.dtype)])
    bootstrap = jnp.concatenate([bootstrap, jnp.repeat(bootstrap[-1:], pad, axis=0)])
    g = bootstrap[n - 1:n - 1 + t]
    for i in reversed(range(n)):
        g = rewards[i:i + t] + discounts[i:i + t] * g
    return g


class QTrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class LearnerState:
    """Everything the learner owns. emb/rnd/rnd_target are None at stage r2d2."""
    q: QTrainState
    emb: TrainState = None
    rnd: TrainState = None
    rnd_target: flax.core.FrozenDict = None


@flax.struct.dataclass
class ActorState:
    """Everything the NUM_ENVS actors carry between steps."""
    env_state: object
    obs: jnp.ndarray
    carry: tuple                  # (c, h), each (num_heads, E, hidden)
    prev_action: jnp.ndarray      # (E,) int32, -1 = episode start
    prev_reward: jnp.ndarray      # (E,)
    prev_intrinsic: jnp.ndarray   # (E,)
    arms: jnp.ndarray             # (E,) int32
    episode_return: jnp.ndarray   # (E,) running extrinsic return, feeds the bandit
    global_step: jnp.ndarray      # () int32, env steps summed over actors
    memory: object = None         # EpisodicMemory
    rnd_stats: RunningStats = None
    bandit: BanditState = None


class Agent57:
    """Static configuration + pure functions. Nothing here holds array state.

    env_step(env_state, actions) -> (next_obs, env_state, reward, done, info), batched.
    """

    def __init__(self, config, action_dim, obs_shape, pixel_based, env_step=None):
        cfg = {**DEFAULTS, **config, "PIXEL_BASED": bool(pixel_based)}
        self.cfg = cfg
        self.stage = cfg["STAGE"]
        assert self.stage in STAGES, f"unknown STAGE: {self.stage}"
        self.use_intrinsic = self.stage != "r2d2"
        self.split_q = self.stage in ("split_q", "agent57")
        self.meta = self.stage == "agent57"
        self.num_heads = 2 if self.split_q else 1

        self.action_dim = int(action_dim)
        self.obs_shape = tuple(obs_shape)
        self.num_envs = int(cfg["NUM_ENVS"])
        self.num_arms = int(cfg["NUM_ARMS"]) if self.use_intrinsic else 1
        self.burn_in = int(cfg["BURN_IN_LENGTH"])
        self.seq_len = int(cfg["SEQUENCE_LENGTH"])
        self.n_step = int(cfg["N_STEP"])
        self.eps = float(cfg["VALUE_RESCALING_EPSILON"])
        self.env_step = env_step
        self.novelty = NoveltyParams.from_config(cfg)

        if self.use_intrinsic:
            self.arm_betas, self.arm_gammas = arm_schedule(
                self.num_arms, cfg["BETA_MAX"], cfg["GAMMA_MAX"], cfg["GAMMA_MIN"])
        else:
            self.arm_betas = jnp.zeros((1,), jnp.float32)
            self.arm_gammas = jnp.full((1,), cfg["GAMMA"], jnp.float32)

        mlp = dict(mlp_width=int(cfg["MLP_WIDTH"]), mlp_depth=int(cfg["MLP_DEPTH"]))
        self.network = SplitQNetwork(
            num_heads=self.num_heads, action_dim=self.action_dim, pixel_based=pixel_based,
            hidden_size=int(cfg["HIDDEN_SIZE"]), dueling_units=int(cfg["DUELING_UNITS"]),
            num_arms=self.num_arms if self.use_intrinsic else 0, **mlp)
        self.emb_model = EmbeddingTrainer(
            action_dim=self.action_dim, pixel_based=pixel_based,
            embedding_dim=int(cfg["EMBEDDING_DIM"]), **mlp)
        self.rnd_net = RNDNetwork(pixel_based=pixel_based, output_dim=int(cfg["RND_OUTPUT_DIM"]), **mlp)

        self.buffer = make_replay_buffer(
            num_envs=self.num_envs, batch_size=int(cfg["BATCH_SIZE"]), burn_in=self.burn_in,
            seq_len=self.seq_len, period=int(cfg["SEQUENCE_PERIOD"]),
            buffer_size=int(cfg["BUFFER_SIZE"]), learning_starts=int(cfg["LEARNING_STARTS"]),
            priority_exponent=float(cfg["PRIORITY_EXPONENT"]))

    # ---- initialisation ------------------------------------------------------
    def init_learner(self, key, dummy_obs):
        k_q, k_emb, k_rnd, k_tgt = jax.random.split(key, 4)
        obs1 = dummy_obs[None]
        i1, f1 = jnp.zeros((1,), jnp.int32), jnp.zeros((1,), jnp.float32)
        params = self.network.init(k_q, self.network.initial_carry(1), obs1, i1, f1, i1, f1)
        cfg = self.cfg
        tx = optax.adam(cfg["LEARNING_RATE"], b1=cfg.get("ADAM_B1", 0.9),
                        b2=cfg.get("ADAM_B2", 0.999), eps=cfg["ADAM_EPS"])
        # Adam is elementwise, so one optimiser over the stacked (Q_e, Q_i)
        # params is exactly two independent optimisers.
        q = QTrainState.create(apply_fn=self.network.apply, params=params,
                               target_params=jax.tree.map(jnp.copy, params), tx=tx)
        if not self.use_intrinsic:
            return LearnerState(q=q)
        side_lr, side_eps = cfg["NGU_LEARNING_RATE"], cfg["NGU_ADAM_EPS"]
        emb = TrainState.create(
            apply_fn=self.emb_model.apply, params=self.emb_model.init(k_emb, obs1, obs1),
            tx=optax.chain(optax.add_decayed_weights(cfg["EMBEDDING_L2"]), optax.adam(side_lr, eps=side_eps)))
        rnd = TrainState.create(apply_fn=self.rnd_net.apply, params=self.rnd_net.init(k_rnd, obs1),
                                tx=optax.adam(side_lr, eps=side_eps))
        return LearnerState(q=q, emb=emb, rnd=rnd, rnd_target=self.rnd_net.init(k_tgt, obs1))

    def init_actor(self, env_state, obs, arms=None):
        e = self.num_envs
        if arms is None:
            # fixed arm per env for ngu/split_q; bandit rounds start from arm 0 for agent57
            arms = jnp.zeros((e,), jnp.int32) if self.meta else jnp.arange(e, dtype=jnp.int32) % self.num_arms
        zeros = jnp.zeros((e,), jnp.float32)
        return ActorState(
            env_state=env_state, obs=obs, carry=self.network.initial_carry(e),
            prev_action=jnp.full((e,), -1, jnp.int32), prev_reward=zeros, prev_intrinsic=zeros,
            arms=arms, episode_return=zeros, global_step=jnp.array(0, jnp.int32),
            memory=init_episodic_memory(e, int(self.cfg["EPISODIC_MEMORY_SIZE"]), int(self.cfg["EMBEDDING_DIM"]))
            if self.use_intrinsic else None,
            rnd_stats=init_running_stats() if self.use_intrinsic else None,
            bandit=init_bandit(e, int(self.cfg["UCB_WINDOW"])) if self.meta else None,
        )

    def init_buffer(self, dummy_obs):
        return self.buffer.init(dummy_timestep(dummy_obs, self.num_heads, int(self.cfg["HIDDEN_SIZE"])))

    # ---- acting --------------------------------------------------------------
    def policy_q(self, q, arms):
        """Heads (H, B, A) -> the Q the policy acts on, (B, A)."""
        if self.split_q:
            return combine_q(q[0], q[1], self.arm_betas[arms], self.eps)
        return q[0]

    def epsilon(self, global_step):
        cfg = self.cfg
        return jnp.interp(global_step, jnp.array([0.0, cfg["EXPLORATION_FRACTION"] * cfg["TOTAL_TIMESTEPS"]]),
                          jnp.array([cfg["START_E"], cfg["END_E"]]))

    def act_step(self, learner: LearnerState, actor: ActorState, rng):
        """One env step for every actor. Returns (actor, TimeStep (E, ...), info)."""
        k_act, k_exp, k_bandit = jax.random.split(rng, 3)
        carry, q = self.network.apply(learner.q.params, actor.carry, actor.obs, actor.prev_action,
                                      actor.prev_reward, actor.arms, actor.prev_intrinsic)
        greedy = self.policy_q(q, actor.arms).argmax(-1)
        explore = jax.random.uniform(k_exp, greedy.shape) < self.epsilon(actor.global_step)
        actions = jnp.where(explore, jax.random.randint(k_act, greedy.shape, 0, self.action_dim), greedy)

        next_obs, env_state, reward, done, info = self.env_step(actor.env_state, actions)
        reward = reward.astype(jnp.float32)

        memory, rnd_stats = actor.memory, actor.rnd_stats
        if self.use_intrinsic:
            r_int, memory, rnd_stats = compute_intrinsic(
                self.emb_model, self.rnd_net, learner.emb.params, learner.rnd.params, learner.rnd_target,
                memory, rnd_stats, next_obs, done, self.novelty)
        else:
            r_int = jnp.zeros_like(reward)

        timestep = TimeStep(
            obs=actor.obs, action=actions, reward=reward, done=done,
            prev_action=actor.prev_action, prev_reward=actor.prev_reward,
            carry=jax.tree.map(lambda c: jnp.swapaxes(c, 0, 1), actor.carry),   # (E, H, hidden)
            arm=actor.arms, intrinsic_reward=r_int, prev_intrinsic=actor.prev_intrinsic)

        episode_return = actor.episode_return + reward
        arms, bandit = actor.arms, actor.bandit
        if self.meta:
            bandit = bandit_update(bandit, arms, episode_return, done)
            new_arms = bandit_select(bandit, self.num_arms, k_bandit,
                                     self.cfg["UCB_BETA"], self.cfg["UCB_EPSILON"])
            arms = jnp.where(done, new_arms, arms)

        actor = actor.replace(
            env_state=env_state, obs=next_obs, carry=carry,
            prev_action=jnp.where(done, -1, actions), prev_reward=jnp.where(done, 0.0, reward),
            prev_intrinsic=jnp.where(done, 0.0, r_int), arms=arms,
            episode_return=jnp.where(done, 0.0, episode_return),
            global_step=actor.global_step + self.num_envs,
            memory=memory, rnd_stats=rnd_stats, bandit=bandit)
        return actor, timestep, info

    def collect(self, learner, actor, rng, num_steps):
        """num_steps actor steps under lax.scan. Trajectory is (E, T, ...), ready for buffer.add."""
        def step(a, k):
            a, ts, info = self.act_step(learner, a, k)
            return a, (ts, info)
        actor, (traj, infos) = jax.lax.scan(step, actor, jax.random.split(rng, num_steps))
        return actor, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj), infos

    # ---- learning ------------------------------------------------------------
    def unroll(self, params, carry, seq: TimeStep):
        """Scan the network over a time-major sequence -> (carry, q (T, H, B, A))."""
        def step(c, x):
            return self.network.apply(params, c, *x)
        return jax.lax.scan(step, carry, (seq.obs, seq.prev_action, seq.prev_reward, seq.arm, seq.prev_intrinsic))

    def q_loss(self, params, target_params, experience: TimeStep, probabilities):
        """[R2D2 + Agent57] h-rescaled n-step double-Q loss over all heads.

        experience: (B, BURN_IN + SEQ_LEN, ...). Returns
        (loss, (priorities (B,), metrics dict)).
        """
        cfg = self.cfg
        start, burn, learn = split_burn_in(experience, self.burn_in)

        # 1. burn-in from the stored state: forward only, no gradient graph
        online_carry, target_carry = start, start
        if self.burn_in > 0:
            online_carry, _ = self.unroll(jax.lax.stop_gradient(params), start, burn)
            target_carry, _ = self.unroll(target_params, start, burn)

        # 2. Q over the learning segment
        _, q_on = self.unroll(params, online_carry, learn)                       # (T, H, B, A)
        _, q_tg = self.unroll(target_params, target_carry, learn)
        q_tg = jax.lax.stop_gradient(q_tg)

        arm = learn.arm                                                           # (T, B)
        beta = self.arm_betas[arm]
        gamma = self.arm_gammas[arm] if self.use_intrinsic else jnp.full(arm.shape, cfg["GAMMA"], jnp.float32)

        # 3. one target action for every head, from the policy's (combined) Q
        q_select = jax.lax.stop_gradient(q_on) if cfg["DOUBLE_Q"] else q_tg
        if self.split_q:
            q_select = combine_q(q_select[:, 0], q_select[:, 1], beta, self.eps)  # (T, B, A)
        else:
            q_select = q_select[:, 0]
        next_action = jnp.argmax(q_select[1:], -1)                                # (T-1, B)
        next_value = jnp.take_along_axis(q_tg[1:], next_action[:, None, :, None], -1)[..., 0]  # (T-1, H, B)

        # 4. per-head rewards and n-step targets in real-return space
        r_e, r_i = learn.reward[:-1], learn.intrinsic_reward[:-1]
        if self.split_q:
            rewards = jnp.stack([r_e, r_i], axis=1)                              # (T-1, 2, B)
        elif self.use_intrinsic:
            rewards = mixed_reward(r_e, r_i, beta[:-1])[:, None]
        else:
            rewards = r_e[:, None]
        discounts = (gamma[:-1] * (1.0 - learn.done[:-1].astype(jnp.float32)))[:, None]
        discounts = jnp.broadcast_to(discounts, rewards.shape)
        returns = n_step_targets(rewards, discounts, h_inv(next_value, self.eps), self.n_step)
        target = jax.lax.stop_gradient(h(returns, self.eps))

        q_taken = jnp.take_along_axis(q_on[:-1], learn.action[:-1, None, :, None], -1)[..., 0]  # (T-1, H, B)
        td = target - q_taken

        # 5. importance-weighted loss: sum over time and heads per sequence
        weights = importance_weights(probabilities, cfg["IMPORTANCE_SAMPLING_EXPONENT"])
        per_seq_head = 0.5 * jnp.square(td).sum(axis=0)                           # (H, B)
        loss = jnp.mean(weights * per_seq_head.sum(axis=0))

        # 6. [R2D2] sequence priority per head; split heads: p_e + beta_arm * p_i
        prio = sequence_priority(jnp.abs(jnp.swapaxes(td, 0, 1)), cfg["PRIORITY_ETA"])   # (H, T-1, B) -> (H, B)
        priorities = prio[0] + (self.arm_betas[arm[0]] * prio[1] if self.split_q else 0.0)

        head_loss = jnp.mean(weights * per_seq_head, axis=-1)
        metrics = {"td_loss_e": head_loss[0], "q_e": q_taken[:, 0].mean()}
        if self.split_q:
            metrics.update(td_loss_i=head_loss[1], q_i=q_taken[:, 1].mean())
        return loss, (jax.lax.stop_gradient(priorities), metrics)

    def side_update(self, learner: LearnerState, experience: TimeStep):
        """[NGU] one step on the embedding (inverse dynamics) and RND predictor,
        using the last NGU_TRAIN_FRAMES transitions of each sampled sequence."""
        k = int(self.cfg["NGU_TRAIN_FRAMES"])
        window = experience.obs[:, -(k + 1):]
        actions = experience.action[:, -(k + 1):-1]
        mask = 1.0 - experience.done[:, -(k + 1):-1].astype(jnp.float32)
        (emb_l, emb_acc), g = jax.value_and_grad(embedding_loss, has_aux=True)(
            learner.emb.params, self.emb_model, window, actions, mask)
        emb = learner.emb.apply_gradients(grads=g)
        next_obs = window[:, 1:].reshape((-1,) + self.obs_shape)
        rnd_l, g = jax.value_and_grad(rnd_loss)(learner.rnd.params, self.rnd_net, learner.rnd_target, next_obs)
        rnd = learner.rnd.apply_gradients(grads=g)
        return learner.replace(emb=emb, rnd=rnd), {"embedding_loss": emb_l, "embedding_accuracy": emb_acc,
                                                    "rnd_loss": rnd_l}

    def learner_update(self, learner: LearnerState, buffer_state, rng):
        """Sample once; train Q heads, write priorities, sync target, train side nets."""
        batch = self.buffer.sample(buffer_state, rng)
        (loss, (priorities, metrics)), grads = jax.value_and_grad(self.q_loss, has_aux=True)(
            learner.q.params, learner.q.target_params, batch.experience, batch.probabilities)
        q = learner.q.apply_gradients(grads=grads)
        # [R2D2] target <- online every TARGET_NETWORK_FREQUENCY learner steps (Polyak if TAU < 1)
        target = jax.lax.cond(
            q.step % self.cfg["TARGET_NETWORK_FREQUENCY"] == 0,
            lambda: optax.incremental_update(q.params, q.target_params, self.cfg["TAU"]),
            lambda: q.target_params)
        learner = learner.replace(q=q.replace(target_params=target))
        buffer_state = self.buffer.set_priorities(buffer_state, batch.indices, priorities)
        metrics = {"loss": loss, **metrics}
        if self.use_intrinsic:
            learner, side_metrics = self.side_update(learner, batch.experience)
            metrics.update(side_metrics)
        return learner, buffer_state, metrics

    def metric_template(self):
        keys = ["loss", "td_loss_e", "q_e"]
        if self.split_q:
            keys += ["td_loss_i", "q_i"]
        if self.use_intrinsic:
            keys += ["embedding_loss", "embedding_accuracy", "rnd_loss"]
        return {k: jnp.float32(0.0) for k in keys}

    def train_iteration(self, carry, _):
        """collect TRAIN_FREQUENCY steps -> buffer.add -> GRADIENT_STEPS updates once warm.
        carry = (learner, buffer_state, actor, rng). Pure; meant for lax.scan under jit."""
        learner, buffer_state, actor, rng = carry
        rng, k_collect, k_update = jax.random.split(rng, 3)
        actor, traj, infos = self.collect(learner, actor, k_collect, int(self.cfg["TRAIN_FREQUENCY"]))
        buffer_state = self.buffer.add(buffer_state, traj)

        def do_update(learner, buffer_state):
            def one(c, k):
                l, b = c
                l, b, m = self.learner_update(l, b, k)
                return (l, b), m
            (learner, buffer_state), m = jax.lax.scan(
                one, (learner, buffer_state), jax.random.split(k_update, max(1, int(self.cfg["GRADIENT_STEPS"]))))
            return learner, buffer_state, jax.tree.map(lambda x: x[-1].astype(jnp.float32), m)

        def no_update(learner, buffer_state):
            return learner, buffer_state, self.metric_template()

        learner, buffer_state, metrics = jax.lax.cond(
            self.buffer.can_sample(buffer_state), do_update, no_update, learner, buffer_state)
        metrics["intrinsic_reward_mean"] = traj.intrinsic_reward.mean()
        metrics["intrinsic_reward_max"] = traj.intrinsic_reward.max()
        metrics["arm_mean"] = traj.arm.astype(jnp.float32).mean()
        return (learner, buffer_state, actor, rng), (infos, metrics)

    def eval_arm(self, actor: ActorState):
        if self.cfg["EVAL_ARM"] is not None:
            return int(self.cfg["EVAL_ARM"])
        if self.meta:
            return int(greedy_arm(actor.bandit, self.num_arms))
        return 0

    def checkpoint(self, learner: LearnerState, actor: ActorState):
        ckpt = {"config": self.cfg, "q_params": learner.q.params, "eval_arm": self.eval_arm(actor)}
        if self.use_intrinsic:
            ckpt.update(emb_params=learner.emb.params, rnd_params=learner.rnd.params,
                        rnd_target=learner.rnd_target, rnd_stats=actor.rnd_stats)
        return ckpt


# ---- environment -------------------------------------------------------------
def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    """Same wrapper stack as agents/dqn (from feat/infra)."""
    import jaxatari
    from jaxatari.wrappers import (AtariWrapper, FlattenObservationWrapper, LogWrapper,
                                   NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper)
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxatari.make(env_id, mods=mods)
        env = AtariWrapper(env, sticky_actions=0.0, episodic_life=not eval, first_fire=True,
                           noop_max=30, full_action_space=False)
        if pixel_based:
            env = PixelObsWrapper(env, do_pixel_resize=True, pixel_resize_shape=(84, 84), grayscale=True,
                                  use_native_downscaling=native_downscaling, smooth_image=False,
                                  frame_stack_size=4, frame_skip=4, max_pooling=True, clip_reward=not eval)
        else:
            env = FlattenObservationWrapper(NormalizeObservationWrapper(
                ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=not eval)))
        return LogWrapper(env)
    return thunk


def batched_env_fns(env, obs_shape):
    """vmapped reset/step returning observations shaped (E, *obs_shape)."""
    def reset(keys):
        obs, state = jax.vmap(env.reset)(keys)
        return obs.reshape((keys.shape[0],) + obs_shape), state

    def step(state, action):
        obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        return obs.reshape((action.shape[0],) + obs_shape), state, reward, jnp.logical_or(terminated, truncated), info
    return reset, step


class _NoOpRTPT:
    """Stand-in when RTPT cannot start (e.g. sandboxed hosts). From feat/r2d2."""
    def start(self):
        pass

    def step(self, *args, **kwargs):
        pass


def _make_rtpt(config, run_name, max_iterations):
    try:
        from rtpt import RTPT
        rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=max_iterations)
        rtpt.start()
        return rtpt
    except Exception as exc:  # pragma: no cover
        print(f"[agent57] RTPT unavailable ({exc}); continuing without it.")
        return _NoOpRTPT()


# ---- entry point -------------------------------------------------------------
def single_run(config: dict):
    import wandb
    from agents.agent57.agent57_eval import evaluate

    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    config = {**DEFAULTS, **config}
    stage = config["STAGE"]

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    env = make_env(config["ENV_ID"], list(config.get("TRAIN_MODS", [])), config["PIXEL_BASED"],
                   config.get("NATIVE_DOWNSCALING", True), False)()
    action_dim = env.action_space().n
    obs_shape = tuple(env.observation_space().shape)
    if config["PIXEL_BASED"]:
        obs_shape = obs_shape[:-1]   # trailing channel axis is not stored
    vmap_reset, vmap_step = batched_env_fns(env, obs_shape)

    agent = Agent57(config, action_dim, obs_shape, config["PIXEL_BASED"], env_step=vmap_step)
    num_envs = agent.num_envs

    key, reset_key, init_key = jax.random.split(key, 3)
    obs, env_state = jax.jit(vmap_reset)(jax.random.split(reset_key, num_envs))
    learner = agent.init_learner(init_key, obs[0])
    buffer_state = agent.init_buffer(obs[0])
    actor = agent.init_actor(env_state, obs)

    obs_bytes = int(np.prod(obs_shape)) * (1 if config["PIXEL_BASED"] else 4)
    carry_bytes = 2 * agent.num_heads * config["HIDDEN_SIZE"] * 4
    print(f"[agent57] stage={stage} env={config['ENV_ID']} {'pixel' if config['PIXEL_BASED'] else 'oc'} "
          f"heads={agent.num_heads} arms={agent.num_arms}")
    print(f"[agent57] action_dim={action_dim} obs_shape={obs_shape} obs dtype={buffer_state.experience.obs.dtype}")
    print(f"[agent57] buffer: {config['BUFFER_SIZE']} transitions ~ "
          f"{config['BUFFER_SIZE'] * (obs_bytes + carry_bytes) / 1e9:.2f} GB")

    @jax.jit
    def scanned_steps(carry):
        return jax.lax.scan(agent.train_iteration, carry, None, length=config["SCAN_STEPS"])

    run_name = (f"{config['ENV_ID']}_{config.get('EXP_NAME', 'agent57')}_{stage}_"
                f"{'pixel' if config['PIXEL_BASED'] else 'oc'}_{config['SEED']}")
    wandb.init(project=config.get("PROJECT", "jaxtari-blines"), entity=config.get("ENTITY", None),
               config=config, name=run_name, save_code=True, mode=config.get("WANDB_MODE", "online"))
    wandb.define_metric("*", step_metric="charts/global_step")

    steps_per_iteration = num_envs * config["TRAIN_FREQUENCY"] * config["SCAN_STEPS"]
    carry = (learner, buffer_state, actor, key)
    print(f"[agent57] compiling ({config['SCAN_STEPS']} scan steps)...")
    t0 = time.perf_counter()
    # AOT-compiled once; calling the executable (not the jit wrapper) means a
    # carry whose types drift raises instead of silently recompiling
    scanned_steps = scanned_steps.lower(carry).compile()
    print(f"[agent57] compile time: {time.perf_counter() - t0:.1f}s")

    rtpt = _make_rtpt(config, run_name, max(1, config["TOTAL_TIMESTEPS"] // steps_per_iteration))
    global_step, avg_return = 0, float("nan")
    run_start = time.perf_counter()
    while global_step < config["TOTAL_TIMESTEPS"]:
        rtpt.step()
        t_iter = time.perf_counter()
        carry, (infos, metrics) = jax.block_until_ready(scanned_steps(carry))
        global_step = int(carry[2].global_step)
        avg_return = float(infos["returned_episode_returns"][-1].mean())
        avg_length = float(infos["returned_episode_lengths"][-1].mean())
        last = {k: float(v[-1]) for k, v in metrics.items()}
        sps = int(steps_per_iteration / (time.perf_counter() - t_iter))
        print(f"[agent57][{stage}] step {global_step} | return {avg_return:.2f} | length {avg_length:.0f} | "
              + " | ".join(f"{k} {v:.4f}" for k, v in last.items())
              + f" | SPS {sps} | total SPS {int(global_step / (time.perf_counter() - run_start))}")
        wandb.log({"charts/avg_episodic_return": avg_return, "charts/avg_episodic_length": avg_length,
                   "charts/SPS": sps, "charts/global_step": global_step,
                   **{f"agent57/{k}": v for k, v in last.items()}}, step=global_step)

    # --- save and evaluate (unclipped returns, comparable to DQN / Rainbow eval)
    learner, _, actor, _ = carry
    model_path = (f'{config.get("SAVE_PATH", "./models")}/{run_name}/'
                  f'{config["EXP_NAME"]}_{global_step}_{int(time.time())}.agent57_model')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(flax.serialization.to_bytes(agent.checkpoint(learner, actor)))
    print(f"[agent57] model saved to {model_path}")

    episodic_returns, _ = evaluate(
        model_path,
        partial(make_env, mods=list(config.get("TRAIN_MODS", [])), pixel_based=config["PIXEL_BASED"],
                native_downscaling=config.get("NATIVE_DOWNSCALING", True), eval=True),
        config["ENV_ID"], eval_episodes=config.get("EVAL_EPISODES", 10), seed=config["SEED"] + 42)
    eval_return = float(jnp.mean(episodic_returns))
    print(f"[agent57] eval return {eval_return:.2f} (train return {avg_return:.2f})")
    wandb.log({"eval/episodic_return": eval_return}, step=global_step)
    wandb.finish()
    return {"default": eval_return}
