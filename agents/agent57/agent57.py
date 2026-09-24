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
# Module layout:
#   networks.py         Q-network (split heads), embedding net, RND
#   replay.py           TimeStep, prioritised sequence buffer, burn-in split
#   intrinsic.py        episodic + lifelong novelty
#   meta_controller.py  arm schedule and UCB bandit
#   agent57.py          (this file) Agent57: act_step / learner_update /
#                       train_iteration, all pure and jit-compatible; single_run.
#   agent57_eval.py     evaluation
import os
import random
import time
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from agents.agent57.intrinsic import (
    NoveltyParams, RunningStats, compute_intrinsic, embedding_loss, init_episodic_memory,
    init_running_stats, mixed_reward, rnd_loss,
)
from agents.agent57.meta_controller import (
    BanditState, arm_schedule, bandit_select, bandit_update, greedy_arm, init_bandit,
)
from agents.agent57.networks import EmbeddingTrainer, RNDNetwork, SplitQNetwork
from agents.agent57.replay import (
    TimeStep, dummy_timestep, importance_weights, make_replay_buffer, sequence_priority,
    split_burn_in,
)

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
