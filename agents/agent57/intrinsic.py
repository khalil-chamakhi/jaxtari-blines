"""[NGU] intrinsic reward: episodic novelty x lifelong novelty.

    r_i = r_episodic * clip(alpha, 1, L)
    r   = r_e + beta_j * r_i        (single-network stages)

r_episodic  k-NN over the controllable-state embeddings seen so far in the
            current episode (NGU Algorithm 1). Batched over envs, no host code.
alpha       RND prediction error, normalised by a running mean/std.

Logic from feat/ngu. Changes: the k-NN distances use the matmul expansion
|a-b|^2 = |a|^2 + |b|^2 - 2ab (no (E, M, D) intermediate); hyperparameters
are a NamedTuple of Python floats instead of the whole config dict; the
embedding/RND updates are fused into one function that embeds each frame once.
"""
from typing import NamedTuple

import flax
import jax
import jax.numpy as jnp


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
