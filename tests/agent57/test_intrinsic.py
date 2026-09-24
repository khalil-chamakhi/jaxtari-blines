"""Episodic + lifelong novelty and reward mixing."""
import jax
import jax.numpy as jnp

from agents.agent57.intrinsic import (
    NoveltyParams, compute_intrinsic, episodic_reward, init_episodic_memory, init_running_stats,
    intrinsic_reward, mixed_reward, rnd_error, rnd_modulator, update_running_stats,
)
from agents.agent57.networks import EmbeddingTrainer, RNDNetwork

E, D, M = 2, 8, 16
P = NoveltyParams(num_neighbours=3)


def test_episodic_reward_finite_nonneg_and_novelty(key):
    mem = init_episodic_memory(E, M, D)
    no_first = jnp.zeros(E, bool)
    rewards = []
    same = jax.random.normal(key, (E, D))
    for t in range(6):
        r, mem = episodic_reward(mem, same, no_first, P)
        rewards.append(r)
    rewards = jnp.stack(rewards)
    assert rewards.shape == (6, E)
    assert bool(jnp.all(jnp.isfinite(rewards))) and bool(jnp.all(rewards >= 0))
    assert bool(jnp.all(rewards[0] == 0))   # empty memory -> 0
    # revisiting the same state keeps getting less novel
    assert bool(jnp.all(rewards[-1] < rewards[1]))
    # a very different embedding is more novel than the repeated one
    r_new, _ = episodic_reward(mem, same + 100.0, no_first, P)
    r_old, _ = episodic_reward(mem, same, no_first, P)
    assert bool(jnp.all(r_new > r_old))
    # first=True wipes that env's memory
    r, mem2 = episodic_reward(mem, same, jnp.array([True, False]), P)
    assert int(mem2.count[0]) == 1 and float(r[0]) == 0.0


def test_lifelong_novelty(key):
    net = RNDNetwork(pixel_based=False, output_dim=8, mlp_width=16, mlp_depth=1)
    obs = jax.random.normal(key, (E, 6))
    k1, k2 = jax.random.split(key)
    err = rnd_error(net, net.init(k1, obs), net.init(k2, obs), obs)
    assert err.shape == (E,) and bool(jnp.all(err >= 0)) and bool(jnp.all(jnp.isfinite(err)))
    stats = update_running_stats(init_running_stats(), err)
    alpha = rnd_modulator(err, stats, 5.0)
    assert bool(jnp.all((alpha >= 1.0) & (alpha <= 5.0)))


def test_mixing_shapes(key):
    r_e = jnp.ones((4, E)); r_i = jnp.full((4, E), 2.0); beta = jnp.array([0.0, 0.5])
    mixed = mixed_reward(r_e, r_i, beta)
    assert mixed.shape == (4, E) and jnp.allclose(mixed[:, 0], 1.0) and jnp.allclose(mixed[:, 1], 2.0)
    assert intrinsic_reward(jnp.ones(E), jnp.full(E, 3.0)).shape == (E,)


def test_compute_intrinsic_pipeline(key):
    emb = EmbeddingTrainer(action_dim=3, pixel_based=False, embedding_dim=D, mlp_width=16, mlp_depth=1)
    rnd = RNDNetwork(pixel_based=False, output_dim=8, mlp_width=16, mlp_depth=1)
    obs = jax.random.normal(key, (E, 6))
    fn = jax.jit(lambda m, s, o, f: compute_intrinsic(emb, rnd, emb.init(key, o, o), rnd.init(key, o),
                                                      rnd.init(jax.random.fold_in(key, 1), o), m, s, o, f, P))
    mem, stats = init_episodic_memory(E, M, D), init_running_stats()
    for t in range(3):
        r, mem, stats = fn(mem, stats, obs + t, jnp.zeros(E, bool))
        assert r.shape == (E,) and bool(jnp.all(jnp.isfinite(r))) and bool(jnp.all(r >= 0))
