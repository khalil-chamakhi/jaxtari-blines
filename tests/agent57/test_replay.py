"""Prioritised sequence replay: insert, sample (burn-in + sequence), priority update."""
import jax
import jax.numpy as jnp

from agents.agent57.agent57 import (
    TimeStep, dummy_timestep, importance_weights, make_replay_buffer, sequence_priority, split_burn_in,
)

E, OBS, HEADS, HID = 2, 6, 2, 8
BURN, SEQ, BATCH = 2, 4, 2


def make_traj(key, t, t0):
    """Chunk of t steps starting at global time t0; reward encodes 100 * env + time."""
    obs = jax.random.normal(key, (E, t, OBS))
    i = jnp.zeros((E, t), jnp.int32)
    f = 100.0 * jnp.arange(E, dtype=jnp.float32)[:, None] + t0 + jnp.arange(t, dtype=jnp.float32)[None]
    c = jnp.ones((E, t, HEADS, HID))
    return TimeStep(obs=obs, action=i, reward=f, done=jnp.zeros((E, t), bool), prev_action=i, prev_reward=f,
                    carry=(c, c), arm=i, intrinsic_reward=f, prev_intrinsic=f)


def test_insert_sample_and_update_priorities(key):
    buf = make_replay_buffer(num_envs=E, batch_size=BATCH, burn_in=BURN, seq_len=SEQ, period=2,
                             buffer_size=64, learning_starts=0, priority_exponent=0.9)
    state = buf.init(dummy_timestep(jnp.zeros(OBS), HEADS, HID))
    assert not bool(buf.can_sample(state))
    for i in range(3):
        state = buf.add(state, make_traj(jax.random.fold_in(key, i), 4, 4 * i))
    assert bool(buf.can_sample(state))

    batch = buf.sample(state, key)
    exp = batch.experience
    assert exp.obs.shape == (BATCH, BURN + SEQ, OBS)
    assert exp.carry[0].shape == (BATCH, BURN + SEQ, HEADS, HID)
    assert batch.probabilities.shape == (BATCH,)
    # consecutive steps inside a sampled window
    assert bool(jnp.all(jnp.diff(exp.reward, axis=1) == 1.0))

    start, burn, learn = split_burn_in(exp, BURN)
    assert burn.obs.shape == (BURN, BATCH, OBS) and learn.obs.shape == (SEQ, BATCH, OBS)
    assert start[0].shape == (HEADS, BATCH, HID)

    new_p = jnp.array([5.0, 0.1])
    state = buf.set_priorities(state, batch.indices, new_p)
    batch2 = buf.sample(state, jax.random.fold_in(key, 7))
    assert batch2.experience.obs.shape == exp.obs.shape


def test_weights_and_priority():
    w = importance_weights(jnp.array([0.1, 0.5, 0.4]), 0.6)
    assert w.dtype == jnp.float32 and jnp.isclose(w.max(), 1.0) and bool(jnp.all(w > 0))
    td = jnp.array([[1.0, 0.0], [3.0, 0.0]])        # (T=2, B=2)
    p = sequence_priority(td, 0.9)
    assert jnp.allclose(p, jnp.array([0.9 * 3 + 0.1 * 2, 0.0]))
