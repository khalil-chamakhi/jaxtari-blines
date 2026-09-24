"""Shared fixtures: a tiny pure-JAX env and tiny Agent57 configs (CPU, seconds)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax
import jax
import jax.numpy as jnp
import pytest

OBS_DIM = 6
ACTION_DIM = 3
EPISODE_LEN = 7


@flax.struct.dataclass
class DummyState:
    t: jnp.ndarray
    ret: jnp.ndarray
    key: jnp.ndarray


def _obs(state):
    return jax.random.normal(jax.random.fold_in(state.key, state.t), (OBS_DIM,))


def dummy_reset(keys):
    """Batched reset: keys (E, 2) -> obs (E, OBS_DIM), state."""
    state = DummyState(t=jnp.zeros(keys.shape[0], jnp.int32), ret=jnp.zeros(keys.shape[0]), key=keys)
    return jax.vmap(_obs)(state), state


def dummy_step(state, action):
    """Batched, auto-resetting step. Reward 1 for action 0. Episodes last EPISODE_LEN steps."""
    reward = (action == 0).astype(jnp.float32)
    t = state.t + 1
    done = t >= EPISODE_LEN
    ret = state.ret + reward
    info = {"returned_episode_returns": jnp.where(done, ret, 0.0),
            "returned_episode_lengths": jnp.where(done, t, 0)}
    new_keys = jax.vmap(lambda k: jax.random.split(k)[0])(state.key)
    state = DummyState(t=jnp.where(done, 0, t), ret=jnp.where(done, 0.0, ret),
                       key=jnp.where(done[:, None], new_keys, state.key))
    return jax.vmap(_obs)(state), state, reward, done, info


def tiny_config(stage="agent57", **overrides):
    cfg = dict(
        STAGE=stage, NUM_ENVS=2, BATCH_SIZE=2, SEQUENCE_LENGTH=4, BURN_IN_LENGTH=2, SEQUENCE_PERIOD=2,
        N_STEP=2, BUFFER_SIZE=64, LEARNING_STARTS=0, PRIORITY_EXPONENT=0.9, PRIORITY_ETA=0.9,
        IMPORTANCE_SAMPLING_EXPONENT=0.6, VALUE_RESCALING_EPSILON=1e-3, GAMMA=0.99,
        TARGET_NETWORK_FREQUENCY=2, TRAIN_FREQUENCY=4, GRADIENT_STEPS=1, SCAN_STEPS=2,
        LEARNING_RATE=1e-3, ADAM_EPS=1e-3, HIDDEN_SIZE=8, DUELING_UNITS=8, MLP_WIDTH=16, MLP_DEPTH=1,
        START_E=1.0, END_E=0.01, EXPLORATION_FRACTION=0.1, TOTAL_TIMESTEPS=1000,
        EMBEDDING_DIM=8, EPISODIC_MEMORY_SIZE=16, NUM_NEIGHBOURS=3, RND_OUTPUT_DIM=8, NGU_TRAIN_FRAMES=2,
        NUM_ARMS=4, BETA_MAX=0.3, GAMMA_MAX=0.99, GAMMA_MIN=0.9, UCB_WINDOW=5, UCB_BETA=1.0, UCB_EPSILON=0.5,
    )
    cfg.update(overrides)
    return cfg


def make_agent(stage="agent57", **overrides):
    from agents.agent57.agent57 import Agent57
    return Agent57(tiny_config(stage, **overrides), ACTION_DIM, (OBS_DIM,), pixel_based=False,
                   env_step=dummy_step)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)
