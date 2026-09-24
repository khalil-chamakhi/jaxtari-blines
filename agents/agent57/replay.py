"""[R2D2] prioritised sequence replay with stored recurrent state and burn-in.

Storage is flashbax's prioritised trajectory buffer (already a dependency on
master): each env's steps are kept in time order, and fixed-length windows of
BURN_IN_LENGTH + SEQUENCE_LENGTH steps starting every SEQUENCE_PERIOD steps
are sampled on device. No per-item Python slicing anywhere.

Every step stores the LSTM carry it was acted with ("stored state"), so a
sampled window can be re-warmed from its real starting state and then burned
in for BURN_IN_LENGTH steps before the loss is taken.
"""
import flashbax as fbx
import flax
import jax
import jax.numpy as jnp


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
