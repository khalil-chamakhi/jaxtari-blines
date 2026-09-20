"""
agents/agent57/agent57.py

CREDIT HEADER
-------------
  Branch: feat/infra, path: agents/agent57/agent57.py
  Commit hash: <FILL IN -- paste the exact commit SHA from
    `git log -1 --format=%H feat/infra -- agents/agent57/agent57.py`
    once you have the branch checked out locally; not available to me
    through the tools used to retrieve this file>
  What I changed vs feat/infra:
    - Torso: pixel branch replaced with the 4-conv-layer spec
      (32/64/128/128, kernels 7/5/5/3, strides 4/2/2/1) instead of the
      3-layer DQN spec that was copied in verbatim (DQN's torso is
      shallower and was flagged as not reusable as-is).
    - Added signed-hyperbolic reward transform + inverse
      (Kapturowski et al., 2019, R2D2).
    - Added RecurrentQNetwork.unroll: stored-state + 40/40 burn-in/trace
      split (Kapturowski et al., 2019, Sec. 3.1).
    - Added TimeStepWithCarry (extends TimeStep with the stored LSTM
      carry so sampled sequences can re-warm from the real collection-time
      state instead of a zero carry).
    - Added R2D2TrainState, create_train_state, r2d2_loss, train_step.
    - Added plain epsilon-greedy acting (not the per-arm bandit -- that
      belongs to the later agent57 STAGE).
    - Extended single_run: STAGE == "r2d2" now actually collects
      rollouts, trains, and updates buffer priorities. Other stages
      (ngu, split_q, agent57) still raise NotImplementedError since
      they depend on modules Aman/Khalil haven't merged yet.
    - Colab smoke-test fixes (2026-09-20): jaxatari's vmapped env.step
      returns (obs, state, reward, terminated, truncated, info) -- a
      6-tuple, not 5; pixel observations carry a trailing singleton
      channel axis that must be squeezed before hitting the conv torso;
      reward must be cast to float32 to match the buffer's dummy dtype;
      r2d2_loss/train_step now take explicit scalar hyperparameters
      instead of the raw config dict, since jax.jit can't trace a dict
      containing strings (e.g. config["ENV_ID"]); RTPT is now optional
      (wrapped so a sandboxed/Colab environment without hostname/user
      permissions can't crash the training loop over a progress bar).

Algorithm references:
  - Kapturowski et al., 2019, "Recurrent Experience Replay in Distributed
    Reinforcement Learning" (R2D2).
  - Badia et al., 2020, "Agent57: Outperforming the Atari Human
    Benchmark" (overall ablation-ladder framing, stage names).
"""
# Agent57 (Badia et al., 2020), built as an ablation ladder.
# STAGE in the config selects how many components are enabled:
# r2d2      Recurrent replay DQN (Kapturowski et al., 2019). LSTM core,
#           sequence replay with burn-in, h-transformed n-step loss,
#           prioritised replay. Exploration is plain epsilon-greedy.
#
# ngu       Never Give Up (Badia et al., 2020). Adds an intrinsic reward
#           from episodic novelty (k-NN in a learned embedding space)
#           multiplied by a lifelong novelty modulator (RND). Trains a
#           family of NUM_ARMS policies, each with its own (beta, gamma).
#
# split_q   Separate Q_e and Q_i networks, acting on Q_e + beta_j * Q_i.
#           Extrinsic and intrinsic rewards differ by orders of magnitude,
#           and one shared network is unstable.
#
# agent57   A sliding-window UCB bandit selects which arm to act with each
#           episode, so the exploration/exploitation trade-off is learned
#           per game rather than fixed.
#
# Each stage is a checkpoint: it must train before the next is enabled.
import os
import random
import time
from functools import partial
from typing import Any, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flashbax as fbx
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper,
)

# from agents.agent57.agent57_eval import evaluate # TODO: enable once eval is written
try:
    from rtpt import RTPT
except ImportError:
    RTPT = None


class _NoOpRTPT:
    """Fallback used when rtpt is unavailable or can't initialise (e.g. a
    sandboxed Colab environment without hostname/user permissions). Keeps
    single_run's control flow identical either way."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def step(self, *args, **kwargs):
        pass


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxatari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )
        env = LogWrapper(env)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Torso
# ---------------------------------------------------------------------------
class Torso(nn.Module):
    """Observation -> flat feature vector.

    Width 512, depth 2 for the OC/MLP branch, from Group 27's sweep (Raban
    confirmed in channel).

    pixel [R2D2] Table 2 architecture: 4-layer conv stack, DEEPER than the
    3-layer "same as DQN" torso from dqn.py. dqn.py's QNetwork used
    32/64/64, kernels 8/4/3, strides 4/2/1 -- that spec is NOT reused here;
    R2D2's own torso is 32/64/128/128, kernels 7/5/5/3, strides 4/2/2/1.
    4 stacked 84x84 frames in.
    """

    pixel_based: bool
    mlp_width: int = 512
    mlp_depth: int = 2

    @nn.compact
    def __call__(self, x):
        if self.pixel_based:
            # observations are stored as (batch, channels, height, width);
            # flax's Conv expects (batch, height, width, channels)
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x.astype(jnp.float32) / 255.0  # network expects float32 in [0,1]

            # R2D2-specific 4-layer torso (deeper than DQN's 3-layer stack).
            conv_specs = [(32, 7, 4), (64, 5, 2), (128, 5, 2), (128, 3, 1)]
            for features, kernel, stride in conv_specs:
                x = nn.Conv(
                    features,
                    kernel_size=(kernel, kernel),
                    strides=(stride, stride),
                    padding="VALID",
                )(x)
                x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
        else:
            for _ in range(self.mlp_depth):
                x = nn.Dense(
                    self.mlp_width,
                    kernel_init=orthogonal(np.sqrt(2.0)),
                    bias_init=constant(0.0),
                )(x)
                x = nn.relu(x)
        return x


# ---------------------------------------------------------------------------
# RecurrentQNetwork
# ---------------------------------------------------------------------------
class RecurrentQNetwork(nn.Module):
    """Torso -> LSTM -> dueling head. The full R2D2 Q-network.

    [R2D2] Table 2: torso, then LSTM(512), then dueling value and advantage
    heads each with a 512 hidden layer. The LSTM also receives the previous
    reward and a one-hot of the previous action.

    Recurrent, so the signature carries state:
        new_carry, q_values = net(carry, obs, prev_action, prev_reward)
    """

    action_dim: int
    pixel_based: bool
    hidden_size: int = 512
    dueling_units: int = 512
    mlp_width: int = 512
    mlp_depth: int = 2

    @nn.compact
    def __call__(self, carry, obs, prev_action, prev_reward):
        x = Torso(
            pixel_based=self.pixel_based, mlp_width=self.mlp_width, mlp_depth=self.mlp_depth
        )(obs)

        # [R2D2] the LSTM input carries the previous action and reward.
        x = jnp.concatenate(
            [x, jax.nn.one_hot(prev_action, self.action_dim), prev_reward[:, None]], axis=-1
        )

        carry, x = nn.OptimizedLSTMCell(self.hidden_size)(carry, x)

        # Dueling head: V says how good the state is, A says how much better
        # each action is than average.
        v = nn.Dense(self.dueling_units)(x)
        v = nn.relu(v)
        v = nn.Dense(1)(v)

        a = nn.Dense(self.dueling_units)(x)
        a = nn.relu(a)
        a = nn.Dense(self.action_dim)(a)

        q = v + (a - a.mean(axis=-1, keepdims=True))
        return carry, q

    @staticmethod
    def initial_carry(batch_size, hidden_size=512):
        """Zeroed (c, h). An LSTM carries two vectors, not one."""
        return nn.OptimizedLSTMCell(hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )

    def unroll(self, params, carry, obs_seq, prev_action_seq, prev_reward_seq, burn_in_length):
        """[R2D2 -- ADDED] Sequence unroll with stored-state + burn-in.

        `carry` must be the STORED carry captured at collection time for
        the first timestep of this sequence (Kapturowski et al., 2019,
        Sec. 3.1), not a zero-initialised one. obs_seq / prev_action_seq /
        prev_reward_seq are (batch, time, ...).

        The first `burn_in_length` steps only refresh the recurrent state
        (under stop_gradient); the remaining steps are used for the loss.
        Returns q-values of shape (batch, time, action_dim); the burn-in
        region is zero-filled and must be masked out by the caller.
        """

        def to_time_major(t):
            return jnp.swapaxes(t, 0, 1)

        obs_tm = to_time_major(obs_seq)
        act_tm = to_time_major(prev_action_seq)
        rew_tm = to_time_major(prev_reward_seq)

        def step_fn(carry, inputs):
            obs_t, act_t, rew_t = inputs
            new_carry, q_t = self.apply(params, carry, obs_t, act_t, rew_t)
            return new_carry, q_t

        total_len = obs_tm.shape[0]

        if burn_in_length > 0:
            for t in range(burn_in_length):
                carry, _ = step_fn(carry, (obs_tm[t], act_tm[t], rew_tm[t]))
            carry = jax.tree_util.tree_map(jax.lax.stop_gradient, carry)

        q_trace = []
        for t in range(burn_in_length, total_len):
            carry, q_t = step_fn(carry, (obs_tm[t], act_tm[t], rew_tm[t]))
            q_trace.append(q_t)
        q_trace = jnp.stack(q_trace, axis=0)

        if burn_in_length > 0:
            pad = jnp.zeros((burn_in_length,) + q_trace.shape[1:], dtype=q_trace.dtype)
            q_full = jnp.concatenate([pad, q_trace], axis=0)
        else:
            q_full = q_trace
        return jnp.swapaxes(q_full, 0, 1)


@flax.struct.dataclass
class TimeStep:
    obs: jnp.ndarray  # observation BEFORE acting. OC: (104,) float32. Pixel: (4,84,84) uint8.
    action: jnp.ndarray  # action index, 0..action_dim-1
    reward: jnp.ndarray  # extrinsic reward, clipped during training
    done: jnp.ndarray  # episode ended here; zeroes the future term in the Q target

    prev_action: jnp.ndarray  # [R2D2] the LSTM input carries the previous
    prev_reward: jnp.ndarray  # action and reward. Must be STORED: a sampled
    # sequence has no access to the step before it.

    arm: jnp.ndarray  # which of the 8 (beta, gamma) policies acted (NGU/agent57 stage; 0 for r2d2)
    intrinsic_reward: jnp.ndarray  # curiosity reward for this step (NGU/agent57 stage; 0 for r2d2)


@flax.struct.dataclass
class TimeStepWithCarry:
    """[R2D2 -- ADDED] TimeStep + the stored LSTM carry at this timestep.

    This is what actually goes into the replay buffer and comes back out
    of buffer.sample(). Kept as a separate struct rather than editing the
    shared TimeStep dataclass, since not every stage needs a carry field
    (flag this in review if you'd rather fold it into TimeStep directly --
    that touches everyone's sampling code, not just R2D2's).
    """

    timestep: TimeStep
    stored_carry_h: jnp.ndarray
    stored_carry_c: jnp.ndarray


# ---------------------------------------------------------------------------
# [R2D2 -- ADDED] reward transform (used instead of reward clipping in the loss)
# ---------------------------------------------------------------------------
def signed_hyperbolic(x: jnp.ndarray, eps: float = 1e-3) -> jnp.ndarray:
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + eps * x


def signed_hyperbolic_inv(x: jnp.ndarray, eps: float = 1e-3) -> jnp.ndarray:
    sign = jnp.sign(x)
    z = jnp.sqrt(1.0 + 4.0 * eps * (jnp.abs(x) + 1.0 + eps)) / (2.0 * eps) - 1.0 / (2.0 * eps)
    return sign * (jnp.square(z) - 1.0)


# ---------------------------------------------------------------------------
# [R2D2 -- ADDED] train state, loss, train step
# ---------------------------------------------------------------------------
class R2D2TrainState(TrainState):
    target_params: Any


def create_train_state(config: dict, action_dim: int, obs_shape, pixel_based: bool, rng):
    net = RecurrentQNetwork(
        action_dim=action_dim,
        pixel_based=pixel_based,
        hidden_size=config.get("LSTM_HIDDEN_SIZE", 512),
        dueling_units=config.get("DUELING_UNITS", 512),
        mlp_width=config.get("MLP_WIDTH", 512),
        mlp_depth=config.get("MLP_DEPTH", 2),
    )
    dummy_obs = jnp.zeros(
        (1,) + tuple(obs_shape), dtype=jnp.uint8 if pixel_based else jnp.float32
    )
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)
    dummy_reward = jnp.zeros((1,), dtype=jnp.float32)
    carry0 = net.initial_carry(1, hidden_size=config.get("LSTM_HIDDEN_SIZE", 512))
    params = net.init(rng, carry0, dummy_obs, dummy_action, dummy_reward)

    tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 40.0)),
        optax.adam(
            learning_rate=config.get("LEARNING_RATE", 1e-4),
            eps=config.get("ADAM_EPS", 1e-3),
        ),
    )
    state = R2D2TrainState.create(apply_fn=net.apply, params=params, target_params=params, tx=tx)
    return state, net


def r2d2_loss(
    net: RecurrentQNetwork,
    params,
    target_params,
    batch: TimeStepWithCarry,
    burn_in: int,
    seq_len: int,
    n_step: int,
    gamma: float,
):
    """[R2D2 -- ADDED] Transformed n-step double-Q loss, burn-in masked out.

    `batch` is a TimeStepWithCarry whose fields are (batch, seq_len, ...),
    sampled from the flashbax prioritised trajectory buffer. Hyperparameters
    are passed explicitly (not via a config dict) so this can be called
    from inside a jax.jit-wrapped train_step -- a dict containing strings
    (e.g. ENV_ID) can't be traced as an array.
    """
    ts = batch.timestep
    carry0 = (batch.stored_carry_h[:, 0], batch.stored_carry_c[:, 0])

    q_online = net.unroll(params, carry0, ts.obs, ts.prev_action, ts.prev_reward, burn_in)
    q_target = net.unroll(target_params, carry0, ts.obs, ts.prev_action, ts.prev_reward, burn_in)
    q_online_stopgrad = jax.lax.stop_gradient(q_online)

    trace = slice(burn_in, seq_len)
    actions_t = ts.action[:, trace]
    rewards_t = ts.reward[:, trace]
    dones_t = ts.done[:, trace].astype(jnp.float32)

    q_sa = jnp.take_along_axis(q_online[:, trace], actions_t[..., None], axis=-1).squeeze(-1)

    next_actions = jnp.argmax(q_online_stopgrad[:, trace], axis=-1)
    q_next_target = jnp.take_along_axis(
        q_target[:, trace], next_actions[..., None], axis=-1
    ).squeeze(-1)

    rewards_transformed = signed_hyperbolic(rewards_t)
    bootstrap = signed_hyperbolic_inv(q_next_target)
    target_raw = rewards_transformed + (gamma ** n_step) * (1.0 - dones_t) * bootstrap
    target = jax.lax.stop_gradient(signed_hyperbolic(target_raw))

    td_error = target - q_sa
    loss = jnp.mean(jnp.square(td_error))
    priorities = 0.9 * jnp.max(jnp.abs(td_error), axis=1) + 0.1 * jnp.mean(jnp.abs(td_error), axis=1)
    return loss, priorities


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
def train_step(
    net: RecurrentQNetwork,
    state: R2D2TrainState,
    batch: TimeStepWithCarry,
    burn_in: int,
    seq_len: int,
    n_step: int,
    gamma: float,
):
    grad_fn = jax.value_and_grad(
        lambda p: r2d2_loss(net, p, state.target_params, batch, burn_in, seq_len, n_step, gamma),
        has_aux=True,
    )
    (loss, priorities), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, priorities


def maybe_sync_target(state: R2D2TrainState, grad_step: int, target_update_interval: int):
    should_update = (grad_step % target_update_interval) == 0
    new_target = jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(should_update, new, old), state.params, state.target_params
    )
    return state.replace(target_params=new_target)


# ---------------------------------------------------------------------------
# [R2D2 -- ADDED] plain epsilon-greedy acting (NOT the per-arm bandit schedule)
# ---------------------------------------------------------------------------
def epsilon_at(step: int, config: dict) -> float:
    start = config.get("EPSILON_START", 1.0)
    end = config.get("EPSILON_END", 0.01)
    decay_steps = config.get("EPSILON_DECAY_STEPS", 1_000_000)
    frac = min(1.0, step / max(1, decay_steps))
    return start + frac * (end - start)


def select_action(net, params, carry, obs, prev_action, prev_reward, epsilon, rng):
    new_carry, q = net.apply(params, carry, obs, prev_action, prev_reward)
    greedy = jnp.argmax(q, axis=-1)
    rng, k1, k2 = jax.random.split(rng, 3)
    random_actions = jax.random.randint(k1, greedy.shape, 0, q.shape[-1])
    explore = jax.random.uniform(k2, greedy.shape) < epsilon
    action = jnp.where(explore, random_actions, greedy)
    return new_carry, action, rng


# ---------------------------------------------------------------------------
# single_run
# ---------------------------------------------------------------------------
def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    stage = config.get("STAGE", "r2d2")
    assert stage in ("r2d2", "ngu", "split_q", "agent57"), f"unknown STAGE: {stage}"

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    env = make_env(
        config["ENV_ID"],
        list(config.get("TRAIN_MODS", [])),
        config["PIXEL_BASED"],
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape

    # Pixel observations arrive with a trailing channel dimension that the
    # buffer does not store.
    if config["PIXEL_BASED"]:
        obs_shape = obs_shape[:-1]

    # Getting this dtype wrong is the classic silent failure: a uint8 buffer
    # rounds a normalised 0.47 to 0 and the agent trains on nothing.
    obs_bytes = int(np.prod(obs_shape)) * (1 if config["PIXEL_BASED"] else 4)
    buffer_gb = config["BUFFER_SIZE"] * obs_bytes / 1e9

    # --- replay buffer -----------------------------------------------------
    num_envs = config["NUM_ENVS"]
    seq_len = config["SEQUENCE_LENGTH"]

    replay_buffer = fbx.make_prioritised_trajectory_buffer(
        add_batch_size=num_envs,
        sample_batch_size=config["BATCH_SIZE"],  # 64 sequences per update
        sample_sequence_length=seq_len,  # [R2D2] m = 80
        period=config["SEQUENCE_PERIOD"],  # stride between starts -> overlap
        min_length_time_axis=seq_len,  # refuse to sample before this exists
        max_length_time_axis=config["BUFFER_SIZE"] // num_envs,
        priority_exponent=config["PRIORITY_EXPONENT"],
    )

    key, reset_key = jax.random.split(key)
    dummy_obs, _ = env.reset(reset_key)
    dummy_obs = dummy_obs.reshape(obs_shape)

    # [R2D2 -- ADDED] stored LSTM carry travels alongside each timestep so
    # a sampled sequence can re-warm from the state it actually had at
    # collection time (Kapturowski Sec. 3.1), not a zero carry.
    lstm_hidden = config.get("LSTM_HIDDEN_SIZE", 512)
    dummy_timestep = TimeStep(
        obs=dummy_obs,
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        done=jnp.zeros((), dtype=jnp.bool_),
        prev_action=jnp.zeros((), dtype=jnp.int32),
        prev_reward=jnp.zeros((), dtype=jnp.float32),
        arm=jnp.zeros((), dtype=jnp.int32),
        intrinsic_reward=jnp.zeros((), dtype=jnp.float32),
    )
    dummy_timestep_with_carry = TimeStepWithCarry(
        timestep=dummy_timestep,
        stored_carry_h=jnp.zeros((lstm_hidden,), dtype=jnp.float32),
        stored_carry_c=jnp.zeros((lstm_hidden,), dtype=jnp.float32),
    )

    buffer_state = replay_buffer.init(dummy_timestep_with_carry)

    print(f"[agent57] stage={stage} env={config['ENV_ID']} " f"{'pixel' if config['PIXEL_BASED'] else 'oc'}")
    print(f"[agent57] action_dim={action_dim} obs_shape={obs_shape} " f"obs_bytes={obs_bytes}")
    print(f"[agent57] buffer: {config['BUFFER_SIZE']} transitions = {buffer_gb:.2f} GB")

    if stage != "r2d2":
        raise NotImplementedError(
            f"STAGE={stage} depends on NGU/episodic-memory/UCB-bandit modules "
            "that live on other branches (Aman: feat/ngu). Merge those first, "
            "then wire the arm-selection + intrinsic-reward path in here."
        )

    # --- [R2D2 -- ADDED] training loop -------------------------------------
    key, net_key = jax.random.split(key)
    state, net = create_train_state(config, action_dim, obs_shape, config["PIXEL_BASED"], net_key)

    reset_keys = jax.random.split(key, num_envs)
    obs, env_state = jax.vmap(lambda k: env.reset(k))(reset_keys)
    if config["PIXEL_BASED"]:
        obs = obs.squeeze(-1)
    carry = net.initial_carry(num_envs, hidden_size=lstm_hidden)
    prev_action = jnp.zeros((num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((num_envs,), dtype=jnp.float32)

    total_timesteps = config["TOTAL_TIMESTEPS"]
    target_update_interval = config.get("TARGET_UPDATE_INTERVAL", 2500)
    train_every = config.get("TRAIN_EVERY", 4)
    log_every = config.get("LOG_EVERY", 100)
    burn_in_length = config["BURN_IN_LENGTH"]
    n_step = config.get("N_STEP", 5)
    gamma = config.get("GAMMA", 0.997)

    rtpt_cls = RTPT if RTPT is not None else _NoOpRTPT
    try:
        rtpt = rtpt_cls(
            name_initials=config.get("RTPT_INITIALS", "PM"),
            experiment_name="agent57_r2d2",
            max_iterations=total_timesteps,
        )
        rtpt.start()
    except Exception as exc:  # pragma: no cover -- sandboxed envs may lack hostname/user perms
        print(f"[agent57] RTPT unavailable ({exc}); continuing without progress tracking.")
        rtpt = _NoOpRTPT()

    grad_step = 0
    act_rng = key
    for step in range(total_timesteps):
        stored_h, stored_c = carry
        eps = epsilon_at(step, config)
        new_carry, action, act_rng = select_action(
            net, state.params, carry, obs, prev_action, prev_reward, eps, act_rng
        )
        next_obs, env_state, reward, terminated, truncated, info = jax.vmap(
            lambda s, a: env.step(s, a)
        )(env_state, action)
        done = jnp.logical_or(terminated, truncated)
        if config["PIXEL_BASED"]:
            next_obs = next_obs.squeeze(-1)

        timestep = TimeStep(
            obs=obs,
            action=action,
            reward=reward.astype(jnp.float32),
            done=done,
            prev_action=prev_action,
            prev_reward=prev_reward,
            arm=jnp.zeros((num_envs,), dtype=jnp.int32),
            intrinsic_reward=jnp.zeros((num_envs,), dtype=jnp.float32),
        )
        # add a singleton time axis: (num_envs, 1, ...)
        timestep_with_carry = TimeStepWithCarry(
            timestep=jax.tree_util.tree_map(lambda x: x[:, None, ...], timestep),
            stored_carry_h=stored_h[:, None, :],
            stored_carry_c=stored_c[:, None, :],
        )
        buffer_state = replay_buffer.add(buffer_state, timestep_with_carry)

        carry = new_carry
        prev_action = action
        prev_reward = reward.astype(jnp.float32)
        obs = next_obs

        if replay_buffer.can_sample(buffer_state) and step % train_every == 0:
            act_rng, sample_key = jax.random.split(act_rng)
            sample = replay_buffer.sample(buffer_state, sample_key)
            batch: TimeStepWithCarry = sample.experience
            state, loss, priorities = train_step(
                net, state, batch, burn_in_length, seq_len, n_step, gamma
            )
            state = maybe_sync_target(state, grad_step, target_update_interval)
            buffer_state = replay_buffer.set_priorities(buffer_state, sample.indices, priorities)
            grad_step += 1

            if grad_step % log_every == 0:
                print(
                    f"[agent57][r2d2] step={step} grad_step={grad_step} "
                    f"loss={float(loss):.4f} eps={eps:.3f}"
                )
                if config.get("USE_WANDB", False):
                    wandb.log({"loss": float(loss), "epsilon": eps, "step": step})

        rtpt.step()

    # main.py expects a dict back from every agent. nan means "no score yet";
    # returning 0 would look like a real score of zero.
    return {"default": float("nan")}
