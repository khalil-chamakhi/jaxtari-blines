# Agent57 (Badia et al., 2020), built as an ablation ladder.
# STAGE in the config selects how many components are enabled:
#   r2d2     Recurrent replay DQN (Kapturowski et al., 2019). LSTM core,
#            sequence replay with burn-in, h-transformed n-step loss,
#            prioritised replay. Exploration is plain epsilon-greedy.
#
#   ngu      Never Give Up (Badia et al., 2020). Adds an intrinsic reward
#            from episodic novelty (k-NN in a learned embedding space) 
#            multiplied by a lifelong novelty modulator (RND). Trains a
#            family of NUM_ARMS policies, each with its own (beta, gamma).
#
#   split_q  Separate Q_e and Q_i networks, acting on Q_e + beta_j * Q_i.
#            Extrinsic and intrinsic rewards differ by orders of magnitude,
#            and one shared network is unstable.
#
#   agent57  A sliding-window UCB bandit selects which arm to act with each
#            episode, so the exploration/exploitation trade-off is learned
#            per game rather than fixed.
#
# Each stage is a checkpoint: it must train before the next is enabled.
import os
import random
import time
from functools import partial

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
    FlattenObservationWrapper
)
# from agents.agent57.agent57_eval import evaluate   # TODO: enable once eval is written
from rtpt import RTPT
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
#----torso--------------------------------------------------------------
class Torso(nn.Module):
    """Observation -> flat feature vector.
    Width 512, depth 2, from Group 27's sweep (Raban confirmed in channel).
   pixel  [R2D2] Table 2: "the same 3-layer convolutional network as DQN".
   Copied verbatim from dqn.py's QNetwork: 32/64/64, kernels 8/4/3,
   strides 4/2/1, padding VALID. 4 stacked 84x84 frames -> 3136 features (7x7x64).
    """
    pixel_based: bool 
    mlp_width: int = 512
    mlp_depth: int = 2
    @nn.compact
    def __call__(self, x):
        if self.pixel_based:
            #observations are stored in this format:(batch, channels, height, width)and flax's Conv expect (batch, height, width, channels)
            x = jnp.transpose(x, (0, 2, 3, 1))
            x= x.astype(jnp.float32) / 255.0 # the netork expects float32 inputs in [0,1]
            x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
        else:
            for i in range(self.mlp_depth):
                x = nn.Dense(self.mlp_width, kernel_init=orthogonal(np.sqrt(2.0)),bias_init=constant(0.0))(x)
                x = nn.relu(x)
        return x

#----RecurrentQnetwork--------------------------------------------------------------
class RecurrentQNetwork(nn.Module):
    """Torso -> LSTM -> dueling head. The full R2D2 Q-network.

    [R2D2] Table 2: torso, then LSTM(512), then dueling value and advantage
    heads each with a 512 hidden layer. The LSTM also receives the previous
    reward and a one-hot of the previous action.

    Recurrent, so the signature carries state:new_carry, q_values = net(carry, obs, prev_action, prev_reward)
    """
    
    action_dim: int
    pixel_based: bool
    hidden_size: int = 512
    dueling_units: int = 512
    mlp_width: int = 512
    mlp_depth: int = 2

    @nn.compact
    def __call__(self, carry, obs, prev_action, prev_reward):
        first = (prev_action < 0)[:, None]
        carry = jax.tree.map(lambda c: jnp.where(first, 0.0, c), carry)
        x = Torso(pixel_based=self.pixel_based,mlp_width=self.mlp_width,mlp_depth=self.mlp_depth)(obs)

        # [R2D2] the LSTM input carries the previous action and reward.
        x = jnp.concatenate([
            x,
            jax.nn.one_hot(prev_action, self.action_dim),
            prev_reward[:, None],
        ], axis=-1)

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



@flax.struct.dataclass
class TimeStep:
   
    obs: jnp.ndarray            # observation BEFORE acting OC: (104,) float32. Pixel: (4,84,84) uint8.
    action: jnp.ndarray         # action index, 0..action_dim-1
    reward: jnp.ndarray         # extrinsic reward, clipped during training
    done: jnp.ndarray           # episode ended here; zeroes the future term  in the Q target

    prev_action: jnp.ndarray    # [R2D2] the LSTM input carries the previous
    prev_reward: jnp.ndarray    #   action and reward, not just the observation. Must be STORED: a sampled sequence has no access to the step before it.

    carry: tuple                     # the LSTM state at this step, stored in the buffer for burn-in 


    arm: jnp.ndarray            # which of the 8 (beta, gamma) policies acted
    intrinsic_reward: jnp.ndarray   # curiosity reward for this step


#---- R2D2 loss ---------------------------------------------------------------
""" Logic follows Acme's R2D2 learner and rlax's transformed n-step Q-learning 
acme:  google-deepmind/acme, acme/agents/jax/r2d2/learning.py @ 89080fe  (Apache-2.0)
rlax:  google-deepmind/rlax, rlax/_src/multistep.py, transforms.py  (Apache-2.0)
Changed: plain JAX instead of rlax; stop_gradient on the burn-in state
(Acme lets gradients flow into the burn-in unroll)."""
def h(x, eps):
    """[R2D2]value rescaling : h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x"""
    return jnp.sign(x)*(jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + eps * x
def h_inv(x, eps):
    """[R2D2] Inverse value rescaling:h^-1(x) = sign(x) * (((sqrt(1 + 4*eps*(|x| + 1 + eps)) - 1) / (2*eps))^2 - 1 )
    """
    a= jnp.sqrt(1.0 + 4.0 * eps * (jnp.abs(x) + 1.0 + eps)) - 1.0
    b= 2.0 * eps
    return jnp.sign(x) * (jnp.square(a / b) - 1.0)
def n_step_targets(rewards, discounts,bootstrap , n ):
    """  G_t = r_t + d_t r_{t+1} + ... + d_t...d_{t+n-1} * bootstrap[t+n-1]
    The last n-1 steps have fewer rewards left and bootstrap from bootstrap[-1]. """
    t= rewards.shape[0]
    pad= n-1 
    rewards= jnp.concatenate([rewards, jnp.zeros((pad,)+ rewards.shape[1:], dtype=rewards.dtype)])
    discounts= jnp.concatenate([discounts, jnp.ones((pad,)+ discounts.shape[1:], dtype=discounts.dtype)])
    bootstrap= jnp.concatenate([bootstrap, jnp.repeat(bootstrap[-1:], pad, axis=0)])
    g= bootstrap[n-1:n-1+t]
    for i in reversed(range(n)):
        g= rewards[i:i+t] + discounts[i:i+t] * g
    return g
        

def unroll(network, params, carry, seq):
    """Run the network over a time-first sequence. Returns (last carry, q of shape (T, B, A))."""
    def step(carry, x):
        obs, prev_action, prev_reward = x
        return network.apply(params, carry, obs, prev_action, prev_reward)
    return jax.lax.scan(step, carry, (seq.obs, seq.prev_action, seq.prev_reward))

def r2d2_loss(params, target_params, batch , probabilities, network, cfg):
    """batch: TimeStep with arrays shaped (B, BURN_IN + SEQ_LEN, ...).
    probabilities: (B,) sampling probabilities from the prioritised buffer.
    Returns loss and (new priorities (B,), mean Q of taken actions)."""
    burn_in = cfg["BURN_IN_LENGTH"]
    eps = cfg["VALUE_RESCALING_EPSILON"]
 
    data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), batch)      # (L, B, ...)
    start = jax.tree.map(lambda c: c[0], data.carry)                 # stored state
    burn = jax.tree.map(lambda x: x[:burn_in], data)
    learn = jax.tree.map(lambda x: x[burn_in:], data)
 
    # 1. burn-in: rebuild the memory from the stored state. No learning here.
    online_carry, _ = unroll(network, params, start, burn)
    target_carry, _ = unroll(network, target_params, start, burn)
    online_carry = jax.lax.stop_gradient(online_carry)
 
    # 2. Q-values on the part we learn from
    _, q_online = unroll(network, params, online_carry, learn)          # (T, B, A)
    _, q_target = unroll(network, target_params, target_carry, learn)
    q_target = jax.lax.stop_gradient(q_target)
 
    # 3. double Q: the online net picks the next action, the target net scores it
    next_action = jnp.argmax(q_online[1:], axis=-1)                    # (T-1, B)
    next_value = jnp.take_along_axis(q_target[1:], next_action[..., None], axis=-1)[..., 0]
 
    # 4. n-step target, computed in real-return space, then squashed with h
    rewards = learn.reward[:-1]
    discounts = cfg["GAMMA"] * (1.0 - learn.done[:-1].astype(jnp.float32))
    returns = n_step_targets(rewards, discounts, h_inv(next_value, eps), cfg["N_STEP"])
    target = jax.lax.stop_gradient(h(returns, eps))
 
    q_taken = jnp.take_along_axis(q_online[:-1], learn.action[:-1, :, None], axis=-1)[..., 0]
    td = target - q_taken                                               # (T-1, B)
 
    # 5. loss: sum over time per sequence, importance-weighted mean over the batch
    per_seq = 0.5 * jnp.square(td).sum(axis=0)                          # (B,)
    weights = (1.0 / (probabilities + 1e-10)) ** cfg["IMPORTANCE_SAMPLING_EXPONENT"]
    weights = weights / weights.max()
    loss = jnp.mean(weights * per_seq)
 
    # 6. [R2D2] priority = eta * max|TD| + (1 - eta) * mean|TD| over the sequence
    abs_td = jnp.abs(td)
    eta = cfg["PRIORITY_ETA"]
    priorities = eta * abs_td.max(axis=0) + (1.0 - eta) * abs_td.mean(axis=0)
 
    return loss, (jax.lax.stop_gradient(priorities), q_taken.mean())
 









def single_run(config:dict):
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
    
    num_envs = config["NUM_ENVS"]
    hidden = config["HIDDEN_SIZE"]
    burn_in = config["BURN_IN_LENGTH"]
    seq_len = config["SEQUENCE_LENGTH"]

    sample_len = burn_in + seq_len

    # Getting this dtype wrong is the classic silent failure: a uint8 buffer
    # rounds a normalised 0.47 to 0 and the agent trains on nothing.
    obs_bytes = int(np.prod(obs_shape)) * (1 if config["PIXEL_BASED"] else 4)
    carry_bytes = 2 * hidden * 4
    buffer_gb = config["BUFFER_SIZE"] * (obs_bytes + carry_bytes) / 1e9
    # --- env helpers (copied from dqn.py) -----------------------------------
    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, info

    # --- network ------------------------------------------------------------

    network = RecurrentQNetwork(
        action_dim=action_dim,
        pixel_based=config["PIXEL_BASED"],
        hidden_size=hidden,
        dueling_units=config["DUELING_UNITS"],
        mlp_width=config.get("MLP_WIDTH", 512),
        mlp_depth=config.get("MLP_DEPTH", 2),
    )
    key, reset_key, net_key = jax.random.split(key, 3)
    dummy_obs, _ = env.reset(reset_key)
    dummy_obs = dummy_obs.reshape(obs_shape)
    params = network.init(
        net_key,
        RecurrentQNetwork.initial_carry(1, hidden),
        dummy_obs[None],
        jnp.zeros((1,), jnp.int32),
        jnp.zeros((1,), jnp.float32),
    )


    # --- replay buffer -----------------------------------------------------
    replay_buffer = fbx.make_prioritised_trajectory_buffer(
        add_batch_size=num_envs,
        sample_batch_size=config["BATCH_SIZE"],       # 64 sequences per update
        sample_sequence_length=sample_len,              
        period=config["SEQUENCE_PERIOD"],             # stride between starts -> 40-step overlap
        min_length_time_axis=max(sample_len, config["LEARNING_STARTS"]// num_envs),                 
        max_length_time_axis=config["BUFFER_SIZE"] // num_envs,
        priority_exponent=config["PRIORITY_EXPONENT"],
    )
    key, reset_key = jax.random.split(key)
    dummy_obs, _ = env.reset(reset_key)
    dummy_obs = dummy_obs.reshape(obs_shape)
    dummy_timestep = TimeStep(
        obs=dummy_obs,
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        done=jnp.zeros((), dtype=jnp.bool_),
        prev_action=jnp.zeros((), dtype=jnp.int32),
        prev_reward=jnp.zeros((), dtype=jnp.float32), carry=jax.tree.map(lambda c: c[0], RecurrentQNetwork.initial_carry(1, hidden)),
        arm=jnp.zeros((), dtype=jnp.int32),
        intrinsic_reward=jnp.zeros((), dtype=jnp.float32),
    )
    buffer_state = replay_buffer.init(dummy_timestep)

     # --- collection ---------------------------------------------------------
     # One call = TRAIN_FREQUENCY steps in every env. Unlike dqn, the whole chunk
     # goes into the buffer at once: the trajectory buffer keeps each env's steps
     # in order and cuts 120-step sequences out of them when sampling.
    total_timesteps = config["TOTAL_TIMESTEPS"]
    steps_per_call = config["TRAIN_FREQUENCY"]
    def collect(params, act_state, rng):
        def act_step(act_state, rng):
            env_state, obs, lstm, prev_action, prev_reward, global_step = act_state
            action_rng, explore_rng = jax.random.split(rng)
            # same schedule as dqn.py
            epsilon = jnp.interp(
                global_step,
                jnp.array([0, config["EXPLORATION_FRACTION"] * total_timesteps]),
                jnp.array([config["START_E"], config["END_E"]]),)

            next_lstm, q_values = network.apply(params, lstm, obs, prev_action, prev_reward)
            greedy_actions = q_values.argmax(axis=-1)
            random_actions = jax.random.randint(action_rng, (num_envs,), 0, action_dim)
            explore_mask = jax.random.uniform(explore_rng, (num_envs,)) < epsilon
            actions = jnp.where(explore_mask, random_actions, greedy_actions)
            next_obs, env_state, rewards, next_done, info = vmap_step(env_state, actions)
            timestep = TimeStep(
                obs=obs,
                action=actions,
                reward=rewards,
                done=next_done,
                prev_action=prev_action,
                prev_reward=prev_reward,
                carry=lstm,                          # state BEFORE this step
                arm=jnp.zeros_like(actions),         # NGU fields, unused in r2d2
                 intrinsic_reward=jnp.zeros_like(rewards),
            )
            # Inputs for the next step. Where the episode just ended, -1 tells
            # the network to start from a zero carry.
            prev_action = jnp.where(next_done, -1, actions)
            prev_reward = jnp.where(next_done, 0.0, rewards)
            act_state = (env_state, next_obs, next_lstm, prev_action, prev_reward, global_step + num_envs)
            return act_state, (timestep, info)
        
        rngs = jax.random.split(rng, steps_per_call)
        act_state, (traj, infos) = jax.lax.scan(act_step, act_state, rngs)
        # scan stacks time first (T, num_envs, ...); the buffer wants (num_envs, T, ...)
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return act_state, traj, infos
    @partial(jax.jit, donate_argnums=(2,))
    def collect_and_store(params, act_state, buffer_state, rng):
        act_state, traj, infos = collect(params, act_state, rng)
        buffer_state = replay_buffer.add(buffer_state, traj)
        return act_state, buffer_state, infos
    
    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    act_state = (
        env_state,
        obs,
        RecurrentQNetwork.initial_carry(num_envs, hidden),
        jnp.full((num_envs,), -1, jnp.int32),      # -1 = first step of an episode
        jnp.zeros((num_envs,), jnp.float32),
        jnp.array(0, jnp.int32),                   # global_step, in env steps
    )
    print(f"[agent57] stage={stage} env={config['ENV_ID']} "f"{'pixel' if config['PIXEL_BASED'] else 'oc'}")
    print(f"[agent57] action_dim={action_dim} obs_shape={obs_shape} " f"obs_bytes={obs_bytes}")
    print(f"[agent57] buffer: {config['BUFFER_SIZE']} transitions = {buffer_gb:.2f} GB")
    print(f"[agent57] buffer storage: {buffer_state.experience.obs.shape} "f"dtype={buffer_state.experience.obs.dtype}")

        # main.py expects a dict back from every agent. nan means "no score yet";
        # returning 0 would look like a real score of zero.
    return {"default": float("nan")}