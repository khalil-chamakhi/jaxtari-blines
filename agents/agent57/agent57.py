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


    arm: jnp.ndarray            # which of the 8 (beta, gamma) policies acted
    intrinsic_reward: jnp.ndarray   # curiosity reward for this step

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


    # Getting this dtype wrong is the classic silent failure: a uint8 buffer
    # rounds a normalised 0.47 to 0 and the agent trains on nothing.
    obs_bytes = int(np.prod(obs_shape)) * (1 if config["PIXEL_BASED"] else 4)
    buffer_gb = config["BUFFER_SIZE"] * obs_bytes / 1e9

    # --- replay buffer -----------------------------------------------------

    num_envs = config["NUM_ENVS"]
    seq_len = config["SEQUENCE_LENGTH"]

    replay_buffer = fbx.make_prioritised_trajectory_buffer(
        add_batch_size=num_envs,
        sample_batch_size=config["BATCH_SIZE"],       # 64 sequences per update
        sample_sequence_length=seq_len,               # [R2D2] m = 80
        period=config["SEQUENCE_PERIOD"],             # stride between starts -> 40-step overlap
        min_length_time_axis=seq_len,                 # refuse to sample before this exists
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
        prev_reward=jnp.zeros((), dtype=jnp.float32),
        arm=jnp.zeros((), dtype=jnp.int32),
        intrinsic_reward=jnp.zeros((), dtype=jnp.float32),
    )

    buffer_state = replay_buffer.init(dummy_timestep)

    print(f"[agent57] stage={stage} env={config['ENV_ID']} "f"{'pixel' if config['PIXEL_BASED'] else 'oc'}")
    print(f"[agent57] action_dim={action_dim} obs_shape={obs_shape} " f"obs_bytes={obs_bytes}")
    print(f"[agent57] buffer: {config['BUFFER_SIZE']} transitions = {buffer_gb:.2f} GB")
    print(f"[agent57] buffer storage: {buffer_state.experience.obs.shape} "f"dtype={buffer_state.experience.obs.dtype}")

    # main.py expects a dict back from every agent. nan means "no score yet";
    # returning 0 would look like a real score of zero.
    return {"default": float("nan")}