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
def single_run(config:dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    stage = config.get("STAGE", "r2d2")
    assert stage in ("r2d2", "ngu", "split_q", "agent57"), f"unknown STAGE: {stage}"

    # Three separate random number generators need seeding, because three
    # different libraries produce randomness here:
    #   random  — Python's own, used by any plain-Python sampling
    #   np      — numpy, used by wandb and some wrappers
    #   key     — JAX's. JAX has no global RNG state; you hold an explicit key
    #             and split it whenever you need fresh randomness. Reusing a
    #             key gives identical "random" numbers, which is a real bug.
    # Seeding all three is what makes a run reproducible.
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    # Build the environment. make_env returns a thunk (a zero-argument
    # function), so the trailing () is what actually constructs it.
    env = make_env(
        config["ENV_ID"],                          # which game
        list(config.get("TRAIN_MODS", [])),        # game modifications, [] = none
        config["PIXEL_BASED"],                     # pixels vs object-centric
        config.get("NATIVE_DOWNSCALING", True),    # pixel mode only
        False,                                     # eval=False: episodic_life on,
                                                   #   rewards clipped for training
    )()

    # Read the shapes FROM THE ENVIRONMENT .
    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape

    # Pixel observations arrive with a trailing channel dimension that the
    # buffer does not store.
    if config["PIXEL_BASED"]:
        obs_shape = obs_shape[:-1]

    # Buffer sizing, printed so we always know where we stand against the
    # 11GB card. Pixel obs are uint8 (1 byte each); object-centric obs go
    # through a normalisation wrapper and are float32 (4 bytes each).
    # Getting this dtype wrong is the classic silent failure: a uint8 buffer
    # rounds a normalised 0.47 to 0 and the agent trains on nothing.
    obs_bytes = int(np.prod(obs_shape)) * (1 if config["PIXEL_BASED"] else 4)
    buffer_gb = config["BUFFER_SIZE"] * obs_bytes / 1e9

    print(f"[agent57] stage={stage} env={config['ENV_ID']} "
          f"{'pixel' if config['PIXEL_BASED'] else 'oc'}")
    print(f"[agent57] action_dim={action_dim} obs_shape={obs_shape} "
          f"obs_bytes={obs_bytes}")
    print(f"[agent57] buffer: {config['BUFFER_SIZE']} transitions = {buffer_gb:.2f} GB")

    # main.py expects a dict back from every agent. nan means "no score yet";
    # returning 0 would look like a real score of zero.
    return {"default": float("nan")}