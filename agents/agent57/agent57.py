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

    print(f"[agent57] stage={stage} env={config['ENV_ID']} "
          f"{'pixel' if config['PIXEL_BASED'] else 'oc'}")
    return {"default": float("nan")}