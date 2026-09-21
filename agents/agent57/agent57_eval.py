"""Evaluation for the recurrent agent.

Same structure as agents/dqn/dqn_eval.py, with the three changes the LSTM forces:
  - the recurrent state is threaded through the episode
  - prev_action / prev_reward are fed back into the network
  - prev_action = -1 marks the first step of an episode, which makes the
    network start from a zero state (same convention as training)

The env is built with eval=True, so rewards are NOT clipped and a lost life is
not an episode end. Training returns use clipped rewards, so for games whose
rewards are not just +-1 the two numbers differ; this is the one comparable to
the DQN and Rainbow eval numbers.

WHAT THE LSTM CHANGES, COMPARED TO dqn_eval.py
    1. The recurrent state (carry) is threaded from step to step, so the agent
       remembers what happened earlier in the episode.
    2. The previous action (one-hot) and previous reward are fed back into the
       network, as R2D2's architecture requires.
    3. prev_action = -1 marks the first step of an episode. The network reads
       that as "no previous action" and wipes its memory. This is the same
       convention the training collection loop uses, so an episode start looks
       identical in both places.
 
RETURNS
    episodic_returns        (eval_episodes,) unclipped return of each episode
    env_states_until_done   states of episode 0 up to its first done, for video capture; None if the env does not expose them.
"""

from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: nn.Module,
    network_kwargs: dict,
    epsilon: float = 0.05,      # same as dqn_eval's default, so the Atari protocol matches
    seed: int = 1,
    max_steps: int = 200_000,
):
#Play `eval_episodes` full games with the trained weights and score them.
   # ---- 1. the evaluation environment ------------------------------------
   # eval=True was bound by the caller: rewards are real game points and only a genuine game over (or after 30 minutes) ends an episode.
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id)()
    action_dim = env.action_space().n
    key = jax.random.PRNGKey(seed)
    # ---- 2. single-episode helpers, then vectorised over episodes ----------
    # squeeze() drops the trailing channel dimension of pixel observations
    # ((4,84,84,1) -> (4,84,84)), matching single_run's obs_shape[:-1].

    def reset_one(key):
        obs, state = env.reset(key)
        return obs.squeeze(), state

    def step_one(state, action):
        obs, state, reward, terminated, truncated, info = env.step(state, action)
        return obs.squeeze(), state, reward, jnp.logical_or(terminated, truncated), info

    # vmap turns one episode into `eval_episodes` episodes running side by side,

    vmap_reset = jax.vmap(reset_one)
    vmap_step = jax.vmap(step_one)
    # ---- 3. rebuild the network and load the trained weights ---------------
    # action_dim comes from the env
    def arm_inputs(n):
        if (network.num_arms > 0) :
            return (jnp.zeros((n,), jnp.int32), jnp.zeros((n,), jnp.float32))
        return ()

    network = Model(action_dim=action_dim, **network_kwargs)

    key, net_key, reset_key = jax.random.split(key, 3)
    dummy_obs, _ = reset_one(reset_key)

    # the network is initialised with random weights first and then overwritten from the file.
    params = network.init(
        net_key,
        Model.initial_carry(1, network_kwargs.get("hidden_size", 512)),
        dummy_obs[None],
        jnp.zeros((1,), jnp.int32),
        jnp.zeros((1,), jnp.float32),
        *arm_inputs(1),
    )
    with open(model_path, "rb") as f:
        (_, params) = flax.serialization.from_bytes((None, params), f.read())
    
    
     # ---- 4. one step of play, for all episodes at once ---------------------
    def step_fn(carry, _):
        obs, env_state, lstm, prev_action, prev_reward, rng = carry
        rng, action_rng, explore_rng = jax.random.split(rng, 3)
        
        # the LSTM state is carried forward; the network wipes it itself wherever
        # prev_action is -1

        lstm, q_values = network.apply(params, lstm, obs, prev_action, prev_reward ,*arm_inputs(obs.shape[0]))
        greedy = q_values.argmax(axis=-1)
        random_actions = jax.random.randint(action_rng, greedy.shape, 0, action_dim)
        explore = jax.random.uniform(explore_rng, greedy.shape) < epsilon
        actions = jnp.where(explore, random_actions, greedy)

        obs, env_state, reward, done, _ = vmap_step(env_state, actions)
        reward = reward.astype(jnp.float32)

        # a finished episode starts the next one with no memory
        prev_action = jnp.where(done, -1, actions)
        prev_reward = jnp.where(done, 0.0, reward)

        # env_state holds all episodes; x[0] takes episode 0's slice. Collected
        # over the scan these frames become the video of one full episode,
        # rendered for the report (CAPTURE_VIDEO), same as dqn_eval.py does.
        first_states = jax.tree.map(lambda x: x[0], env_state)
        return (obs, env_state, lstm, prev_action, prev_reward, rng), (first_states, done, reward)

    @jax.jit
    def scanned_steps(carry):
        return jax.lax.scan(step_fn, carry, None, length=1000)


    # ---- 5. run until every episode has finished at least once -------------
    key, rng = jax.random.split(key)
    reset_keys = jax.random.split(rng, eval_episodes)
    obs, env_states = vmap_reset(reset_keys)

    carry = (
        obs,
        env_states,
        Model.initial_carry(eval_episodes, network_kwargs.get("hidden_size", 512)),
        jnp.full((eval_episodes,), -1, jnp.int32),      # -1 = first step
        jnp.zeros((eval_episodes,), jnp.float32),
        key,
    )

    all_first_states, all_dones, all_rewards = [], [], []
    done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)
    steps = 0

    while not jnp.all(done_ever) and steps < max_steps:
        carry, (first_states_chunk, dones_chunk, rewards_chunk) = scanned_steps(carry)
        all_first_states.append(first_states_chunk)
        all_dones.append(dones_chunk)
        all_rewards.append(rewards_chunk)
        done_ever = done_ever | jnp.any(dones_chunk, axis=0)
        steps += 1000

    first_states_history = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_first_states)
    dones = jnp.concatenate(all_dones, axis=0)
    rewards = jnp.concatenate(all_rewards, axis=0)

     # ---- 6. score the FIRST episode of each parallel run -------------------
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    episodic_returns = jnp.sum(rewards * (1 - mask_after_first_done), axis=0)

    finished = int(jnp.sum(done_ever))
    print(
        f"[agent57_eval] {finished}/{eval_episodes} episodes finished | "
        f"mean return {episodic_returns.mean():.2f} | std {episodic_returns.std():.2f}"
    )

     # ---- 7. states of episode 0, for the report video ----------------------
    first_done = jnp.argmax(dones, axis=0)
    try:
        states = first_states_history.atari_state.atari_state.env_state
        env_states_until_done = jax.tree.map(lambda x: x[: first_done[0] + 1], states)
    except AttributeError:
        # wrapper chain does not expose that nesting; skip the video rather than
        # failing an evaluation that otherwise succeeded
        env_states_until_done = None

    return episodic_returns, env_states_until_done