"""Evaluation for Agent57.

Same protocol as agents/dqn/dqn_eval.py (epsilon 0.05, env built with
eval=True so rewards are unclipped and a lost life does not end the episode),
with what the recurrent UVFA agent needs on top:
  - the LSTM carry is threaded through the episode; prev_action = -1 marks an
    episode start and zeroes it (same convention as training)
  - one fixed arm is played (EVAL_ARM, or for STAGE=agent57 the arm with the
    best windowed return in the bandits at the end of training)
  - the network's prev_intrinsic input is the real intrinsic reward, computed
    with the trained embedding / RND nets and frozen RND statistics (feat/ngu
    found that feeding a constant 0 halves the Pong score of arm 0)
  - Q_e and Q_i are combined exactly as while acting: h(h^-1 Q_e + beta h^-1 Q_i)

evaluate_policy is pure and takes an Agent57 + arrays, so it runs on any env
exposing batched reset/step (the smoke tests use a dummy env).
"""
import flax
import jax
import jax.numpy as jnp

from agents.agent57.intrinsic import compute_intrinsic


def evaluate_policy(agent, params, env_reset, env_step, eval_episodes, key, arm=0,
                    epsilon=0.05, max_steps=200_000, chunk=1000):
    """Play eval_episodes episodes in parallel; score the FIRST episode of each.

    agent:   Agent57 (only its static networks/config are used)
    params:  dict with q_params and, for stages with intrinsic reward,
             emb_params, rnd_params, rnd_target, rnd_stats
    env_reset(keys) -> (obs (E, ...), state); env_step as in Agent57.
    Returns (episodic_returns (E,), dones (T, E), per-step env states of episode 0).
    """
    n = eval_episodes
    arms = jnp.full((n,), arm, jnp.int32)

    def step_fn(carry, _):
        obs, env_state, lstm, prev_a, prev_r, prev_i, memory, rng = carry
        rng, k_act, k_exp = jax.random.split(rng, 3)
        lstm, q = agent.network.apply(params["q_params"], lstm, obs, prev_a, prev_r, arms, prev_i)
        greedy = agent.policy_q(q, arms).argmax(-1)
        explore = jax.random.uniform(k_exp, greedy.shape) < epsilon
        actions = jnp.where(explore, jax.random.randint(k_act, greedy.shape, 0, agent.action_dim), greedy)
        obs, env_state, reward, done, _ = env_step(env_state, actions)
        reward = reward.astype(jnp.float32)
        if agent.use_intrinsic:
            r_int, memory, _ = compute_intrinsic(
                agent.emb_model, agent.rnd_net, params["emb_params"], params["rnd_params"],
                params["rnd_target"], memory, params["rnd_stats"], obs, done, agent.novelty,
                update_stats=False)
            prev_i = jnp.where(done, 0.0, r_int)
        prev_a = jnp.where(done, -1, actions)
        prev_r = jnp.where(done, 0.0, reward)
        first_states = jax.tree.map(lambda x: x[0], env_state)
        return (obs, env_state, lstm, prev_a, prev_r, prev_i, memory, rng), (first_states, done, reward)

    @jax.jit
    def run_chunk(carry):
        return jax.lax.scan(step_fn, carry, None, length=chunk)

    key, reset_key = jax.random.split(key)
    obs, env_state = env_reset(jax.random.split(reset_key, n))
    zeros = jnp.zeros((n,), jnp.float32)
    memory = None
    if agent.use_intrinsic:
        from agents.agent57.intrinsic import init_episodic_memory
        memory = init_episodic_memory(n, int(agent.cfg["EPISODIC_MEMORY_SIZE"]), int(agent.cfg["EMBEDDING_DIM"]))
    carry = (obs, env_state, agent.network.initial_carry(n), jnp.full((n,), -1, jnp.int32),
             zeros, zeros, memory, key)

    states, dones, rewards = [], [], []
    done_ever = jnp.zeros((n,), jnp.bool_)
    steps = 0
    # one host check per chunk, not per step
    while not bool(jnp.all(done_ever)) and steps < max_steps:
        carry, (s, d, r) = run_chunk(carry)
        states.append(s)
        dones.append(d)
        rewards.append(r)
        done_ever = done_ever | jnp.any(d, axis=0)
        steps += chunk

    states = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *states)
    dones = jnp.concatenate(dones, axis=0)
    rewards = jnp.concatenate(rewards, axis=0)
    finished_before = jnp.pad(jax.lax.cummax(dones.astype(jnp.int32), axis=0)[:-1], ((1, 0), (0, 0)))
    episodic_returns = jnp.sum(rewards * (1 - finished_before), axis=0)
    return episodic_returns, dones, states


def load_checkpoint(model_path):
    with open(model_path, "rb") as f:
        return flax.serialization.msgpack_restore(f.read())


def evaluate(model_path, make_env, env_id, eval_episodes, epsilon=0.05, seed=1, max_steps=200_000):
    """Load an .agent57_model checkpoint and evaluate it on make_env(env_id)().

    Returns (episodic_returns, env_states_until_done of episode 0 or None).
    """
    from agents.agent57.agent57 import Agent57, batched_env_fns

    raw = load_checkpoint(model_path)
    config = raw["config"]
    env = make_env(env_id)()
    obs_shape = tuple(env.observation_space().shape)
    if config["PIXEL_BASED"]:
        obs_shape = obs_shape[:-1]
    env_reset, env_step = batched_env_fns(env, obs_shape)
    agent = Agent57({**config, "NUM_ENVS": eval_episodes}, env.action_space().n, obs_shape,
                    config["PIXEL_BASED"], env_step=env_step)

    # restore into correctly-structured templates
    key = jax.random.PRNGKey(seed)
    obs, _ = env_reset(jax.random.split(key, 1))
    learner = agent.init_learner(key, obs[0])
    params = {"q_params": flax.serialization.from_state_dict(learner.q.params, raw["q_params"])}
    if agent.use_intrinsic:
        from agents.agent57.intrinsic import init_running_stats
        params.update(
            emb_params=flax.serialization.from_state_dict(learner.emb.params, raw["emb_params"]),
            rnd_params=flax.serialization.from_state_dict(learner.rnd.params, raw["rnd_params"]),
            rnd_target=flax.serialization.from_state_dict(learner.rnd_target, raw["rnd_target"]),
            rnd_stats=flax.serialization.from_state_dict(init_running_stats(), raw["rnd_stats"]),
        )
    arm = int(raw.get("eval_arm", 0))

    episodic_returns, dones, states = evaluate_policy(
        agent, params, env_reset, env_step, eval_episodes, key, arm=arm, epsilon=epsilon, max_steps=max_steps)
    print(f"[agent57_eval] arm {arm} | {int(jnp.any(dones, 0).sum())}/{eval_episodes} episodes finished | "
          f"mean return {episodic_returns.mean():.2f} | std {episodic_returns.std():.2f}")

    first_done = jnp.argmax(dones[:, 0])
    try:
        env_states = states.atari_state.atari_state.env_state
        env_states_until_done = jax.tree.map(lambda x: x[: first_done + 1], env_states)
    except AttributeError:
        env_states_until_done = None
    return episodic_returns, env_states_until_done
