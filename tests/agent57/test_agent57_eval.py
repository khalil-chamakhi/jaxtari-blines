"""Evaluation loop on the dummy env, directly and through a saved checkpoint."""
import jax
import jax.numpy as jnp
import pytest

from agents.agent57.agent57_eval import evaluate_policy
from tests.agent57.conftest import EPISODE_LEN, dummy_reset, dummy_step, make_agent


def params_for(agent, learner, actor):
    ckpt = agent.checkpoint(learner, actor)
    return {k: v for k, v in ckpt.items() if k not in ("config", "eval_arm")}


@pytest.mark.parametrize("stage", ["r2d2", "ngu", "agent57"])
def test_evaluate_policy(stage, key):
    agent = make_agent(stage)
    obs, env_state = dummy_reset(jax.random.split(key, agent.num_envs))
    learner = agent.init_learner(key, obs[0])
    actor = agent.init_actor(env_state, obs)
    returns, dones, _ = evaluate_policy(agent, params_for(agent, learner, actor), dummy_reset, dummy_step,
                                        eval_episodes=3, key=key, arm=agent.num_arms - 1, chunk=16)
    assert returns.shape == (3,) and bool(jnp.all(jnp.isfinite(returns)))
    assert bool(jnp.all(jnp.any(dones, axis=0)))                   # every episode finished
    assert bool(jnp.all((returns >= 0) & (returns <= EPISODE_LEN)))  # only the first episode is scored


def test_evaluate_from_checkpoint(key, tmp_path, monkeypatch):
    """Save a checkpoint the way single_run does and evaluate it through evaluate()."""
    import flax
    from agents.agent57 import agent57 as a57
    from agents.agent57.agent57_eval import evaluate

    agent = make_agent("agent57")
    obs, env_state = dummy_reset(jax.random.split(key, agent.num_envs))
    learner, actor = agent.init_learner(key, obs[0]), agent.init_actor(env_state, obs)
    path = tmp_path / "model.agent57_model"
    path.write_bytes(flax.serialization.to_bytes(agent.checkpoint(learner, actor)))

    class Env:
        def action_space(self):
            return type("S", (), {"n": 3})()

        def observation_space(self):
            return type("S", (), {"shape": (6,)})()

    monkeypatch.setattr(a57, "batched_env_fns", lambda env, shape: (dummy_reset, dummy_step))
    returns, _ = evaluate(str(path), lambda env_id: (lambda: Env()), "dummy", eval_episodes=2, seed=3)
    assert returns.shape == (2,) and bool(jnp.all(jnp.isfinite(returns)))
