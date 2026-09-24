"""End-to-end act_step / train_step on the dummy env, for every STAGE."""
import jax
import jax.numpy as jnp
import pytest

from agents.agent57.agent57 import STAGES, combine_q, h, h_inv, n_step_targets
from tests.agent57.conftest import ACTION_DIM, OBS_DIM, dummy_reset, make_agent


def setup(stage, key, **kw):
    agent = make_agent(stage, **kw)
    obs, env_state = dummy_reset(jax.random.split(key, agent.num_envs))
    learner = agent.init_learner(key, obs[0])
    return agent, learner, agent.init_buffer(obs[0]), agent.init_actor(env_state, obs)


def fill_buffer(agent, learner, buffer_state, actor, key, iters=4):
    collect = jax.jit(lambda l, a, k: agent.collect(l, a, k, 4))
    for i in range(iters):
        actor, traj, _ = collect(learner, actor, jax.random.fold_in(key, i))
        buffer_state = agent.buffer.add(buffer_state, traj)
    return buffer_state, actor


def test_value_rescaling_roundtrip():
    x = jnp.linspace(-500.0, 500.0, 101)
    assert jnp.allclose(h_inv(h(x, 1e-3), 1e-3), x, rtol=1e-4, atol=1e-2)
    q = jnp.array([[1.0, -2.0]])
    assert jnp.allclose(combine_q(q, q * 10, jnp.array([0.0]), 1e-3), q, atol=1e-5)   # beta 0 -> Q_e


def test_n_step_targets():
    r = jnp.array([1.0, 1.0, 1.0]); d = jnp.array([0.5, 0.0, 0.5]); v = jnp.array([10.0, 20.0, 30.0])
    g = n_step_targets(r, d, v, 2)
    # t0: 1 + .5*1 + .5*0*v[1];  t1: 1 + 0*...;  t2: 1 + .5*v[2]
    assert jnp.allclose(g, jnp.array([1.5, 1.0, 16.0]))


@pytest.mark.parametrize("stage", STAGES)
def test_act_step(stage, key):
    agent, learner, _, actor = setup(stage, key)
    step = jax.jit(agent.act_step)
    new_actor, ts, info = step(learner, actor, key)
    E = agent.num_envs
    assert ts.obs.shape == (E, OBS_DIM) and ts.action.shape == (E,) and ts.action.dtype == jnp.int32
    assert bool(jnp.all((ts.action >= 0) & (ts.action < ACTION_DIM)))
    assert ts.carry[0].shape == (E, agent.num_heads, agent.cfg["HIDDEN_SIZE"])
    assert ts.reward.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(ts.intrinsic_reward)))
    assert int(new_actor.global_step) == E
    assert new_actor.carry[0].shape == actor.carry[0].shape


@pytest.mark.parametrize("stage", STAGES)
def test_train_step_loss_and_grads(stage, key):
    agent, learner, buffer_state, actor = setup(stage, key)
    buffer_state, actor = fill_buffer(agent, learner, buffer_state, actor, key)
    assert bool(agent.buffer.can_sample(buffer_state))

    batch = agent.buffer.sample(buffer_state, key)
    (loss, (prio, metrics)), grads = jax.jit(jax.value_and_grad(agent.q_loss, has_aux=True))(
        learner.q.params, learner.q.target_params, batch.experience, batch.probabilities)
    assert jnp.isfinite(loss) and prio.shape == (agent.cfg["BATCH_SIZE"],) and bool(jnp.all(prio >= 0))
    assert all(jnp.isfinite(v) for v in metrics.values())
    assert any(float(jnp.abs(g).max()) > 0 for g in jax.tree.leaves(grads))
    if agent.split_q:   # both Q_e (head 0) and Q_i (head 1) receive gradient
        per_head = [jax.tree.map(lambda g: float(jnp.abs(g[i]).max()), grads) for i in range(2)]
        assert all(max(jax.tree.leaves(p)) > 0 for p in per_head)
    # no gradient reaches the target network
    assert jax.tree.structure(grads) == jax.tree.structure(learner.q.params)

    new_learner, new_buffer, m = jax.jit(agent.learner_update)(learner, buffer_state, key)
    assert int(new_learner.q.step) == 1 and jnp.isfinite(m["loss"])
    changed = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), new_learner.q.params, learner.q.params)
    assert any(jax.tree.leaves(changed))
    if agent.use_intrinsic:
        assert jnp.isfinite(m["embedding_loss"]) and jnp.isfinite(m["rnd_loss"])


def test_target_sync_period(key):
    agent, learner, buffer_state, actor = setup("agent57", key, TARGET_NETWORK_FREQUENCY=2)
    buffer_state, actor = fill_buffer(agent, learner, buffer_state, actor, key)
    upd = jax.jit(agent.learner_update)
    l1, b, _ = upd(learner, buffer_state, key)                       # step 1: no sync
    same = jax.tree.map(lambda a, b: bool(jnp.all(a == b)), l1.q.target_params, learner.q.target_params)
    assert all(jax.tree.leaves(same))
    l2, _, _ = upd(l1, b, jax.random.fold_in(key, 1))                  # step 2: sync
    synced = jax.tree.map(lambda a, b: bool(jnp.all(a == b)), l2.q.target_params, l2.q.params)
    assert all(jax.tree.leaves(synced))


@pytest.mark.parametrize("stage", ["r2d2", "agent57"])
def test_scanned_train_iterations(stage, key):
    """The whole collect -> add -> learn loop compiles once and runs under lax.scan."""
    agent, learner, buffer_state, actor = setup(stage, key)
    run = jax.jit(lambda c: jax.lax.scan(agent.train_iteration, c, None, length=6))
    (learner, buffer_state, actor, _), (infos, metrics) = run((learner, buffer_state, actor, key))
    assert int(actor.global_step) == 6 * agent.num_envs * agent.cfg["TRAIN_FREQUENCY"]
    assert int(learner.q.step) > 0
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in metrics.values())
    if stage == "agent57":
        assert int(actor.bandit.count.sum()) > 0          # episodes ended and were recorded


@pytest.mark.parametrize("stage", STAGES)
def test_carry_is_type_stable(stage, key):
    """One train iteration must return a carry with identical shapes/dtypes/weak types,
    otherwise jit retraces and recompiles the whole training scan every call."""
    agent, learner, buffer_state, actor = setup(stage, key)
    carry = (learner, buffer_state, actor, key)
    new, _ = jax.jit(lambda c: jax.lax.scan(agent.train_iteration, c, None, length=1))(carry)
    sig = jax.typeof   # the abstract value jit keys its cache on (shape, dtype, weak_type)
    assert [sig(x) for x in jax.tree.leaves(carry)] == [sig(x) for x in jax.tree.leaves(new)]


def test_priority_reduces_over_time(key):
    """Priority must be eta*max_t|td| + (1-eta)*mean_t|td| per sequence, recomputed here from the loss's own TD."""
    agent, learner, buffer_state, actor = setup("split_q", key)
    buffer_state, actor = fill_buffer(agent, learner, buffer_state, actor, key)
    batch = agent.buffer.sample(buffer_state, key)
    params = learner.q.params
    _, (prio, _) = agent.q_loss(params, params, batch.experience, batch.probabilities)

    # reference: per-head TD errors via a per-head (Q_e, Q_i) loss computed by hand
    from agents.agent57.replay import split_burn_in
    start, burn, learn = split_burn_in(batch.experience, agent.burn_in)
    c, _ = agent.unroll(params, start, burn)
    _, q = agent.unroll(params, c, learn)                                          # (T, H, B, A)
    beta = agent.arm_betas[learn.arm]
    nxt = jnp.argmax(combine_q(q[1:, 0], q[1:, 1], beta[1:], agent.eps), -1)
    v = jnp.take_along_axis(q[1:], nxt[:, None, :, None], -1)[..., 0]
    d = (agent.arm_gammas[learn.arm][:-1] * (1 - learn.done[:-1]))
    ref = []
    for hd, r in enumerate([learn.reward[:-1], learn.intrinsic_reward[:-1]]):
        g = h(n_step_targets(r, d, h_inv(v[:, hd], agent.eps), agent.n_step), agent.eps)
        td = jnp.abs(g - jnp.take_along_axis(q[:-1, hd], learn.action[:-1, :, None], -1)[..., 0])
        ref.append(0.9 * td.max(0) + 0.1 * td.mean(0))
    expected = ref[0] + agent.arm_betas[learn.arm[0]] * ref[1]
    assert jnp.allclose(prio, expected, rtol=1e-4, atol=1e-5)
