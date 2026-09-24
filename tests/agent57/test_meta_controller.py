"""Arm schedule and sliding-window UCB bandit."""
import jax
import jax.numpy as jnp
import pytest

from agents.agent57.agent57 import arm_schedule, bandit_select, bandit_update, greedy_arm, init_bandit

E, A, W = 3, 4, 5


def test_arm_schedule():
    betas, gammas = arm_schedule(8, 0.3, 0.997, 0.99)
    assert betas.shape == gammas.shape == (8,)
    assert float(betas[0]) == 0.0 and jnp.isclose(betas[-1], 0.3)
    assert jnp.isclose(gammas[0], 0.997) and jnp.isclose(gammas[-1], 0.99)
    assert bool(jnp.all(jnp.diff(gammas) < 0))
    for n in (1, 2, 3):   # tiny populations must not produce NaN
        b, g = arm_schedule(n, 0.3, 0.997, 0.99)
        assert bool(jnp.all(jnp.isfinite(b))) and bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.parametrize("fill", [0, 2, W, 3 * W])
def test_select_in_range_for_any_window_fill(key, fill):
    bandit = init_bandit(E, W)
    for i in range(fill):
        bandit = bandit_update(bandit, jnp.full(E, i % A), jnp.full(E, float(i)), jnp.ones(E, bool))
    assert bool(jnp.all(bandit.count == min(fill, W)))
    arms = bandit_select(bandit, A, key)
    assert arms.shape == (E,) and arms.dtype == jnp.int32
    assert bool(jnp.all((arms >= 0) & (arms < A)))


def test_update_only_where_done():
    bandit = init_bandit(E, W)
    bandit = bandit_update(bandit, jnp.array([1, 2, 3]), jnp.array([5.0, 6.0, 7.0]), jnp.array([True, False, True]))
    assert bandit.count.tolist() == [1, 0, 1]
    assert bandit.arms[0, 0] == 1 and bandit.returns[2, 0] == 7.0


def test_every_arm_selected_and_best_arm_preferred(key):
    """Every (beta, gamma) arm gets played; with no epsilon the bandit settles on the best arm."""
    step = jax.jit(lambda b, arm, k: bandit_update(b, arm, (arm == 2).astype(jnp.float32) * 10.0,
                                                    jnp.ones(E, bool)))
    bandit = init_bandit(E, 20)
    arm = bandit_select(bandit, A, key, ucb_epsilon=0.0)
    seen = set()
    counts = jnp.zeros(A)
    for t in range(40):
        seen.update(arm.tolist())
        counts = counts.at[arm].add(1)
        bandit = step(bandit, arm, key)
        arm = bandit_select(bandit, A, jax.random.fold_in(key, t), ucb_epsilon=0.0)
    assert seen == set(range(A))
    assert int(jnp.argmax(counts)) == 2
    assert int(greedy_arm(bandit, A)) == 2
