"""[Agent57] policy population and sliding-window UCB meta-controller.

Population: NUM_ARMS policies j = 0..N-1, each a (beta_j, gamma_j) pair. All
arms share one set of network weights (UVFA: the arm is a network input), so
the population dimension is handled by gathering arm_betas[arm] /
arm_gammas[arm] per env and per replayed step — never a Python loop over arms.

Meta-controller: one sliding-window UCB bandit per actor (env). At the start
of every episode it picks the arm with the best
    mean_return_k + UCB_BETA * sqrt(1 / N_k)
over the last UCB_WINDOW episodes, or a random arm with probability
UCB_EPSILON. Arms absent from the window score +inf, so every arm is tried
first. All state is a fixed-shape pytree; updates are pure jnp.

arm_schedule is from feat/ngu, the bandit from feat/split-q.
"""
import flax
import jax
import jax.numpy as jnp


def arm_schedule(num_arms, beta_max, gamma_max, gamma_min):
    """[NGU] (beta_j, gamma_j) for every arm, two (num_arms,) float32 arrays.

    beta_0 = 0, beta_{N-1} = beta_max, sigmoid in between.
    1 - gamma_j interpolated log-linearly between 1 - gamma_max and 1 - gamma_min.
    num_arms = 1 gives the single exploitative arm (beta 0, gamma_max).
    """
    n = int(num_arms)
    if n == 1:
        return jnp.zeros((1,), jnp.float32), jnp.full((1,), gamma_max, jnp.float32)
    j = jnp.arange(n, dtype=jnp.float32)
    inner = beta_max * jax.nn.sigmoid(10.0 * (2.0 * j - (n - 2)) / max(n - 2, 1))
    betas = jnp.where(j == 0, 0.0, jnp.where(j == n - 1, beta_max, inner))
    log_1mg = ((n - 1 - j) * jnp.log(1.0 - gamma_max) + j * jnp.log(1.0 - gamma_min)) / (n - 1)
    return betas.astype(jnp.float32), (1.0 - jnp.exp(log_1mg)).astype(jnp.float32)


@flax.struct.dataclass
class BanditState:
    """Ring buffers over EPISODES (not steps), one row per env."""
    arms: jnp.ndarray      # (E, W) arm played in each remembered episode
    returns: jnp.ndarray   # (E, W) extrinsic return of that episode
    write: jnp.ndarray     # (E,) next slot
    count: jnp.ndarray     # (E,) filled slots, capped at W


def init_bandit(num_envs, window):
    return BanditState(
        arms=jnp.zeros((num_envs, window), jnp.int32),
        returns=jnp.zeros((num_envs, window), jnp.float32),
        write=jnp.zeros((num_envs,), jnp.int32),
        count=jnp.zeros((num_envs,), jnp.int32),
    )


def arm_statistics(bandit: BanditState, num_arms):
    """Per-env visit counts and mean returns inside the window, both (E, A)."""
    window = bandit.arms.shape[1]
    filled = jnp.arange(window)[None, :] < bandit.count[:, None]
    is_arm = (bandit.arms[:, :, None] == jnp.arange(num_arms)[None, None, :]) & filled[:, :, None]
    n_k = jnp.sum(is_arm, axis=1)
    mean_k = jnp.sum(bandit.returns[:, :, None] * is_arm, axis=1) / jnp.maximum(n_k, 1)
    return n_k, mean_k


def bandit_select(bandit: BanditState, num_arms, rng, ucb_beta=1.0, ucb_epsilon=0.5):
    """Arm for each env's next episode, (E,) int32 in [0, num_arms)."""
    n_k, mean_k = arm_statistics(bandit, num_arms)
    score = jnp.where(n_k == 0, jnp.inf, mean_k + ucb_beta * jnp.sqrt(1.0 / jnp.maximum(n_k, 1)))
    greedy = jnp.argmax(score, axis=-1)
    k1, k2 = jax.random.split(rng)
    random_arm = jax.random.randint(k1, greedy.shape, 0, num_arms)
    explore = jax.random.uniform(k2, greedy.shape) < ucb_epsilon
    return jnp.where(explore, random_arm, greedy).astype(jnp.int32)


def bandit_update(bandit: BanditState, arm, episode_return, done):
    """Record (arm, return) for every env whose episode just ended (done)."""
    window = bandit.arms.shape[1]
    idx = bandit.write
    put = lambda buf, i, v, d: buf.at[i].set(jnp.where(d, v, buf[i]))
    return BanditState(
        arms=jax.vmap(put)(bandit.arms, idx, arm.astype(jnp.int32), done),
        returns=jax.vmap(put)(bandit.returns, idx, episode_return.astype(jnp.float32), done),
        write=jnp.where(done, (idx + 1) % window, idx),
        count=jnp.where(done, jnp.minimum(bandit.count + 1, window), bandit.count),
    )


def greedy_arm(bandit: BanditState, num_arms):
    """Best arm by windowed mean return, pooled over envs (used for evaluation)."""
    n_k, mean_k = arm_statistics(bandit, num_arms)
    n = n_k.sum(0)
    mean = (mean_k * n_k).sum(0) / jnp.maximum(n, 1)
    return jnp.argmax(jnp.where(n > 0, mean, -jnp.inf)).astype(jnp.int32)
