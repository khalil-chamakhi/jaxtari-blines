"""Shapes and finiteness of every Agent57 network on random input."""
import jax
import jax.numpy as jnp
import pytest

from agents.agent57.agent57 import EmbeddingTrainer, RNDNetwork, SplitQNetwork, Torso

B, OBS, A, H = 2, 6, 3, 8


def finite(x):
    return bool(jnp.all(jnp.isfinite(x)))


def test_torso_oc_and_pixel(key):
    oc = Torso(pixel_based=False, mlp_width=16, mlp_depth=2)
    x = jax.random.normal(key, (B, OBS))
    y = oc.apply(oc.init(key, x), x)
    assert y.shape == (B, 16) and finite(y)

    px = Torso(pixel_based=True)
    img = jax.random.randint(key, (B, 4, 84, 84), 0, 255).astype(jnp.uint8)
    y = px.apply(px.init(key, img), img)
    assert y.shape == (B, 7 * 7 * 64) and finite(y)   # DQN / R2D2 conv stack


@pytest.mark.parametrize("num_heads", [1, 2])
def test_split_q_heads(key, num_heads):
    net = SplitQNetwork(num_heads=num_heads, action_dim=A, pixel_based=False, hidden_size=H,
                        dueling_units=8, mlp_width=16, mlp_depth=1, num_arms=4)
    obs = jax.random.normal(key, (B, OBS))
    prev_a = jnp.array([-1, 1]); prev_r = jnp.zeros(B); arm = jnp.array([0, 3]); prev_i = jnp.ones(B)
    params = net.init(key, net.initial_carry(B), obs, prev_a, prev_r, arm, prev_i)
    # heads have separate weights: leading axis = num_heads on every leaf
    assert all(l.shape[0] == num_heads for l in jax.tree.leaves(params))
    carry = jax.tree.map(lambda c: c + 1.0, net.initial_carry(B))
    new_carry, q = net.apply(params, carry, obs, prev_a, prev_r, arm, prev_i)
    assert q.shape == (num_heads, B, A) and finite(q)
    assert new_carry[0].shape == (num_heads, B, H) and finite(new_carry[1])
    if num_heads == 2:
        assert not jnp.allclose(q[0], q[1])   # Q_e and Q_i differ


def test_episode_start_resets_carry(key):
    """prev_action = -1 must make the output independent of the incoming carry."""
    net = SplitQNetwork(num_heads=1, action_dim=A, pixel_based=False, hidden_size=H, mlp_width=16, mlp_depth=1)
    obs = jax.random.normal(key, (B, OBS)); z = jnp.zeros(B); a = jnp.full((B,), -1)
    params = net.init(key, net.initial_carry(B), obs, a, z, a, z)
    _, q0 = net.apply(params, net.initial_carry(B), obs, a, z, a, z)
    _, q1 = net.apply(params, jax.tree.map(lambda c: c + 5.0, net.initial_carry(B)), obs, a, z, a, z)
    assert jnp.allclose(q0, q1)


def test_embedding_net(key):
    m = EmbeddingTrainer(action_dim=A, pixel_based=False, embedding_dim=8, mlp_width=16, mlp_depth=1)
    obs = jax.random.normal(key, (B, OBS))
    params = m.init(key, obs, obs)
    emb = m.apply(params, obs, method="embed")
    logits = m.apply(params, obs, obs)
    assert emb.shape == (B, 8) and finite(emb)
    assert logits.shape == (B, A) and finite(logits)


def test_rnd_pair(key):
    net = RNDNetwork(pixel_based=False, output_dim=8, mlp_width=16, mlp_depth=1)
    obs = jax.random.normal(key, (B, OBS))
    k1, k2 = jax.random.split(key)
    target, predictor = net.init(k1, obs), net.init(k2, obs)
    yt, yp = net.apply(target, obs), net.apply(predictor, obs)
    assert yt.shape == yp.shape == (B, 8) and finite(yt) and finite(yp)
    assert not jnp.allclose(yt, yp)   # independent initialisations
