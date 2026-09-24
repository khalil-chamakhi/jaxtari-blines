"""Agent57 networks.

    Torso               observation -> features (DQN conv stack or MLP)
    RecurrentQNetwork   torso -> LSTM -> dueling head, one R2D2 Q-network
    SplitQNetwork       NUM_HEADS copies of RecurrentQNetwork with separate
                        weights, evaluated in ONE call via nn.vmap.
                        Head 0 = Q_e (extrinsic), head 1 = Q_i (intrinsic).
    EmbeddingTrainer    NGU controllable-state embedding + inverse dynamics
    RNDNetwork          NGU lifelong novelty (frozen target / trained predictor)

Provenance: Torso, RecurrentQNetwork, EmbeddingTrainer and RNDNetwork come
from feat/infra and feat/ngu. SplitQNetwork replaces feat/split-q's two
independent network.apply calls (one per Q) with a single vmapped module.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class Torso(nn.Module):
    """Observation -> flat feature vector.

    pixel  [R2D2] Table 2: "the same 3-layer convolutional network as DQN":
           32/64/64, kernels 8/4/3, strides 4/2/1, VALID. Input (B, 4, 84, 84) uint8.
    oc     MLP, width 512 and depth 2 by default.
    """
    pixel_based: bool
    mlp_width: int = 512
    mlp_depth: int = 2

    @nn.compact
    def __call__(self, x):
        if self.pixel_based:
            # stored as (B, C, H, W); flax Conv expects (B, H, W, C)
            x = jnp.transpose(x, (0, 2, 3, 1)).astype(jnp.float32) / 255.0
            x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
            x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
            x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.astype(jnp.float32)
            for _ in range(self.mlp_depth):
                x = nn.Dense(self.mlp_width, kernel_init=orthogonal(np.sqrt(2.0)),
                             bias_init=constant(0.0))(x)
                x = nn.relu(x)
        return x


class RecurrentQNetwork(nn.Module):
    """[R2D2] torso -> LSTM -> dueling head.

    The LSTM input is [features, one_hot(prev_action), prev_reward] and, when
    num_arms > 0, [NGU] one_hot(arm) and the previous intrinsic reward (UVFA:
    one set of weights plays every (beta, gamma) arm).

    prev_action < 0 marks the first step of an episode; the carry is zeroed
    there, both while acting and while unrolling replayed sequences.
    """
    action_dim: int
    pixel_based: bool
    hidden_size: int = 512
    dueling_units: int = 512
    mlp_width: int = 512
    mlp_depth: int = 2
    num_arms: int = 0

    @nn.compact
    def __call__(self, carry, obs, prev_action, prev_reward, arm, prev_intrinsic):
        first = (prev_action < 0)[:, None]
        carry = jax.tree.map(lambda c: jnp.where(first, 0.0, c), carry)
        x = Torso(self.pixel_based, self.mlp_width, self.mlp_depth)(obs)
        inputs = [x, jax.nn.one_hot(prev_action, self.action_dim), prev_reward[:, None]]
        if self.num_arms > 0:
            inputs += [jax.nn.one_hot(arm, self.num_arms), prev_intrinsic[:, None]]
        carry, x = nn.OptimizedLSTMCell(self.hidden_size)(carry, jnp.concatenate(inputs, -1))

        # dueling head; layer order (and so parameter names) as in feat/infra
        v = nn.relu(nn.Dense(self.dueling_units)(x))
        v = nn.Dense(1)(v)
        a = nn.relu(nn.Dense(self.dueling_units)(x))
        a = nn.Dense(self.action_dim)(a)
        return carry, v + (a - a.mean(axis=-1, keepdims=True))


class SplitQNetwork(nn.Module):
    """[Agent57 sec 3.1] num_heads independent RecurrentQNetworks in one call.

    Params and carry get a leading head axis; every other input is shared.
      carry: (c, h) each (num_heads, B, hidden)
      returns carry of the same shape and q of shape (num_heads, B, A)
    num_heads = 1 for the r2d2/ngu stages, 2 (Q_e, Q_i) for split_q/agent57.
    """
    num_heads: int
    action_dim: int
    pixel_based: bool
    hidden_size: int = 512
    dueling_units: int = 512
    mlp_width: int = 512
    mlp_depth: int = 2
    num_arms: int = 0

    @nn.compact
    def __call__(self, carry, obs, prev_action, prev_reward, arm, prev_intrinsic):
        heads = nn.vmap(
            RecurrentQNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(0, None, None, None, None, None),
            out_axes=0,
            axis_size=self.num_heads,
        )
        return heads(
            action_dim=self.action_dim, pixel_based=self.pixel_based,
            hidden_size=self.hidden_size, dueling_units=self.dueling_units,
            mlp_width=self.mlp_width, mlp_depth=self.mlp_depth, num_arms=self.num_arms,
        )(carry, obs, prev_action, prev_reward, arm, prev_intrinsic)

    def initial_carry(self, batch_size):
        z = jnp.zeros((self.num_heads, batch_size, self.hidden_size), jnp.float32)
        return (z, z)


class EmbeddingTrainer(nn.Module):
    """[NGU] controllable-state embedding f(x) + inverse-dynamics classifier.

    embed(obs)               -> (B, embedding_dim), linear output (compared
                                with euclidean distances, so no ReLU)
    classify(emb_t, emb_tp1) -> (B, action_dim) logits, 128 hidden units
    __call__(obs, next_obs)  -> logits (used for init)
    """
    action_dim: int
    pixel_based: bool
    embedding_dim: int = 32
    hidden: int = 128
    mlp_width: int = 512
    mlp_depth: int = 2

    def setup(self):
        self.torso = Torso(self.pixel_based, self.mlp_width, self.mlp_depth)
        self.proj = nn.Dense(self.embedding_dim)
        self.cls_hidden = nn.Dense(self.hidden)
        self.cls_out = nn.Dense(self.action_dim)

    def embed(self, obs):
        return self.proj(self.torso(obs))

    def classify(self, emb_t, emb_tp1):
        x = nn.relu(self.cls_hidden(jnp.concatenate([emb_t, emb_tp1], axis=-1)))
        return self.cls_out(x)

    def __call__(self, obs, next_obs):
        return self.classify(self.embed(obs), self.embed(next_obs))


class RNDNetwork(nn.Module):
    """[NGU] observation -> feature vector; used as frozen target and trained predictor."""
    pixel_based: bool
    output_dim: int = 128
    mlp_width: int = 512
    mlp_depth: int = 2

    @nn.compact
    def __call__(self, obs):
        return nn.Dense(self.output_dim)(Torso(self.pixel_based, self.mlp_width, self.mlp_depth)(obs))
