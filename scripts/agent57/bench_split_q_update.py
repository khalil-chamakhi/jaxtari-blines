"""Benchmark + equivalence check: feat/split-q's update_split_q vs the merged Agent57 learner.

Extracts feat/split-q's agent57.py from git into a temp module, builds both
implementations at the same sizes, checks the merged loss/priorities equal
feat/split-q's given identical weights, then times one compiled update of each.

    git fetch origin feat/split-q
    JAX_PLATFORMS=cpu uv run python scripts/agent57/bench_split_q_update.py
"""
import os, subprocess, sys, tempfile, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.getcwd())
_src = subprocess.run(["git", "show", "origin/feat/split-q:agents/agent57/agent57.py"],
                      check=True, capture_output=True, text=True).stdout
_src = _src.replace("from agents.agent57.agent57_eval import evaluate", "evaluate = None")
_tmp = tempfile.mkdtemp()
open(os.path.join(_tmp, "old_a57.py"), "w").write(_src)
sys.path.insert(0, _tmp)
import jax, jax.numpy as jnp, optax
import old_a57 as old
from agents.agent57.agent57 import Agent57
from agents.agent57.replay import TimeStep

B, BURN, SEQ, OBS, A, HID, ARMS = 16, 20, 40, 104, 6, 128, 8
L = BURN + SEQ
cfg = dict(STAGE="split_q", NUM_ENVS=8, BATCH_SIZE=B, SEQUENCE_LENGTH=SEQ, BURN_IN_LENGTH=BURN, SEQUENCE_PERIOD=20,
           N_STEP=5, BUFFER_SIZE=8000, LEARNING_STARTS=0, PRIORITY_EXPONENT=0.9, PRIORITY_ETA=0.9,
           IMPORTANCE_SAMPLING_EXPONENT=0.6, VALUE_RESCALING_EPSILON=1e-3, GAMMA=0.99, TARGET_NETWORK_FREQUENCY=2500,
           TRAIN_FREQUENCY=4, LEARNING_RATE=1e-4, ADAM_EPS=1e-3, ADAM_B1=0.9, ADAM_B2=0.999, HIDDEN_SIZE=HID,
           DUELING_UNITS=128, MLP_WIDTH=256, MLP_DEPTH=2, NUM_ARMS=ARMS, BETA_MAX=0.3, GAMMA_MAX=0.99, GAMMA_MIN=0.98,
           START_E=1., END_E=.01, EXPLORATION_FRACTION=.1, TOTAL_TIMESTEPS=1)
k = jax.random.PRNGKey(0)
obs = jax.random.normal(k, (B, L, OBS)); act = jax.random.randint(k, (B, L), 0, A); f = jax.random.normal(k, (B, L))
arm = jax.random.randint(k, (B, L), 0, ARMS); done = jnp.zeros((B, L), bool); probs = jnp.full((B,), 1. / B)
zc = jnp.zeros((B, L, HID))

# --- new
agent = Agent57(cfg, A, (OBS,), False)
learner = agent.init_learner(k, obs[0, 0])
new_exp = TimeStep(obs=obs, action=act, reward=f, done=done, prev_action=act, prev_reward=f,
                   carry=(jnp.zeros((B, L, 2, HID)),) * 2, arm=arm, intrinsic_reward=f, prev_intrinsic=f)
def new_step(q, exp):
    (loss, (p, m)), g = jax.value_and_grad(agent.q_loss, has_aux=True)(q.params, q.target_params, exp, probs)
    return q.apply_gradients(grads=g), p

# --- old (feat/split-q), same sizes; replay write stubbed out
net = old.RecurrentQNetwork(action_dim=A, pixel_based=False, hidden_size=HID, dueling_units=128, mlp_width=256,
                            mlp_depth=2, num_arms=ARMS)
c1 = old.RecurrentQNetwork.initial_carry(1, HID)
pe = net.init(jax.random.PRNGKey(1), c1, obs[:1, 0], act[:1, 0], f[:1, 0], arm[:1, 0], f[:1, 0])
pi = net.init(jax.random.PRNGKey(2), c1, obs[:1, 0], act[:1, 0], f[:1, 0], arm[:1, 0], f[:1, 0])
tx = optax.adam(1e-4, eps=1e-3)
se = old.Agent57TrainState.create(apply_fn=net.apply, params=pe, target_params=pe, tx=tx)
si = old.Agent57TrainState.create(apply_fn=net.apply, params=pi, target_params=pi, tx=tx)
old_exp = old.TimeStep(obs=obs, action=act, reward=f, done=done, prev_action=act, prev_reward=f,
                       carry_e=(zc, zc), carry_i=(zc, zc), arm=arm, intrinsic_reward=f, prev_intrinsic=f)
betas, gammas = old.arm_schedule(ARMS, 0.3, 0.99, 0.98)
class Batch:  # what update_split_q reads from a flashbax sample
    def __init__(s, e): s.experience, s.probabilities, s.indices = e, probs, jnp.arange(B)
class NoBuf:
    def set_priorities(s, b, i, p): return b
def old_step(se, si, exp):
    out = old.update_split_q(se, si, jnp.zeros(()), Batch(exp), net, cfg, betas, gammas, NoBuf())
    return out[0], out[1], out[2]

inner = list(learner.q.params["params"].keys())[0]
stack = lambda a, b: jax.tree.map(lambda x, y: jnp.stack([x, y]), a, b)
params = {"params": {inner: stack(pe["params"], pi["params"])}}
# identical target params -> identical math
nl, (np_, nm) = agent.q_loss(params, params, new_exp, probs)
na = old.compute_shared_next_action(net, pe, pi, old_exp, cfg, betas)
le, (pre, _) = old.r2d2_loss(pe, pe, old_exp, probs, net, cfg, arm_gammas=gammas, next_action=na, reward_key="reward", carry_key="carry_e")
li, (pri, _) = old.r2d2_loss(pi, pi, old_exp, probs, net, cfg, arm_gammas=gammas, next_action=na, reward_key="intrinsic_reward", carry_key="carry_i")
# feat/split-q passes exp.arm[:, 0] (first burn-in step); the merged code weights p_i by the
# arm of the first LEARNED step (docs/AGENT57_DESIGN.md), so compare at index BURN
op = old.split_q_priority_from(pre, pri, old_exp.arm[:, BURN], betas)
print("loss new", float(nl), "old", float(le + li), "rel.err", float(abs(nl - le - li) / abs(le + li)))
print("prio max abs diff", float(jnp.abs(np_ - op).max()))

def bench(name, fn, args, n=10):
    fn = jax.jit(fn)
    t = time.perf_counter(); c = fn.lower(*args).compile(); ct = time.perf_counter() - t
    jax.block_until_ready(c(*args))
    t = time.perf_counter()
    for _ in range(n): jax.block_until_ready(c(*args))
    print(f"{name}: compile {ct:.1f}s | {1000 * (time.perf_counter() - t) / n:.1f} ms/update")
    return c
bench("feat/split-q update_split_q", old_step, (se, si, old_exp))
bench("merged learner (q_loss)   ", new_step, (learner.q, new_exp))
