# Config decisions — why each number is what it is

Working notes for the report's adaptations section. Every value we did not simply
inherit needs a one-line justification there; this is where those lines come from.

Keep this updated as values change. Reconstructing reasoning in week six does not work.

---

## Measured, not guessed

Observation shapes, read from the environment on Pong (2026-08):

| Mode | Shape | Elements | Dtype | Bytes/transition |
|---|---|---|---|---|
| Object-centric | `(104,)` | 104 | float32 | ~441 |
| Pixel | `(4, 84, 84)` | 28,224 | uint8 | ~28,250 |

Object-centric is 26 features x 4 stacked frames. Pixel is 4 stacked 84x84 grayscale
frames. The ~25 bytes on top of the observation are action, reward, done, prev_action
and prev_reward.

**Pixel observations are 270x heavier than object-centric.** Every memory decision
below follows from that one fact.

Note the element count varies per game — Montezuma has more objects than Pong, so its
OC observation is larger. Immaterial for memory, but it is why `obs_shape` must be read
from the env and never hardcoded.

---

## BUFFER_SIZE

**Object-centric: 1,000,000** (unchanged from `dqn_oc_original.yaml`)

```
441 bytes x 1,000,000 = 441 MB
```

Comfortably fits. No reason to reduce, and matching DQN keeps the stage-1 gate
comparison clean. R2D2's own value is 4x10^6 observations; we keep dqn's 1x10^6.

**Pixel: 100,000** (reduced from 1,000,000)

```
28,250 bytes x 1,000,000 = 28.2 GB   -> impossible on any hardware we have
28,250 bytes x   100,000 =  2.8 GB   -> fits comfortably on the 10GB pool
```

Not our invention: `dqn_rgb_original.yaml`, `rainbow_oc_original.yaml` and
`rainbow_rgb_original.yaml` all ship with `# BUFFER_SIZE: 1000000` commented out and
100,000 in its place. We follow the same precedent for the same reason.

Considered raising the pixel buffer to ~150,000, which the 10GB pool would support.
Rejected: dqn_rgb and rainbow_rgb both use 100,000, and matching them keeps the stage-1
gate ("R2D2 beats DQN at equal steps") free of a memory confound. Revisit only if a
specific experiment justifies it.

All runs happen on the 10GB student pool, not the local 6GB card. The pixel buffer at
100,000 (~2.8 GB) leaves comfortable headroom there.


**Unit warning for the report:** `BUFFER_SIZE` counts *transitions*, the same unit as
dqn and rainbow. The sequence count is derived in code. An earlier draft had it count
sequences, which would have made our config silently incomparable to the neighbouring
ones.

**The uint8 dtype is load-bearing.** Storing pixel observations as float32 turns 2.8 GB
into 11.3 GB — instantly impossible. This is what the `rainbow.py:244` pattern
(`uint8 if PIXEL_BASED else float32`) protects. Object-centric observations pass through
a normalisation wrapper and genuinely are float32; storing *those* as uint8 rounds 0.47
to 0 and silently destroys them. The dtype must follow the mode, both directions.

---

## R2D2 hyperparameters — all verified against the paper's table

No UNVERIFIED markers remain in either config.

| Key | Value | Note |
|---|---|---|
| `SEQUENCE_LENGTH` | 80 | the paper's m = 80 |
| `SEQUENCE_PERIOD` | 40 | stride between sequence starts, giving the 40-step overlap |
| `BURN_IN_LENGTH` | 40 | paper uses l = 40. A *separate* parameter from the overlap, which happens to share the value |
| `BATCH_SIZE` | 64 | 64 sequences, i.e. batches of 64 x 80 observations |
| `TARGET_NETWORK_FREQUENCY` | 2500 | counts **learner** steps, not env steps |
| `N_STEP` | 5 | n-step double Q-learning |
| `GAMMA` | 0.997 | dqn uses 0.99 |
| `LEARNING_RATE` | 2e-4 | dqn uses 1e-4 |
| `ADAM_EPS` | 1e-6 | `dqn.py` hardcodes 1e-4; rainbow reads it from config |
| `ADAM_B1` / `ADAM_B2` | 0.9 / 0.999 | identical to optax's defaults; listed explicitly so the config is a complete record. **Must be passed to `optax.adam` in `single_run`** — dqn and rainbow pass no betas at all |
| `PRIORITY_EXPONENT` | 0.9 | rainbow uses 0.5 |
| `PRIORITY_ETA` | 0.9 | priority = eta*max\|TD\| + (1-eta)*mean\|TD\| over the sequence |
| `IMPORTANCE_SAMPLING_EXPONENT` | 0.6 | fixed, **not** rainbow's 0.4 -> 1.0 anneal |

### Network architecture

```
torso -> pre-LSTM linear -> LSTM -> post-LSTM linear -> dueling head
```

| Key | Value |
|---|---|
| `PRE_LSTM_UNITS` | 512 |
| `HIDDEN_SIZE` | 512 (LSTM units) |
| `POST_LSTM_UNITS` | 256 |
| `DUELING_UNITS` | 256 (each of the V and A branches) |

**Pixel torso is 4 conv layers**: channels 32/64/128/128, kernels 7/5/5/3, strides
4/2/2/1. This is *deeper than dqn.py's 3-layer torso* — Pratik cannot reuse `QNetwork`
unchanged.

**Caveat on the source.** The table we read was reproduced in a later DeepMind paper
describing its own R2D2 variant (Schaul et al., "The Phenomenon of Policy Churn",
arXiv:2206.00730, Appendix B.2). Its target-update interval reads 400 updates, whereas
the original paper says 2500 learner steps — so that table has been modified for a
reduced-scale setup. Architecture numbers are trustworthy; prefer the original paper
wherever training hyperparameters disagree.

---

## Confirmed externally

**`MLP_WIDTH: 512`, `MLP_DEPTH: 2`** (object-centric only) — Group 27 ran a width sweep
on Pong in object-centric mode with everything else held fixed:

| Network | Final return at 10M frames |
|---|---|
| 512 x 512 | +18.0, solved and plateaued |
| 128 -> 64 -> 32 | +12.4, solved late, still rising |
| 64 -> 32 -> 16 | -2.4, mostly flat |

Only 512x512 plateaued within the budget. Raban confirmed 512x512 in the channel. We
do not need to repeat this experiment — cite Group 27 and the channel confirmation.

These two keys are absent from the pixel config, which uses the CNN torso instead.

---

## Inherited from the repo, unchanged

`LEARNING_STARTS: 80000`, `SCAN_STEPS: 1000`, `GRADIENT_STEPS: 1`, `NUM_ENVS: 1`,
`TRAIN_FREQUENCY: 4`, `TAU: 1.0`, `TOTAL_TIMESTEPS: 10000000`.

These match `dqn_oc_original.yaml` / `rainbow_oc_original.yaml`.

**Why matching matters:** the stage-1 gate is "R2D2 beats DQN on Pong at equal steps."
If anything other than the algorithm differs — buffer size, step budget, number of
envs, wrapper chain — the comparison cannot distinguish "the LSTM helped" from "we gave
it more resources."

`NUM_ENVS: 1` is the repo default across every existing config. We override to 8 on the
command line for faster local iteration, but the file matches the others.

`make_env` is copied byte-identical from `dqn.py` for the same reason. If it ever needs
changing, change it in both files and say so in the PR.

---

## Deliberate deviations from the paper

These go straight into the report's adaptations section.

| What | Paper | Ours | Why |
|---|---|---|---|
| Replay buffer | 4e6 observations | 1e6 (OC) / 1e5 (RGB) | VRAM, and matching dqn/rainbow keeps the gate fair |
| Epsilon | fixed 0.01 | anneal 1.0 -> 0.01 | matches DQN's schedule; pending Raban's answer |
| Actors | 256 distributed | 1 vectorised env | single-GPU synchronous architecture — the core adaptation of the whole project |

The distributed-to-synchronous translation is the architectural contribution worth
claiming in the PR. Acme (JAX R2D2) and DRLearner (JAX Agent57) both exist, but both
are distributed — Reverb replay servers, Launchpad, many actor processes. Nobody has
built this as a single-file, end-to-end-jitted, single-GPU agent.

---

## Ours — reduced scale (each needs a report line)

Not yet in the config; these arrive at stage 2 and beyond.

| Key | Paper | Ours | Reason |
|---|---|---|---|
| `NUM_ARMS` | 32 | 8 | VRAM: each arm is a policy trained in parallel |
| `EPISODIC_MEMORY_SIZE` | 30,000 | 2,048 | VRAM, and JAX needs a fixed-size circular buffer |
| `UCB_WINDOW` | 160 | 30 | scales with the reduced step budget |
| `TOTAL_TIMESTEPS` | 10M+ | TBD | GPU hours available, not a design choice |

`STAGE` (`r2d2` / `ngu` / `split_q` / `agent57`) is entirely ours — it does not exist
in the paper. It selects how many components are enabled so that all four ablation
rungs come from one file rather than four near-identical copies that drift apart.

---

## Open question for Raban

`rainbow_rgb_original.yaml` contains this precedent:

```yaml
# TARGET_NETWORK_FREQUENCY: 8000 
TARGET_NETWORK_FREQUENCY: 1000  #Adapted to fit to DQN (cleanRL HP's)
```

They abandoned Rainbow's own paper value in order to match DQN's — prioritising
comparability over fidelity.

**Ask:** for the ablation to be fair, should we match DQN's hyperparameters where they
differ from R2D2's, or keep the paper's?

This now affects **three** values, not one:

| Key | R2D2 | DQN | We currently use |
|---|---|---|---|
| `GAMMA` | 0.997 | 0.99 | R2D2's |
| `LEARNING_RATE` | 2e-4 | 1e-4 | R2D2's |
| epsilon | fixed 0.01 | anneal 1.0 -> 0.01 | DQN's |

We are currently inconsistent — following the paper on two and DQN on one. Worth
resolving before generating any results.

---

## Caveat to state in the report

`BATCH_SIZE: 64` in our config means 64 *sequences* of 80 steps = 5,120 transitions per
gradient update. DQN's `BATCH_SIZE: 32` means 32 transitions.

So "equal environment steps" does not mean equal compute — we do roughly 160x more work
per update. This asymmetry is inherent to R2D2 rather than something we introduced, but
it must be stated explicitly or a reviewer will raise it.

Related unit trap: environment steps, learner steps and frames are three different
counters. `frame_skip=4`, so 10M frames = 2.5M agent steps. Group 27 reports frames;
this repo counts steps. State the conversion once in the report.

---

## Local environment notes

`config/config.yaml` ships with `NAME_INITIALS: "RE"` (Raban's). Changed locally to
`"KC"` but **deliberately not committed** — committing it would stamp Khalil's initials
on Pratik's and Aman's runs too. Pass `NAME_INITIALS=KC` on the command line instead.

`ENTITY: "jaxatari"` is a W&B org we do not yet have access to, so runs are offline
until that is granted. Note that the `WANDB_MODE` config key does not reach
`wandb.init()` — use the `WANDB_MODE` environment variable instead. Possibly a real bug
worth fixing in the PR.
