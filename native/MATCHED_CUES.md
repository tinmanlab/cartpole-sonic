# Matched-training cue comparison and observation boundary

## What this closes

The previous temporal audit changed the input of an already trained future Encoder. A weak history/current/endpoint result could therefore be due to an unfamiliar input format, not necessarily to insufficient information. This experiment **trains a separate Encoder for each declared cue condition** and compares the final models on new whole physical reference pairs.

This is not a new PPO implementation, a claim of globally optimal baselines, or an attempt to make SONIC win every comparison. The original SONIC `UniversalTokenModule`, `BaseModule`, FSQ and the already trained shared Decoder are reused. Only the selected new Encoder is trainable. Original controller/Decoder weights remain fixed, and the previous native/browser baselines are unchanged.

## Protocol fixed before results

`matched_cue_protocol.json` owns the exact case lists, random seeds, budgets and metrics. The input conditions are full future, current reference only, causal reference history, and current/terminal reference. There are joint-state and marker-coordinate input branches. Each has three independently initialized Encoders, with identical initial learned parameters across cue conditions within that branch/seed. This is **three Encoder initializations conditional on one shared pretrained Decoder**, not three independent whole-controller/PPO training runs.

Each condition receives the same supervised recorded-action loss, optimizer settings, training sample stream and fixed update count. Both signed branches of a reference pair are always present together in a minibatch. This prevents a blind model from exploiting accidental sign imbalance. The final fixed-budget checkpoint is evaluated; validation is not used to choose the seed, epoch, checkpoint or learning rate.

Training matches the recorded demonstration's action at its recorded state. Closed-loop replay separately tests the resulting feedback behavior, including a small initial-state perturbation. Low supervised error does not itself establish stable control. No hidden stabilizer, action relabelling, target changes or new PPO training are added.

## Physical data and independent split

The existing MuJoCo pair generator supplies actual nonlinear trajectories, not invented frame sequences. The terminal full-state, force, interior-displacement and solver-budget checks are retained. All generated/rejected candidates are recorded. Both signs and all time windows of a pair remain in one split.

The protocol declares 36 training candidates, 12 validation candidates and 24 test candidates. Test amplitudes differ from the training amplitudes; initial positions also use a separate seed. They remain within the declared small local task family, so the test is not a broader morphology or hardware transfer experiment. A solver rejection is not proof of physical infeasibility and is never silently replaced by a selected easy case.

Future tails and causal histories are generated using the existing physical replay helpers. No missing states are padded, no reset boundary is crossed, and no current simulator's future is supplied as a reference. At the branch point, current/history/endpoint inputs are identical for the signed pair; canonicalization removes only sub-tolerance endpoint roundoff, without altering raw physics records.

## What the measurements mean

Report both recorded-action prediction and actual closed-loop behavior. The paired common-force predictor gives a squared-error lower bound only for reproducing the two recorded first actions when observations are identical. It is not a universal optimal-control bound: a successful controller need not reproduce a particular demonstration's force sequence.

The selected Encoder is trained anew for its own information format, so the previous frozen-input-format confound is removed. Equal budget does **not** prove global optimizer convergence or equal best attainable performance. Differences can still include finite-data/finite-training effects and limitations of the shared frozen Decoder.

Report individual seeds, mean and sample standard deviation, with surviving/completed counts and actual horizons. Sample standard deviation over three Encoder seeds is not a confidence interval, and correlated time windows are not independent episodes. The 0.56-second paired task does not extend previous five/ten-second survival claims.

## Missing or noisy marker information

The full cart-plus-tip position/velocity representation is invertible in this planar task and has an analytic baseline. Tip-only observations are not: two different cart/pole states can give the same tip position/velocity. The full-observation path therefore rejects missing marker information rather than silently replacing an absent measurement with a zero. No neural Encoder can be assumed to recover information that the supplied observation does not identify.

The noise probe perturbs only active marker-reference coordinates and velocities, using the declared physical-unit noise scales and identical draws across models. Actual proprioception and target actions remain unchanged. This is **offline sensitivity to reference-coordinate noise**, not a demonstration of a noisy-proprioception closed-loop controller, camera perception, occlusion handling or hardware safety. Missing-data rejection and ambiguity evidence close the information-contract question, not sensor-estimator research.

## Reuse and pivot decision

The successful/unsuccessful outcomes of this bounded comparison are evaluated under `native/PIVOT.md`. A lack of convergence, a shared-decoder limitation or an inadequately learned motion family is an experiment/implementation issue unless a required structure is actually absent. Independent actuators, continuous kinematic redundancy, foot-contact transitions and general 3D retargeting are still not represented by the current CartPole plant.

No new robot is installed in this comparison. When the admitted question requires one of those absent structures, reproduce the appropriate pinned upstream small example and port only robot-specific configuration/data/units. Retain this CartPole suite as a regression environment; never promote its learned weights or force normalization into universal robot defaults.

## Checkpoint integrity correction found during review

The initial new fingerprint helper used only `state_dict()`. The installed FSQ registers its level/basis tensors as nonpersistent buffers, so its `state_dict()` is empty. A regression test changed `_levels` and demonstrated that the initial hash failed to notice. The helper now includes `named_buffers()` as well as persistent parameters/buffers. The failing test then passed, before the full experiment was executed. This corrects the new experiment's integrity check; it is not described as a defect in upstream FSQ or a change to quantization behavior.

## Measured final-protocol result

All 24 declared models completed 1,200 selected-Encoder updates each. Data acceptance was **27/36 training pairs, 9/12 validation pairs and 20/24 test pairs**. All 16 rejected generation attempts are retained in `pairs.npz`; the test set has 40 trajectories (20 signed pairs). No test result selected a model, checkpoint or hyperparameter.

For the three independently initialized Encoders in each condition, the test first-force prediction MSE was:

| Trained cue condition | Joint Encoder [N²], mean ± sample SD | Marker Encoder [N²], mean ± sample SD |
|---|---:|---:|
| Full future | **0.00436 ± 0.00081** | **0.02491 ± 0.00419** |
| Current only | 0.19155 ± 0.00140 | 0.19189 ± 0.00157 |
| History only | 0.19155 ± 0.00068 | 0.19584 ± 0.00376 |
| Endpoints only | 0.19155 ± 0.00084 | 0.19467 ± 0.00614 |

The paired-common first-force prediction bound is **0.18748 N²** for this test set. The trained reduced-cue models approach it but cannot identify the signed branch from their identical branch-point information. Full future information allows much lower error for these recorded actions. This is stronger than merely observing different output after corrupting an unfamiliar input, but it remains an imitation/information statement—not a proof that every successful control strategy must use these forces.

All-window force RMSE was 0.569 N (joint full-future) versus 0.692–0.702 N (its reduced-cue conditions), and 0.700 N (marker full-future) versus 0.756–0.844 N. Training deliberately samples the branch point with probability 0.5 in every condition; the report includes both branch-point and all-window metrics so that emphasis is visible.

Closed-loop cart MAE on the 40 unperturbed test trajectories was:

| Trained cue condition | Joint Encoder [mm], mean ± sample SD | Marker Encoder [mm], mean ± sample SD |
|---|---:|---:|
| Full future | **0.393 ± 0.012** | **0.594 ± 0.032** |
| Current only | 1.021 ± 0.019 | 0.793 ± 0.012 |
| History only | 0.871 ± 0.017 | 0.796 ± 0.012 |
| Endpoints only | 1.023 ± 0.023 | 0.807 ± 0.014 |

Every seed/condition completed 40/40 over **0.56 seconds**. This denominator is the same 40 physical test trajectories evaluated under three Encoder initializations, not 120 independent environments or independently trained base controllers. The nominal full-future advantage occurred for every tested Encoder seed in both modalities.

With the predeclared actual-state offset `[0,0.02,0.001,-0.02]`, differences largely collapsed: full-future MAE was **2.629 ± 0.058 mm** (joint) and **2.651 ± 0.148 mm** (marker). Reduced-cue means ranged from 2.650 to 2.710 mm for joints and 2.618 to 2.695 mm for markers. Therefore the experiment does **not** establish robust full-future superiority under state disturbances; marker history was slightly lower on the aggregate. The fixed Decoder, finite data and training support remain material limits.

For the full-future marker Encoder, clean all-window force RMSE averaged 0.700 N. Adding 1 mm reference-position noise and 0.01 m/s reference-velocity noise gave approximately **0.717 N**; using 0.05 m/s reference-velocity noise gave about **1.061 N**. All four declared position/velocity-noise combinations and individual seeds are recorded. This identifies reference-velocity sensitivity, not a validated perception filter or noisy-state control capability.

## Reproduce, inspect and use the artifacts

Use the existing `native/training.lock` environment and the pinned official source. No new package installation is needed beyond that environment.

```bash
export SONIC_UPSTREAM=/path/to/pinned/GR00T-WholeBodyControl
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
python -m pytest native/test_matched_cues.py -q
python native/matched_cues.py --output-dir /tmp/sonic-matched-cues --steps 1200
```

Repository evidence is under `evidence/matched_cues/`. `summary.json` is a derived readable projection; `matched.json.gz` is the losslessly compressed complete execution report (including all predictions, training losses and per-pair rollouts); `pairs.npz` preserves all accepted/rejected physics and solver records; 24 small checkpoint files preserve the selected Encoders. The original Decoder is not duplicated into each checkpoint. Compression avoids committing a nearly 10 MB pretty-printed raw report; no measurements are discarded. Reproduction emits the uncompressed report, whose SHA-256 is recorded alongside the compressed artifact hash.

Tests verify the source/protocol/checkpoint chain, all 24 reloads and test predictions, initialization/sample-stream equality across conditions, original learned Decoder and FSQ invariance, representative physical replay for every modality/condition, and summary arithmetic. The expensive full training comparison is not repeated in every CI run.

## Resolution of the admitted gaps

**Resolved within this protocol:** separate cue-trained baselines, equal data/budgets, paired sign balance, new trajectory/amplitude split, three conditional Encoder initializations, physically valid references, explicit missing-marker rejection, reference-noise sensitivity and frozen-checkpoint integrity. The earlier frozen-input audit remains valid for its own question; the new comparison removes its unfamiliar-input-format confound rather than retroactively rewriting its result.

**Not claimed:** global optimality of any baseline; independent retraining of the full base controller; long-horizon/noisy-proprioception robustness; camera/VR perception; multi-actuator/contact/3D embodiment competence; hardware or sim-to-real readiness. These are outside this local acceptance contract, not hidden successes or reasons to keep repeating the same comparison. The next embodiment is selected by the missing physical property under `PIVOT.md`, not by demanding an unlimited sequence of CartPole optimization wins.
