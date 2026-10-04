# Same-present / same-endpoint temporal-information audit

## Question

The previous concept experiment corrupted frame order and used eight previously inspected diagnostic trajectories. This experiment instead constructs a **new physically realized motion family** with shared stationary history/current state and equal terminal full state, but distinct interior motion. The Encoders, shared native Decoder and FSQ are frozen before any case is evaluated.

This is the next CartPole information test from issue #7, not new PPO training or a robot pivot. `temporal_audit_protocol.json` declares cases, tolerances, checkpoint hash, reference conditions and actual-state perturbations before policy outcomes are inspected.

## Physical construction, not an invented reference

`temporal_pairs.py` uses **MuJoCo's installed `minimize.least_squares`**, not a new solver or dependency. Seven bounded normalized actuator values are each held for four control steps. Each control step is two original 0.01-second MuJoCo steps. The 28-step maneuver lasts 0.56 seconds.

The declared candidates are the Cartesian product of interior ticks 8, 12, 16 and cart-displacement amplitudes 1, 2, 4, 6 mm; initial cart positions come from seed 73421 in [-0.25,0.25] m. Starts are stationary and upright. The solver matches the terminal **full state** `[x, x_dot, theta, theta_dot]` to the start and the specified interior displacement. Controls are bounded at ±0.25, equivalent to ±2.5 N in this plant. A mirrored control sequence supplies the paired motion and is independently rerun in the nonlinear simulator.

Full-state endpoint absolute tolerance is `1e-7` in each component's native unit. Interior position difference must exceed 0.1 mm. The first recorded action difference is separately checked against `1e-5` normalized units. A solver termination message is not sufficient: physical replay, endpoint and interior constraints, failure bounds and stationary prehistory are checked directly. All rejected candidates and optimization diagnostics remain in `physical_pairs.npz`; no rejected case is replaced by a synthetic success.

Official solver reference: https://mujoco.readthedocs.io/en/stable/python.html#minimize

## What information is actually identical?

At the branch point, the two motions have identical actual initial state, current reference and physically generated zero-control prehistory. They also have the same intended terminal equilibrium; measured terminal differences are retained, never overwritten in the physical data.

For the **initial-action information comparison only**, terminal input roundoff is replaced by the canonical common endpoint, after enforcing the declared tolerance. Raw-versus-canonical action differences and endpoint changes are recorded. This prevents tiny solver residuals from revealing the branch to an ostensibly endpoint-only input.

The fixed model receives four input conditions: true future frames; repeated current reference; causal reference history; or interpolation using only the current and terminal reference. History is supplied in an explicitly documented current-to-past layout. These reduced-cue paths are **interventions on the same previously trained future Encoder**, not models independently trained to make optimal use of current/history/endpoints.

The paired mean of the two recorded first forces minimizes squared prediction error for any single common force prediction. The report computes this elementary bound. It is only a bound on **predicting these recorded demonstration controls**, not a proof that every successful feedback strategy must copy those forces or that this SONIC policy is an optimal anticipatory controller.

## Closed-loop replay and missing-future handling

The same frozen model is replayed from matched initial states with two actual-state settings: unperturbed and the declared small `[0, .02, .001, -.02]` offset. Desired reference data remain unchanged. Both joint-state and marker-coordinate Encoders feed the exact same saved native Decoder.

The evaluation lasts 28 controls (0.56 seconds). For later fixed-size previews, another 28 zero-control steps are actually simulated from each recorded terminal state. This is a measured tail, not repeated-state padding. Policy physics and reference generation are separate; no online actual-future state is passed into the policy. Short-horizon completion, per-component errors, terminal errors and actual forces are recorded, including failures.

A new motion family can lie outside the training distribution even if individual state components have familiar numerical ranges. Failure here is not by itself an implementation defect, evidence that preview is useless, or proof that CartPole cannot express the concept. Conversely, passing a 0.56-second replay does not extend earlier five/ten-second stability evidence.

## Reproduction and ownership

Use the existing `native/training.lock` environment and pinned SONIC upstream. No dependencies, services, controller training or other robot assets are added.

```bash
export SONIC_UPSTREAM=/path/to/pinned/GR00T-WholeBodyControl
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
python -m pytest native/test_temporal_pairs.py native/test_temporal_audit.py -q
python native/temporal_audit.py --output-dir /tmp/sonic-temporal-audit
```

`audit.json` records protocol/source/checkpoint hashes, all candidates, accepted and rejected counts, frozen-model verification, per-pair probes and full replay outcomes. `physical_pairs.npz` is numeric/pickle-free and retains raw states, controls, histories and solver traces. Scientific claims depend on the measured results and scope, not a test requiring the policy to win.

`native/PIVOT.md` continues to own the embodiment decision. The existence of feasible matched-endpoint CartPole pairs addresses an experiment-design gap; it does not require an arm or humanoid. Multi-actuator coordination, continuous redundancy, contact support and 3D retargeting remain separate missing-structure questions.

## Observed results

The declared 12 candidates produced **9 accepted physical pairs / 3 rejected candidates**. All nine accepted pairs have distinct recorded first actions. The largest accepted full-state endpoint component error was about `2.10e-16`, well inside the predeclared `1e-7` tolerance. Cases `(tick 8, amplitude 4 mm)`, `(8, 6 mm)` and `(16, 6 mm)` failed the endpoint/interior constraints with this bounded solver/profile family. Their traces are retained; these failures are **not a proof that those motions are physically impossible**.

At identical initial state/history/canonical endpoint, each new Encoder produced different full-preview tokens and actions for **6 of 9 pairs**. Current-only, history-only and endpoint-only inputs were identical across the pair at that branch point. Thus these six differences cannot be attributed just to different present states or a slightly different destination.

However, distinguishing the two futures did not mean predicting the recorded first force correctly. Mean full-preview first-force prediction MSE was `0.2340 N²` (joint Encoder) / `0.2277 N²` (marker Encoder), versus `0.1596 N²` for the optimal single common prediction of the paired recorded forces. Neither Encoder beat that paired common-prediction bound on any accepted pair. This concerns the recorded force labels, **not** a universal successful-control lower bound.

The 0.56-second replay contained 18 accepted trajectories per condition (nine pairs). Both Encoders completed all 18 unperturbed and all 18 perturbed replays in every input condition. The unperturbed cart MAEs were:

| Input condition | Joint Encoder | Marker Encoder |
|---|---:|---:|
| Full future | 1.294 mm | 1.229 mm |
| Current reference only | 1.585 mm | 1.402 mm |
| Causal reference history | 1.378 mm | 1.299 mm |
| Current + terminal reference only | **1.114 mm** | **0.992 mm** |

With the declared actual-state perturbation, full-preview MAE was 2.992 / 2.958 mm, while endpoint-only MAE was 2.857 / 2.852 mm. The full future therefore did **not** dominate the reduced-cue interventions in this new local maneuver family. All completions concern a short 0.56-second horizon and small excursions; they are not an added long-duration stability result.

This closes a prior **experimental construction gap**: CartPole can supply physically valid same-present/same-endpoint but different-interior references. It also narrows interpretation of earlier frame-corruption results: the frozen policy can react to temporal detail, yet that detail does not automatically improve its control on a newly specified motion family. A matched retraining study, broader trajectory support and appropriate causal baselines remain separate work. No new robot or additional policy training was used to conceal this result.

Artifacts: [`audit.json`](../evidence/temporal_audit/audit.json) contains every accepted/rejected case and per-pair/condition result. [`physical_pairs.npz`](../evidence/temporal_audit/physical_pairs.npz) contains raw physics and solver diagnostics. Source and protocol hashes bind the code and declared experiment to these measurements.

### Numerical equality in regression tests

Equal-cue input arrays and quantized token vectors are compared exactly. Neural force outputs use a `1e-7 N` absolute equality tolerance, below the `1e-6 N` action-distinguishability criterion used by the audit. The first clean CI run exposed a `3.7253e-8 N` output difference between identical rows; the test had incorrectly required bitwise zero. Only that regression assertion was corrected. Neither recorded physics, endpoint acceptance, model weights nor reported outcomes were changed.

## Follow-up: separately trained cue baselines

The frozen-input comparison above is preserved unchanged. Its unfamiliar-input-format limitation is addressed by [the matched-training experiment](MATCHED_CUES.md), which separately trains current/history/endpoint and full-future Encoders under the same budget, holds the shared native Decoder fixed, and uses new whole-pair test data plus three Encoder initializations. It reports nominal improvements alongside weak disturbance separation and reference-noise sensitivity. This does not turn either experiment into a global-optimality, full-controller multi-seed or hardware claim. The raw and projected results are owned by `evidence/matched_cues/`.
