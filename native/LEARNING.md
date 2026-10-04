# Native learning validation

This is a bounded experiment around the original SONIC trainer, not another PPO implementation. The two-iteration integration test remains separate in `TRAINING.md`.

## Questions and controls

1. Does more native training actually learn stable balance?
2. Does the policy use the future reference, rather than only reacting to current state?
3. Does coarse FSQ precision prevent different references from reaching the decoder?

The established teacher-free task, initial weights, network widths, observation normalization, reward and physics are retained. The declared precision comparison is five versus 32 levels per scalar, with the same two 2D tokens. FSQ has no trainable parameters here; the test verifies that all initial learned parameters are identical between these conditions. The precision variant replaces only the parameter-free quantizer with a newly configured instance of the same official FSQ class; no quantizer forward or trainer method is patched.

Learning-rate experiments are reported separately from the precision comparison. They are diagnostic configurations, not a claim of optimal hyperparameters. A learning-rate change affects both actor and critic under the established shared optimizer.

## A concrete long-run defect

The initial longer run stopped in the official `process_ep_infos` aggregator: the local environment emitted a `length` metric only on steps with a termination, while the aggregator assumes consistent keys. A two-iteration smoke had not encountered that sequence. The fix removes this duplicate sparse metric; the original trainer already computes episode length from termination flags. No reward, reset, physics or PPO equation was changed. A regression test reproduces a failed step followed by a nonfailed step and calls the original aggregator.

## Reference interventions

The evaluator holds initial conditions and true target trajectories fixed. It compares the normal policy input against zeroed or sign-reversed `tokenizer` input. The target is **not** changed when the input is intervened on. A policy that returns the same behavior after reference removal has not demonstrated reference-conditioned tracking.

`token_probe` evaluates 65 settled target references spanning [-0.4,0.4] m while keeping proprioception fixed. It reports unique quantized vectors and action spread. A response is necessary, not sufficient, for correct tracking; it is not a test of all possible trajectories.

The evaluator reports survival, pre-failure MAE, survivor-conditioned complete-episode/tail MAE and push reachability. Push is applied at control step 120 only to still-alive initial episodes. No reached push means `post_push_mae=null`, never zero error or successful recovery. Evaluation creates its own environments and does not train the policy or change Torch RNG state.

## Execution and checkpoints

Use the existing CPU environment from `native/training.lock` and the same pinned upstream checkout. From the repository root:

```bash
export SONIC_UPSTREAM=/path/to/pinned/GR00T-WholeBodyControl
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1
.venv-native/bin/python native/learning.py --output-dir /tmp/native-learning \
  --iterations 128 --seed 9101 --fsq-levels 5 --learning-rate 0.0003
```

`--iterations` is bounded to 1..512. Checkpoints include the FSQ level configuration because the quantizer's nonpersistent buffers are not carried by a plain weights-only state dictionary. A reload must reconstruct the right quantizer, not silently fall back to five levels. The driver verifies exact same-input action parity after reload. Checkpoints still are not exact-resume snapshots of optimizer, RNG and simulator state.

`learning.json` contains source hashes, declared settings, original-method call counts, actual optimizer/physics counts, validation checkpoints and reference interventions. CLI failures replace an older success report with `execution=FAILED`. `execution=COMPLETE` asserts only completion of the requested computation, not successful control. `progress.jsonl` and console logs are transient run outputs, not project authority.

The fixed checkpoint evaluation seed 8080 is a **validation** set. It can be used for diagnosing/selecting a configuration. The separate final audit uses other declared seeds and is not used to choose checkpoints or hyperparameters. Results from a single model initialization do not establish seed-robust superiority or sim-to-real readiness.

## Fixed final-audit protocol

The final audit uses the 512-iteration snapshot of the 32-level / learning-rate 0.003 run. It is the final budget checkpoint, not the best checkpoint selected from validation. Two separate evaluation seeds, **28081 and 28082**, each define 16 initial episodes. Run clean, push, zero-reference and negated-reference conditions on the same per-seed initial states and true targets. Do not retune or choose another checkpoint using those audit outcomes. These 32 evaluation episodes still correspond to **one training initialization**, not 32 independently trained policies.

Use `load_policy(upstream, checkpoint)` and `audit_policy(policy, seed, episodes=16)` from `learning.py` to reproduce the audit. Report each condition and each seed, including failure and missing post-push data. Passing a response probe, a logging test or an optimizer-count assertion never implies a successful tracking controller.

## Measured results

[Summary](../evidence/native_learning/summary.json), [raw final run](../evidence/native_learning/combined512.json), [per-episode holdout](../evidence/native_learning/holdout.json), and [native weights](../evidence/native_learning/weights-final.pt) are stored together with SHA-256 checks. These weights were trained locally from random initialization; they are not an NVIDIA humanoid checkpoint.

At the same **128-iteration budget**, every tested setting still had clean survival 0/8:

| FSQ levels per scalar | Shared optimizer LR | Mean survival [s] | Unique token vectors in the 65-goal probe |
|---:|---:|---:|---:|
| 5 | 0.0003 | 1.270 | 1 |
| 32 | 0.0003 | 1.405 | 3 |
| 5 | 0.003 | 1.980 | 2 |
| 32 | 0.003 | 1.495 | 9 |

The first baseline was repeated with the finalized driver and reproduced the same numerical outcome. The original default two-iteration integration result also reproduces unchanged after the logging fix; only its recorded environment-source hash changes.

For the **32-level / LR 0.003** run, the validation trajectory was:

| PPO iterations | Clean survival | Mean survival [s] | Complete-episode MAE [m] |
|---:|---:|---:|---:|
| 128 | 0/8 | 1.495 | not observed |
| 256 | 0/8 | 3.495 | not observed |
| 512 | 8/8 | 10.000 | 0.0948 |

The final run consumed **524,288 control transitions, 1,048,576 MuJoCo steps and 2,048 optimizer updates**. Original SONIC PPO/GAE/auxiliary methods ran throughout; no teacher or separate stabilizing controller was introduced. Its fixed-state probe used 13 distinct token vectors.

The fixed final checkpoint was then evaluated on the two separate audit seeds (16 cases each):

| Policy input / disturbance condition | 10-second survival | Tracking MAE over complete episodes [m] |
|---|---:|---:|
| Correct future reference | 32/32 | 0.1073 |
| Correct reference + specified velocity impulse | 32/32 | 0.1300 |
| Reference input erased; true target unchanged | 32/32 | 0.2267 |
| Reference input sign reversed; true target unchanged | 32/32 | 0.3590 |

All 32 pushed episodes reached the impulse at step 120 and survived to step 500. Mean **post-push** MAE was 0.1269 m. This is an observed survival/tracking result under that protocol, not a measurement of sustained-recovery time or a stability certificate.

The counterfactual policies also stayed upright but tracked worse. Thus, survival alone would miss the role of the reference; the paired input interventions provide evidence of useful reference-conditioned control in this trained policy. They do not prove universal superiority of this architecture or configuration. In particular, **a 512-iteration five-level comparison was not performed**, so 32 levels have not been shown necessary or optimal.

Remaining boundaries: only one training initialization; near-upright starts; narrow goal range; fixed plant and accurate simulated state; one impulse protocol; 10-second horizon. No hardware, delay/noise, morphology variation, humanoid performance or long-horizon robustness claim is made. The browser continues to use its separate JS teaching policy; these native weights are not silently installed into its controller selector.
