# Small-environment sufficiency and pivot contract

## 목적

작은 환경은 버리는 데모가 아니라 **구조적 오개념·좌표·시간·학습 경로 오류를 빠르게 잡는 검증 장치**다. 큰 로봇에서 잘될 것이라는 주장을 위해 CartPole 결과를 과장하지 않는다. 반대로 PPO가 한 번 실패했거나 offline imitation이 불안정하다는 이유만으로 로봇을 바꾸지 않는다.

This file owns code-coupled embodiment limits and adapter acceptance. Live task/PR status belongs in GitHub; measured numbers belong in `evidence/native_concepts/`, not this decision rule.

## What constitutes a pivot trigger?

A pivot is justified when the **next admitted concept requires a physical or information structure absent from the current task**. It is not justified solely by an optimizer seed, failed training, a plotting problem, insufficient data, an unimplemented loss or an unrun experiment.

Before declaring a CartPole limitation, record the exact question, which current physical/information feature makes that question unidentifiable, the smallest added feature that resolves it, what evidence/tests transfer, and which new claims remain unvalidated. Classify the item as one of:

| Classification | Meaning | Action |
|---|---|---|
| IMPLEMENTATION_GAP | Code, coordinate, timing, checkpoint, routing or logging defect | Fix in the existing CartPole path |
| EXPERIMENT_GAP | A feasible CartPole comparison/data distribution has not been run | Run one bounded informative comparison |
| ENVIRONMENT_LIMIT | Required actuator/contact/observation structure is absent | Prepare the minimal suitable embodiment |
| VERIFIED_LIMITED | Measured property holds only for the stated scope | Preserve evidence; do not enlarge the claim |

**No success-to-pivot shortcut:** proving reference dependence is not proving future-aware optimal control, multimodal human retargeting, multi-actuator coordination, contact robustness or sim-to-real readiness.

## Sufficiency map

| Concept | CartPole can test | What it cannot certify | Smallest later environment |
|---|---|---|---|
| Reference/proprioception separation | Freeze external motion, perturb actual state, inspect tokens/action/reconstruction | Sensor estimation quality or hardware safety | Stay in CartPole |
| Temporal reference use | Shared-current-state physical branches, matched current-only baseline, preserve endpoints while corrupting interior frames | Benefit of an online planner without its separate benchmark | Stay in CartPole |
| Shared latent, multiple encoders | Joint-state and marker coordinate representations of the same measured motion, one frozen shared decoder, mixed-row routing | Human morphology/retargeting, visual occlusion, real multimodal semantic equivalence | Stay for interface tests; expand only for actual missing ambiguity |
| FSQ and reconstruction | Quantization grids/gradients, held-out reconstruction, no-aux comparisons, input interventions | Universal necessity or optimal token counts | Stay in CartPole |
| Multiple independently actuated joints | Current plant has one input and an unactuated pole | Independent joint coordination, output-order permutation across actuators | Fixed-base planar 2R Reacher |
| Continuous kinematic redundancy for planar x/y target | Current task has no redundant actuation | Null-space behavior and multiple continuously variable postures for the same endpoint | Planar 3R arm, not a 2R arm |
| Hybrid contact/support transitions | Current model disables contacts and uses a rail constraint | Foot impacts, alternating support, slip, contact inference, floating-base balance | Small planar biped/walker, not a fixed-base arm |
| General 3D orientation and partial human observations | Current motion and coordinate transforms are planar and invertible | SO(3) conventions, heading invariance, 3D body mapping, human-to-robot ambiguity | Minimal 3D articulated model first; humanoid only when its body/contact structure is the question |

A 2R arm is **not continuously redundant** for a generic two-dimensional endpoint position task. It can expose multiple inverse-kinematic branches, but that is not the same as a continuous null space. A planar walker is **not** a general 3D humanoid. Adding more bodies without a missing-property reason is not progress.

## Preferred reuse candidates (prepared, not installed or implemented here)

Use the existing MuJoCo/native SONIC path. Do not create another PPO, simulator service, new repository, generic robot-schema system or orchestration layer.

**Independent-joint/coordinate lane:** DeepMind Control Suite `reacher.xml` + `reacher.py` from `google-deepmind/dm_control`. Candidate source snapshot inspected: `a04e3e4cf56c12117d2294bb090f9acec21e5c67`; `reacher.xml` Git blob `343f799c01ec76abd396e74bcb011b7088f02273`. Preserve the original task and assets as a reproducibility baseline before adding a reference-tracking adaptation. Its original point-reaching reward is not already a SONIC motion-tracking task.

- https://github.com/google-deepmind/dm_control/blob/a04e3e4cf56c12117d2294bb090f9acec21e5c67/dm_control/suite/reacher.xml
- https://github.com/google-deepmind/dm_control/blob/a04e3e4cf56c12117d2294bb090f9acec21e5c67/dm_control/suite/reacher.py

**Contact lane, only when contact is the question:** the same project's planar `walker.xml` / `walker.py`. This has a mobile planar torso and actuated legs; it introduces contact rather than pretending the cart rail is a foot-contact equivalent. Load the original stand/walk task and document its control/physics rates before any SONIC adaptation.

- https://github.com/google-deepmind/dm_control/blob/a04e3e4cf56c12117d2294bb090f9acec21e5c67/dm_control/suite/walker.xml
- https://github.com/google-deepmind/dm_control/blob/a04e3e4cf56c12117d2294bb090f9acec21e5c67/dm_control/suite/walker.py

The candidate snapshot is a source reference, not evidence that this environment has been installed, trained or validated in this project. Resolve and pin its tested dependency set when that lane is admitted. Keep upstream Apache-2.0 notices and all included assets; do not copy a model fragment and lose its includes or attribution.

## What carries over, and what must change?

Carry over original pinned SONIC neural/trainer implementations, provenance checks, source/weight/config coupling, episode-wise data splits, immutable reference replay, branch routing tests, missing-data metrics, gradient/freezing assertions and honest claim boundaries.

Do **not** carry over CartPole weights, state normalization, one-output force scaling, the rail model, pole-tip formulas, action indices, failure angles, rewards or dataset dimensions as universal defaults. Architecture reuse is not zero-shot weight transfer.

For a 2R adaptation, define joint order `[q1,q2,qdot1,qdot2]`, body/site names, two actuator orders and units, local/world endpoint frame, angle wrapping, velocity Jacobians, joint/effort limits, resets and physically feasible motion references. Choose direct torque or position-target-plus-PD explicitly; the current ±10 N cart-force output is neither two torques nor two joint-position targets. A desired endpoint path alone may not specify a unique joint path; the paired representation test must state whether an elbow marker, branch label or trajectory context resolves that ambiguity.

The official SONIC new-embodiment guide emphasizes robot assets, joint/body ordering, actuator/action scales, retargeted motion and reward/termination body names. Reusing its module graph does not remove these responsibilities:
https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/new_embodiments.html

## Bounded first implementation packet for an admitted pivot

1. Reproduce the selected upstream example **unmodified**, including assets and timestep. Capture baseline observations/actions and test name/order roundtrips; no new learning claim.
2. Add only the robot-specific environment/config/loss bindings to the current native path. Preserve the original trainer and shared-latent model. Reject invalid dimensions/actuator names and unobservable paired inputs rather than filling fake humanoid fields.
3. Generate and replay a small physically verified motion bank. Split by whole trajectory. Reference and sensed current state remain separate; terminal, truncation and task completion are named correctly.
4. Re-run the concept tests, then a bounded original-trainer update and fixed-weights replay. Evaluate the **new missing property**, not only survival or reward. Retain CartPole tests as regression tests; do not delete them after pivot.

Stop and escalate before new hardware control, credentials/privileges, deployment or material compute expansion. Planning a pivot does not itself authorize those effects. Remaining feasible CartPole tests may continue independently; they do not need to wait for a new robot, and a new robot is not an excuse to leave a known local defect unfixed.
