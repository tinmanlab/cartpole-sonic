# CartPole SONIC

> A small, interactive teaching lab for understanding the information flow behind NVIDIA GEAR-SONIC: **Encoder → latent → VQ/VQ-VAE → FSQ motion token → Dynamic Decoder → physical control**.

[**Open the live lab →**](https://tinmanlab.github.io/cartpole-sonic/) · [FSQ lesson](https://tinmanlab.github.io/cartpole-sonic/?lesson=fsq) · [Whole SONIC view](https://tinmanlab.github.io/cartpole-sonic/?lesson=sonic)

<p align="center">
  <a href="https://tinmanlab.github.io/cartpole-sonic/">
    <img src="media/cartpole-sonic-demo.gif" alt="CartPole SONIC interactive learning lab" width="100%">
  </a>
</p>

[High-resolution MP4](media/cartpole-sonic-demo.mp4)

## What this project is

This is **not a reimplementation of GEAR-SONIC's humanoid capability**.

It is a structural teaching model that keeps the control graph intact while replacing a high-DoF humanoid with one CartPole:

```text
high-level goal
      ↓
planner / future reference
      ↓
Encoder
      ↓
latent z
      ↓
VQ or FSQ
      ↓
motion token q
      │
      │      current robot proprioception
      └──────────────┐
                     ↓
              Dynamic Decoder
                     ↓
                   action
                     ↓
              MuJoCo physics
                     ↓
               measured state
```

Two training-only paths are shown separately:

```text
token → Kinematic Decoder → future-motion reconstruction  [auxiliary]
rollout → Critic → GAE / PPO                              [training only]
```

The point is to make each role visible before returning to a full humanoid system.

---

## The first question: why compress and then reconstruct?

This is the core idea behind the Encoder/Decoder lesson.

Suppose the planner provides eight future frames:

```text
[x₁, ẋ₁, x₂, ẋ₂, ... x₈, ẋ₈] = 16 numbers
```

The Encoder maps them into a much smaller latent:

```text
16-D future reference
        ↓ Encoder
     [z₁, z₂]
        2-D
```

Why deliberately throw information away?

Because the bottleneck forces the network to find a **compact representation of what matters for the motion**, instead of copying every input number independently. A compact latent is also easier to discretize into reusable motion tokens.

Then why decode it back to 16-D?

```text
[z₁, z₂]
   ↓ Kinematic Decoder
reconstructed future reference
```

**Not because deployment needs the original input again.**

The reconstruction path is an auxiliary training test:

> “Did the small latent actually retain enough motion information?”

If the Decoder cannot reconstruct the reference, the latent probably discarded useful motion information.

The deployed physical controller uses a different decoder:

```text
motion token + current robot state
              ↓
        Dynamic Decoder
              ↓
            action
```

This distinction is one of the main reasons this lab exists.

---

## Learning path

The main path is intentionally short:

```text
Encoder / Autoencoder
        ↓
Vector Quantization
        ↓
VQ-VAE
        ↓
FSQ
        ↓
What actually learns?
        ↓
Motion token
        ↓
Dynamic Decoder
        ↓
PPO physical tracking
        ↓
Full SONIC structure
```

**VAE is an optional side branch**, not a required prerequisite for FSQ:

```text
Autoencoder
├── VAE      : probabilistic continuous latent (μ, σ, KL)
└── VQ-VAE   : discrete latent via learned vector codebook
                 ↓
                FSQ : discrete latent without a learned vector codebook
```

The page now uses one fixed teaching shell in every chapter:

```text
[ same CartPole simulation ] [ same-size main visualization ] [ lesson guide ]
                              ↓
                    [ stable SONIC process strip ]
```

The main path is:

| Step | Lesson | Main question |
|---:|---|---|
| 1 | Encoder / AE | Why compress 16-D to 2-D and reconstruct it? |
| 2 | VQ | Why does a learned vector codebook help create discrete symbols? |
| 3 | VQ-VAE | Why are STE, commitment, and codebook learning needed? |
| 4 | FSQ | What does FSQ remove from VQ? |
| 5 | What learns? | Does FSQ learn? Which modules actually receive parameter updates? |
| 6 | Motion token | What exactly is being tokenized? |
| 7 | Dynamic Decoder | Why does the token still need proprioception? |
| 8 | PPO training | What actually trains physical tracking? |
| 9 | GEAR-SONIC | How do the toy blocks map back to SONIC? |

`VAE` is shown as an **optional branch from Autoencoder**, then the learner returns to VQ. It is not presented as a prerequisite for FSQ.

---

## Reference, token, and robot state are different things

The live UI deliberately separates them.

```text
Reference world
planner future motion
      ↓
Encoder
      ↓
continuous latent z
      ↓
FSQ
      ↓
motion token q
```

The blue future reference is **upstream of FSQ**. It is not the command sent to the motor.

```text
motion token q       actual robot proprioception
      │                        │
      └──────────┬─────────────┘
                 ↓
          Dynamic Decoder
                 ↓
              action
                 ↓
          actual MuJoCo robot
```

So the semantic roles are:

- **reference** = desired future motion, before Encoder/FSQ
- **latent z** = continuous Encoder output
- **token q** = post-FSQ compact motion representation
- **proprioception** = measured state of the actual robot
- **action** = Dynamic Decoder output applied to the actual robot

For that reason, the Simulation panel now draws only the actual robot. Reference trajectories live in the separate center visualization instead of being overlaid as a ghost robot.

---

## FSQ in one minute

VQ uses a learned vector codebook:

```text
continuous z
     ↓ nearest learned vector
codebook: ● ● ● ● ● ● ● ●
     ↓
discrete q
```

FSQ removes the learned vector lookup.

For this toy, each latent scalar uses five finite levels. The implementation follows the odd-level form of the FSQ reference logic:

```text
qᵢ = round(1.998 · tanh(zᵢ)) / 2
```

So each normalized scalar lands on:

```text
-1, -0.5, 0, 0.5, 1
```

With two scalar dimensions, that gives an implicit (5 × 5 = 25) code space.

FSQ still uses a straight-through estimator for the non-differentiable rounding operation, but it does not need a learned vector codebook, codebook reseeding, or VQ-style commitment machinery.

---

## Visualization behavior

All lessons share the same fixed layout:

- left: **actual MuJoCo robot only**, always the same location and controls
- center: lesson-specific **reference / latent / token / training** visualization
- right: lesson guide
- bottom: stable SONIC information-flow strip

The reference and robot worlds are intentionally separated:

- the Simulation panel never draws a future-reference ghost robot
- planner future motion is **pre-FSQ reference**, not a motor command
- FSQ/VQ produces motion token `q`
- `q + actual proprioception → Dynamic Decoder → action`

For VQ/FSQ latent plots:

- gray points are only the fixed FSQ grid or learned VQ codebook
- the blue trail is the **actual z history generated during the current live episode**
- there is no hypothetical goal sweep mixed into the live graph
- the short dashed `z → q` segment is only the current quantization displacement
- latent x/y axes use equal unit scale (1:1 data aspect)

Canvas backing resolution follows displayed CSS size × device-pixel ratio, so plots are not stretched by mismatched canvas dimensions.

When **Live** is enabled, the robot, current reference, latent, token-dependent visualizations, proprioception, and action are redrawn from the same control ticks. If the CartPole reaches a termination condition, the teaching demo automatically starts a new episode and keeps Live running instead of silently stopping.

**Push** is a robot-only impulse: it changes actual velocity/proprioception without advancing the planner/reference. This makes the Dynamic Decoder experiment explicit:

```text
same reference
same motion token q
      +
changed actual proprioception
      ↓
different action
```

PPO training plots redraw after each training iteration.


---

## What actually learns?

A useful distinction is:

> **FSQ participates in training, but the standard FSQ quantizer itself is not parameter-learned.**

The default SONIC quantizer is `vector_quantize_pytorch.FSQ`, instantiated from a fixed `levels` configuration. It has no learned vector codebook. The rounding operation uses a straight-through estimator (STE), so gradients can pass through the discrete bottleneck to the Encoder.

| Component | Learned? | Main learning signal |
|---|---:|---|
| Motion Encoder(s) | **Yes** | PPO through the Dynamic Decoder + reconstruction auxiliary loss + cross-encoder latent-alignment losses |
| FSQ finite levels / grid | **No** | Fixed hyperparameters; STE provides a surrogate backward path |
| G1 Dynamic Decoder | **Yes** | PPO physical-tracking objective |
| G1 Kinematic Decoder | **Yes** | Future-motion reconstruction auxiliary loss |
| Critic | **Yes** | Value / return loss for PPO |
| VQ learned codebook (comparison) | **Yes** | Codebook/EMA-style update; this is one of the mechanisms FSQ removes |

The subtle point is that the representation still learns even though FSQ does not:

```text
loss
 ↓
Decoder
 ↓
quantized q
 ↓  STE through round()
Encoder weights change
 ↓
next z is placed more usefully relative to the fixed FSQ bins
```

So it is better to say:

- “the **Encoder learns to use FSQ**”
- not “FSQ learns its codebook”

### Actual SONIC token shape

The release configuration uses:

```text
token_dim = 32 scalar dimensions
levels per scalar = 32 fixed values
max_num_tokens = 2

flattened decoder input from tokens = 2 × 32 = 64 values
```

The implementation config names are slightly confusing: `num_fsq_levels: 32` is assigned to `token_dim`, while `fsq_level_list: 32` is broadcast to 32 scalar dimensions.

The implicit Cartesian-product code space is enormous, but it is **not explicitly stored as a table**. Each scalar is quantized independently.

The SONIC decoder consumes the quantized numeric token vectors; “token” here should not be interpreted as necessarily one integer ID like an LLM vocabulary token.

### Why do different encoders produce a universal token?

FSQ alone does not guarantee that a G1 Encoder, SMPL Encoder, and teleoperation Encoder give the same semantic token.

SONIC trains these encoders with auxiliary latent-alignment losses in addition to physical PPO and reconstruction. The released auxiliary configuration includes G1↔SMPL, G1↔teleop, teleop↔SMPL, and re-encoded SMPL↔G1 alignment terms.

This is a key part of the word **universal**.

Official implementation references:

- [UniversalTokenModule](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/trl/modules/universal_token_modules.py)
- [SONIC training code](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/references/training_code.md)
- [FSQ config](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/config/actor_critic/quantizers/fsq.yaml)
- [Universal-token config](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml)
- [Auxiliary-loss config](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/config/aux_losses/universal_token/g1_recon_and_all_latent.yaml)

---

## Questions the learner should now be able to ask

These are intentional checkpoints, not extra jargon:

1. **Why compress at all?** What information do we want the bottleneck to retain?
2. **Why reconstruct if the deployment controller does not need reconstruction?**
3. **What is the difference between latent dimension, FSQ levels per dimension, number of tokens, and flattened token dimension?**
4. **Does FSQ learn anything? If not, how does gradient reach the Encoder?**
5. **Why can a fixed quantizer improve during training?**
6. **Is a SONIC motion token an integer ID or a quantized numeric vector?**
7. **Why does a motion token not contain the current robot state?**
8. **Why does the Dynamic Decoder need proprioception in addition to the token?**
9. **What teaches the Dynamic Decoder: reconstruction or PPO?**
10. **What teaches the Kinematic Decoder?**
11. **How do G1, SMPL, and teleop Encoders learn a compatible shared token space?**
12. **Where does the reference come from, and is that reference source itself part of the SONIC tracker?**
13. **How is temporal information compressed into multiple tokens?**
14. **Which parts exist at deployment, and which exist only during training?**
15. **What does this CartPole toy preserve, and what humanoid behavior can it not validate?**

If those questions can be answered from the UI without reading source code, the teaching lab is doing its job.

---

## Quick start

### 1. Fastest: use GitHub Pages

Open:

**https://tinmanlab.github.io/cartpole-sonic/**

Useful direct links:

- Encoder / bottleneck: https://tinmanlab.github.io/cartpole-sonic/?lesson=ae
- VAE optional branch: https://tinmanlab.github.io/cartpole-sonic/?lesson=vae
- VQ: https://tinmanlab.github.io/cartpole-sonic/?lesson=vq
- VQ-VAE: https://tinmanlab.github.io/cartpole-sonic/?lesson=vqvae
- FSQ: https://tinmanlab.github.io/cartpole-sonic/?lesson=fsq
- What learns?: https://tinmanlab.github.io/cartpole-sonic/?lesson=learning-graph
- Motion token: https://tinmanlab.github.io/cartpole-sonic/?lesson=motion-token
- Dynamic Decoder: https://tinmanlab.github.io/cartpole-sonic/?lesson=dynamic-decoder
- PPO training: https://tinmanlab.github.io/cartpole-sonic/?lesson=ppo
- Full SONIC structure: https://tinmanlab.github.io/cartpole-sonic/?lesson=sonic

### 2. Run locally

No build step is required for the web app.

```bash
git clone https://github.com/tinmanlab/cartpole-sonic.git
cd cartpole-sonic
python3 serve.py
```

Open:

```text
http://localhost:8765
```

For development checks:

```bash
npm ci
npm run check
npm run verify:teacher
npm run verify:bootstrap
npm run verify:ppo
```

---

## Native MuJoCo WASM

The browser physics is real MuJoCo, not a hand-written CartPole equation.

The repo vendors the official MuJoCo 3.14.0 JavaScript/WASM build under:

```text
vendor/mujoco/
├── mujoco.js
├── mujoco.wasm
└── LICENSE
```

The default model follows the relevant MuJoCo Playground / dm_control CartPole settings:

- simulation timestep: 10 ms
- cart mass: 1.0 kg
- pole mass: 0.1 kg
- slide range: ±1.8 m
- motor force interface: ±10 N

Two additional bounded variants are included for dynamics/morphology experiments:

- Long pole
- Heavy pole

These are teaching variants, not official Playground benchmark tasks.

---

## WebMCP

The page exposes a semantic control surface instead of requiring screen-coordinate automation.

When the MCP-B browser runtime is available, the page registers one consistent semantic API:

```text
course_get_outline
course_get_state
course_navigate
course_run_lesson_action
simulation_control
experiment_set_goal
training_run
simulation_set_model
```

`course.js` is the single source of truth for the curriculum. The UI and WebMCP both read the same lesson IDs, prerequisites, questions, experiments, and SONIC mappings.

`course_get_state` returns a structured teaching snapshot:

```text
course
  current lesson + outline
experiment
  goal + live/model state
signals
  reference + latent + token + proprioception + action
training
  PPO iterations + held-out evidence
backends
  MuJoCo + WebGPU + WebMCP
```

This means an agent can navigate the course, run the canonical experiment for a chapter, manipulate the shared simulation, and then read back the exact evidence the learner sees—without screen-coordinate automation.

The WebMCP pattern was informed by the related **tinmanlab/web-mcp-gpu** experiments.

---

## WebGPU

WebGPU is intentionally separated from the tiny PPO trainer so the project does not pretend that GPU acceleration is necessary for this small model.

After the lesson UI is visible, `webgpu_fsq.js` asynchronously runs the actual FSQ scalar transform in a WebGPU compute shader:

```wgsl
dst[i] = round(tanh(src[i]) * 1.998) / 2.0;
```

It compares the GPU result with the CPU reference and reports parity in the top status badge.

Current roles:

- **MuJoCo WASM**: physical simulation
- **JavaScript/CPU**: tiny PPO / teacher / MLP training
- **WebGPU**: real FSQ compute parity and browser-GPU capability verification
- **WebMCP**: semantic agent control

This keeps each technology's role explicit.

---

## Fast startup checkpoints

The teaching UI does not recompute the 500-step teacher bootstrap on every page load.

Three deterministic bootstrap checkpoints are bundled for the default continuous/VQ/FSQ students. They are loaded immediately so the first visualization is usable as soon as MuJoCo WASM is ready. PPO iterations run after that remain live browser-side learning.

If a checkpoint is unavailable, the lab can fall back to the teacher bootstrap path.

---

## Bundled teacher and PPO

A frozen CartPole PPO artifact is included only as a **lab bootstrap teacher**.

It is not a GEAR-SONIC component.

Why use it?

A completely random discrete-token policy often fails before the student can learn a useful physical controller. The teacher gives the toy student a stable initial action prior; the student is then improved using its own SONIC-like tracking PPO loop.

Deterministic native-MuJoCo evidence is stored in `evidence/`.

Current checked results:

| Stage | Result |
|---|---:|
| frozen teacher stability | 15 / 15 episodes survive 10 s |
| teacher ±0.8 m precision | ~0.42 m MAE |
| FSQ student after bootstrap | 12 / 12 survive 10 s, ~0.201 m MAE |
| FSQ student + 20 PPO iterations | ~0.144 m MAE |
| FSQ student + 30 PPO iterations | ~0.142 m MAE |
| FSQ student + 50 PPO iterations | regresses to ~0.192 m MAE |

These are **toy CartPole measurements**, not NVIDIA SONIC benchmark results.

---

## Mapping back to GEAR-SONIC

| CartPole SONIC | GEAR-SONIC role |
|---|---|
| future [x, ẋ] trajectory | future whole-body motion reference |
| tiny reference Encoder | G1 / SMPL / teleop motion Encoder |
| 2-D latent | learned motion latent |
| VQ / FSQ | discrete motion bottleneck |
| compact q | universal motion-token representation |
| [x, ẋ, θ, θ̇] | robot proprioception |
| Dynamic Decoder | G1 dynamic decoder |
| scalar force | whole-body joint action |
| MuJoCo CartPole | humanoid physics / robot |
| Kinematic Decoder | future-motion reconstruction auxiliary path |
| Critic + GAE + PPO | physical tracking policy optimization |

What this toy **does not** reproduce:

- G1 morphology and multi-contact dynamics
- the production SONIC network size
- multi-encoder latent alignment
- large-scale motion data
- real actuator/sensor dynamics
- sim-to-real deployment quality
- VLA / teleoperation production interfaces

---

## Repository structure

```text
.
├── index.html                 # interactive UI
├── app.js                     # fixed teaching shell + WebMCP orchestration
├── course.js                  # canonical curriculum SSOT
├── sonic_toy.js               # SONIC-like planner/encoder/token/decoder/PPO
├── mujoco_sim.js              # native MuJoCo WASM CartPole wrapper
├── webgpu_fsq.js              # WebGPU FSQ parity kernel
├── teacher_policy.js          # frozen bootstrap teacher adapter
├── assets/
│   ├── teacher_cartpole_ppo.json
│   ├── student_ae_bootstrap.json
│   ├── student_vq_bootstrap.json
│   └── student_fsq_bootstrap.json
├── evidence/                  # deterministic evaluation snapshots
├── scripts/                   # structure and evaluation checks
├── vendor/mujoco/             # pinned MuJoCo 3.14.0 JS/WASM
└── media/
    ├── cartpole-sonic-demo.gif
    └── cartpole-sonic-demo.mp4
```

---

## References

- NVIDIA GEAR-SONIC: https://nvlabs.github.io/GEAR-SONIC/
- NVIDIA GR00T Whole-Body Control: https://github.com/NVlabs/GR00T-WholeBodyControl
- ProtoMotions: https://github.com/NVlabs/ProtoMotions
- FSQ — *Finite Scalar Quantization: VQ-VAE Made Simple*: https://arxiv.org/abs/2309.15505
- MuJoCo: https://github.com/google-deepmind/mujoco
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground

## Scope

This repository is a learning and research prototype. Its goal is to make the architecture understandable and inspectable, not to claim performance equivalence with GEAR-SONIC or a production humanoid controller.
