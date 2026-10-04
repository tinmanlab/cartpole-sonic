# Native SONIC core: what is actually connected?

This page describes `core_smoke.py`, the **first native-code gate**, not a trained CartPole controller.
For the subsequent actual trainer/physics integration, see [TRAINING.md](TRAINING.md).
It imports the unmodified official `UniversalTokenModule`, its `BaseModule` Encoder/Decoders, and the actual `vector_quantize_pytorch.FSQ` dependency. No local neural-network, FSQ or PPO replacement is used in this path.

## 쉽게 이해하기

기존 브라우저는 **SONIC의 구조를 설명하려고 따로 만든 작은 모델**입니다. 이 디렉터리는 **공식 SONIC 부품을 직접 연결해 전기 신호가 통하는지 확인하는 단계**에 해당합니다.

```text
8개 미래 [위치, 속도] 프레임 (16개 수)
    ↓ 공식 BaseModule Encoder
공식 FSQ → 실제 양자화된 token
    ├─ + 현재 [위치, 속도, 기울기, 각속도]
    │      ↓ 공식 BaseModule Dynamic Decoder
    │   CartPole용 출력 1개 — 아직 물리에 적용하지 않음
    └─ 공식 BaseModule Kinematic Decoder
        미래 프레임 복원 (16개 수)
```

동일한 미래 reference에서 현재 상태만 바꾸면 **token은 그대로이고 출력만 달라집니다**. token은 현재 상태 자체가 아니라 원하는 움직임의 표현이라는 분리를 실제 공식 코드 경계에서 확인합니다.

| 단계 | 이 경로가 확인하는 것 | 확인하지 않는 것 |
|---|---|---|
| 브라우저 설명용 모델 | 기존 JS/MuJoCo 학습장은 별도로 유지 | 공식 코드로 바뀌었다고 재표기하지 않음 |
| 공식 모듈 연결 | 원본 클래스·FSQ 호출·입출력 크기·역전파·reference/state 분리 | 학습된 제어 성능 |
| 공식 학습기 + 물리 CartPole | 후속 gate | 이번 코드에서 실행하지 않음 |

**순전파**는 입력을 넣어 출력을 계산하는 것입니다. **역전파**는 출력의 오차로부터 앞쪽 Encoder까지 학습 신호가 돌아가는 것입니다. 이것이 통과해도 넘어지지 않는 제어기를 얻었다는 뜻은 아닙니다. 이번에는 optimizer update, 공식 PPO, physical rollout을 모두 0회 실행합니다.

## Provenance and intentional adaptations

Upstream: [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), pinned to `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`.

- [UniversalTokenModule](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic/trl/modules/universal_token_modules.py) is imported directly from the checkout. Its source file, BaseModule and instantiation helper must be byte-equal to the pinned Git blobs. Wrong revision, dirty source or imports from another location fail.
- [BaseModule](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic/trl/modules/base_module.py) and upstream encoder/dynamic/kinematic YAML configurations supply the actual network construction. All hidden widths are reduced to `[32,32]`; SiLU activation is retained. CartPole dimensions and every resolved setting appear in the report.
- Only one encoder, `cartpole`, is configured. No fake SMPL, teleoperation or humanoid motion data is supplied.
- The dynamic decoder retains the upstream **legacy dispatch key `g1_dyn`**, because `forward()` extracts `action_mean` from that name. The name does not mean a G1 observation, checkpoint or actuator is used. The reconstruction decoder is `cartpole_kin`.
- FSQ is the actual pinned PyPI package `vector-quantize-pytorch==1.31.6`. The optional `return_indices=False` setting avoids integer-index/implicit-codebook-table construction; quantization itself remains enabled and its invocation is measured. The public code is not patched.
- The `release-token-shape` case uses **2 tokens × 32 scalars, 32 levels per scalar**. It does **not** reproduce release network widths, all observation features, data, weights, modalities or performance.
- Inputs are deterministic synthetic CartPole reference/state fixtures. Reference normalization is `[1.8,3.0]`; state normalization is `[1.8,3.0,0.55,4.0]`. Batch 4 and sequence length 2 exercise temporal handling. The output is one raw network coordinate, not a calibrated or safely bounded physical force command.
- Independent action-squared and reconstruction-MSE probes check gradient paths. They are test losses, **not official SONIC PPO/auxiliary training**. All weights remain unchanged.

## Run without changing a global Python environment

The reproducible CI target is Python 3.10, Linux x86_64, CPU only. No CUDA/Isaac Lab/hardware, motion datasets, model downloads, services or W&B login are needed. `wandb` and `torchvision` are installed because upstream modules import them; this probe does not initialize tracking or download image models.

From this repository root:

```bash
python3.10 -m venv .venv-native
.venv-native/bin/python -m pip install \
  'torch==2.7.0+cpu' 'torchvision==0.22.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
.venv-native/bin/python -m pip install -r native/requirements.lock
.venv-native/bin/python -m pip check

# Use a separate empty path; do not reset or repair an existing user checkout.
git clone --filter=blob:none --no-checkout \
  https://github.com/NVlabs/GR00T-WholeBodyControl.git .upstream-sonic
git -C .upstream-sonic sparse-checkout set \
  gear_sonic/trl/modules gear_sonic/trl/utils gear_sonic/config/actor_critic
git -C .upstream-sonic checkout --detach b042411fae38ee4d1af9aac82a37a1f8d14d6dd0

export SONIC_UPSTREAM="$PWD/.upstream-sonic"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WANDB_MODE=disabled
.venv-native/bin/python -m pytest native/test_core_smoke.py -q
.venv-native/bin/python native/core_smoke.py --output /tmp/native-core-result.json
```

A missing dependency is a nonzero exit/FAIL, never a skipped success. The command atomically replaces its specified report with FAIL on an ordinary computation/import failure, so a stale PASS at that path cannot survive such a failed run. Only explicitly regenerate `evidence/native_core_smoke.json` when reviewing a changed experiment; tests compare its source/configuration and gradient measurements to fresh computation.

The full-framework installer is intentionally not used: this is a narrow direct-module smoke. Dependency versions are locked to the tested CPU environment; compatibility with arbitrary versions or platforms is not claimed. One upstream `docker-pycreds`/W&B Python 3.10 deprecation warning is known; it does not affect the tensor tests.

## Checks and evidence

[Stored execution report](../evidence/native_core_smoke.json) includes source SHA-256 values, actual dependency versions, resolved configurations, quantizer invocation counts, output shapes, separate gradient norms, state perturbation results and explicit scope flags. CI executes the tests and runner afresh and uploads its own report; it does not merely read a stored PASS.

| Case | Tokens per control input | Scalar levels | Output | Reconstruction |
|---|---|---|---|---|
| reduced-one | 1 × 2 | 5 | 1 coordinate | 8 × 2 |
| reduced-two | 2 × 2 | 5 | 1 coordinate | 8 × 2 |
| release-token-shape | 2 × 32 | 32 | 1 coordinate | 8 × 2 |

For each case: action-loss gradients reach Encoder + Dynamic Decoder, reconstruction gradients reach Encoder + Kinematic Decoder, and unrelated decoder gradients are zero. FSQ has zero trainable parameters in these configurations but still propagates the gradient through its upstream implementation.

## Next dependent gate

Connect the official trainer at a supported CartPole environment/task boundary. Define observation ordering, feasible reference distribution, action scale/units, timing, reset, reward and termination explicitly. Only a real rollout followed by an official trainer update can promote the claim to **native SONIC training framework adapted to CartPole**. A zero-action balance policy or this module smoke is not that evidence. See [issue #2](https://github.com/tinmanlab/cartpole-sonic/issues/2).
