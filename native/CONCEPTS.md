# SONIC concepts in a small physical environment

## 목적과 구현 경계

PPO 점수를 더 높이는 것이 아니라 **미래 reference, 공유 token, 입력별 Encoder, 현재 상태를 읽는 Dynamic Decoder, 복원용 Kinematic Decoder의 역할**을 작은 환경에서 분리해 검사한다.

기존 `training.py` / `learning.py`와 브라우저 JS 제어기는 바꾸지 않았다. 이 실험은 원본 `UniversalTokenModule`, `BaseModule`, 실제 FSQ와 기존에 학습한 native 제어기를 재사용한다. 입력 표현과 로봇별 loss binding은 로컬 코드다. Offline imitation/latent alignment를 원본 SONIC PPO 학습이라고 부르지 않는다. 실제 PPO를 실행한 별도 단계는 `concept_control.py`의 원본 `TRLAuxLossPPOTrainer.train()` 호출로 구분한다.

공식 구조 참고: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/docs/source/references/training_code.md

## 작은 데이터부터 물리적으로 확인

이미 학습된 native 정책에 정착·진동·방향 전환·이동 후 복귀 형태의 네 종류 command를 제공하고 실제 MuJoCo 상태와 action을 기록했다. **요청한 곡선을 완벽히 따라갔다고 가정하지 않는다.** 학습 reference는 요청곡선이 아니라 그때 실제로 실행된 물리 궤적이다.

32개의 300-step 기록 중 24개는 학습, 8개는 gradient에 사용하지 않는 진단 평가로 분리한다. 같은 trajectory의 겹치는 window를 train/test 양쪽에 나누지 않는다. 총 2,184 training windows와 728 diagnostic windows이며, 각 window는 `[현재 reference, +0.08, ..., +0.56초]`의 8프레임이다. 이 실험에서는 첫 프레임이 현재 reference라는 점이 이전 command-only 정책의 `+0.08 ... +0.64초` 입력과 다르다.

기록된 실제 상태만 사용하며 terminal/reset을 넘는 window나 빠진 미래를 복제한 padding은 만들지 않는다. 이번 기록은 32개 모두 300 step까지 생성되었다. 재생 시 reference는 불변 파일이고, 실제 proprioception만 진행 중인 MuJoCo에서 읽는다. **현재 시뮬레이터의 미래 정답 상태를 관측으로 주는 oracle은 아니다.**

원시 데이터: `evidence/native_concepts/motions.npz`. Source checkpoint/model 해시와 episode split은 `concepts.json`에 기록된다.

## 같은 움직임의 두 표현

Joint 표현: `[cart x, cart velocity, pole angle, pole angular velocity]`.

Marker 표현: `[cart x, pole-tip x, pole-tip z, cart velocity, tip x velocity, tip z velocity]`. Tip z는 pivot 높이 기준이다. 막대 길이 1 m에 대한 변환을 MuJoCo body transform 및 Jacobian으로 독립 확인하고 역변환 roundtrip도 검사한다.

이는 정확한 수치 좌표의 두 표현이지 camera/VR/SMPL 데이터가 아니다. 전체 cart+tip 좌표는 이 범위에서 역변환 가능하다. 따라서 실제 사용 목적이 단순 좌표 변환뿐이면 **분석적 역변환이 더 단순한 기준선**이다. 학습 Encoder는 공유 인터페이스 방법을 검증하기 위한 것이지, 여기서 신경망 좌표 변환이 반드시 필요하다는 주장이 아니다.

반면 tip만 남기면 CartPole에서도 모호성이 생긴다. 서로 다른 `[x,theta]`가 같은 tip 위치·속도를 만들 수 있음을 테스트한다. 관측에서 정보를 지운 뒤 정답을 추정하지 못하는 것을 곧바로 Encoder 버그나 로봇 크기의 문제라고 하지 않는다.

## 세 경로를 섞지 않는다

### 1. 새로운 공유 Decoder를 학습하는 concept model

`concepts.py`는 두 official Encoder와 공유 Dynamic/Kinematic Decoder를 구성한다. Joint 경로를 offline action imitation + reconstruction으로 학습한 뒤, joint Encoder와 Decoder를 고정하고 marker Encoder를 같은 latent에 정렬한다. 같은 sample의 route가 하나만 활성화되도록 입력을 검사한다.

실제 학습 후 marker latent MSE는 약 **0.01047 → 0.000131**로 줄었고, 동작 출력 차이 RMSE는 **0.535 N → 0.135 N**로 줄었다. 그러나 새 모델의 5초 폐루프 재생은 두 branch 모두 **0/8**이다. **표현 정렬과 낮은 오프라인 오차가 안정적인 feedback control을 보장하지 않는다는 음성 결과를 보존**한다.

미래 window를 그대로 쓰는 모델과 현재 프레임만 반복해 학습하는 모델을 같은 초기화·900 update 예산에서 비교했다. Held-out action RMSE는 각각 **0.0858 N / 0.2131 N**이었다. 둘 다 폐루프 성공을 입증하지 못했으므로 이것을 미래 기반 제어기의 보편적 우월성이라고 해석하지 않는다.

### 2. 같은 concept model을 원본 SONIC PPO로 보완

`concept_control.py`는 불변 기록 궤적을 추적하는 유한 길이 task를 원본 학습기에 연결한다. Episode 완료는 기록 task의 진짜 종료이며, continuing task에 인위적인 time-limit을 추가한 것으로 취급하지 않는다. Critic은 time-to-go를 받는다. 실패와 정상 motion 완료를 별도 집계한다.

192 iterations / 147,456 physical control transitions / 768 original optimizer updates 후에도 이번 설정은 폐루프 **0/8**이다. Marker 재정렬로 action 차이가 더 줄어도 생존은 해결되지 않았다. 이 bounded 실패를 숨기거나 CartPole의 부적합성으로 바꾸어 설명하지 않는다. 보고서는 `refinement.json`이다.

### 3. 검증된 native Decoder는 그대로, 새 Encoder만 연결

`encoder_transfer.py`는 **이전에 성공한 native Encoder, FSQ, Dynamic Decoder와 Kinematic Decoder를 그대로 고정**한다. 같은 원본 UniversalTokenModule에 joint-motion / marker-motion Encoder를 추가하고, 이들만 기존 token 공간에 정렬한다. 기존 command branch를 선택하면 저장된 원래 정책의 출력과 정확히 일치한다.

새 branch를 실행할 때 원래 command-reference 공간은 비활성 placeholder다. 원래 command를 변조해도 선택한 새 Encoder의 token/action이 바뀌지 않는 검사를 통해 label 누출을 막는다. 실제 실행에 원래 command를 요구하는 숨은 경로는 없다.

이 단계에서는 Decoder를 다시 학습하지 않았고 PPO도 새로 돌리지 않았다. Shared Decoder의 가중치 해시가 그대로인지 확인한다. 새 Encoder 학습은 local paired latent/action distillation이며, G1/SMPL 전용 loss를 가짜 데이터에 적용하지 않는다.

## 고정 Decoder 재사용 경로의 측정 결과

동일한 8개 gradient-excluded 진단 trajectory를 250 control steps, 즉 **5초** 재생했다. 아래 오차는 **기록된 실제 카트 궤적에 대한 MAE**다. 이전 10초 실험의 요청 목표 추적 MAE와 목표·시간·외란 크기가 다르므로 직접적인 성능 향상률로 비교하지 않는다.

| 입력 경로/조건 | Joint-motion Encoder | Marker-motion Encoder |
|---|---:|---:|
| 정상 reference, 5초 완주 | 8/8 | 8/8 |
| 정상 reference, 기록 궤적 MAE | 2.10 mm | 1.89 mm |
| step 70에 작은 속도 impulse, 완주 | 8/8 | 8/8 |
| impulse 포함 전체 MAE | 11.39 mm | 10.81 mm |
| 현재 reference 프레임만 반복, MAE | 45.59 mm | 52.37 mm |
| 처음/끝을 유지하고 중간 프레임 순서 반전, MAE | 21.83 mm | 46.24 mm |

Impulse는 cart velocity +0.2 m/s, pole angular velocity -0.3 rad/s이다. 두 branch 모두 8개에서 실제 impulse에 도달했다. 이는 sustained recovery time이나 formal stability certificate는 아니다.

기존 token과의 완전한 vector 일치율은 새 Joint/Marker Encoder 각각 약 **84.3% / 83.9%**, 출력 차이 RMSE는 각각 **0.0556 N / 0.0563 N**이었다. Token이 100% 같아야만 유효한 제어가 되는 것은 아니며, token 일치율만으로 제어 성공을 판단하지 않는다.

이 비교는 미래 내부 프레임이 실제 출력/추적 결과에 영향을 준다는 증거다. 다만 순서를 반전한 입력은 물리적으로 실행 가능한 새 reference가 아니라 의도적인 corruption이고, OOD 효과도 포함한다. **동일 endpoint의 물리적 경로들에 대해 미래 예측 제어가 최적임을 증명하거나, 짧은 reference로 재학습한 최적 baseline을 이겼다고 주장하지 않는다.** 진단 평가 궤적은 이전 단계에서도 관찰했으므로 untouched final holdout으로 부르지 않는다.

## Reconstruction 역할도 별도 비교

`concept_ablation.py`는 첫 concept model에서 같은 가중치 초기화·sample·900 update·학습률을 유지하고 reconstruction 계수만 0.5에서 0으로 바꾼다.

위치 reconstruction MAE는 auxiliary loss 사용 시 **8.4 mm**, 미사용 시 **132.7 mm**였다. 반면 action prediction RMSE는 각각 **0.0858 N / 0.0766 N**이며, 둘 다 폐루프 0/8이다. 이 조건에서 reconstruction은 의도한 복원 능력을 키웠지만, action 오차나 제어 성능을 자동으로 개선하지 않았다. `no_aux.json`을 확인한다.

## 재현 및 검증

기존 `native/training.lock` 환경과 고정 upstream을 사용한다. 추가 simulator service, 새 PPO, 새 의존성 환경은 만들지 않는다.

```bash
export SONIC_UPSTREAM=/path/to/pinned/GR00T-WholeBodyControl
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1
python -m pytest native/test_concepts.py native/test_concept_control.py native/test_encoder_transfer.py -q
python native/concepts.py --output-dir /tmp/sonic-concepts --steps 900
python native/concept_ablation.py --concept-dir /tmp/sonic-concepts --output-dir /tmp/sonic-noaux
python native/concept_control.py --concept-dir /tmp/sonic-concepts --output-dir /tmp/sonic-ppo-refine --iterations 192
python native/encoder_transfer.py --concept-dir /tmp/sonic-concepts --output-dir /tmp/sonic-transfer --steps 1200
```

새 code/data 결과는 `evidence/native_concepts/`에 함께 보존한다. Native control checkpoint나 재현용 source hash를 바꾸면 연결된 증거를 재검증한다. Tests는 source/routing/geometry/gradient/freeze/replay 계약을 검사하며 특정 성능 승자를 강제하지 않는다.

## Pivot 판정

**이번 음성 결과는 ENVIRONMENT_LIMIT이 아니라 학습/인터페이스 실험의 차이였다.** 고정된 기존 Decoder를 재사용하는 경로로 CartPole 안에서 공유 표현과 제어를 더 검증할 수 있었다. 따라서 아직 새 로봇으로 실행을 옮기지 않는다.

다음으로 필요한 개념이 두 개 이상의 독립 actuator, 연속 kinematic redundancy, 발 접촉 전환 또는 일반적인 3D frame/heading이라면 CartPole에서 이를 검증했다고 주장할 수 없다. 그때 적용할 최소 환경·재사용 요소·porting checklist와 중단 조건은 [PIVOT.md](PIVOT.md)에 있다. 테스트하지 않은 항목과 현재 환경으로 표현할 수 없는 항목을 구분한다.

## Subsequent same-endpoint audit

The previously missing physically feasible same-current/same-terminal comparison is now implemented in [`TEMPORAL_AUDIT.md`](TEMPORAL_AUDIT.md). Nine of twelve declared paired maneuvers passed the full-state physics tolerances; three rejected candidates are preserved. The frozen transferred policy distinguished six accepted pairs from their interior future frames, but did not outperform endpoint-only inputs on the short-horizon cart-error aggregate. This refines—not erases—the earlier diagnostic result: temporal sensitivity is not automatic temporal-control superiority. No policy retraining or embodiment change was used in this audit.
