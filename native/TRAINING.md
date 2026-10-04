# Official SONIC trainer + physical CartPole

## 무엇이 이번에 연결됐나?

`core_smoke.py`는 공식 신경망 부품의 입출력·역전파만 검사합니다. **`training.py`는 공식 SONIC 학습기 자체를 실행하며, 실제 MuJoCo CartPole의 결과로 가중치를 갱신합니다.** 기존 브라우저의 JavaScript PPO로 바꿔 실행한 것이 아닙니다.

```text
원하는 미래 위치·속도 ─→ 공식 Encoder ─→ 실제 FSQ token
                                              + 현재 MuJoCo 상태
                                              ↓
                                    공식 Dynamic Decoder / Actor
                                              ↓
                                    카트 힘 ─→ 실제 MuJoCo step
                                              ↓
                                    reward · next observation · done
                                              ↓
공식 RolloutStorage → 공식 GAE → 공식 PPO + auxiliary-loss 결합
                                              ↓
                                     실제 optimizer 갱신
```

공식 `TRLAuxLossPPOTrainer.train()`을 직접 호출합니다. 원본 `Actor`, `Critic`, `UniversalTokenModule`, `BaseModule`, rollout storage, GAE, PPO clipping과 trainer loop를 복사하거나 override하지 않았습니다. 공식 소스는 고정된 upstream commit의 Git blob과 비교하고, 실제 호출된 함수 횟수는 수동 실행 코드 대신 Python의 수동 변경 없는 profiling과 PyTorch optimizer hook으로 측정합니다.

## 현재 결과를 읽는 법

기본 실행은 **8개 환경 × 128 control steps × 2 PPO iterations = 2,048개 물리 전이**, **4,096회 MuJoCo step**, **8회 optimizer update**입니다. Encoder, Dynamic Decoder, Kinematic Decoder, Critic 모두 실제 가중치 변화가 생겼습니다. 저장한 가중치를 새 공식 Actor/Critic에 불러온 뒤 같은 입력에 대한 출력 차이는 0입니다.

그러나 이 테스트의 PASS는 **학습 연결의 PASS**이지 제어 성능의 PASS가 아닙니다. 저장된 2회 학습 결과에서 500-step 평가 완주는 학습 전후 모두 **0/8**입니다. 평균 실행 길이는 약 61.6 → 59.1 steps이며, 120번째 step의 push까지 도달한 episode가 없어 **외란 회복은 평가되지 않았습니다**. 짧게 실행하고 넘어져서 MAE가 작게 나오는 것을 좋은 추적 성능으로 표시하지 않습니다.

초기화는 난수이며 teacher bootstrap, LQR/PD/수식 기반 안정화기, 외부 humanoid checkpoint는 사용하지 않습니다. 이 단계로 “SONIC 원본 학습기를 CartPole에 연결했다”는 것은 말할 수 있지만 “학습된 안정적인 CartPole 추적 제어기를 확보했다”는 것은 말할 수 없습니다. 여러 seed, 장시간 학습, 최종 holdout 성능, reference-conditioned tracking, sim-to-real은 아직 검증 대상입니다.

## 공식 코드와 로컬 구현의 경계

| 구성 | 소유 코드 |
|---|---|
| Encoder/Decoder/FSQ, Actor/Critic | 원본 SONIC 모듈과 실제 FSQ 라이브러리 |
| rollout storage, GAE, PPO loss, gradient clipping, 학습 루프 | 원본 SONIC `TRLAuxLossPPOTrainer` 및 상위 클래스 |
| 물리 | MuJoCo 3.14.0 |
| CartPole XML, 관측 순서, reference, reward, reset/termination, action 단위 | 이 저장소의 로봇별 환경 adapter |
| reconstruction 대상 선택과 MSE | 로컬 `CartPoleReconstructionLoss`, 공식 auxiliary-loss 확장점에 등록 |
| reconstruction과 PPO의 가중합·역전파 | 원본 SONIC auxiliary trainer |

G1 기구학·회전·SMPL alignment loss는 CartPole 데이터에 맞지 않으므로 그대로 쓰거나 가짜 데이터를 공급하지 않습니다. 로컬 MSE binding은 명시된 task adaptation입니다. 정렬 loss나 원래 G1 loss를 검증했다고 표시하지 않습니다.

Upstream pin: `NVlabs/GR00T-WholeBodyControl@b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`.

- [원본 trainer](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic/trl/trainer/ppo_trainer.py)
- [원본 auxiliary trainer](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic/trl/trainer/ppo_trainer_aux_loss.py)
- [원본 Actor/Critic](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic/trl/modules/actor_critic_modules.py)

이 연결은 **CPU MuJoCo 환경 adapter**이며 Isaac Lab의 CartPole task를 그대로 실행한 것은 아닙니다. 공식 trainer의 `reset_all`, `step(policy_state_dict)`, 관측/configuration, 평가 모드 인터페이스를 제공하는 방식입니다. 공식 Hydra humanoid entrypoint나 대규모 분산 학습 전체를 검증한 것도 아닙니다.

## 명시된 물리·학습 조건

`cartpole.xml`: 카트 질량 1.0 kg, 막대 질량 0.1 kg, 길이 1.0 m. 상태 순서는 `[x, x_dot, theta, theta_dot]`, action은 원본 Gaussian sample입니다. PPO log probability용 sample은 보존하고, **물리 입력에만 [-1,1] clipping 후 10 N 배율**을 적용합니다. dt=0.01 s에서 물리 2 step마다 action을 바꾸므로 control rate는 50 Hz입니다.

관측은 actual-state 4개와 미래 `[x,x_dot]` 8프레임 16개를 분리합니다. Critic은 두 벡터를 합친 20개를 받습니다. Reference는 실제 로봇 위치를 따라 다시 쓰지 않고 독립적으로 진행하는 critically damped trajectory이며 goal은 [-0.4,0.4] m입니다. 막대 각도를 독립 구동할 수 있는 것처럼 별도 명령하지 않습니다.

Reward는 `1 - 0.65*min((x-x_ref)^2,4) - 8*theta^2 - 0.02*x_dot^2 - 0.02*theta_dot^2 - 0.002*u^2`입니다. `|x|>1.78` 또는 `|theta|>0.65`에서 해당 환경만 실패 종료·리셋합니다. NaN/Inf action이나 상태는 학습을 중단시킵니다.

**Training은 continuing task입니다.** 인위적 episode time limit은 두지 않습니다. 따라서 `time_outs=false`이고, 128-step rollout이 끝나도 done을 만들지 않습니다. 그 경계의 value bootstrap과 GAE는 원본 trainer가 수행합니다. 별도의 500-step evaluation cutoff를 training termination으로 혼동하지 않습니다. 원본 trainer의 time-limit bootstrap 경로, time-limit이 있는 task의 적합성은 이 gate에서 검증하지 않았습니다.

Reduced network: 2 tokens × 2 scalars, 5 scalar levels, `[32,32]` MLPs. Actor std=0.12 고정, learning rate=3e-4, gamma=.99, lambda=.95, 2 PPO epochs, 2 minibatches, batch당 4 environment sequences입니다. Reconstruction MSE의 계수는 0.1입니다. 이 숫자가 SONIC의 최적 hyperparameter라는 주장은 없습니다.

알려진 표시 한계: 이 upstream trainer의 console printer는 `log_std`를 쓰면 noise-std를 0으로 표시하는 fallback이 있습니다. 실제 distribution의 std=0.12를 별도로 테스트합니다. 출력 문자열을 고치기 위해 원본 trainer를 patch하지 않습니다. TRL experimental API 경고와 Python 3.10 W&B 의존성 deprecation warning도 발생할 수 있습니다.

## 실행과 검증

기존 core-only `requirements.lock`과 **별도의 대안 환경**으로 `training.lock`을 사용합니다. 두 lock을 합치지 않습니다. HF datasets 의존성의 fsspec 상한 때문에 core-only 전체 lock과 학습기 의존성을 한 번에 설치하면 resolver가 충돌합니다. 공통 신경망 패키지 버전은 보존되고, 이 환경에서도 기존 core tests가 통과합니다. 글로벌 Python, CUDA, Isaac Lab, 호스트 서비스는 변경하지 않습니다.

```bash
python3.10 -m venv .venv-native
.venv-native/bin/python -m pip install 'torch==2.7.0+cpu' 'torchvision==0.22.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
.venv-native/bin/python -m pip install -r native/training.lock
.venv-native/bin/python -m pip check

# README.md의 별도 pinned upstream checkout을 사용합니다.
git -C "$SONIC_UPSTREAM" sparse-checkout add \
  gear_sonic/trl/trainer gear_sonic/trl/callbacks gear_sonic/utils
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
.venv-native/bin/python -m pytest native/test_core_smoke.py native/test_training.py -q
.venv-native/bin/python native/training.py --output-dir /tmp/native-ppo-run
```

실행은 기본 2 iterations이며 integration 목적의 1..10 범위만 허용합니다. 결과는 `native_training.json`, checkpoint는 `native-weights.pt`입니다. Checkpoint는 **가중치 저장/재로딩**으로, exact-resume에 필요한 optimizer/RNG/환경 전체 snapshot은 아닙니다. 임의의 외부 pickle checkpoint는 불러오지 않으며, 자신이 생성한 파일을 `weights_only=True`로 검사합니다.

[저장된 측정 결과](../evidence/native_training.json)는 원본과 로컬 소스 해시, 버전, 실제 함수 호출·step 수, loss, 가중치 변화, 평가 실패까지 담습니다. CI는 같은 단일 native workflow에서 core와 training 테스트를 실행하고 새 보고서 및 가중치 artifact를 생성합니다. 브라우저 화면이나 기존 JS 성능 곡선을 이 결과로 대체하지 않습니다.

다음 성능 작업의 기준은 이 연결을 유지한 채 안정화와 reference-conditioned tracking을 실제로 학습시키고, 독립 평가에서 생존·추적·외란 도달/회복을 확인하는 것입니다. 현재의 작은 연결 확인용 실행을 성능 결과로 승격하지 않습니다.
