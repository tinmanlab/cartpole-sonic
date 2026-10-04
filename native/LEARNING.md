# Native 정책은 균형과 목표 추적을 학습했는가?

원본 SONIC 학습기를 사용한 제한된 실험이다. 2회 학습으로 연결만 확인하는 검사는 [TRAINING.md](TRAINING.md)에 있다.

## 질문과 실제 방법

학습을 늘리면 균형을 유지하는가? 미래 reference를 지워도 같은 목표를 따라가는가? FSQ 정밀도가 다른 reference를 Decoder에 전달하는 데 영향을 주는가?

Teacher 없이 시작하는 기존 task의 초기 가중치·네트워크 폭·정규화·보상·물리를 유지했다. 두 개의 2D token에서 scalar당 5단계와 32단계를 비교했다. FSQ 자체에는 학습할 파라미터가 없으며,
주변 신경망의 초기 학습 파라미터는 두 조건에서 동일한지 검사했다. 정밀도만 원본 FSQ 클래스의 설정으로 바꾸고 forward나 trainer를 수정하지 않았다.

학습률 비교는 정밀도 비교와 구분한다. 공유 optimizer의 학습률은 Actor와 Critic 모두에 적용된다. 이 설정들을 최적 hyperparameter라고 부르지 않는다.

평가에서는 초기 물리 상태와 실제 목표 궤적을 고정하고 정책의 `tokenizer` 입력만 정상·0·부호 반전으로 바꾼다. 입력을 지워도 목표는 그대로다. `token_probe`는 proprioception을
고정한 채 [-0.4,0.4] m의 정착 목표 65개에 대한 token 종류와 행동 범위를 측정한다. 출력 반응만으로 정확한 추적이 증명되지는 않는다.

생존, 실패 전 MAE, 완주 episode 및 마지막 구간의 MAE, 외란 도달 여부를 따로 기록한다. Push는 초기 episode 중 step 120까지 살아 있는 경우에만 적용한다. 도달하지 못하면
`post_push_mae=null`이며 오차 0이나 회복 성공으로 해석하지 않는다. 평가용 환경은 학습 환경과 별도이며 Torch RNG 상태도 바꾸지 않는다.

## 측정 결과

[요약](../evidence/native_learning/summary.json), [최종 학습 원시 보고서](../evidence/native_learning/combined512.json),
[episode별 별도 평가](../evidence/native_learning/holdout.json), [가중치](../evidence/native_learning/weights-final.pt)에
SHA-256 검사가 연결돼 있다. 가중치는 로컬 난수 초기화에서 학습한 것으로 NVIDIA 휴머노이드 checkpoint가 아니다.

같은 **128 iterations**에서 모든 설정의 정상 조건 완주는 **0/8**이었다.

| scalar당 FSQ 단계 | 공유 optimizer 학습률 | 평균 생존 시간 [s] | 65개 목표에서 서로 다른 token 수 |
|---:|---:|---:|---:|
| 5 | 0.0003 | 1.270 | 1 |
| 32 | 0.0003 | 1.405 | 3 |
| 5 | 0.003 | 1.980 | 2 |
| 32 | 0.003 | 1.495 | 9 |

**32단계 / 학습률 0.003**의 validation 결과는 다음과 같다.

| PPO 반복 수 | 정상 조건 완주 | 평균 생존 시간 [s] | 완주 episode MAE [m] |
|---:|---:|---:|---:|
| 128 | 0/8 | 1.495 | 미관측 |
| 256 | 0/8 | 3.495 | 미관측 |
| 512 | 8/8 | 10.000 | 0.0948 |

최종 실행은 **524,288 control transitions, 1,048,576 MuJoCo steps, 2,048 optimizer updates**를 사용했다. 원본 PPO·GAE·auxiliary 경로를
실행했고 teacher나 별도 안정화 제어기는 추가하지 않았다. 고정 상태 probe에서는 서로 다른 token 13개를 사용했다.

최종 평가는 32단계 / 학습률 0.003 실행의 **512회 최종 예산 checkpoint**를 사용했다. Validation에서 가장 좋은 시점을 고른 것이 아니다. 별도 평가 seed **28081,
28082**에서 각각 16 episode를 만들고, 각 seed의 같은 초기 상태·실제 목표로 네 조건을 비교했다.

| 정책 입력 / 외란 조건 | 10초 완주 | 완주 episode 추적 MAE [m] |
|---|---:|---:|
| 정상 미래 reference | 32/32 | 0.1073 |
| 정상 reference + 지정 속도 impulse | 32/32 | 0.1300 |
| 입력 제거; 실제 목표 유지 | 32/32 | 0.2267 |
| 입력 부호 반전; 실제 목표 유지 | 32/32 | 0.3590 |

외란 조건의 32개 모두 step 120의 impulse에 도달하고 step 500까지 생존했다. 외란 이후 평균 MAE는 **0.1269 m**다. 지속적인 회복에 걸린 시간이나 안정성 증명은 측정하지 않았다.

입력을 지운 정책도 균형은 유지했지만 추적 오차가 커졌다. 따라서 이 정책에서는 reference가 추적에 유용하다는 증거가 있다. 생존만 측정하면 이 차이를 놓친다.

## 한계

학습 초기화는 seed **9101** 하나다. 32개 평가 episode는 독립적으로 학습한 정책 32개가 아니다. Validation seed **8080**은 설정 진단에 사용하고, 별도 평가 결과로
checkpoint나 학습률을 다시 고르지 않는다.

**5단계 FSQ의 512회 학습 비교는 수행하지 않았다.** 32단계가 필요하거나 최적이라고 결론 내릴 수 없다. 조건은 직립 근처 초기 상태, 좁은 목표 범위, 고정 물리, 정확한 시뮬레이션 상태, 지정
impulse 하나, 10초다. 센서 잡음·지연·다른 형태·휴머노이드·하드웨어·장시간 강건성은 검증하지 않았다. 브라우저 구조 탐색은 별도 JavaScript 정책을 사용한다.

## 실패 분석: 긴 실행에서 드러난 집계 오류

초기 장기 실행은 원본 `process_ep_infos`에서 중단됐다. 로컬 환경이 종료된 step에만 `length`를 내보냈지만 집계기는 같은 key가 계속 존재한다고 가정했다. 중복된 희소 지표를 제거했다.
Episode 길이는 원본 trainer가 종료 flag로 이미 계산한다. 회귀 검사는 실패 step 뒤 정상 step을 원본 집계기에 넣는다. 보상·reset·물리·PPO 식은 바뀌지 않았다.

확정된 driver로 반복한 첫 기준 실행은 같은 수치를 재현했다. 2회 연결 검사의 수치도 유지됐으며, 수정된 환경 소스의 기록 hash만 달라졌다.

## 재현과 checkpoint 해석

기존 `native/training.lock` CPU 환경과 고정 upstream을 사용한다. 저장소 루트에서 실행한다.

```bash
export SONIC_UPSTREAM=/path/to/pinned/GR00T-WholeBodyControl
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1
.venv-native/bin/python native/learning.py --output-dir /tmp/native-learning \
  --iterations 128 --seed 9101 --fsq-levels 5 --learning-rate 0.0003
```

`--iterations`는 1..512다. 위 명령은 128회 비교 조건을 재현한다. 최종 평가에는 `learning.py`의 `load_policy(upstream, checkpoint)`와
`audit_policy(policy, seed, episodes=16)`를 사용한다. 조건·seed별 실패 및 외란 이후 미관측도 함께 읽는다.

Checkpoint는 FSQ 단계 설정도 저장한다. FSQ의 비영속 buffer는 weights-only state dictionary만으로 복원되지 않으므로 올바른 quantizer를 다시 구성해야 한다. 재로딩
뒤 같은 입력의 행동 출력이 정확히 일치하는지 검사한다. Optimizer·RNG·물리 전체의 exact-resume snapshot은 아니다.

`learning.json`에는 설정, 소스 hash, 원본 메서드 호출 수, optimizer·물리 횟수와 평가가 있다. 실패하면 `execution=FAILED`로 이전 성공 보고서를 대체한다.
`execution=COMPLETE`는 요청한 계산 완료만 뜻한다.
