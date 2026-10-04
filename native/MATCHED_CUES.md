# 입력별로 따로 학습해도 미래 정보의 차이가 남는가?

## 질문

[TEMPORAL_AUDIT.md](TEMPORAL_AUDIT.md)는 기존 미래 Encoder에 낯선 축소 입력을 넣었다. 결과가 정보 부족 때문인지 입력 형식 때문인지 구분하려고, 여기서는 **각 입력 조건의
Encoder를 따로 학습**하고 새로운 물리 궤적 쌍으로 비교한다.

## 실제 방법

원본 `UniversalTokenModule`, `BaseModule`, FSQ와 기존 공유 Decoder를 재사용한다. 선택한 새 Encoder만 학습한다. PPO나 새 안정화 제어기를 추가하지 않으며, 기존 제어기와 Decoder는 고정한다.

`matched_cue_protocol.json`에 전체 미래·현재만·인과적 이력·현재와 끝점의 네 조건, joint / marker 두 표현, 데이터·seed·예산·지표를 선언했다. Encoder seed는
**52101, 52102, 52103**이다. 같은 표현·seed의 네 입력 조건은 동일 학습 파라미터로 시작한다. 이는 **한 고정 Decoder에 대한 Encoder 초기화 세 번**이며 전체 제어기/PPO
독립 학습 세 번이 아니다.

조건별로 같은 기록 행동의 지도학습 loss, optimizer, sample 순서와 고정 update 횟수를 사용한다. 한 쌍의 양쪽 부호를 같은 minibatch에 넣어 방향 불균형을 이용하지 못하게 한다. 최종
예산 checkpoint를 평가하며 validation으로 seed·시점·학습률을 고르지 않는다. 기록 상태에서 행동을 맞추는 지도학습과 실제 폐루프 제어는 별도로 측정한다.

### 데이터와 분리

기존 MuJoCo 쌍 생성기로 비선형 물리 궤적을 만든다. 종료 전체 상태·힘·중간 변위·solver 예산 검사를 유지하고 탈락 후보도 기록한다. 양쪽 부호와 모든 window는 같은 split에 둔다.

선언된 후보는 학습 36개, validation 12개, test 24개다. 위치 seed는 각각 **94101, 94102, 94103**이다. Test 변위는 학습 변위와 다르지만 같은 작은 국소 task 집합
안이다. 탈락을 쉬운 성공 사례로 바꾸지 않으며 solver 실패를 물리적 불가능이라 부르지 않는다.

미래 꼬리와 과거 이력은 물리 재생으로 생성한다. Padding이나 reset 경계 넘김은 없으며 현재 시뮬레이터의 실제 미래를 reference로 주지 않는다. 분기점에서는 현재·이력·끝점 입력이 양쪽 부호에
동일하다. 끝점의 허용오차 이하 잔차만 입력에서 공통화하고 원시 물리는 보존한다.

### 관측 정보와 잡음

전체 cart+tip 위치·속도는 이 평면 task에서 분석적으로 역변환할 수 있다. Tip만 있으면 다른 cart/pole 상태가 같은 tip 위치·속도를 만들 수 있다. **좌표 역변환 가능성과 신경망의 실제
센서 인식 능력은 별개**다. 전체 관측 경로는 빠진 marker를 0으로 채우지 않고 거부한다.

잡음 검사는 활성 marker-reference 좌표·속도만 선언된 물리 단위로 교란하며 모델마다 같은 잡음을 사용한다. 실제 proprioception과 정답 행동은 그대로다. 이는 **reference 잡음에
대한 오프라인 민감도**이며 잡음 상태의 폐루프 제어·카메라·가림 처리·센서 추정 검증이 아니다.

## 측정 결과

24개 모델 모두 선택 Encoder를 **1,200 updates** 학습했다. 허용된 데이터는 **학습 27/36쌍, validation 9/12쌍, test 20/24쌍**이다. 탈락한 16개도
`pairs.npz`에 있다. Test는 양쪽 부호를 포함한 **40개 궤적**이다.

첫 힘 예측 MSE의 평균 ± 표본 표준편차(Encoder seed 세 개)는 다음과 같다.

| 학습한 입력 조건 | Joint Encoder [N²], 평균 ± 표본 SD | Marker Encoder [N²], 평균 ± 표본 SD |
|---|---:|---:|
| 전체 미래 | **0.00436 ± 0.00081** | **0.02491 ± 0.00419** |
| 현재만 | 0.19155 ± 0.00140 | 0.19189 ± 0.00157 |
| 이력만 | 0.19155 ± 0.00068 | 0.19584 ± 0.00376 |
| 현재·끝점 | 0.19155 ± 0.00084 | 0.19467 ± 0.00614 |

이 test의 공통 첫 힘 예측 하한은 **0.18748 N²**다. 분기점 정보가 같은 축소 입력 모델은 방향을 구별할 수 없어 하한에 가까워진다. 전체 미래는 기록 행동을 훨씬 낮은 오차로 예측했다. 이 하한은
기록된 첫 행동을 모사하는 문제에만 적용되며 모든 성공 제어 전략의 하한이 아니다.

전체 window 힘 RMSE는 joint 전체 미래 **0.569 N** 대 축소 입력 **0.692–0.702 N**, marker 전체 미래 **0.700 N** 대 **0.756–0.844 N**이다. 모든
조건에서 분기점 sample 확률을 0.5로 정했으므로 분기점과 전체 window 지표를 함께 읽는다.

정상 조건 40개 궤적의 폐루프 카트 MAE는 다음과 같다.

| 학습한 입력 조건 | Joint Encoder [mm], 평균 ± 표본 SD | Marker Encoder [mm], 평균 ± 표본 SD |
|---|---:|---:|
| 전체 미래 | **0.393 ± 0.012** | **0.594 ± 0.032** |
| 현재만 | 1.021 ± 0.019 | 0.793 ± 0.012 |
| 이력만 | 0.871 ± 0.017 | 0.796 ± 0.012 |
| 현재·끝점 | 1.023 ± 0.023 | 0.807 ± 0.014 |

모든 seed·조건은 **0.56초에서 40/40 완주**했다. 같은 40개 물리 궤적을 Encoder 초기화 세 번으로 평가한 것이며 독립 환경 120개나 독립 학습 제어기 120개가 아니다. 정상 조건의 전체
미래 이점은 두 표현의 모든 평가 seed에서 나타났다. 표시 수업의 세 쌍·한 seed는 [GUIDED_LESSON.md](GUIDED_LESSON.md)의 별도 사례다.

실제 초기 상태 오차 `[0,0.02,0.001,-0.02]` (단위 `[m,m/s,rad,rad/s]`)에서는 차이가 대부분 줄었다. 전체 미래 MAE는 joint **2.629 ± 0.058 mm**,
marker **2.651 ± 0.148 mm**다. 축소 입력 평균 범위는 joint **2.650–2.710 mm**, marker **2.618–2.695 mm**이며 marker 이력의 평균은 조금 더
낮았다. 상태 외란에서도 전체 미래가 강건하게 우월하다는 결과는 아니다.

Marker 전체 미래의 깨끗한 입력 전체 window RMSE는 평균 **0.700 N**이다. Reference 위치 잡음 1 mm와 속도 잡음 0.01 m/s에서는 약 **0.717 N**, 속도 잡음 0.05
m/s에서는 약 **1.061 N**이었다. 선언된 위치·속도 잡음 네 조합과 개별 seed는 원시 보고서에 있다. Reference 속도 민감도를 보여주며 인식 필터 성능을 입증하지 않는다.

## 한계와 무결성 검사

동일 예산은 전역 최적 수렴이나 각 조건의 최선 성능을 보장하지 않는다. 유한 데이터·학습 및 고정 Decoder의 한계가 남는다. 세 seed의 표본 SD는 신뢰구간이 아니며 겹치는 window도 독립
episode가 아니다. 0.56초 완주를 장시간 안정성으로 확대하지 않는다.

초기 fingerprint가 `state_dict()`만 사용해 FSQ의 비영속 level/basis buffer를 놓쳤다. `_levels`를 바꾸는 회귀 검사로 이를 확인해 `named_buffers()`
까지 포함했다. 이는 실험의 무결성 검사 수정이며 upstream FSQ 결함이나 양자화 행동 변경이 아니다.

전체 제어기 독립 재학습, 장시간·잡음 proprioception 강건성, 카메라/VR, 다중 actuator·접촉·3D, 하드웨어·sim-to-real은 검증 범위 밖이다. 필요한 물리 구조가 없을 때의 최소 로봇
선택은 [PIVOT.md](PIVOT.md)에 있다.

## 재현과 자료

기존 `native/training.lock` 환경과 고정 공식 소스를 사용한다.

```bash
export SONIC_UPSTREAM=/path/to/pinned/GR00T-WholeBodyControl
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
python -m pytest native/test_matched_cues.py -q
python native/matched_cues.py --output-dir /tmp/sonic-matched-cues --steps 1200
```

[summary.json](../evidence/matched_cues/summary.json)은 읽기용 요약이다.
[matched.json.gz](../evidence/matched_cues/matched.json.gz)는 예측·loss·쌍별 재생을 모두 담은 무손실 압축 보고서다.
[pairs.npz](../evidence/matched_cues/pairs.npz)는 허용·탈락 물리와 solver 기록을 보존한다. Encoder checkpoint 24개에는 공유 Decoder를 중복
저장하지 않는다. 재현 실행은 비압축 보고서를 생성하며 원본·압축 SHA-256을 함께 기록한다.

검사는 소스·선언·checkpoint 연결, 24개 재로딩과 예측, 조건 간 초기화·sample 동일성, 기존 Decoder·FSQ 불변성, 표현·조건별 물리 재생과 요약 계산을 확인한다. 전체 학습 비교를 매 CI마다 반복하는 검사는 아니다.
