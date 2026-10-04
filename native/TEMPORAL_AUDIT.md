# 현재와 끝점이 같아도 미래 입력을 구별하는가?

## 질문

앞선 [CONCEPTS.md](CONCEPTS.md)의 프레임 순서 반전은 입력 손상 실험이었다. 여기서는 **현재·정지 이력·종료 전체 상태는 같고 중간 경로는 다른 실제 물리 궤적**을 새로 만든다.
Encoder·공유 native Decoder·FSQ를 고정해 미래 내부 프레임에 반응하는지 검사한다.

## 실제 방법

`temporal_audit_protocol.json`에 후보·허용오차·checkpoint hash·입력 조건·초기 실제 상태 오차를 결과 확인 전에 선언했다. 정책을 재학습하지 않는다.

`temporal_pairs.py`는 기존 MuJoCo의 `minimize.least_squares`를 사용한다. 정규화된 제어값 7개를 각각 4 control steps씩 유지한다. Control step은
0.01초 물리 step 두 번이므로 **28 steps / 0.56초**다.

후보는 중간 tick 8, 12, 16과 변위 1, 2, 4, 6 mm의 조합이다. Seed **73421**로 [-0.25,0.25] m의 시작 카트 위치를 정하고 직립·정지에서 출발한다. Solver는 종료 상태
`[x, x_dot, theta, theta_dot]`를 시작 상태에 맞추면서 지정 중간 변위도 맞춘다. 제어 제한 ±0.25는 이 물리에서 ±2.5 N이다. 부호를 반전한 제어열도 비선형 시뮬레이터에서 별도로
실행한다.

종료 오차 허용값은 각 성분의 고유 단위에서 `1e-7`이다: 위치 m, 속도 m/s, 각도 rad, 각속도 rad/s. 중간 위치 차이는 0.1 mm를 넘어야 하며 첫 기록 행동 차이는 정규화 단위 `1e-5`
로 별도 검사한다. Solver 종료 메시지만 믿지 않고 재생·끝점·중간 변위·실패 기준·정지 이력을 확인한다. 탈락 후보도 보존한다.

Solver 참고: https://mujoco.readthedocs.io/en/stable/python.html#minimize

### 두 경로에서 같은 정보와 다른 정보

분기점의 실제 초기 상태·현재 reference·물리적으로 생성한 무입력 과거 이력은 같다. 종료 상태의 작은 수치 잔차는 원시 데이터에서 지우지 않는다. **첫 행동 정보 비교에만** 허용오차를 검사한 뒤 공통
끝점을 입력해 잔차로 방향을 알아맞히지 못하게 한다. 원시 입력과 공통 끝점 입력의 출력 차이도 기록한다.

입력은 전체 미래, 현재 프레임 반복, 인과적 reference 이력, 현재·끝점 사이 보간의 네 조건이다. 이력 순서는 현재부터 과거다. 축소 입력은 **기존 미래 Encoder의 입력 개입**이며 각 입력 형식으로 따로 학습한 기준 정책이 아니다.

두 기록 첫 힘의 평균은 같은 힘 하나를 예측해야 할 때 제곱 오차를 최소화한다. 이 하한은 **두 기록 행동을 재현하는 예측 문제**에만 해당한다. 성공적인 feedback 제어가 반드시 그 힘을 복제해야 한다는 하한은 아니다.

### 폐루프 재생

Joint와 marker Encoder는 같은 저장 Decoder를 사용한다. 실제 초기 상태를 정상 또는 `[0, .02, .001, -.02]`만큼 바꾼다. 성분별 단위는
`[m, m/s, rad, rad/s]` 다. 목표 reference는 그대로 유지한다.

평가는 28 controls / 0.56초다. 이후 preview에 필요한 꼬리는 각 기록 종료 상태에서 무입력 28 steps를 실제로 더 계산한다. 마지막 상태를 복제하는 padding이 아니다. Native
입력 시각은 현재부터 `0.00, 0.08, …, 0.56초`이며 브라우저 command preview의 `0.08, …, 0.64초`와 다르다. Reference 생성과 정책 물리는 별도라 실제 실행 중인
로봇의 미래 정답을 입력하지 않는다.

## 결과

12개 후보 중 **9쌍 허용 / 3개 탈락**이다. 허용된 9쌍 모두 기록 첫 행동이 다르다. 종료 상태의 성분별 절댓값 차이를 수치상 합친 최대는 약 `2.10e-16`이며 각 성분은 해당 고유 단위의
`1e-7` 기준 안에 있다. 혼합 단위 전체를 하나의 m 오차로 읽지 않는다.

탈락은 `(tick 8, amplitude 4 mm)`, `(8, 6 mm)`, `(16, 6 mm)`다. 이 제한된 solver·제어열 조건에서 제약을 만족하지 못한 것으로, 물리적 불가능의 증명은 아니다.

동일 현재·이력·공통 끝점에서 각 Encoder는 전체 미래 입력으로 **9쌍 중 6쌍**의 token과 행동을 구별했다. 축소 입력 세 조건은 분기점에서 쌍의 입력이 동일했다.

그러나 전체 미래의 첫 힘 예측 MSE는 joint **0.2340 N²**, marker **0.2277 N²**로, 공통 예측 하한 **0.1596 N²**보다 컸다. 어느 허용 쌍에서도 이 하한보다 낮지
않았다. 미래를 구별하는 것과 기록된 힘을 정확히 예측하는 것은 다르다.

조건마다 **18개 궤적(9쌍)**을 재생했다. 두 Encoder 모두 정상·상태 오차 조건의 모든 입력에서 18/18 완주했다. 정상 조건 카트 MAE는 다음과 같다.

| 입력 조건 | Joint Encoder | Marker Encoder |
|---|---:|---:|
| 전체 미래 | 1.294 mm | 1.229 mm |
| 현재 reference만 | 1.585 mm | 1.402 mm |
| 인과적 reference 이력 | 1.378 mm | 1.299 mm |
| 현재·종료 reference만 | **1.114 mm** | **0.992 mm** |

상태 오차 조건에서 전체 미래 MAE는 joint / marker **2.992 / 2.958 mm**, 끝점만 사용한 경우 **2.857 / 2.852 mm**였다. 이 궤적 집합에서는 전체 미래가 축소 입력 모두보다 낫지 않았다.

## 한계와 후속 비교

CartPole에서도 같은 현재·끝점과 다른 중간 경로를 물리적으로 만들 수 있었다. 고정 정책이 미래 차이에 반응한다는 결과가 새 궤적에서의 제어 우월성을 보장하지는 않는다. 익숙한 상태 범위라도 궤적 형태는 학습
분포 밖일 수 있다. 0.56초 완주는 이전 5초·10초 안정성 결과를 확장하지 않는다.

입력 형식마다 별도 Encoder를 같은 예산으로 학습한 후속 비교는 [MATCHED_CUES.md](MATCHED_CUES.md)에 있다. 위 고정 입력 개입 결과는 그대로 보존한다. 로봇 전환 기준은 [PIVOT.md](PIVOT.md)에 있다.

## 수치 검사와 재현

같은 입력 배열과 양자화 token은 정확히 비교한다. 신경망 힘 출력의 동등성 허용오차는 `1e-7 N`으로, 행동 구별 기준 `1e-6 N` 보다 작다. 동일 행에서도 `3.7253e-8 N` 차이가 발생해
bitwise zero를 요구하던 회귀 단언을 수정했다. 물리·끝점 허용 기준·가중치·측정 결과는 바뀌지 않았다.

기존 `native/training.lock` 환경과 고정 upstream을 사용한다.

```bash
export SONIC_UPSTREAM=/path/to/pinned/GR00T-WholeBodyControl
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
python -m pytest native/test_temporal_pairs.py native/test_temporal_audit.py -q
python native/temporal_audit.py --output-dir /tmp/sonic-temporal-audit
```

[audit.json](../evidence/temporal_audit/audit.json)은 선언·소스·checkpoint hash, 모든 후보와 조건별 재생 결과를 담는다.
[physical_pairs.npz](../evidence/temporal_audit/physical_pairs.npz)는 pickle 없이 원시 상태·제어·이력·solver 기록을 보존한다. 과학적 결론은 정책의
승리를 강제하는 테스트가 아니라 이 측정과 범위에 근거한다.
