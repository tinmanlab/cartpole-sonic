# 언제 CartPole을 넘어 다른 로봇이 필요한가?

## 질문과 판단 방법

CartPole은 구조·좌표·시간·학습 경로의 오류를 빠르게 확인하는 작은 환경이다. 학습 실패만으로 로봇을 바꾸지 않는다. **다음 질문에 필요한 물리·관측 구조가 현재 task에 없을 때** 최소 환경으로 전환한다.

질문, 현재 구조로 답할 수 없는 이유, 필요한 최소 추가 요소, 재사용할 검사, 새로 검증할 주장을 먼저 적는다. 이 문서는 환경 한계와 adapter 요구사항을 맡고 측정값은 각 실험 문서와 증거 파일에 둔다.

| 분류 | 의미 | 다음 행동 |
|---|---|---|
| IMPLEMENTATION_GAP | 코드·좌표·시각·checkpoint·입력 경로·기록 오류 | 기존 CartPole 경로에서 수정 |
| EXPERIMENT_GAP | 가능한 비교나 데이터 조건을 아직 실행하지 않음 | 범위가 정해진 비교 실행 |
| ENVIRONMENT_LIMIT | 필요한 actuator·접촉·관측 구조가 없음 | 최소 적합 환경 준비 |
| VERIFIED_LIMITED | 명시한 범위에서만 측정된 성질 | 증거와 범위 보존 |

Reference 의존성이 확인됐다는 것만으로 미래 기반 최적 제어, 사람 동작 retargeting, 다중 actuator 협응, 접촉 강건성이나 sim-to-real이 증명되지는 않는다.

## CartPole로 답할 수 있는 질문과 한계

| 개념 | CartPole에서 확인하는 방법 | 확인할 수 없는 성질 / 필요한 최소 환경 |
|---|---|---|
| Reference와 proprioception 분리 | 외부 동작을 고정하고 실제 상태만 바꿔 token·힘·복원을 확인 | 센서 추정·하드웨어 안전은 별도 검증 |
| 시간 정보 사용 | 같은 현재의 물리 경로, 별도 학습한 축소 입력 비교, 끝점을 유지한 내부 프레임 개입 | 온라인 planner 이점은 별도 benchmark 필요 |
| 여러 Encoder와 공유 latent | 같은 동작의 joint / marker 표현, 고정 Decoder, 혼합 행의 경로 선택 | 사람 형태·시각 가림·실제 다중 센서의 의미 일치는 별도 |
| FSQ와 복원 | grid·gradient·분리 평가·auxiliary 제거·입력 개입 | 보편적 필요성이나 최적 token 수는 증명하지 못함 |
| 독립적으로 구동되는 여러 관절 | 현재는 힘 입력 하나와 수동 막대 | 독립 관절 협응·actuator 출력 순서: 고정 기반 평면 2R Reacher |
| 평면 x/y 목표의 연속 기구학 여유도 | 현재는 여분의 구동 자유도 없음 | 같은 끝점의 연속 자세·null space: 평면 3R arm |
| 접촉·지지 전환 | 현재는 접촉을 끈 rail 모델 | 발 충돌·지지 교대·미끄럼·접촉 추정: 작은 평면 biped/walker |
| 일반 3D 회전과 부분적 사람 관측 | 현재 변환은 평면이며 전체 marker는 역변환 가능 | SO(3)·heading·3D 매핑·사람/로봇 모호성: 최소 3D 관절 모델 |

평면 2R arm은 일반적인 2차원 끝점 위치 목표에 **연속 여유도**가 없다. 여러 역기구학 분기가 있을 수 있지만 연속 null space와 다르다. 평면 walker도 일반 3D 휴머노이드 검증을 대신하지
않는다. 휴머노이드는 그 신체·접촉 구조 자체가 질문일 때 선택한다.

## 재사용 후보와 확인된 범위

기존 MuJoCo/native SONIC 경로에 로봇별 요소만 연결한다. 아래는 검토한 **소스 후보**이며 이 프로젝트에 설치·학습·검증했다는 뜻은 아니다.

독립 관절·좌표 질문에는 DeepMind Control Suite의 `reacher.xml` / `reacher.py`를 후보로 둔다. 검토 snapshot은
`a04e3e4cf56c12117d2294bb090f9acec21e5c67`, `reacher.xml` Git blob은 `343f799c01ec76abd396e74bcb011b7088f02273`이다. 원래 점
도달 보상은 SONIC 동작 추적 task가 아니므로 원본 task·asset 재현을 먼저 보존한다.

- https://github.com/google-deepmind/dm_control/blob/a04e3e4cf56c12117d2294bb090f9acec21e5c67/dm_control/suite/reacher.xml
- https://github.com/google-deepmind/dm_control/blob/a04e3e4cf56c12117d2294bb090f9acec21e5c67/dm_control/suite/reacher.py

접촉이 질문이면 같은 프로젝트의 평면 `walker.xml` / `walker.py`를 검토한다. 이동하는 평면 몸통과 구동 다리로 접촉을 추가한다. 원래 stand/walk task의 제어·물리 주기를 먼저 확인한다.

- https://github.com/google-deepmind/dm_control/blob/a04e3e4cf56c12117d2294bb090f9acec21e5c67/dm_control/suite/walker.xml
- https://github.com/google-deepmind/dm_control/blob/a04e3e4cf56c12117d2294bb090f9acec21e5c67/dm_control/suite/walker.py

해당 작업을 시작할 때 시험한 의존성 집합을 고정한다. 원본 Apache-2.0 고지와 포함 asset을 보존하며, 모델 조각만 복사해 include나 출처를 잃지 않는다.

## 재사용할 것과 로봇마다 정의할 것

고정 SONIC 신경망·trainer, 출처 검사, 소스·가중치·설정 연결, 궤적 단위 split, 불변 reference 재생, 경로 선택·결측 관측·gradient·동결 검사는 재사용한다.

CartPole 가중치·정규화·단일 힘 배율·rail·tip 식·action index·실패 각도·보상·데이터 차원은 로봇 공통 기본값이 아니다. 구조 재사용은 zero-shot 가중치 전이가 아니다.

2R에서는 상태 순서 `[q1,q2,qdot1,qdot2]`, body/site 이름, 두 actuator 순서와 단위, 끝점 local/world 좌표, 각도 wrapping, 속도 Jacobian, 관절·힘
제한, reset, 물리적으로 가능한 reference를 정의한다. 직접 torque인지 위치 목표+PD인지도 정한다. 현재 ±10 N 카트 힘은 두 torque나 두 관절 목표로 그대로 옮길 수 없다. 끝점
경로만으로 관절 경로가 유일하지 않으면 elbow marker·분기 label·궤적 문맥 중 무엇이 모호성을 해소하는지 명시한다.

기존에 참조한 공식 새 로봇 안내: https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/new_embodiments.html

## 전환 시 재현 순서와 한계

1. 선택한 upstream 예제를 asset·timestep까지 **수정 없이** 재현한다. 관측·행동 및 이름·순서 roundtrip을 기록한다.
2. 기존 native 경로에 환경·설정·loss binding만 추가한다. 잘못된 차원·actuator 이름·식별 불가능한 입력은 거부한다.
3. 물리적으로 확인한 작은 motion bank를 생성·재생하고 궤적 전체로 split한다. Reference와 실제 상태, 실패·절단·정상 완료를 구분한다.
4. 개념 검사, 제한된 원본 trainer 갱신, 고정 가중치 재생으로 **새로 필요한 성질**을 평가한다. CartPole 회귀 검사는 유지한다.

이 계획은 새 환경의 성능 증거가 아니다. 하드웨어·자격 증명·배포·큰 계산 확장은 별도 작업 범위다. 기존 환경의 가능한 비교나 알려진 오류는 로봇 전환과 독립적으로 해결한다.
