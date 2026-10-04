# 공식 SONIC 모듈 연결 검증

이 문서는 **`core_smoke.py`라는 작은 연결 검사**를 설명합니다. 전체 native 경로의 최신 성능 보고서는 아닙니다. 실제 학습기 연결은 [TRAINING.md](TRAINING.md), 학습된
정책은 [LEARNING.md](LEARNING.md), 표현·공유 Decoder 비교는 [CONCEPTS.md](CONCEPTS.md)를 참고하세요.

브라우저는 SONIC 원리를 설명하는 별도 JavaScript 모델입니다. 이 Python 검사는 고정된 공식 소스의 `UniversalTokenModule`과 `BaseModule`, 실제
`vector_quantize_pytorch.FSQ`를 불러옵니다. 이름만 같은 신경망·양자화 함수를 다시 작성한 경로가 아닙니다.

## 이 검사에서 실제로 하는 일

```text
미래 위치·속도 8프레임 — 테스트 입력
               ↓
       공식 Encoder와 실제 FSQ
               ↓
       양자화된 token
          ┌────┴────┐
          ↓         ↓
현재 상태와 결합   Kinematic Decoder
          ↓         ↓
 Dynamic Decoder  미래 프레임 복원
          ↓
        출력 1개
```

**순전파**는 입력으로부터 출력을 계산하는 과정입니다. **역전파**는 출력 오차의 gradient가 필요한 앞쪽 모듈에 전달되는지 계산하는 과정입니다. 이 검사에서는 둘 다 실행하지만 가중치를 갱신하거나 물리
로봇을 구동하지 않습니다. 출력 하나가 계산된다고 안정적인 힘 제어기를 얻은 것은 아닙니다.

현재 상태만 바꿨을 때 reference-derived token은 유지되고 행동 출력은 달라지는지 검사합니다. 행동 오차는 Encoder와 Dynamic Decoder로, 복원 오차는 Encoder와
Kinematic Decoder로 전달되는지도 따로 확인합니다.

| 검사 구성 | Token 크기 | 각 scalar의 단계 수 | 출력 / 복원 |
|---|---:|---:|---|
| 축소형 1-token | 1 × 2 | 5 | 출력 1개 / 8 × 2 프레임 |
| 축소형 2-token | 2 × 2 | 5 | 출력 1개 / 8 × 2 프레임 |
| 공식 release와 같은 token 크기 | 2 × 32 | 32 | 출력 1개 / 8 × 2 프레임 |

마지막 행은 **token 크기만** 대응합니다. 전체 신경망 크기, 입력 종류, 데이터, 가중치와 휴머노이드 성능까지 같다는 뜻은 아닙니다. 이 검사에서 hidden layer는 `[32,32]`로 줄였고, 입력은 결정적인 합성 테스트 데이터입니다.

## 공식 코드 재사용과 변경 사항

공식 소스는
[NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0)
의 지정 commit에 고정합니다. 원본 파일 바이트와 실제 import 위치를 검사하며, 수정된 원본이나 다른 버전이면 실패합니다.

원본 설정에서 CartPole의 reference·상태·출력 차원과 작은 layer 폭을 지정했습니다. `g1_dyn`은 공식 코드가 행동 출력을 선택할 때 사용하는 **호환용 이름**입니다. G1 관측이나 가중치를 사용한다는 뜻이 아닙니다.

FSQ의 `return_indices=False`는 사용하지 않는 정수 index와 암시적 codebook table 생성을 생략합니다. FSQ의 단계는 고정되어 있고 학습된 codebook은 없습니다. 주변
Encoder·Decoder의 신경망 가중치가 학습됩니다. 숫자 양자화 자체는 실행되며, 실제 호출과 출력 grid를 검사합니다. 입력 형식 하나를 확인하는 검사이므로 가짜 SMPL·VR·카메라 데이터를 채워 넣지
않습니다.

Reference 정규화는 `[1.8,3.0]`, 상태 정규화는 `[1.8,3.0,0.55,4.0]`이며 해석 단위는 m, m/s, rad, rad/s입니다. 이것은 CartPole 전용 설정이지 다른 로봇의 보편적 기본값이 아닙니다.

브라우저 축소 예제의 scalar 양자화는 `q_i = round(1.998*tanh(z_i))/2`입니다.
`z_i`는 Encoder 출력의 i번째 성분, `q_i`는 무차원 양자화 값이며 가능한 값은 `{-1,-0.5,0,0.5,1}`입니다.
이 식을 upstream의 모든 FSQ 단계 설정(예: 32단계)에 그대로 적용하지 않습니다.
두 token slot도 가까운/먼 미래로 역할이 지정되지 않습니다.

## 별도 CPU 환경에서 실행

검증 대상은 Python 3.10 / Linux x86_64 / CPU 환경입니다. 전역 Python이나 GPU 드라이버를 변경할 필요가 없습니다. 의존성 누락은 명시적 실패이며, 저장된 PASS로 대체하지 않습니다.

```bash
python3.10 -m venv .venv-native
.venv-native/bin/python -m pip install \
  'torch==2.7.0+cpu' 'torchvision==0.22.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
.venv-native/bin/python -m pip install -r native/requirements.lock
.venv-native/bin/python -m pip check

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

Clone 대상은 기존 checkout을 덮어쓰지 않는 새로운 경로로 지정합니다.

## 결과의 범위

[원시 실행 보고서](../evidence/native_core_smoke.json)에는 소스 hash, 의존성, 실제 적용된 설정, 양자화 호출 횟수, 입출력 shape, gradient와 상태 변경 결과가 있습니다.

이 검사의 “optimizer 0회, 물리 rollout 없음”은 올바른 범위 표시입니다. **다른 파일에 이미 구현된 공식 trainer 연결까지 미구현이라는 뜻은 아닙니다.** 학습·물리 실행은
[TRAINING.md](TRAINING.md)를 참고하세요.
