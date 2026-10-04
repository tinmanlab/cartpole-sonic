export const COURSE_VERSION = "2.10";

export const SONIC_FLOW = [
  {
    id:"task",
    nav:"목표 지정",
    official:"Interactive control / VR / video / VLA intent",
    toy:"CartPole goal x*",
    title:"무엇을 하고 싶은가?",
    viz:"task",
    input:"user / higher-level task",
    output:"high-level intent",
    why:"SONIC은 motor action을 직접 받는 것이 아니라 먼저 '무슨 움직임을 원하나'를 받는다.",
    misconception:"goal 자체가 FSQ token이 아니다.",
    question:"이 단계의 출력은 joint action인가, 아니면 아직 high-level intent인가?",
    action:"change-goal",
    concepts:[]
  },
  {
    id:"generator",
    nav:"궤적 생성",
    official:"Kinematic Planner / VR toolkit / motion generator",
    toy:"critically-damped future planner",
    title:"목표를 시간에 따른 궤적으로 바꾸기",
    viz:"reference",
    input:"high-level intent",
    output:"원하는 미래 궤적 (reference)",
    why:"Encoder가 읽을 수 있도록 task intent를 시간축을 가진 motion reference로 바꾼다.",
    misconception:"planner reference는 measured robot trajectory가 아니다.",
    question:"지금 보이는 future curve는 robot이 실제로 움직인 궤적인가?",
    action:"change-goal",
    concepts:[]
  },
  {
    id:"reference",
    nav:"원하는 궤적",
    official:"Robot motion / VR keypoints+legs / human motion",
    toy:"8 future [x, ẋ] frames",
    title:"Encoder가 실제로 읽는 입력",
    viz:"reference-vector",
    input:"future motion sequence",
    output:"structured tokenizer observation",
    why:"SONIC은 현재 state를 token화하는 것이 아니라 원하는 미래 motion을 token화한다.",
    misconception:"reference와 proprioception은 서로 다른 입력이다.",
    question:"reference에는 actual robot state가 섞여 있는가?",
    action:"change-goal",
    concepts:[]
  },
  {
    id:"encoder",
    nav:"표현 압축",
    official:"Robot / Hybrid / Human encoders",
    toy:"16D future reference → 2D latent",
    title:"원하는 궤적을 작은 표현으로 압축",
    viz:"encoder",
    input:"future reference",
    output:"압축된 연속 표현 z (latent)",
    why:"bottleneck(좁은 표현 공간)은 복사할 수 있는 정보를 제한한다. 중요한 정보가 남는지는 학습 목표와 검증 결과로 확인해야 한다.",
    misconception:"Encoder는 robot-state observer가 아니다.",
    question:"이 기본 실험에서 왜 16D를 2D로 줄일까? 복원 검증은 무엇을 확인하며 제어 성능과 어떻게 다를까?",
    action:"change-goal",
    concepts:[
      {id:"core",label:"기본 구조"},
      {id:"ae",label:"Autoencoder"},
      {id:"vae",label:"VAE · 선택 배경"}
    ]
  },
  {
    id:"quantizer",
    nav:"양자화",
    official:"Finite Scalar Quantization",
    toy:"2 scalar dims × 5 fixed levels",
    title:"연속 표현 z를 정해진 단계 q로 바꾸기",
    viz:"quantizer",
    input:"압축된 연속 표현 z (latent)",
    output:"quantized q",
    why:"연속 latent를 제한된 discrete representation으로 만들어 modality 사이에서 공유하기 쉬운 bottleneck을 만든다.",
    misconception:"FSQ의 finite levels는 learned VQ codebook처럼 움직이지 않는다.",
    question:"VQ, VQ-VAE, FSQ는 정확히 무엇이 다른가?",
    action:"live",
    concepts:[
      {id:"vq",label:"VQ"},
      {id:"vqvae",label:"VQ-VAE"},
      {id:"fsq",label:"FSQ"}
    ]
  },
  {
    id:"token",
    nav:"모션 토큰",
    official:"shared motion-token space",
    toy:"q=[q₁,q₂]",
    title:"FSQ를 통과한 '원하는 움직임' 표현",
    viz:"token",
    input:"quantized code vector(s)",
    output:"motion token",
    why:"downstream decoder가 원래 입력 modality 대신 하나의 공통 motion representation을 사용하게 한다.",
    misconception:"token은 꼭 LLM처럼 integer ID 하나가 아니다. SONIC decoder는 quantized numeric vectors를 사용한다.",
    question:"실제 SONIC의 64-D motion token은 어디서 나오는가?",
    action:"live",
    concepts:[
      {id:"core",label:"현재 토큰"},
      {id:"temporal",label:"표현 1개와 2개"},
      {id:"temporal-control",label:"제어기 1·2-token 비교"}
    ]
  },
  {
    id:"motion-decoder",
    nav:"동작 복원",
    official:"Robot Motion Decoder",
    toy:"Kinematic Decoder",
    title:"토큰에 남은 정보를 복원으로 확인",
    viz:"motion-decoder",
    input:"motion token",
    output:"reconstructed future motion",
    why:"작은 token이 원래 desired motion 정보를 유지하도록 auxiliary reconstruction signal을 준다.",
    misconception:"이 decoder가 physical motor command를 내는 것은 아니다.",
    question:"deployment action decoder와 reconstruction decoder는 같은 역할인가?",
    action:"change-goal",
    concepts:[]
  },
  {
    id:"control-decoder",
    nav:"행동 계산",
    official:"Robot Control Decoder",
    toy:"Dynamic Decoder",
    title:"원하는 움직임과 현재 상태로 힘 계산",
    viz:"control-decoder",
    input:"motion token + proprioception",
    output:"action / force",
    why:"같은 desired motion이어도 실제 robot pose/velocity가 다르면 즉시 필요한 action은 달라진다.",
    misconception:"q 자체가 motor command가 아니다.",
    question:"왜 token 외에 proprioception이 다시 필요한가?",
    action:"push",
    concepts:[]
  },
  {
    id:"robot",
    nav:"로봇 상태",
    official:"whole-body robot control",
    toy:"MuJoCo CartPole",
    title:"힘이 상태를 바꾸고, 바뀐 상태를 다시 읽기",
    viz:"tracking",
    input:"action",
    output:"measured actual state",
    why:"closed-loop controller이므로 실제 state가 다시 다음 Dynamic Decoder 입력이 된다.",
    misconception:"reference가 움직였다고 robot이 자동으로 그 reference와 일치하는 것은 아니다.",
    question:"현재 reference와 actual tracking error는 얼마나 되는가?",
    action:"live",
    concepts:[]
  }
];

export const TRAINING_TOPICS = [
  {
    id:"loss-flow",
    label:"손실의 흐름",
    title:"SONIC은 어떤 loss로 전체 구조를 학습하나?",
    viz:"training-flow",
    input:"rollouts + reference targets",
    output:"gradients for policy / representation",
    why:"physical tracking PPO와 representation auxiliary loss가 서로 다른 역할을 맡아 같은 network를 함께 학습시킨다.",
    misconception:"reconstruction loss 하나가 physical controller 전체를 학습하는 것이 아니다.",
    question:"PPO loss와 auxiliary reconstruction/alignment loss는 각각 어느 module을 가르치는가?"
  },
  {
    id:"what-learns",
    label:"학습되는 부분",
    title:"FSQ도 학습될까? 어떤 parameter가 실제로 바뀌나?",
    viz:"learning-graph",
    input:"loss gradients",
    output:"parameter updates",
    why:"FSQ 자체와 FSQ를 사용하는 Encoder/Decoder를 구분해야 VQ와 FSQ의 차이를 정확히 이해할 수 있다.",
    misconception:"FSQ의 fixed levels가 VQ codebook처럼 loss로 이동한다고 생각하면 안 된다.",
    question:"Encoder, FSQ, Dynamic Decoder, Kinematic Decoder 중 어느 것이 trainable parameter를 갖는가?"
  },
  {
    id:"alignment",
    label:"입력 표현 정렬",
    title:"왜 서로 다른 Encoder가 같은 Universal Token 의미를 만들까?",
    viz:"alignment",
    input:"paired representations of the same motion",
    output:"aligned shared latent space",
    why:"같은 FSQ를 쓴다고 서로 다른 Encoder의 latent 의미가 자동으로 같아지지 않기 때문이다. 이 toy는 같은 future motion을 16D full trajectory와 8D sparse keypoints라는 두 표현으로 만들어 실제 alignment를 학습한다.",
    misconception:"FSQ 하나만 붙이면 universal token이 자동으로 생기는 것이 아니다. 또한 이 toy의 두 표현은 G1/SMPL/teleop 자체가 아니라 alignment mechanism을 재현하기 위한 analogue다.",
    question:"같은 motion의 서로 다른 표현이 같은 q/action 의미로 수렴하려면 어떤 alignment signal이 필요한가?"
  },
  {
    id:"optimizer-sensitivity",
    label:"최적화 조건 비교",
    title:"왜 2-token controller는 같은 PPO 설정에서 더 흔들렸을까?",
    viz:"optimizer-evidence",
    input:"matched bootstrap controllers + controlled PPO ablations",
    output:"paired-seed / matched-budget / actual Adam-step evidence",
    why:"token slot 수를 늘리면 representation capacity뿐 아니라 policy interface와 optimization sensitivity도 바뀔 수 있기 때문이다.",
    misconception:"2-token이 나빴다는 한 결과만 보고 token 수 자체가 원인이라고 결론내리면 안 된다.",
    question:"seed 한 개와 gradient 크기만으로 token 수의 우열이나 원인을 확정할 수 있는가?"
  },
  {
    id:"ppo",
    label:"PPO",
    title:"보상으로 추적 정책을 어떻게 개선하나?",
    viz:"training",
    input:"physics rollout + tracking reward",
    output:"updated control policy",
    why:"reference를 실제 physics에서 따라가는 능력은 reconstruction이 아니라 closed-loop rollout reward가 가르친다.",
    misconception:"PPO training curve는 quantization quality 자체를 직접 측정하는 그래프가 아니다.",
    question:"PPO가 줄이려는 physical error는 무엇이며, 어느 network가 그 gradient를 받는가?"
  }
];


export const RUNTIME_DETAILS = {
  task:{
    easy:"사용자에게서 원하는 목표를 받는다. 아직 관절 행동이나 힘을 결정하지 않는다.",
    mechanism:"브라우저는 목표 위치 x* 하나를 받는다. 궤적 생성기가 이 목표를 사용하며, FSQ나 제어 Decoder가 직접 받는 입력은 아니다.",
    sonic:"실제 SONIC 앞단은 interactive control, VR, video/VLA 등 서로 다른 task interface가 될 수 있다. 이 단계의 출력 형식은 interface마다 다르며 motor command가 아니다.",
    ifMissing:"외부 task와 controller 사이의 의미 연결이 사라져 무엇을 수행해야 하는지 지정할 수 없다.",
    next:"high-level intent만으로는 시간에 따른 자세/속도를 알 수 없으므로 Motion Generator가 future motion으로 확장한다.",
    toyShape:"1 scalar goal x*",
    sonicShape:"interface-dependent intent / targets"
  },
  generator:{
    easy:"목표 한 점을 '앞으로 어떻게 움직일지'라는 짧은 계획으로 펼치는 단계다.",
    mechanism:"toy planner는 plannerContext와 goal에서 critically-damped future trajectory를 만들고 8 future frame의 [x,ẋ]를 출력한다. policy/control은 50 Hz로 진행된다.",
    sonic:"SONIC에서는 Kinematic Planner, VR toolkit, 다른 motion generator가 이 역할을 맡을 수 있다. generator는 control policy와 분리된 upstream motion source다.",
    ifMissing:"Encoder는 시간 구조가 없는 goal만 받아 '어떤 속도와 궤적으로' 움직여야 하는지 알 수 없다.",
    next:"생성된 future motion을 Encoder가 읽을 수 있는 구조화된 Motion Reference로 전달한다.",
    toyShape:"8 future frames × [x,ẋ]",
    sonicShape:"generator-dependent future motion representation"
  },
  reference:{
    easy:"원하는 미래 궤적(reference)이다. 실제로 관측한 로봇 궤적과 구분한다.",
    mechanism:"미래 8개 시각의 위치 x와 속도 ẋ를 이어 붙인 16개 값이다. 현재 로봇 상태(proprioception)는 여기에 포함하지 않고 제어 Decoder가 따로 받는다.",
    sonic:"실제 SONIC은 robot motion, human/SMPL motion, VR keypoint+leg target 등 modality별 motion representation을 Encoder 입력으로 사용한다.",
    ifMissing:"Encoder가 'desired future motion' 대신 현재 상태나 단일 goal만 보게 되어 motion intent와 robot state의 분리가 무너진다.",
    next:"16D 이상의 긴 reference를 compact shared representation으로 만들기 위해 Encoder가 필요하다.",
    toyShape:"16D = 8 × 2",
    sonicShape:"modality-specific future motion features × time"
  },
  encoder:{
    easy:"원하는 미래 궤적을 작은 연속 표현(latent) z로 압축한다.",
    mechanism:"toy MLP Encoder는 16D reference를 2D continuous z로 압축한다. Encoder weight는 PPO, reconstruction auxiliary, 그리고 실제 SONIC에서는 cross-encoder alignment gradient를 받을 수 있다.",
    sonic:"실제 SONIC은 Robot/G1, Human/SMPL, teleop 계열 등 여러 Encoder를 사용하고 같은 semantic latent/token space로 정렬한다. toy의 2D는 시각화를 위한 의도적 축소다.",
    ifMissing:"각 modality의 큰 원본 reference를 decoder가 직접 처리해야 하고, shared compact interface를 만들기 어렵다.",
    next:"Encoder output z는 아직 continuous이므로 Quantizer가 finite/discrete representation으로 바꾼다.",
    toyShape:"16D → 2D z",
    sonicShape:"modality-specific input → 32 scalar dims per token (release configuration)"
  },
  quantizer:{
    easy:"연속 표현 z를 정해진 수준으로 바꿔 양자화된 표현(token) q를 만든다.",
    mechanism:"VQ는 z에 가장 가까운 학습 코드 벡터를 고른다. FSQ는 각 성분을 고정 수준으로 반올림한다. 브라우저는 성분 2개에 각각 5수준을 쓴다. STE는 불연속 선택 대신 기울기를 전달하는 학습 근사다.",
    sonic:"release 계열 config는 token당 32 scalar dims, scalar당 32 fixed levels, 최대 2 tokens를 사용한다. FSQ의 finite levels는 trainable codebook이 아니다.",
    ifMissing:"latent가 완전히 continuous라 modality 사이 shared discrete bottleneck의 제약이 사라지고 universal token interface가 덜 규격화된다.",
    next:"Quantizer output q를 downstream decoder가 사용할 공통 Universal Token representation으로 해석한다.",
    toyShape:"z∈R² → q∈{5 levels}²",
    sonicShape:"2 tokens × 32 scalar dims, 32 fixed levels/scalar"
  },
  token:{
    easy:"원하는 움직임을 전달하는 양자화된 수치 표현이다. 모터에 적용할 힘은 아니다.",
    mechanism:"브라우저 token은 q₁, q₂ 두 값이다. 원하는 미래 궤적을 표현하며, 현재 로봇 상태는 제어 Decoder에 따로 전달한다. 같은 고정 수준을 쓰는 것만으로 서로 다른 입력의 의미가 정렬되지는 않는다.",
    sonic:"release 구조는 2 temporal tokens × 32 scalar dims = 64 flattened numeric values를 control decoder interface로 사용한다. universal 의미는 FSQ만이 아니라 multi-encoder alignment 학습까지 필요하다.",
    ifMissing:"decoder가 다시 원래 modality별 reference를 직접 이해해야 해서 하나의 공통 control interface를 만들기 어렵다.",
    next:"같은 token은 두 용도로 해석된다: motion information을 복원하는 Motion Decoder와 실제 action을 만드는 Control Decoder.",
    toyShape:"1 token × 2 scalar values",
    sonicShape:"2 temporal tokens × 32 scalar dims = 64 flattened values"
  },
  "motion-decoder":{
    easy:"양자화된 표현에서 미래 궤적을 복원해 입력 정보가 얼마나 남았는지 검사한다.",
    mechanism:"q에서 미래 위치·속도 16개 값을 복원한다. 정규화된 입력과 복원값의 평균 제곱 오차(MSE)를 보조 손실로 사용한다. 이 무차원 오차는 제어 추적 오차와 다르다.",
    sonic:"공식 SONIC의 Robot Motion Decoder 역할과 대응시키되, 이 CartPole 구현에서는 Kinematic Decoder를 training auxiliary path로 사용한다. physical motor action은 별도 Control Decoder가 낸다.",
    ifMissing:"이 실험에서는 복원 보조 손실이 사라져 입력 정보의 보존을 직접 확인하기 어렵다. 모든 제어기에 복원이 필수인 것은 아니며 제어 성능은 따로 평가해야 한다.",
    next:"future motion을 복원하는 것만으로 robot을 안정적으로 움직일 수 없으므로 actual proprioception을 포함한 Robot Control Decoder가 필요하다.",
    toyShape:"q(2D) → future reconstruction(16D)",
    sonicShape:"universal token → robot-motion representation"
  },
  "control-decoder":{
    easy:"양자화된 움직임 표현(token)과 현재 로봇 상태(proprioception)를 합쳐 다음 힘을 계산한다.",
    mechanism:"toy Dynamic Decoder는 q₁,q₂ + normalized [x,ẋ,θ,θ̇] 4D를 받아 scalar force를 출력한다. Push는 q/reference를 유지하고 state만 바꿔 decoder가 다른 action을 내는지 검증한다.",
    sonic:"공식 G1 dynamic decoder는 token_flattened + proprioception → action 구조다. 이 경로가 physical tracking PPO의 핵심 policy path다.",
    ifMissing:"token만 보는 open-loop decoder는 같은 motion intent라도 실제 robot이 넘어지거나 밀렸을 때 correction을 만들 수 없다.",
    next:"action은 실제 physics에 적용되고 measured state가 다시 feedback되어 closed loop가 완성된다.",
    toyShape:"q(2D)+state(4D) → force(1D)",
    sonicShape:"flattened token + robot proprioception → whole-body action"
  },
  robot:{
    easy:"계산한 힘을 물리 시뮬레이션에 적용하고, 다음 로봇 상태를 제어기에 돌려준다.",
    mechanism:"브라우저 MuJoCo는 10ms마다 물리를 계산한다. 정책은 20ms마다 힘을 결정해 두 물리 단계 동안 유지한다. 위치·속도·각도·각속도가 다음 제어 입력이다.",
    sonic:"실제 deployment에서는 humanoid body, actuator, contact, sensors가 이 closed-loop plant를 구성한다. 이 toy는 multi-contact/sim2real을 재현하지 않는다.",
    ifMissing:"physics feedback이 없으면 reference tracking이 실제로 되는지 검증할 수 없고 disturbance correction도 불가능하다.",
    next:"feedback state가 다시 Robot Control Decoder 입력으로 돌아가며 loop가 반복된다.",
    toyShape:"force → MuJoCo → 4D measured state",
    sonicShape:"whole-body action → robot/contact dynamics → measured proprioception"
  }
};

export const TRAINING_DETAILS = {
  "loss-flow":{
    easy:"보상으로 정책을 개선하는 경로와, 입력 복원·표현 정렬을 돕는 보조 학습 경로를 구분한다.",
    mechanism:"브라우저 제어기는 PPO(정책 개선)와 reconstruction(입력 복원) 보조 손실로 갱신한다. alignment(표현 정렬)는 별도 실험이다. 공식 구조의 보조 손실과 이 실험의 갱신 경로를 구분한다.",
    sonic:"release training은 PPO와 token reconstruction / cross-encoder latent-alignment auxiliary losses를 결합한다.",
    ifMissing:"이 PPO 경로를 제거하면 보상 기반 개선은 없어지지만, 지도학습·모방학습 등 대안이 있다. 복원 오차만으로 추적 성능을 입증할 수는 없다.",
    toyShape:"PPO + reconstruction aux",
    sonicShape:"PPO + weighted reconstruction/alignment auxiliaries",
    highlights:["encoder","quantizer","token","motion-decoder","control-decoder","robot"]
  },
  "what-learns":{
    easy:"FSQ 칸 자체가 배우는 것이 아니라 Encoder/Decoder가 그 고정 칸을 잘 쓰는 법을 배운다.",
    mechanism:"Encoder, Dynamic Decoder, Kinematic Decoder, Critic은 trainable parameter를 가진다. FSQ levels는 fixed hyperparameter이고 round backward는 STE가 대리한다.",
    sonic:"release FSQ는 learned VQ codebook 대신 fixed scalar levels를 사용한다. Encoder가 PPO/auxiliary gradient를 받아 bin 사용을 바꾼다.",
    ifMissing:"FSQ와 VQ의 핵심 차이를 놓치고 'FSQ codebook이 학습된다'는 잘못된 모델을 갖게 된다.",
    toyShape:"trainable MLPs around fixed FSQ",
    sonicShape:"multiple trainable encoders/decoders around fixed FSQ",
    highlights:["encoder","quantizer","motion-decoder","control-decoder"]
  },
  alignment:{
    easy:"같은 움직임을 서로 다른 표현으로 보더라도 같은 token 의미가 되도록 두 Encoder를 맞추는 학습이다.",
    mechanism:"toy는 같은 future motion을 full trajectory 16D와 sparse keypoints 8D로 만든다. primary Encoder A를 anchor로 고정하고 secondary Encoder B의 latent가 z_A에 가까워지도록 MSE alignment loss로 실제 학습한다. 같은 FSQ와 같은 Control Decoder를 사용해 token agreement와 action difference도 측정한다.",
    sonic:"실제 SONIC은 G1/SMPL/teleop 등 여러 modality Encoder를 auxiliary latent-alignment loss로 공동 정렬한다. toy의 frozen-anchor 방식은 mechanism을 안정적으로 보여주기 위한 단순화이며 실제 modality/공동학습을 그대로 재현하지 않는다.",
    ifMissing:"같은 FSQ quantizer를 써도 Encoder마다 latent 좌표의 의미가 달라져 같은 motion이 다른 q/action으로 해석될 수 있다.",
    toyShape:"full 16D → Encoder A(anchor) ; sparse 8D → Encoder B(trainable) → same 2D latent / same FSQ",
    sonicShape:"multiple modality encoders → aligned shared latent → shared FSQ / Universal Token",
    highlights:["encoder","quantizer","token"]
  },
  "optimizer-sensitivity":{
    easy:"한 번의 학습 결과만으로 token 수의 우열을 정할 수 없다. 같은 학습 횟수에서 여러 PPO 실행을 비교하고, 모델이 실제로 얼마나 바뀌었는지 확인한다.",
    mechanism:"전체 파라미터와 actor 파라미터를 따로 센다. 같은 bootstrap을 사용한 3개 PPO seed에서 1·2-token과 actor LR ×1/×0.05를 +10/+50회 각각 비교한다. gradient 크기를 이동량으로 간주하지 않고 실제 Adam 가중치 변화, 같은 입력에서의 token 변화율, 정책 분포 변화(KL)를 측정한다.",
    sonic:"이것은 fixed-bootstrap CartPole 조건부 실험이다. 독립적인 모델 초기화, 전체 motion 데이터, humanoid dynamics까지 검증한 결과가 아니다. 작은 LR이 SONIC에도 최적이라는 결론은 내리지 않는다.",
    ifMissing:"seed 한 개나 서로 다른 학습 예산을 비교하고, 전체 파라미터 수가 같으면 같은 모델 능력이라고 오해할 수 있다.",
    toyShape:"total / actor: 1-token 1220 / 607; 2-token 1366 / 705; total-match 1220 / 643; actor-match 1268 / 607",
    sonicShape:"representation design, optimizer and independent validation are separate questions",
    highlights:["encoder","token","control-decoder","robot"]
  },
  ppo:{
    formula:"L = E[min(rₜ Aₜ, clip(rₜ, 0.8, 1.2) Aₜ)] (최대화)\nrₜ = π새(aₜ|sₜ)/π이전(aₜ|sₜ), Aₜ: 예상보다 좋은 정도(이점). a: 행동, s: 정책 입력, E: 표본 평균. 브라우저는 이 목표의 음수를 최소화한다. 물리 미분은 하지 않는다.",
    easy:"원하는 미래 궤적을 따라가고 균형을 유지하도록 보상으로 정책을 개선하는 학습이다.",
    mechanism:"MuJoCo 실행에서 추적·균형·행동 비용에 따른 보상을 모은다. 이점(advantage)과 행동 로그확률로 Encoder와 제어 정책을 갱신한다. 물리 시뮬레이터를 통과해 미분하지 않는다. Critic은 기대 보상을 예측하도록 학습한다.",
    sonic:"실제 SONIC은 대규모 병렬 humanoid physics tracking PPO를 사용한다. toy의 작은 PPO는 역할만 보존한다.",
    ifMissing:"이 PPO 경로를 제거하면 보상 기반 개선은 없지만 지도학습 등 대안이 있다. 표현의 복원만으로 실제 제어 성능을 입증할 수는 없다.",
    toyShape:"8 env × 96 control steps / iteration",
    sonicShape:"large-scale parallel humanoid tracking rollouts",
    highlights:["encoder","control-decoder","robot"]
  }
};

export const CONCEPT_TEXT = {
  "ae": {
    "formula": "MSE = (1/N) Σⱼ (rⱼ − r̂ⱼ)²\nN: 성분 수, r: 정규화된 목표, r̂: 복원값. 위치·속도를 각각 정규화한 평균이므로 무차원이며 mm가 아니다.",
    "title": "Autoencoder — 압축 후 무엇이 남았는가?",
    "input": "원하는 미래 궤적 (reference)",
    "output": "연속 표현 z와 복원한 궤적",
    "short": "Encoder는 원하는 미래 궤적(reference)을 작은 연속 표현 z로 압축하고, Decoder는 입력을 복원한다.",
    "why": "작은 표현으로 줄인 뒤에도 원래 궤적 정보를 얼마나 되찾는지 확인한다.",
    "mechanism": "입력 → Encoder → 연속 표현 z → Decoder → 복원값의 순서다. 아래 정규화 MSE로 Encoder와 Decoder를 함께 학습한다.",
    "sonic": "SONIC에서는 이 생각이 motion Encoder + Robot Motion/Kinematic Decoder auxiliary path로 이어진다. physical action은 별도 Control Decoder가 낸다.",
    "ifMissing": "latent가 motion semantic을 보존하는지 직접 검사할 보조 신호가 약해진다.",
    "question": "왜 작은 latent를 다시 복원해 보는 것이 bottleneck의 품질 검사가 되는가?",
    "toyShape": "16D reference → 2D z → 16D reconstruction",
    "sonicShape": "motion Encoder → shared latent/token → Robot Motion Decoder auxiliary reconstruction",
    "key": "AE는 SONIC deployment 구조 그 자체가 아니라 Encoder/auxiliary Decoder를 이해하기 위한 기본 개념이다."
  },
  "vae": {
    "title": "VAE — 확률적 표현을 만드는 선택 배경",
    "input": "input/reference features",
    "output": "평균 μ, 표준편차 σ → 확률적으로 뽑은 연속 표현 z",
    "short": "Encoder가 평균과 표준편차를 내고, 그 분포에서 연속 표현 z를 뽑는다.",
    "why": "continuous latent를 regularized probabilistic space로 만드는 계열이다.",
    "mechanism": "Encoder가 μ,σ를 출력하고 z=μ+σ·ε로 sample한다. reconstruction loss와 KL divergence를 함께 최적화한다.",
    "sonic": "SONIC runtime은 VAE를 핵심 block으로 사용하지 않는다. VAE는 VQ-VAE라는 이름의 계보를 이해하기 위한 선택 배경지식이다.",
    "ifMissing": "SONIC 이해에는 치명적이지 않다. 이 탭을 건너뛰어도 VQ→VQ-VAE→FSQ 본선을 이해할 수 있다.",
    "question": "VAE의 μ/σ/KL은 왜 FSQ 이해의 필수 선행조건이 아닌가?",
    "toyShape": "concept-only: μ,σ → sampled z (not implemented in toy runtime)",
    "sonicShape": "optional background; not a SONIC runtime block",
    "key": "FSQ의 직접 선행 개념은 VAE보다 VQ/VQ-VAE다. VAE는 선택 배경지식이다.",
    "formula": "z = μ + σ ⊙ ε, ε ~ N(0,I)\nμ: 평균, σ: 표준편차, ε: 표준정규 표본, ⊙: 성분별 곱. 복원 오차와 분포 차이(KL)를 함께 다룬다. 이 탭은 설명이며 VAE 학습을 실행하지 않는다."
  },
  "vq": {
    "formula": "k* = argminₖ ‖z − eₖ‖²; q = e[k*]\nz: 연속 표현, eₖ: k번째 학습 코드, q: 선택된 벡터. 가장 가까운 코드의 제곱거리를 최소화한다.",
    "title": "VQ — 가장 가까운 학습 벡터 선택",
    "input": "압축된 연속 표현 z (latent)",
    "output": "가장 가까운 학습 벡터 q",
    "short": "압축된 연속 표현 z를 학습되는 코드 목록에서 가장 가까운 벡터 q로 바꾼다.",
    "why": "연속 표현을 유한한 대표 벡터 중 하나로 바꾼다.",
    "mechanism": "아래 수식처럼 z와 각 코드 벡터의 제곱거리를 비교한다. 가장 가까운 코드를 고르는 연산은 불연속이므로 Encoder 기울기와 코드 갱신을 따로 처리한다.",
    "sonic": "SONIC은 learned VQ codebook 대신 FSQ를 사용한다. VQ는 FSQ가 무엇을 단순화했는지 이해하기 위한 직접 비교 기준이다.",
    "ifMissing": "FSQ의 장점인 'learned codebook 제거'가 왜 중요한지 이해하기 어렵다.",
    "question": "VQ에서 z가 codebook의 어느 vector로 가는지는 어떻게 결정되고, 그 vector 자체는 학습되는가?",
    "toyShape": "z∈R² → nearest of 8 learned 2D code vectors",
    "sonicShape": "comparison baseline; SONIC runtime uses FSQ instead of a learned VQ codebook",
    "key": "브라우저는 선택된 코드 벡터를 해당 z의 평균 쪽으로 이동시킨다. FSQ의 고정 수준과 다르다."
  },
  "vqvae": {
    "formula": "k* = argminₖ ‖z − eₖ‖²; q = e[k*]\nz: 연속 표현, eₖ: 학습 코드. 원 논문: 복원 + 코드북 + commitment 손실. 브라우저: STE + z 근접 항 + 코드 평균 갱신. FSQ에는 학습 코드북이 없다.",
    "title": "VQ-VAE — 양자화와 복원을 함께 학습",
    "input": "원하는 궤적 → 연속 표현 z",
    "output": "선택한 코드 q → 복원한 궤적",
    "short": "입력을 압축하고, 가까운 코드를 선택하고, 그 코드로 입력을 복원한다.",
    "why": "코드 선택이 불연속이어도 앞단을 학습하고, 연속 표현과 대표 벡터가 서로 멀어지지 않도록 해야 한다.",
    "mechanism": "순방향은 z → 가장 가까운 코드 q → 복원이다. 역방향은 STE(불연속 선택을 대신하는 기울기)로 Encoder를 갱신한다. 브라우저 VQ는 z를 q에 가까이 두는 항과 선택된 코드의 이동 평균 갱신을 사용한다. 원 논문의 코드북·commitment 손실을 그대로 실행하는 것은 아니다.",
    "sonic": "SONIC FSQ는 이 구조에서 learned vector codebook/commitment/dead-code 관리 부담을 줄인 대안으로 이해하면 된다.",
    "ifMissing": "FSQ가 단순히 'VQ보다 다른 rounding' 정도로만 보여 왜 연구적으로 의미가 있는지 놓치게 된다.",
    "question": "대체 기울기, z와 코드의 거리, 대표 벡터 갱신은 각각 무엇을 해결하는가?",
    "toyShape": "16D → z2 → nearest learned q2 → recon16",
    "sonicShape": "VQ-VAE predecessor concept → SONIC FSQ removes learned codebook machinery",
    "key": "forward에서는 q를 쓰고 backward에서는 STE로 Encoder까지 gradient를 보낸다. 이것이 '그냥 diagram'이 아니라 학습 메커니즘의 핵심이다."
  },
  "fsq": {
    "formula": "브라우저 근사: qᵢ = round(1.998 × tanh(zᵢ))/2\nzᵢ: 연속 표현의 i번째 성분, qᵢ: 양자화 값. 수준은 {−1, −0.5, 0, 0.5, 1}. 이 화면은 정규화된 q를 표시한다. 공식 32수준 설정과 다르며, FSQ만으로 의미 정렬이 생기지는 않는다.",
    "title": "FSQ — 각 성분을 고정된 단계로 양자화",
    "input": "연속 표현 z의 각 숫자",
    "output": "고정 단계로 반올림한 q",
    "short": "압축된 연속 표현(latent)의 각 성분을 제한하고 반올림해 양자화된 표현(token)을 만든다.",
    "why": "학습되는 대표 벡터 목록 없이, 각 숫자가 사용할 단계를 미리 정한다.",
    "mechanism": "각 z 성분을 tanh로 제한한 뒤 아래 식으로 반올림한다. 순방향은 정해진 단계의 값을 내고, 역방향은 대체 기울기(STE)를 앞단 Encoder로 전달한다. 반올림 자체를 매끈한 함수라고 가정하는 것은 아니다.",
    "sonic": "release 계열 config는 2 temporal tokens × 32 scalar dims, scalar당 32 fixed levels를 사용해 flattened 64-D token interface를 만든다.",
    "ifMissing": "continuous latent를 그대로 쓰게 되어 finite shared bottleneck의 규격성과 discrete token interface를 잃는다.",
    "question": "단계 값은 고정인데 Encoder는 무엇을 바꾸며 학습하는가?",
    "toyShape": "z∈R² → q on 5×5 implicit fixed grid",
    "sonicShape": "2 tokens × 32 scalar dims, 32 fixed levels/scalar → 64 flattened values",
    "key": "격자의 단계 값은 고정된다. Encoder와 Decoder가 그 단계를 유용하게 쓰는 방법을 학습한다."
  },
  "temporal": {
    "title": "1 token vs 2 token slots — 왜 future window를 여러 token으로 표현할까?",
    "input": "same whole future window",
    "output": "1-slot vs 2-slot FSQ representations + reconstructions",
    "short": "전체 future window를 한꺼번에 읽되, 출력 representation을 token slot 하나가 아니라 여러 slot으로 나누어 더 많은 discrete capacity를 쓴다.",
    "why": "하나의 작은 token이 표현할 수 있는 finite combinations는 제한적이다. 여러 token slot을 쓰면 whole-window 정보를 더 풍부하게 보존할 수 있다.",
    "mechanism": "toy는 같은 16D future window를 두 모델에 넣는다. 1-token model은 2 scalar FSQ, 2-token model은 2×2 scalar FSQ를 출력한다. 둘 다 whole window를 공동으로 읽으며 reconstruction loss로 학습된다. token 1=근미래, token 2=원미래 같은 역할은 미리 지정하지 않는다.",
    "sonic": "공식 SONIC MLP Encoder는 temporal input을 flatten해 읽고, 출력은 max_num_tokens=2 temporal slots로 reshape한다. release config는 token_dim=32이므로 2×32=64 flattened values다. slot 의미는 architecture/config이 정하는 shape이지 near/far semantic을 하드코딩한 것이 아니다.",
    "ifMissing": "token slot 수와 token dimension을 같은 것으로 오해하거나, token 1/2에 임의의 시간 의미를 붙이게 된다.",
    "question": "왜 token을 두 개 쓰는가? 그리고 token 1과 token 2가 각각 near/far를 담당한다고 말해도 되는가?",
    "toyShape": "same 16D window → 1×2 FSQ vs 2×2 FSQ",
    "sonicShape": "whole future window → 2 token slots × 32 scalar dims → 64 flattened values",
    "key": "여러 표현 칸은 용량을 늘린다. 각 칸이 근미래·원미래를 뜻하도록 미리 지정되지는 않는다."
  },
  "temporal-control": {
    "title": "1 token vs 2 token control — 표현력 증가가 실제 제어도 좋아지게 할까?",
    "input": "same future reference + same actual proprioception",
    "output": "1-token vs 2-token Dynamic Decoder actions → 브라우저 MuJoCo WASM",
    "short": "두 controller를 같은 teacher/bootstrap과 같은 PPO budget으로 학습해 실제 tracking과 disturbance recovery를 비교한다.",
    "why": "reconstruction이 좋아졌다는 사실만으로 physical control이 좋아졌다고 결론낼 수 없기 때문이다.",
    "mechanism": "1-token controller는 16D reference→2D→FSQ q2, 2-token controller는 16D→4D→reshape 2×2→FSQ q4를 사용한다. 각 q는 같은 4D proprioception과 결합되어 별도 Dynamic Decoder가 force를 낸다. 두 controller는 동일 300-step teacher imitation과 동일 PPO hyperparameter/budget을 받는다.",
    "sonic": "실제 SONIC Robot Control Decoder는 flattened multi-token representation과 proprioception을 받아 action을 낸다. 이 toy는 token-slot 수가 representation capacity뿐 아니라 policy optimization 난이도에도 영향을 줄 수 있음을 분리해 보여준다.",
    "ifMissing": "'2-token reconstruction이 더 좋다 → 2-token controller도 반드시 더 좋다'는 잘못된 결론을 내리게 된다.",
    "question": "2-token reconstruction 이득이 같은 PPO 조건에서 closed-loop tracking 이득으로 그대로 이어지는가?",
    "toyShape": "1-token: q2+state4→force1 ; 2-token: q4+state4→force1",
    "sonicShape": "flattened multi-token vector + robot proprioception → whole-body action",
    "key": "matched-budget ablation, not a claim that one or two tokens is globally optimal."
  }
};

export function getNode(id){
  return SONIC_FLOW.find(x=>x.id===id)||SONIC_FLOW[0];
}
export function getTrainingTopic(id){
  return TRAINING_TOPICS.find(x=>x.id===id)||TRAINING_TOPICS[0];
}
export function getSystemOutline(){
  return {
    version:COURSE_VERSION,
    runtime:SONIC_FLOW.map(({id,nav,official,toy,input,output})=>({
      id,nav,official,toy,input,output,
      explanation:RUNTIME_DETAILS[id]||null,
      concepts:(SONIC_FLOW.find(x=>x.id===id)?.concepts||[]).map(c=>({
        ...c,
        explanation:c.id==="core"?null:(CONCEPT_TEXT[c.id]||null)
      }))
    })),
    training:TRAINING_TOPICS.map(({id,label,title,input,output})=>({
      id,label,title,input,output,explanation:TRAINING_DETAILS[id]||null
    }))
  };
}
