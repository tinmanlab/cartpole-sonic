export const COURSE_VERSION = "2.7";

export const SONIC_FLOW = [
  {
    id:"task",
    nav:"Task / Interface",
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
    nav:"Motion Generator",
    official:"Kinematic Planner / VR toolkit / motion generator",
    toy:"critically-damped future planner",
    title:"의도를 실제 미래 motion으로 바꾸기",
    viz:"reference",
    input:"high-level intent",
    output:"future motion reference",
    why:"Encoder가 읽을 수 있도록 task intent를 시간축을 가진 motion reference로 바꾼다.",
    misconception:"planner reference는 measured robot trajectory가 아니다.",
    question:"지금 보이는 future curve는 robot이 실제로 움직인 궤적인가?",
    action:"change-goal",
    concepts:[]
  },
  {
    id:"reference",
    nav:"Motion Reference",
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
    nav:"Encoder(s)",
    official:"Robot / Hybrid / Human encoders",
    toy:"16D future reference → 2D latent",
    title:"긴 motion을 compact latent로 압축",
    viz:"encoder",
    input:"future reference",
    output:"continuous latent z",
    why:"bottleneck(좁은 표현 공간)은 복사할 수 있는 정보를 제한한다. 중요한 정보가 남는지는 학습 목표와 검증 결과로 확인해야 한다.",
    misconception:"Encoder는 robot-state observer가 아니다.",
    question:"이 기본 실험에서 왜 16D를 2D로 줄일까? 복원 검증은 무엇을 확인하며 제어 성능과 어떻게 다를까?",
    action:"change-goal",
    concepts:[
      {id:"core",label:"Core"},
      {id:"ae",label:"Autoencoder"},
      {id:"vae",label:"VAE · optional"}
    ]
  },
  {
    id:"quantizer",
    nav:"Quantizer · FSQ",
    official:"Finite Scalar Quantization",
    toy:"2 scalar dims × 5 fixed levels",
    title:"continuous z를 finite motion representation으로 바꾸기",
    viz:"quantizer",
    input:"continuous latent z",
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
    nav:"Universal Token",
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
      {id:"core",label:"Current token"},
      {id:"temporal",label:"1 vs 2 token slots"},
      {id:"temporal-control",label:"Closed-loop 1 vs 2"}
    ]
  },
  {
    id:"motion-decoder",
    nav:"Robot Motion Decoder",
    official:"Robot Motion Decoder",
    toy:"Kinematic Decoder",
    title:"token이 future motion 정보를 보존했는지 복원으로 검사",
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
    nav:"Robot Control Decoder",
    official:"Robot Control Decoder",
    toy:"Dynamic Decoder",
    title:"motion intent와 실제 robot state를 action으로 결합",
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
    nav:"Robot / Feedback",
    official:"whole-body robot control",
    toy:"MuJoCo CartPole",
    title:"action이 실제 physics를 바꾸고 다시 feedback이 된다",
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
    label:"Loss flow",
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
    label:"What learns?",
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
    label:"Multi-encoder alignment",
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
    label:"Optimizer sensitivity",
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
    title:"physical tracking controller는 무엇으로 학습되나?",
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
    easy:"운전자에게서 '어디로 갈지'를 받는 단계다. 아직 robot joint나 force를 결정하지 않는다.",
    mechanism:"CartPole toy에서는 하나의 scalar goal x*만 사용한다. 이 값은 Motion Generator의 목표가 되며 FSQ나 Dynamic Decoder에 직접 들어가지 않는다.",
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
    easy:"앞으로 몇 순간 동안 원하는 움직임을 적어 둔 짧은 motion script다.",
    mechanism:"toy에서는 8 frame × 2 feature = 16D vector다. 각 frame은 future x와 ẋ를 포함하고 실제 CartPole proprioception은 포함하지 않는다.",
    sonic:"실제 SONIC은 robot motion, human/SMPL motion, VR keypoint+leg target 등 modality별 motion representation을 Encoder 입력으로 사용한다.",
    ifMissing:"Encoder가 'desired future motion' 대신 현재 상태나 단일 goal만 보게 되어 motion intent와 robot state의 분리가 무너진다.",
    next:"16D 이상의 긴 reference를 compact shared representation으로 만들기 위해 Encoder가 필요하다.",
    toyShape:"16D = 8 × 2",
    sonicShape:"modality-specific future motion features × time"
  },
  encoder:{
    easy:"긴 motion script를 downstream에서 다루기 쉬운 작은 latent 좌표로 요약하는 단계다.",
    mechanism:"toy MLP Encoder는 16D reference를 2D continuous z로 압축한다. Encoder weight는 PPO, reconstruction auxiliary, 그리고 실제 SONIC에서는 cross-encoder alignment gradient를 받을 수 있다.",
    sonic:"실제 SONIC은 Robot/G1, Human/SMPL, teleop 계열 등 여러 Encoder를 사용하고 같은 semantic latent/token space로 정렬한다. toy의 2D는 시각화를 위한 의도적 축소다.",
    ifMissing:"각 modality의 큰 원본 reference를 decoder가 직접 처리해야 하고, shared compact interface를 만들기 어렵다.",
    next:"Encoder output z는 아직 continuous이므로 Quantizer가 finite/discrete representation으로 바꾼다.",
    toyShape:"16D → 2D z",
    sonicShape:"modality-specific input → 32 scalar dims per token (release configuration)"
  },
  quantizer:{
    easy:"연속적으로 움직이는 latent를 정해진 '칸'에 놓아 제한된 motion 표현으로 만드는 단계다.",
    mechanism:"VQ는 z 전체와 가장 가까운 learned vector code를 고른다. FSQ는 각 scalar를 fixed finite level에 bound→round하고 STE로 backward gradient를 통과시킨다. toy FSQ는 2 scalar × 5 levels다.",
    sonic:"release 계열 config는 token당 32 scalar dims, scalar당 32 fixed levels, 최대 2 tokens를 사용한다. FSQ의 finite levels는 trainable codebook이 아니다.",
    ifMissing:"latent가 완전히 continuous라 modality 사이 shared discrete bottleneck의 제약이 사라지고 universal token interface가 덜 규격화된다.",
    next:"Quantizer output q를 downstream decoder가 사용할 공통 Universal Token representation으로 해석한다.",
    toyShape:"z∈R² → q∈{5 levels}²",
    sonicShape:"2 tokens × 32 scalar dims, 32 fixed levels/scalar"
  },
  token:{
    easy:"'어떤 움직임을 원하는가'를 짧게 전달하는 공통 motion message다.",
    mechanism:"token은 FSQ/VQ output numeric vector(s)다. toy는 q₁,q₂ 두 scalar를 사용한다. token에는 actual robot state가 포함되지 않으며 proprioception은 별도로 Control Decoder에 들어간다.",
    sonic:"release 구조는 2 temporal tokens × 32 scalar dims = 64 flattened numeric values를 control decoder interface로 사용한다. universal 의미는 FSQ만이 아니라 multi-encoder alignment 학습까지 필요하다.",
    ifMissing:"decoder가 다시 원래 modality별 reference를 직접 이해해야 해서 하나의 공통 control interface를 만들기 어렵다.",
    next:"같은 token은 두 용도로 해석된다: motion information을 복원하는 Motion Decoder와 실제 action을 만드는 Control Decoder.",
    toyShape:"1 token × 2 scalar values",
    sonicShape:"2 temporal tokens × 32 scalar dims = 64 flattened values"
  },
  "motion-decoder":{
    easy:"token을 다시 future motion으로 풀어 보고, token이 원하는 움직임 정보를 잃지 않았는지 검사하는 decoder다.",
    mechanism:"toy Kinematic Decoder는 q에서 16D future [x,ẋ] reconstruction을 출력하고 reconstruction MSE auxiliary signal을 제공한다.",
    sonic:"공식 SONIC의 Robot Motion Decoder 역할과 대응시키되, 이 CartPole 구현에서는 Kinematic Decoder를 training auxiliary path로 사용한다. physical motor action은 별도 Control Decoder가 낸다.",
    ifMissing:"이 실험에서는 복원 보조 손실이 사라져 입력 정보의 보존을 직접 확인하기 어렵다. 모든 제어기에 복원이 필수인 것은 아니며 제어 성능은 따로 평가해야 한다.",
    next:"future motion을 복원하는 것만으로 robot을 안정적으로 움직일 수 없으므로 actual proprioception을 포함한 Robot Control Decoder가 필요하다.",
    toyShape:"q(2D) → future reconstruction(16D)",
    sonicShape:"universal token → robot-motion representation"
  },
  "control-decoder":{
    easy:"'원하는 움직임(token)'과 '지금 몸 상태(proprioception)'를 합쳐 바로 실행할 action을 만드는 controller다.",
    mechanism:"toy Dynamic Decoder는 q₁,q₂ + normalized [x,ẋ,θ,θ̇] 4D를 받아 scalar force를 출력한다. Push는 q/reference를 유지하고 state만 바꿔 decoder가 다른 action을 내는지 검증한다.",
    sonic:"공식 G1 dynamic decoder는 token_flattened + proprioception → action 구조다. 이 경로가 physical tracking PPO의 핵심 policy path다.",
    ifMissing:"token만 보는 open-loop decoder는 같은 motion intent라도 실제 robot이 넘어지거나 밀렸을 때 correction을 만들 수 없다.",
    next:"action은 실제 physics에 적용되고 measured state가 다시 feedback되어 closed loop가 완성된다.",
    toyShape:"q(2D)+state(4D) → force(1D)",
    sonicShape:"flattened token + robot proprioception → whole-body action"
  },
  robot:{
    easy:"controller가 낸 action을 실제 dynamics에 적용하고 결과를 다시 센서 state로 돌려주는 단계다.",
    mechanism:"toy는 브라우저 MuJoCo WASM 10 ms physics를 사용하고 50 Hz policy action을 두 physics step 동안 hold한다. x,ẋ,θ,θ̇가 다음 Control Decoder proprioception이 된다.",
    sonic:"실제 deployment에서는 humanoid body, actuator, contact, sensors가 이 closed-loop plant를 구성한다. 이 toy는 multi-contact/sim2real을 재현하지 않는다.",
    ifMissing:"physics feedback이 없으면 reference tracking이 실제로 되는지 검증할 수 없고 disturbance correction도 불가능하다.",
    next:"feedback state가 다시 Robot Control Decoder 입력으로 돌아가며 loop가 반복된다.",
    toyShape:"force → MuJoCo → 4D measured state",
    sonicShape:"whole-body action → robot/contact dynamics → measured proprioception"
  }
};

export const TRAINING_DETAILS = {
  "loss-flow":{
    easy:"SONIC에는 두 종류의 선생이 있다: 실제로 잘 움직이게 만드는 PPO와, representation을 잘 정리하는 auxiliary loss.",
    mechanism:"브라우저 제어기는 PPO(정책 개선)와 reconstruction(입력 복원) 보조 손실로 갱신한다. alignment(표현 정렬)는 별도 실험이다. 공식 구조의 보조 손실과 이 실험의 갱신 경로를 구분한다.",
    sonic:"release training은 PPO와 token reconstruction / cross-encoder latent-alignment auxiliary losses를 결합한다.",
    ifMissing:"PPO만 있으면 representation semantic이 약해질 수 있고, aux만 있으면 실제 physics tracking 능력을 학습할 수 없다.",
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
    easy:"robot이 실제로 넘어지지 않고 reference를 따라가게 만드는 시행착오 학습이다.",
    mechanism:"MuJoCo rollout에서 tracking/balance/action cost reward를 모아 GAE advantage와 PPO ratio/clipping으로 Encoder+Dynamic Decoder policy를 update한다. Critic은 value loss로 학습된다.",
    sonic:"실제 SONIC은 대규모 병렬 humanoid physics tracking PPO를 사용한다. toy의 작은 PPO는 역할만 보존한다.",
    ifMissing:"representation을 잘 복원해도 실제 dynamics, disturbance, balance에 맞는 action을 내는 법을 배우지 못한다.",
    toyShape:"8 env × 96 control steps / iteration",
    sonicShape:"large-scale parallel humanoid tracking rollouts",
    highlights:["encoder","control-decoder","robot"]
  }
};

export const CONCEPT_TEXT = {
  ae:{
    title:"Autoencoder — bottleneck이 정보를 보존하는지 확인",
    input:"future motion reference",
    output:"latent z + reconstructed reference",
    short:"Encoder가 입력을 작은 z로 압축하고 Decoder가 입력을 복원한다.",
    why:"작은 latent가 motion 정보를 잃지 않았는지 reconstruction error로 확인한다.",
    mechanism:"x → Encoder → z → Decoder → x̂, 그리고 reconstruction loss ||x-x̂||²로 Encoder/Decoder를 같이 학습한다.",
    sonic:"SONIC에서는 이 생각이 motion Encoder + Robot Motion/Kinematic Decoder auxiliary path로 이어진다. physical action은 별도 Control Decoder가 낸다.",
    ifMissing:"latent가 motion semantic을 보존하는지 직접 검사할 보조 신호가 약해진다.",
    question:"왜 작은 latent를 다시 복원해 보는 것이 bottleneck의 품질 검사가 되는가?",
    toyShape:"16D reference → 2D z → 16D reconstruction",
    sonicShape:"motion Encoder → shared latent/token → Robot Motion Decoder auxiliary reconstruction",
    key:"AE는 SONIC deployment 구조 그 자체가 아니라 Encoder/auxiliary Decoder를 이해하기 위한 기본 개념이다."
  },
  vae:{
    title:"VAE — optional background",
    input:"input/reference features",
    output:"μ, σ → sampled continuous latent z",
    short:"z 하나를 직접 내는 대신 μ,σ를 내고 확률분포에서 latent를 sample한다.",
    why:"continuous latent를 regularized probabilistic space로 만드는 계열이다.",
    mechanism:"Encoder가 μ,σ를 출력하고 z=μ+σ·ε로 sample한다. reconstruction loss와 KL divergence를 함께 최적화한다.",
    sonic:"SONIC runtime은 VAE를 핵심 block으로 사용하지 않는다. VAE는 VQ-VAE라는 이름의 계보를 이해하기 위한 선택 배경지식이다.",
    ifMissing:"SONIC 이해에는 치명적이지 않다. 이 탭을 건너뛰어도 VQ→VQ-VAE→FSQ 본선을 이해할 수 있다.",
    question:"VAE의 μ/σ/KL은 왜 FSQ 이해의 필수 선행조건이 아닌가?",
    toyShape:"concept-only: μ,σ → sampled z (not implemented in toy runtime)",
    sonicShape:"optional background; not a SONIC runtime block",
    key:"FSQ의 직접 선행 개념은 VAE보다 VQ/VQ-VAE다. VAE는 선택 배경지식이다."
  },
  vq:{
    title:"Vector Quantization — learned vector dictionary",
    input:"continuous latent z",
    output:"nearest learned code vector q",
    short:"continuous z를 가장 가까운 learned codebook vector q로 치환한다.",
    why:"continuous representation을 discrete code로 바꾼다.",
    mechanism:"각 codebook vector e_k와 z의 거리를 계산해 argmin_k ||z-e_k||²를 선택한다. 선택 연산은 불연속이므로 학습 때 별도 gradient 처리와 codebook update가 필요하다.",
    sonic:"SONIC은 learned VQ codebook 대신 FSQ를 사용한다. VQ는 FSQ가 무엇을 단순화했는지 이해하기 위한 직접 비교 기준이다.",
    ifMissing:"FSQ의 장점인 'learned codebook 제거'가 왜 중요한지 이해하기 어렵다.",
    question:"VQ에서 z가 codebook의 어느 vector로 가는지는 어떻게 결정되고, 그 vector 자체는 학습되는가?",
    toyShape:"z∈R² → nearest of 8 learned 2D code vectors",
    sonicShape:"comparison baseline; SONIC runtime uses FSQ instead of a learned VQ codebook",
    key:"codebook vector 자체가 학습/EMA update된다."
  },
  vqvae:{
    title:"VQ-VAE — discrete bottleneck을 실제로 학습시키는 구조",
    input:"reference → continuous z",
    output:"discrete q + reconstruction",
    short:"Encoder → nearest code q → Decoder reconstruction을 함께 학습한다.",
    why:"nearest lookup은 미분 불가능하고 z와 learned codebook을 함께 맞춰야 하므로 STE·commitment·codebook update가 각각 필요하다.",
    mechanism:"forward: z→nearest q→Decoder. backward: STE가 Encoder gradient를 통과시키고, commitment는 z를 q 근처에 붙이며, codebook update는 q 자체를 움직인다. 선택되지 않는 code는 dead code가 된다.",
    sonic:"SONIC FSQ는 이 구조에서 learned vector codebook/commitment/dead-code 관리 부담을 줄인 대안으로 이해하면 된다.",
    ifMissing:"FSQ가 단순히 'VQ보다 다른 rounding' 정도로만 보여 왜 연구적으로 의미가 있는지 놓치게 된다.",
    question:"VQ-VAE에서 STE, commitment, codebook update는 각각 어떤 학습 문제를 해결하는가?",
    toyShape:"16D → z2 → nearest learned q2 → recon16",
    sonicShape:"VQ-VAE predecessor concept → SONIC FSQ removes learned codebook machinery",
    key:"forward에서는 q를 쓰고 backward에서는 STE로 Encoder까지 gradient를 보낸다. 이것이 '그냥 diagram'이 아니라 학습 메커니즘의 핵심이다."
  },
  fsq:{
    title:"FSQ — VQ codebook을 없애고 scalar별 finite level을 사용",
    input:"continuous latent scalars z",
    output:"fixed-level quantized scalars q",
    short:"각 latent scalar를 fixed finite level로 bound→round한다.",
    why:"learned vector codebook 관리 없이 discrete bottleneck을 만든다.",
    mechanism:"toy는 q_i = round(1.998·tanh(z_i))/2를 사용한다. round는 forward에서 discrete value를 만들고 backward에서는 STE로 gradient를 Encoder에 전달한다.",
    sonic:"release 계열 config는 2 temporal tokens × 32 scalar dims, scalar당 32 fixed levels를 사용해 flattened 64-D token interface를 만든다.",
    ifMissing:"continuous latent를 그대로 쓰게 되어 finite shared bottleneck의 규격성과 discrete token interface를 잃는다.",
    question:"FSQ level은 고정인데 Encoder representation은 어떻게 계속 좋아질 수 있는가?",
    toyShape:"z∈R² → q on 5×5 implicit fixed grid",
    sonicShape:"2 tokens × 32 scalar dims, 32 fixed levels/scalar → 64 flattened values",
    key:"FSQ levels는 보통 학습되지 않는다. Encoder가 fixed bins를 유용하게 사용하는 법을 배운다."
  }
,
  temporal:{
    title:"1 token vs 2 token slots — 왜 future window를 여러 token으로 표현할까?",
    input:"same whole future window",
    output:"1-slot vs 2-slot FSQ representations + reconstructions",
    short:"전체 future window를 한꺼번에 읽되, 출력 representation을 token slot 하나가 아니라 여러 slot으로 나누어 더 많은 discrete capacity를 쓴다.",
    why:"하나의 작은 token이 표현할 수 있는 finite combinations는 제한적이다. 여러 token slot을 쓰면 whole-window 정보를 더 풍부하게 보존할 수 있다.",
    mechanism:"toy는 같은 16D future window를 두 모델에 넣는다. 1-token model은 2 scalar FSQ, 2-token model은 2×2 scalar FSQ를 출력한다. 둘 다 whole window를 공동으로 읽으며 reconstruction loss로 학습된다. token 1=근미래, token 2=원미래 같은 역할은 미리 지정하지 않는다.",
    sonic:"공식 SONIC MLP Encoder는 temporal input을 flatten해 읽고, 출력은 max_num_tokens=2 temporal slots로 reshape한다. release config는 token_dim=32이므로 2×32=64 flattened values다. slot 의미는 architecture/config이 정하는 shape이지 near/far semantic을 하드코딩한 것이 아니다.",
    ifMissing:"token slot 수와 token dimension을 같은 것으로 오해하거나, token 1/2에 임의의 시간 의미를 붙이게 된다.",
    question:"왜 token을 두 개 쓰는가? 그리고 token 1과 token 2가 각각 near/far를 담당한다고 말해도 되는가?",
    toyShape:"same 16D window → 1×2 FSQ vs 2×2 FSQ",
    sonicShape:"whole future window → 2 token slots × 32 scalar dims → 64 flattened values",
    key:"multiple token slots increase representational capacity, but slot semantics are learned/not pre-assigned."
  },
  "temporal-control":{
    title:"1 token vs 2 token control — 표현력 증가가 실제 제어도 좋아지게 할까?",
    input:"same future reference + same actual proprioception",
    output:"1-token vs 2-token Dynamic Decoder actions → 브라우저 MuJoCo WASM",
    short:"두 controller를 같은 teacher/bootstrap과 같은 PPO budget으로 학습해 실제 tracking과 disturbance recovery를 비교한다.",
    why:"reconstruction이 좋아졌다는 사실만으로 physical control이 좋아졌다고 결론낼 수 없기 때문이다.",
    mechanism:"1-token controller는 16D reference→2D→FSQ q2, 2-token controller는 16D→4D→reshape 2×2→FSQ q4를 사용한다. 각 q는 같은 4D proprioception과 결합되어 별도 Dynamic Decoder가 force를 낸다. 두 controller는 동일 300-step teacher imitation과 동일 PPO hyperparameter/budget을 받는다.",
    sonic:"실제 SONIC Robot Control Decoder는 flattened multi-token representation과 proprioception을 받아 action을 낸다. 이 toy는 token-slot 수가 representation capacity뿐 아니라 policy optimization 난이도에도 영향을 줄 수 있음을 분리해 보여준다.",
    ifMissing:"'2-token reconstruction이 더 좋다 → 2-token controller도 반드시 더 좋다'는 잘못된 결론을 내리게 된다.",
    question:"2-token reconstruction 이득이 같은 PPO 조건에서 closed-loop tracking 이득으로 그대로 이어지는가?",
    toyShape:"1-token: q2+state4→force1 ; 2-token: q4+state4→force1",
    sonicShape:"flattened multi-token vector + robot proprioception → whole-body action",
    key:"matched-budget ablation, not a claim that one or two tokens is globally optimal."
  }};

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
