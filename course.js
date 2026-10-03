export const COURSE_VERSION = "2.0";

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
    why:"bottleneck이 모든 입력값을 그대로 복사하지 못하게 하고 motion에 중요한 정보를 compact하게 만든다.",
    misconception:"Encoder는 robot-state observer가 아니다.",
    question:"왜 16D를 2D로 줄이고, 왜 reconstruction으로 다시 확인할까?",
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
      {id:"core",label:"Core"},
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
    concepts:[]
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
    input:"G1 / SMPL / teleop representations of related motion",
    output:"aligned shared latent space",
    why:"같은 FSQ를 쓴다고 서로 다른 Encoder의 latent 의미가 자동으로 같아지지 않기 때문이다.",
    misconception:"FSQ 하나만 붙이면 universal token이 자동으로 생기는 것이 아니다.",
    question:"G1, SMPL, teleop Encoder의 latent를 같은 의미로 맞추는 training signal은 무엇인가?"
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

export const CONCEPT_TEXT = {
  ae:{
    title:"Autoencoder — bottleneck이 정보를 보존하는지 확인",
    short:"Encoder가 입력을 작은 z로 압축하고 Decoder가 입력을 복원한다.",
    why:"작은 latent가 motion 정보를 잃지 않았는지 reconstruction error로 확인한다.",
    key:"AE는 SONIC deployment 구조 그 자체가 아니라 Encoder/auxiliary Decoder를 이해하기 위한 기본 개념이다."
  },
  vae:{
    title:"VAE — optional background",
    short:"z 하나를 직접 내는 대신 μ,σ를 내고 확률분포에서 latent를 sample한다.",
    why:"continuous latent를 regularized probabilistic space로 만드는 계열이다.",
    key:"FSQ의 직접 선행 개념은 VAE보다 VQ/VQ-VAE다. VAE는 선택 배경지식이다."
  },
  vq:{
    title:"Vector Quantization — learned vector dictionary",
    short:"continuous z를 가장 가까운 learned codebook vector q로 치환한다.",
    why:"continuous representation을 discrete code로 바꾼다.",
    key:"codebook vector 자체가 학습/EMA update된다."
  },
  vqvae:{
    title:"VQ-VAE — discrete bottleneck을 실제로 학습시키는 구조",
    short:"Encoder → nearest code q → Decoder reconstruction을 함께 학습한다.",
    why:"nearest lookup은 미분 불가능하므로 STE가 필요하고, Encoder가 code에 붙도록 commitment가 필요하며, codebook 자체도 update해야 한다.",
    key:"forward에서는 q를 쓰고 backward에서는 STE로 Encoder까지 gradient를 보낸다. 이것이 '그냥 diagram'이 아니라 학습 메커니즘의 핵심이다."
  },
  fsq:{
    title:"FSQ — VQ codebook을 없애고 scalar별 finite level을 사용",
    short:"각 latent scalar를 fixed finite level로 bound→round한다.",
    why:"learned vector codebook 관리 없이 discrete bottleneck을 만든다.",
    key:"FSQ levels는 보통 학습되지 않는다. Encoder가 fixed bins를 유용하게 사용하는 법을 배운다."
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
    runtime:SONIC_FLOW.map(({id,nav,official,toy,input,output})=>({id,nav,official,toy,input,output})),
    training:TRAINING_TOPICS.map(({id,label,title})=>({id,label,title}))
  };
}
