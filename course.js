export const COURSE_VERSION = "1.0";

export const PRIMARY_PATH = [
  "ae",
  "vq",
  "vqvae",
  "fsq",
  "motion-token",
  "dynamic-decoder",
  "ppo",
  "sonic",
];

export const OPTIONAL_BRANCHES = {
  ae: ["vae"],
};

export const LESSONS = {
  ae: {
    id: "ae",
    step: 1,
    world: "reference",
    nav: "Encoder / AE",
    title: "왜 차원을 줄였다가 다시 복원할까?",
    mode: "ae",
    viz: "reconstruction",
    prerequisites: [],
    next: "vq",
    optionalNext: ["vae"],
    question: "16개의 미래 reference를 왜 2개의 latent로 줄이고, Decoder로 다시 16개를 복원할까?",
    answer: "좁은 bottleneck이 motion의 핵심만 남기도록 강제하고, 복원은 그 작은 latent에 원래 motion 정보가 실제로 남았는지 확인하는 학습 시험이다.",
    why: "입력을 그대로 action network에 넣으면 내부 표현이 무엇을 담았는지 보기 어렵다. 16→2 bottleneck은 중요한 정보를 고르게 만들고, 2→16 reconstruction error는 무엇을 잃었는지 눈으로 확인하게 한다.",
    watch: [
      "파란 planner reference와 보라 reconstruction이 얼마나 겹치는지 본다.",
      "goal을 바꾸면 16D reference가 바뀌고 2D latent도 함께 움직이는지 본다.",
    ],
    try: "goal slider를 -0.8, 0, +0.8로 움직여 reconstruction과 latent가 같이 바뀌는지 본다.",
    takeaway: "Decoder는 deployment에서 원본을 다시 쓰기 위한 장치가 아니다. latent가 motion 정보를 보존하게 만드는 auxiliary learning signal이다.",
    sonic: "SONIC의 motion Encoder와 Kinematic Decoder auxiliary path에 대응한다.",
    highlights: ["reference", "encoder", "latent", "kinematic"],
  },

  vae: {
    id: "vae",
    step: "2A",
    world: "concept",
    nav: "VAE (optional)",
    title: "VAE는 어디에 있는가? — 선택 분기",
    mode: "ae",
    viz: "vae-branch",
    optional: true,
    prerequisites: ["ae"],
    returnsTo: "vq",
    question: "VAE를 꼭 알아야 FSQ를 이해할 수 있을까?",
    answer: "아니다. VAE는 Autoencoder에서 갈라지는 probabilistic continuous-latent 분기다. SONIC의 FSQ 계보에는 VQ가 더 직접적인 선행 개념이다.",
    why: "VAE의 μ, σ, sampling, KL은 'latent를 확률분포로 만드는 법'을 설명한다. 반면 VQ/FSQ는 'latent를 discrete symbol로 만드는 법'이 핵심이다.",
    watch: [
      "AE에서 VAE와 VQ가 서로 다른 방향으로 갈라지는 계보만 확인한다.",
      "VAE는 본선이 아니라 선택 분기라는 점을 확인한다.",
    ],
    try: "이 장은 개념 분기만 보고 바로 VQ로 돌아가도 된다.",
    takeaway: "FSQ 이해에 VAE의 reparameterization/KL을 깊게 공부할 필요는 없다.",
    sonic: "직접 대응되는 SONIC block은 없다. 배경 개념이다.",
    highlights: ["latent"],
  },

  vq: {
    id: "vq",
    step: 2,
    world: "reference-token",
    nav: "VQ",
    title: "VQ는 왜 codebook을 만들까?",
    mode: "vq",
    viz: "latent",
    prerequisites: ["ae"],
    next: "vqvae",
    question: "continuous latent z를 왜 가장 가까운 대표 vector 하나로 바꿀까?",
    answer: "무한히 많은 실수 벡터 대신 제한된 대표 motion symbol을 사용하기 위해서다.",
    why: "VQ는 k-means처럼 learned codebook에서 z와 가장 가까운 vector를 선택한다. 그래서 continuous latent를 discrete code로 바꿀 수 있다.",
    watch: [
      "회색 reference sweep이 latent 공간에서 어디에 놓이는지 본다.",
      "파란 z가 어느 learned code로 이동해 빨간 q가 되는지 본다.",
    ],
    try: "goal을 천천히 움직여 z는 연속적으로 움직이지만 q는 특정 code 사이에서 점프하는지 본다.",
    takeaway: "VQ = learned vector dictionary + nearest-neighbor assignment.",
    sonic: "FSQ를 이해하기 위한 직접적인 비교 기준이다.",
    highlights: ["reference", "encoder", "latent", "quantizer"],
  },

  vqvae: {
    id: "vqvae",
    step: 3,
    world: "reference-token",
    nav: "VQ-VAE",
    title: "VQ-VAE는 discrete code를 어떻게 학습할까?",
    mode: "vq",
    viz: "vqvae",
    prerequisites: ["vq"],
    next: "fsq",
    question: "nearest code 선택은 미분할 수 없는데 Encoder는 어떻게 학습될까?",
    answer: "forward에서는 discrete q를 사용하고 backward에서는 STE로 gradient를 통과시킨다. VQ에는 commitment와 codebook update도 필요하다.",
    why: "단순 nearest lookup만 넣으면 Encoder와 codebook이 함께 안정적으로 학습되지 않는다. VQ-VAE는 reconstruction, STE, commitment/codebook learning을 하나의 학습 구조로 묶는다.",
    watch: [
      "reference → Encoder z → VQ q → Kinematic Decoder reconstruction 경로를 본다.",
      "실선 forward와 점선 backward/auxiliary 역할을 구분한다.",
    ],
    try: "VQ와 reconstruction 그래프를 함께 보고 code가 달라도 future reference의 핵심 형태가 복원되는지 확인한다.",
    takeaway: "VQ-VAE의 핵심 부담은 learned codebook을 잘 유지하면서 discrete bottleneck을 학습하는 것이다.",
    sonic: "SONIC이 VQ 대신 FSQ를 사용하는 이유를 이해하기 위한 직전 단계다.",
    highlights: ["reference", "encoder", "quantizer", "kinematic"],
  },

  fsq: {
    id: "fsq",
    step: 4,
    world: "reference-token",
    nav: "FSQ",
    title: "FSQ는 VQ에서 무엇을 없앴을까?",
    mode: "fsq",
    viz: "latent",
    prerequisites: ["vqvae"],
    next: "motion-token",
    question: "learned vector codebook 없이 어떻게 discrete representation을 만들까?",
    answer: "저차원 latent의 각 scalar를 finite level로 bound하고 round한다. scalar 조합이 implicit codebook을 만든다.",
    why: "VQ의 learned codebook lookup, dead-code 관리, reseeding 같은 부담을 줄이면서 discrete bottleneck을 유지한다.",
    watch: [
      "파란 z가 고정된 FSQ grid의 빨간 q로 이동하는지 본다.",
      "VQ와 달리 회색 code point가 학습되어 이동하지 않는다는 점을 본다.",
    ],
    try: "goal을 움직이며 z는 연속적으로 움직이고 q는 5×5 finite grid를 따라 바뀌는지 본다.",
    takeaway: "FSQ = low-dimensional scalar quantization + implicit codebook + STE.",
    sonic: "GEAR-SONIC universal motion token의 핵심 bottleneck에 대응한다.",
    highlights: ["reference", "encoder", "latent", "quantizer", "token"],
  },

  "motion-token": {
    id: "motion-token",
    step: 5,
    world: "reference-token",
    nav: "Motion token",
    title: "무엇을 token으로 만드는가?",
    mode: "fsq",
    viz: "reconstruction",
    prerequisites: ["fsq"],
    next: "dynamic-decoder",
    question: "FSQ가 압축하는 것은 현재 robot state일까, 미래 motion일까?",
    answer: "미래 motion reference다. 현재 robot state는 token과 별도로 Dynamic Decoder에 들어간다.",
    why: "원하는 움직임과 현재 몸 상태를 분리해야 같은 motion intent를 여러 실제 상태에서 재사용할 수 있다.",
    watch: [
      "planner reference가 Encoder 입력이고 actual CartPole state는 입력이 아님을 확인한다.",
      "같은 goal/reference라도 push로 실제 state만 바꾸면 token보다 action이 더 크게 달라지는지 본다.",
    ],
    try: "goal을 고정한 채 Push를 눌러 actual state만 바꿔본다.",
    takeaway: "motion token = '무엇을 하고 싶은가'의 compact discrete representation.",
    sonic: "G1 / SMPL / teleop reference가 공유 motion-token space로 들어가는 개념의 최소 예다.",
    highlights: ["reference", "encoder", "quantizer", "token"],
  },

  "dynamic-decoder": {
    id: "dynamic-decoder",
    step: 6,
    world: "bridge",
    nav: "Dynamic decoder",
    title: "token만으로 왜 action을 만들 수 없을까?",
    mode: "fsq",
    viz: "decoder",
    prerequisites: ["motion-token"],
    next: "ppo",
    question: "같은 motion token인데 왜 현재 proprioception이 또 필요할까?",
    answer: "같은 의도라도 현재 위치·속도·기울기가 다르면 지금 줘야 할 action이 달라지기 때문이다.",
    why: "token은 desired motion, proprioception은 actual state, Dynamic Decoder는 둘을 합쳐 immediate action을 만든다.",
    watch: [
      "Push 전후 token과 force를 비교한다.",
      "token이 거의 같아도 state가 변하면 action이 달라지는지 본다.",
    ],
    try: "goal을 고정하고 Push → 1 Step을 반복해 token/state/action 관계를 본다.",
    takeaway: "Dynamic Decoder = motion intent + actual body state → next action.",
    sonic: "공식 SONIC의 g1_dyn: token_flattened + proprioception → action에 대응한다.",
    highlights: ["token", "proprioception", "dynamic", "action", "physics"],
  },

  ppo: {
    id: "ppo",
    step: 7,
    world: "training",
    nav: "PPO training",
    title: "무엇이 실제 physical controller를 학습할까?",
    mode: "fsq",
    viz: "training",
    prerequisites: ["dynamic-decoder"],
    next: "sonic",
    question: "reconstruction loss가 controller를 학습하는 본체일까?",
    answer: "아니다. 실제 physics rollout의 tracking reward → GAE → PPO가 Encoder와 Dynamic Decoder를 physical control에 맞게 학습한다.",
    why: "Kinematic reconstruction은 representation 보조 신호다. 실제로 넘어지지 않고 reference를 추적하는 법은 physical rollout reward가 가르친다.",
    watch: [
      "PPO iteration 전후 held-out tracking MAE 변화를 본다.",
      "Critic과 Kinematic Decoder가 training-only라는 점을 확인한다.",
    ],
    try: "+10 PPO iter를 누르고 held-out MAE가 어떻게 바뀌는지 본다.",
    takeaway: "tracking PPO가 본체, Kinematic reconstruction과 Critic은 training support.",
    sonic: "SONIC의 large-scale physical tracking optimization에 대응하는 최소 CartPole 버전이다.",
    highlights: ["physics", "reward", "ppo", "encoder", "dynamic"],
  },

  sonic: {
    id: "sonic",
    step: 8,
    world: "mapping",
    nav: "GEAR-SONIC",
    title: "CartPole에서 이해한 구조를 GEAR-SONIC으로 되돌리기",
    mode: "fsq",
    viz: "sonic-map",
    prerequisites: ["ppo"],
    question: "이 toy에서 배운 각 block은 실제 SONIC에서 무엇이 될까?",
    answer: "CartPole의 future reference / Encoder / FSQ token / proprioception / Dynamic Decoder / PPO가 whole-body motion / universal token / G1 action control로 확장된다.",
    why: "이 단계에서는 새 개념을 추가하지 않고 이전 장의 역할을 실제 SONIC 용어로 치환한다.",
    watch: [
      "toy block과 SONIC block을 1:1 역할 기준으로 대응시킨다.",
      "toy가 재현하지 않는 multi-encoder alignment, multi-contact, scale, sim2real 범위를 확인한다.",
    ],
    try: "각 block을 가리키며 '무엇을 입력받고 무엇을 출력하는가?'를 설명할 수 있으면 완료다.",
    takeaway: "FSQ는 핵심 bottleneck이지만 SONIC은 reference→token→state-conditioned action→physics 전체 tracking architecture다.",
    sonic: "최종 mapping lesson.",
    highlights: ["reference", "encoder", "quantizer", "token", "proprioception", "dynamic", "action", "physics"],
  },
};

export function getLesson(id) {
  return LESSONS[id] || LESSONS.ae;
}

export function getCourseOutline() {
  return {
    version: COURSE_VERSION,
    primary: PRIMARY_PATH.map(id => ({id, title: LESSONS[id].title, nav: LESSONS[id].nav})),
    branches: Object.fromEntries(
      Object.entries(OPTIONAL_BRANCHES).map(([from, ids]) => [
        from,
        ids.map(id => ({id, title: LESSONS[id].title, nav: LESSONS[id].nav, returnsTo: LESSONS[id].returnsTo})),
      ])
    ),
  };
}
