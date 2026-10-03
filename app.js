import { MuJoCoCartPole } from "./mujoco_sim.js";
import { SonicToyTrainer, planReference, advancePlannerContext, referenceFramesPhysical, SONIC_TOY_CONSTANTS } from "./sonic_toy.js";
import { loadTeacherPolicy } from "./teacher_policy.js";
import { probeWebGPUFSQ } from "./webgpu_fsq.js";

const $ = (id) => document.getElementById(id);
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const fmt = (v,n=3) => Number.isFinite(v) ? Number(v).toFixed(n) : "—";

const query = new URLSearchParams(location.search);
let lesson = query.has("lesson") ? clamp(Math.floor(Number(query.get("lesson"))||0),0,8) : 0;
const requestedMode=query.get("mode");
const lessonDefaultMode=lesson===1?"ae":(lesson===2||lesson===3)?"vq":"fsq";
let mode = ["ae","vq","fsq"].includes(requestedMode) ? requestedMode : lessonDefaultMode;
let preset = ["playground","long","heavy"].includes(query.get("preset")) ? query.get("preset") : "playground";
let goal = query.has("goal") && Number.isFinite(Number(query.get("goal"))) ? clamp(Number(query.get("goal")),-1.2,1.2) : 0.8;
const startupTrain = query.has("train") ? clamp(Math.floor(Number(query.get("train"))||0),0,20) : 0;

const sim = new MuJoCoCartPole();
let trainer = null;
let teacher = null;
let bootstrapEval = null;
let heldoutEval = null;
let physicsReady = false;
let live = false;
let busy = false;
let lastForce = 0;
let plannerContext = [0,0];
let webmcpMode = "unavailable";
let webmcpTools = [];
let webgpuStatus = {available:false,ok:false,reason:"not probed"};
let lastFrame = performance.now();
let accumulator = 0;

const methodText = {
  ae: "<b>Continuous latent.</b> FSQ/VQ 없이 planner reference의 latent를 그대로 Dynamic Decoder에 보낸다. quantization 효과를 비교하는 기준선.",
  vq: "<b>VQ.</b> Encoder의 2D z 전체와 가장 가까운 <b>학습되는 vector code</b> 하나를 고른다. STE + commitment + codebook update가 필요하다.",
  fsq: "<b>FSQ.</b> learned vector codebook 없이 각 scalar를 finite level로 만든다. 이 toy는 L=[5,5]의 paper-style tanh bound → round를 사용한다."
};


const LESSONS = [
  {
    title:"0. 왜 굳이 CartPole로 축소하는가?",
    intro:"가능하다. 단, GEAR-SONIC의 성능을 재현하는 것이 아니라 구조와 정보 흐름을 재현한다. G1의 29개 관절·접촉·retargeting·대규모 데이터까지 한꺼번에 보면 FSQ가 무슨 역할을 하는지 분리해서 볼 수 없다.",
    why:"축소의 목적은 변수를 없애는 것이 아니라 인과관계를 보이게 하는 것이다. '원하는 미래 움직임'과 '현재 로봇 상태'가 어디서 만나 action이 되는지 한 화살표씩 확인한다.",
    keep:"절대 보존: reference는 robot state와 독립 · Encoder는 reference를 읽음 · quantizer는 latent를 token으로 바꿈 · Dynamic Decoder는 token+proprioception을 받음 · physics feedback · Kinematic Decoder/Critic은 training-only.",
    cut:"줄여도 됨: 29 joints→1 actuator, 수백 차원 reference→8×[x,ẋ], 큰/multi-token latent→2 scalars, 수천 env→8 env. 따라서 capability·sim2real·humanoid contact 성능은 이 toy에서 주장하지 않는다.",
    chain:"Goal\n  ↓\nPlanner / reference\n  ↓\nEncoder → FSQ token\n  ↓       + actual state\nDynamic Decoder\n  ↓\nMuJoCo",
    panels:["goal","runtime","reference"]
  },
  {
    title:"1. Encoder / Decoder / latent / Autoencoder",
    intro:"Encoder는 '현재 CartPole 관측기'가 아니다. 이 학습장에서는 앞으로 원하는 reference 8 frame(16개 숫자)을 읽어 2개의 latent 숫자로 압축한다.",
    why:"latent는 사람이 z₁='속도', z₂='방향'처럼 이름 붙인 값이 아니다. 학습이 future motion을 실행·복원하는 데 유용한 내부 좌표를 찾는다. bottleneck은 모든 세부값을 그대로 복사하지 못하게 만든다.",
    keep:"SONIC 대응: modality-specific motion encoder → latent. Kinematic Decoder가 token에서 future motion을 복원하는 경로는 representation을 학습시키는 보조 역할이다.",
    cut:"Autoencoder는 개념의 출발점일 뿐 SONIC deployment의 본체가 아니다. 실제 deployed Dynamic Decoder는 원본 motion을 복원하지 않고 action을 출력한다.",
    chain:"future reference 16D\n   ↓ Encoder\nlatent z 2D\n   ↓ Kinematic Decoder\nfuture reference reconstruction\n(auxiliary only)",
    panels:["reference","latent"],
    hint:"권장: Live tracker에서 continuous를 선택하고 reference→z→Kinematic reconstruction을 먼저 본다."
  },
  {
    title:"2. VAE는 어디에 있고, VQ는 왜 필요한가?",
    intro:"VAE는 continuous probabilistic latent(μ, σ, sampling, KL)를 배우는 옆가지다. SONIC의 FSQ를 이해하려면 VAE 수식보다 k-means/VQ 직관이 더 중요하다.",
    why:"VQ는 continuous z를 learned codebook의 대표 vector 하나로 바꾼다. k-means의 '가장 가까운 대표점'을 neural latent에 붙였다고 생각하면 된다.",
    keep:"배워야 할 것: codebook, nearest-vector assignment, discrete code. 이 세 가지를 이해하면 VQ-VAE와 FSQ의 차이가 바로 보인다.",
    cut:"VAE의 reparameterization trick·KL divergence는 배경지식으로 충분하다. SONIC FSQ 경로를 이해하는 데 필수 단계가 아니다.",
    chain:"Autoencoder latent\n├─ VAE: μ,σ → sample z + KL\n└─ VQ: z → nearest learned code\n             ↓\n          discrete q",
    panels:["latent","reference"],
    hint:"권장: VQ를 선택하고 회색 goal sweep, learned code 점, 현재 z→q 이동을 본다."
  },
  {
    title:"3. VQ-VAE: discrete latent를 어떻게 학습하나?",
    intro:"VQ-VAE는 Encoder → VQ → Decoder를 한 시스템으로 학습한다. 문제는 nearest/round가 미분 불가능하다는 점이고, 여기서 STE가 등장한다.",
    why:"forward에서는 진짜 discrete code를 고르고, backward에서는 그 선택이 identity였던 것처럼 gradient를 흘린다. VQ에는 encoder가 code에 붙도록 commitment term과 codebook update가 필요하고 dead code 문제도 생길 수 있다.",
    keep:"SONIC 이해에 필수: STE의 역할, learned codebook, commitment, dead code/codebook collapse. 이걸 알아야 FSQ가 무엇을 단순화했는지 이해된다.",
    cut:"이 toy의 VQ path는 VQ 메커니즘을 보여주지만 대규모 VQ-VAE benchmark 재현이 아니다. reconstruction은 Kinematic Decoder의 auxiliary 경로로 연결한다.",
    chain:"reference → Encoder z\n              ↓\n          VQ code q\n        ↙          ↘\nDyn Decoder     Kin Decoder\n action          recon loss",
    panels:["latent","reference"],
    hint:"권장: VQ 모드에서 learned codebook, z→q 이동, 그리고 planner reference↔Kinematic Decoder reconstruction을 같이 본다."
  },
  {
    title:"4. FSQ: GEAR-SONIC을 보기 위한 핵심 bottleneck",
    intro:"FSQ는 learned vector codebook을 없애고 저차원 latent의 각 scalar를 finite level로 quantize한다. 이 lab은 L=[5,5]라 25개의 implicit 조합이 생긴다.",
    why:"VQ의 codebook lookup·commitment·dead-code 관리 없이 discrete representation을 만든다. round에는 STE를 쓰지만 codebook을 학습하거나 reseed할 필요가 없다.",
    keep:"원문식 핵심: bound(z) → round → normalized code. 현재 qᵢ = round(1.998·tanh(zᵢ))/2. '그냥 가까운 숫자 선택'보다 이 표현이 정확하다.",
    cut:"FSQ가 SONIC 성공의 유일한 이유는 아니다. 대규모 tracking data, PPO, future reference, proprioception, multi-encoder alignment가 함께 필요하다.",
    chain:"z₁,z₂\n ↓ tanh bound\nfinite scalar levels\n ↓ round + STE\nq₁,q₂\n= motion token",
    panels:["live","latent","runtime"],
    hint:"권장: FSQ를 선택하고 z→bound→round→q를 직접 읽는다."
  },
  {
    title:"5. Motion token: 무엇을 token으로 만드는가?",
    intro:"가장 중요한 교정점이다. token은 현재 robot state의 압축본이 아니다. Planner/motion library가 제공한 '앞으로 원하는 움직임'의 압축 표현이다.",
    why:"원하는 움직임과 실제 몸 상태를 분리해야 같은 motion intent를 서로 다른 순간의 상태에서도 실행할 수 있다. goal 자체도 token이 아니라 planner가 만든 future trajectory가 Encoder 입력이다.",
    keep:"SONIC: G1/SMPL/teleop 같은 서로 다른 reference source → encoder → shared motion-token space. 이 toy는 그중 reference source 하나만 사용한다.",
    cut:"이 toy의 planner는 단순한 critically-damped reference generator다. SONIC의 learned kinematic planner나 BONES-SEED motion library를 재현하는 것은 아니다.",
    chain:"high-level goal\n ↓\nfuture reference frames\n ↓ Encoder\nz\n ↓ FSQ\nmotion token q",
    panels:["goal","reference","latent"]
  },
  {
    title:"6. Dynamic Decoder: token만으로는 로봇을 제어할 수 없다",
    intro:"같은 motion token이라도 현재 CartPole이 오른쪽으로 기울었는지, 속도가 어떤지에 따라 지금 줄 force는 달라야 한다. 그래서 Dynamic Decoder는 token과 proprioception을 함께 본다.",
    why:"token은 '무엇을 하고 싶은가', proprioception은 '지금 몸이 어떤가', action은 '바로 지금 무엇을 해야 하나'다. 이 분리가 SONIC tracker의 제어 의미다.",
    keep:"공식 SONIC의 g1_dyn도 token_flattened + proprioception → action 구조다. Kinematic Decoder와 혼동하면 안 된다.",
    cut:"CartPole action은 1개의 force뿐이다. G1은 full joint action vector와 훨씬 복잡한 body/contact dynamics를 다룬다.",
    chain:"motion token q ──┐\n                 ├→ Dynamic Decoder → action\nactual state ────┘\n                       ↓\n                    physics",
    panels:["goal","runtime","live"]
  },
  {
    title:"7. 실제 학습: PPO가 본체, reconstruction은 보조",
    intro:"SONIC-like controller는 '잘 복원하는 autoencoder'가 목적이 아니다. physics rollout에서 reference를 실제로 따라가도록 tracking reward로 policy를 학습하는 것이 본체다.",
    why:"PPO는 Encoder와 Dynamic Decoder가 실제 action을 잘 내도록 업데이트한다. Critic은 advantage를 안정화하고 deployment에서 사라진다. Kinematic Decoder reconstruction은 token이 motion 정보를 유지하도록 돕는 auxiliary loss다.",
    keep:"teacher bootstrap은 이 lab의 실용적 초기화 장치일 뿐 SONIC 필수 요소가 아니다. 기존 tinmanlab PPO teacher는 MuJoCo에서 안정화 prior로 검증됐고 이후 student PPO가 tracking을 개선한다.",
    cut:"teacher가 ±0.8 m goal을 정밀하게 풀지는 않는다. bootstrap 이후 30 PPO iter 부근이 현재 검증 sweep의 최선이며 more-is-better가 아니다.",
    chain:"rollout → reward\n   ↓\nGAE → PPO ─→ Encoder + Dyn Decoder\n       │\n       ├→ Critic (training only)\n       └→ Kin recon aux",
    panels:["train","live"]
  },
  {
    title:"8. 전체 연결: 이것으로 GEAR-SONIC을 어디까지 이해할 수 있나?",
    intro:"이제 전체 6개 패널을 한 번에 본다. CartPole은 SONIC의 구조적 축소 모델이다. core tracker의 정보 흐름과 training-role separation은 재현할 수 있지만 humanoid foundation-controller의 성능과 규모는 재현하지 않는다.",
    why:"이 정도를 이해하면 논문의 block diagram과 코드에서 Planner / Encoder / FSQ / token / g1_dyn / g1_kin / proprioception / PPO가 각각 왜 존재하는지 추적할 수 있다.",
    keep:"충분히 이해 가능한 핵심: reference→token, FSQ가 VQ를 어떻게 단순화하는지, token+state→action, auxiliary reconstruction과 PPO의 역할, planner와 tracker의 경계.",
    cut:"남은 고급 주제: 여러 encoder가 같은 token space로 align되는 법, temporal downsampling/multi-token, 대규모 motion scaling, G1 contact/actuator/sim2real, VLA가 token을 직접 생성하는 경로.",
    chain:"Planner / Motion / Teleop\n        ↓\n     Encoder(s)\n        ↓\n       FSQ\n        ↓\n universal motion token\n        + proprioception\n        ↓\n Dynamic Decoder → Robot",
    panels:["goal","runtime","live","reference","latent","train"],
    full:true
  }
];

const LESSON_MODES={1:"ae",2:"vq",3:"vq",4:"fsq",5:"fsq",6:"fsq",7:"fsq",8:"fsq"};

function state(){ return physicsReady ? sim.getState() : [0,0,.1,0]; }
function currentReference(){ return planReference(plannerContext,goal); }
function preview(){ return trainer ? trainer.preview(state(),currentReference(),goal) : null; }

function setBadge(id,text,ok=true){
  const el=$(id); if(!el)return; el.textContent=text; el.className="badge "+(ok?"ok":"warn");
}
function setStatus(id,text,ok=true){
  const dot=$(id+"Dot"),label=$(id+"Status"); if(dot)dot.className="dot "+(ok?"ok":"bad"); if(label)label.textContent=text;
}

function resetRobot(){
  if(!physicsReady)return;
  sim.reset({x:0,xDot:0,theta:.1,thetaDot:0});
  plannerContext=[0,0];lastForce=0;accumulator=0;render();
}
function pushRobot(){
  if(!physicsReady)return;
  sim.stepForce(10,8);lastForce=10;render();
}
function policyStep(){
  if(!trainer||!physicsReady)return;
  const p=preview(); sim.stepForce(p.force,2); plannerContext=advancePlannerContext(plannerContext,goal,.02); lastForce=p.force; render();
}
function setGoal(v){
  goal=clamp(Number(v),-1.2,1.2);$("goal").value=goal;render();
}

async function rebuildTrainer(){
  live=false;busy=true;render();
  trainer?.delete?.();
  trainer=new SonicToyTrainer(sim,{seed:20261003,mode,n:8,horizon:96,epochs:4,batch:128});
  bootstrapEval=null;heldoutEval=null;
  if(teacher){
    trainer.bootstrapFromTeacher(teacher,{steps:500,batch:256,lr:.0015,auxCoef:.08});
    bootstrapEval=trainer.evaluate(12);
    heldoutEval=bootstrapEval;
  }
  busy=false;
  setStatus("ppo",teacher?"teacher bootstrap complete · PPO ready":"PPO ready · no teacher",true);
  setBadge("trainBadge","tracker ready",true);
  render();
}

async function changePreset(p){
  if(!MuJoCoCartPole.presets[p]||busy)return;
  live=false;busy=true;render();
  preset=p;await sim.loadPreset(preset);resetRobot();await rebuildTrainer();
  setBadge("physicsBadge",sim.backend,true);setStatus("physics",sim.backend+" · "+sim.snapshot().model,true);
  $("simPreset").value=preset;busy=false;render();
}
async function changeMode(m){
  if(!["ae","vq","fsq"].includes(m)||busy)return;
  mode=m;
  const url=new URL(location.href);url.searchParams.set("mode",mode);history.replaceState(null,"",url);
  await rebuildTrainer();
}

async function runIterations(n){
  if(!trainer||busy)return;
  busy=true;live=false;render();
  try{
    for(let i=0;i<n;i++){
      trainer.iteration();
      setBadge("trainBadge","PPO · iter "+trainer.iter,true);
      render();
      await new Promise(r=>setTimeout(r,0));
    }
    heldoutEval=trainer.evaluate(12);
  }finally{busy=false;render();}
}

function renderCart(){
  const c=$("cart"),ctx=c.getContext("2d"),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);
  const centerX=320,pivotY=202,railY=251,wheelY=242,scale=110,cartW=78,cartH=28;
  ctx.strokeStyle="#cbd5e1";ctx.lineWidth=6;ctx.lineCap="round";ctx.beginPath();ctx.moveTo(56,railY);ctx.lineTo(584,railY);ctx.stroke();
  ctx.fillStyle="#7b8492";ctx.font="10px ui-monospace,monospace";ctx.textAlign="center";ctx.fillText("−1.8 m",122,railY+20);ctx.fillText("0",320,railY+20);ctx.fillText("+1.8 m",518,railY+20);

  const refs=referenceFramesPhysical(plannerContext,goal);
  refs.forEach((r,i)=>drawGhost(r.x,0,"#315dc9",.08+.055*i,1.5));
  const p=preview();
  if(p){
    for(let i=0;i<SONIC_TOY_CONSTANTS.REF_FRAMES;i++){
      const xr=p.kinRecon[i*2]*SONIC_TOY_CONSTANTS.STATE_SCALE[0];
      drawGhost(xr,0,"#795fc5",.05+.035*i,1);
    }
  }
  const s=state();drawActual(s[0],s[2]);
  const gx=centerX+goal*scale;ctx.strokeStyle="#16805d";ctx.lineWidth=2;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(gx,70);ctx.lineTo(gx,railY);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle="#16805d";ctx.font="600 10px system-ui";ctx.fillText("goal",gx,62);

  function drawGhost(x,theta,color,alpha,lw){
    const cx=centerX+x*scale;ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.strokeRect(cx-cartW/2,pivotY,cartW,cartH);ctx.beginPath();ctx.moveTo(cx,pivotY);ctx.lineTo(cx+Math.sin(theta)*100,pivotY-Math.cos(theta)*100);ctx.stroke();ctx.restore();
  }
  function drawActual(x,theta){
    const cx=centerX+x*scale,L=132*(sim.spec?.poleLength||1),tx=cx+Math.sin(theta)*L,ty=pivotY-Math.cos(theta)*L;
    ctx.strokeStyle="#dc5b60";ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(cx,pivotY);ctx.lineTo(tx,ty);ctx.stroke();
    ctx.fillStyle="#334155";ctx.fillRect(cx-cartW/2,pivotY,cartW,cartH);ctx.fillStyle="#1e293b";for(const dx of [-24,24]){ctx.beginPath();ctx.arc(cx+dx,wheelY,9,0,Math.PI*2);ctx.fill();}
  }
}

function renderReference(){
  const p=preview(),refs=referenceFramesPhysical(plannerContext,goal),c=$("reference"),ctx=c.getContext("2d"),w=c.width,h=c.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  const pad={l:34,r:12,t:14,b:24},xmin=-1.4,xmax=1.4;
  const y=(x)=>pad.t+(xmax-x)/(xmax-xmin)*(h-pad.t-pad.b),xx=(i)=>pad.l+i/(7)*(w-pad.l-pad.r);
  ctx.strokeStyle="#e2e5ea";ctx.beginPath();ctx.moveTo(pad.l,y(0));ctx.lineTo(w-pad.r,y(0));ctx.stroke();
  ctx.strokeStyle="#315dc9";ctx.lineWidth=2.2;ctx.beginPath();refs.forEach((r,i)=>{const X=xx(i),Y=y(r.x);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();
  if(p){ctx.strokeStyle="#795fc5";ctx.lineWidth=1.6;ctx.setLineDash([5,3]);ctx.beginPath();for(let i=0;i<8;i++){const X=xx(i),Y=y(p.kinRecon[i*2]*SONIC_TOY_CONSTANTS.STATE_SCALE[0]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}ctx.stroke();ctx.setLineDash([]);}
  ctx.fillStyle="#667085";ctx.font="9px system-ui";ctx.fillText("x reference [m]",4,12);ctx.fillText("future →",w-50,h-7);
  $("frames").innerHTML=refs.map((r,i)=>'<div class="frame"><b>'+fmt(r.x,2)+'</b><span>t+'+fmt(r.t,2)+'s</span><span>v '+fmt(r.xd,2)+'</span></div>').join("");
}

function latentPoint(z,c){const pad=25,s=1.25;return[pad+(z[0]+s)/(2*s)*(c.width-2*pad),c.height-pad-(z[1]+s)/(2*s)*(c.height-2*pad)];}
function renderLatent(){
  const c=$("latent"),ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  const o=latentPoint([0,0],c);ctx.strokeStyle="#e2e5ea";ctx.beginPath();ctx.moveTo(15,o[1]);ctx.lineTo(w-15,o[1]);ctx.moveTo(o[0],12);ctx.lineTo(o[0],h-15);ctx.stroke();ctx.fillStyle="#7b8492";ctx.font="10px system-ui";ctx.fillText("z₁",w-26,o[1]-4);ctx.fillText("z₂",o[0]+5,14);
  if(!trainer)return;
  const curState=state(),zs=[],codes=new Set();
  for(let g=-1.2;g<=1.2001;g+=.08){const ref=planReference(plannerContext,g),out=trainer.preview(curState,ref,g);zs.push(out.z);codes.add(out.q.map(v=>v.toFixed(2)).join(","));const p=latentPoint(out.z,c);ctx.fillStyle="rgba(130,140,154,.28)";ctx.beginPath();ctx.arc(p[0],p[1],2.2,0,Math.PI*2);ctx.fill();}
  if(mode==="fsq"){for(const a of [-1,-.5,0,.5,1])for(const b of [-1,-.5,0,.5,1]){const p=latentPoint([a,b],c);ctx.fillStyle="#c8cdd5";ctx.beginPath();ctx.arc(p[0],p[1],3,0,Math.PI*2);ctx.fill();}}
  if(mode==="vq"){for(let k=0;k<8;k++){const p=latentPoint([trainer.policy.codebook[k*2],trainer.policy.codebook[k*2+1]],c);ctx.fillStyle="#8c949f";ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.fill();ctx.fillStyle="#596273";ctx.font="9px monospace";ctx.fillText(String(k),p[0]+7,p[1]-4);}}
  const out=preview();if(out){const pz=latentPoint(out.z,c),pq=latentPoint(out.q,c);if(mode!=="ae"){ctx.strokeStyle="#a2a9b3";ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(pz[0],pz[1]);ctx.lineTo(pq[0],pq[1]);ctx.stroke();ctx.setLineDash([]);}ctx.fillStyle="#315dc9";ctx.beginPath();ctx.arc(pz[0],pz[1],7,0,Math.PI*2);ctx.fill();if(mode!=="ae"){ctx.fillStyle="#d64f4f";ctx.beginPath();ctx.arc(pq[0],pq[1],7,0,Math.PI*2);ctx.fill();}}
  $("latentUsage").textContent=mode==="fsq"?"goal sweep activates "+codes.size+"/25 FSQ tokens":mode==="vq"?"goal sweep activates "+codes.size+"/8 VQ codes":"continuous latent";
}

function renderTokenDetails(){
  const p=preview();if(!p)return;
  $("zDetail").textContent="z = ["+fmt(p.z[0],3)+", "+fmt(p.z[1],3)+"]";
  if(mode==="fsq"){
    const b=p.z.map(z=>.999*Math.tanh(z));$("tokenMethod").textContent="FSQ · L=[5,5]";$("boundDetail").textContent="bound_norm = ["+fmt(b[0],3)+", "+fmt(b[1],3)+"]";$("qDetail").textContent="q = ["+fmt(p.q[0],2)+", "+fmt(p.q[1],2)+"]";$("tokenMethodText").textContent="qᵢ = round(1.998·tanh(zᵢ))/2. round는 STE로 학습.";
  }else if(mode==="vq"){
    $("tokenMethod").textContent="VQ · 8 learned vectors";$("boundDetail").textContent="nearest code index = "+p.qIndex;$("qDetail").textContent="q = ["+fmt(p.q[0],3)+", "+fmt(p.q[1],3)+"]";$("tokenMethodText").textContent="vector 전체가 가장 가까운 learned code 하나로 이동.";
  }else{
    $("tokenMethod").textContent="No quantizer";$("boundDetail").textContent="q = z";$("qDetail").textContent="q = ["+fmt(p.q[0],3)+", "+fmt(p.q[1],3)+"]";$("tokenMethodText").textContent="continuous latent를 그대로 Dynamic Decoder에 전달.";
  }
}

function renderRuntime(){
  const p=preview(),s=state(),refs=referenceFramesPhysical(plannerContext,goal);
  $("pipeGoal").textContent=(goal>=0?"+":"")+fmt(goal,2)+"m";$("nextRef").textContent=fmt(refs[0].x,2)+"m";
  if(!p)return;
  kvRows($("zVals"),[["z₁",fmt(p.z[0])],["z₂",fmt(p.z[1])]]);
  kvRows($("qVals"),[["q₁",fmt(p.q[0])],["q₂",fmt(p.q[1])]]);
  kvRows($("dynVals"),[["q",fmt(p.q[0],2)+","+fmt(p.q[1],2)],["state","4D"],["μ",fmt(p.mu,3)],["force",fmt(p.force,2)+"N"]]);
  $("quantTitle").textContent=mode==="fsq"?"FSQ":mode==="vq"?"VQ":"Continuous";$("forceRead").textContent="force "+fmt(lastForce,2)+" N";
  $("mu").textContent=fmt(p.mu,3);$("force").textContent=fmt(p.force,2)+" N";$("tokenShort").textContent="["+fmt(p.q[0],2)+","+fmt(p.q[1],2)+"]";$("goalErr").textContent=fmt(Math.abs(s[0]-goal),3)+" m";
  if($("teacherForce"))$("teacherForce").textContent=teacher?fmt(teacher.forward(s,goal).forceN,1)+" N":"—";
  if($("bootstrapEval"))$("bootstrapEval").textContent=heldoutEval?(heldoutEval.successes+"/"+heldoutEval.episodes+" · "+fmt(heldoutEval.trackingMae,3)+"m"):"—";
}
function kvRows(el,rows){el.innerHTML=rows.map(([k,v])=>'<div class="kv"><span>'+k+'</span><code>'+v+'</code></div>').join("");}

function renderTraining(){
  if(!trainer)return;const l=trainer.last;$("iter").textContent=trainer.iter;$("reward").textContent=fmt(l.reward,3);$("trackMae").textContent=fmt(l.tracking,3);$("auxLoss").textContent=fmt(l.auxLoss,4);$("piLoss").textContent=fmt(l.piLoss,4);$("iterMs").textContent=fmt(l.ms,0)+" ms";
  const boot=trainer.teacherBootstrap;
  if($("teacherMse"))$("teacherMse").textContent=boot?fmt(boot.actionMse,3):"—";
  if($("teacherMseMetric"))$("teacherMseMetric").textContent=boot?fmt(boot.actionMse,3):"—";
  if($("teacherSource"))$("teacherSource").textContent=boot?"cartpole-ppo · iter "+boot.teacher.source_iteration:"not loaded";
  if($("bootstrapEvalTrain"))$("bootstrapEvalTrain").textContent=bootstrapEval?(bootstrapEval.successes+"/"+bootstrapEval.episodes+" · "+fmt(bootstrapEval.trackingMae,3)+"m"):"—";
  if(lesson===7)setBadge("trainBadge",boot?"Teacher bootstrap · PPO "+trainer.iter:"PPO · iter "+trainer.iter,true);
  else if(lesson===8)setBadge("trainBadge","PPO · iter "+trainer.iter,true);
  else setBadge("trainBadge",lesson<=4?"representation ready":"tracker ready",true);
}

function render(){
  const s=state();$("goal").value=goal;$("goalVal").textContent=(goal>=0?"+":"")+fmt(goal,2)+" m";$("vX").textContent=fmt(s[0],3)+" m";$("vXd").textContent=fmt(s[1],3)+" m/s";$("vTh").textContent=fmt(s[2]*180/Math.PI,1)+"°";$("vThd").textContent=fmt(s[3]*180/Math.PI,1)+"°/s";
  document.querySelectorAll("[data-mode]").forEach(b=>b.classList.toggle("active",b.dataset.mode===mode));$("methodExplain").innerHTML=methodText[mode];$("runLive").classList.toggle("active",live);$("runLive").textContent="live tracking "+(live?"ON":"OFF");
  renderCart();renderReference();renderLatent();renderTokenDetails();renderRuntime();renderTraining();
}


function renderLesson(){
  const cfg=LESSONS[lesson]||LESSONS[0];
  document.body.dataset.lesson=String(lesson);
  document.body.dataset.panelCount=String(cfg.panels.length);
  document.body.classList.toggle("lesson-view",!cfg.full);
  document.body.classList.toggle("full-view",!!cfg.full);
  const card=$("lessonCard");
  card.classList.toggle("visible",!cfg.full);
  if(!cfg.full){
    $("lessonTitle").textContent=cfg.title;
    $("lessonIntro").textContent=cfg.intro;
    $("lessonWhy").textContent=cfg.why;
    $("lessonKeep").innerHTML="<p><b>보존해야 하는 것:</b> "+cfg.keep+"</p>";
    $("lessonCut").innerHTML="<p><b>축소/한계:</b> "+cfg.cut+"</p>";
    $("lessonChain").textContent=cfg.chain;
    $("lessonBoundary").textContent=(cfg.hint?cfg.hint+" · ":"")+"GEAR-SONIC 구조 학습용 축소 모델 · 성능 동등성 주장이 아님";
  }

  const liveTitle=$("liveTitle"),liveSub=$("liveSub"),latentTitle=$("latentTitle"),latentSub=$("latentSub"),latentImportant=$("latentImportant");
  const panelText={
    1:{live:["Representation mode","먼저 continuous latent를 보고 quantization이 없는 기준선을 이해"],latent:["④ Encoder → latent z","16D future reference가 2D bottleneck으로 어떻게 요약되는지 확인"],important:"<b>Decoder(aux)는 왜 있나?</b><br>latent만 보고 future reference를 복원하게 해 representation에 motion 정보가 남도록 돕는다. 실제 action은 별도 Dynamic Decoder가 낸다."},
    2:{live:["Quantizer 선택","VQ를 선택해 continuous latent와 learned codebook의 차이를 비교"],latent:["④ Encoder z → VQ code q","continuous z 전체가 가장 가까운 learned vector code로 이동"],important:"<b>VQ에서 봐야 할 것</b><br>회색 reference sweep → 파란 z → 빨간 q와 learned codebook 점의 관계를 본다. codebook은 학습되는 대표 벡터 사전이다."},
    3:{live:["VQ-VAE bottleneck","VQ code를 forward에 쓰면서 STE·commitment가 왜 필요한지 연결"],latent:["④ VQ-VAE: z → learned code q","nearest-vector 선택과 discrete bottleneck을 시각화"],important:"<b>학습의 핵심</b><br>forward는 discrete q를 쓰고 backward는 STE로 Encoder까지 gradient를 전달한다. Kinematic Decoder reconstruction은 representation 보조 loss다."},
    4:{live:["VQ ↔ FSQ 비교","FSQ가 VQ의 learned codebook 관리에서 무엇을 제거했는지 확인"],latent:["④ Encoder z → FSQ token q","각 scalar에 bound → round를 적용해 finite token을 생성"],important:"<b>FSQ의 핵심</b><br>learned vector codebook 없이 finite scalar 조합이 implicit codebook을 만든다. round에는 STE를 쓴다."},
    6:{live:["Action test","같은 motion intent라도 현재 proprioception에 따라 action이 달라지는지 확인"],latent:["④ Motion token","이 lesson의 핵심은 token 자체보다 token+proprioception 결합"],important:"<b>역할 분리</b><br>token='무엇을 할지', proprioception='지금 몸 상태', Dynamic Decoder='지금 할 action'이다."},
    7:{live:["PPO 전후 policy 확인","held-out tracking과 live action으로 학습 결과를 확인"],latent:["④ Motion representation","PPO 중에도 Encoder/FSQ token이 action 경로에 계속 존재"],important:"<b>본체와 보조</b><br>tracking PPO가 physical control을 학습하고 Kinematic Decoder reconstruction은 representation 보조다."}
  };
  const pt=panelText[lesson];
  const codeLegend=$("codeLegend"),quantizedLegend=$("quantizedLegend");
  if(codeLegend)codeLegend.lastChild.textContent=mode==="vq"?"learned VQ code":mode==="fsq"?"fixed FSQ grid":"no quantizer";
  if(quantizedLegend)quantizedLegend.hidden=mode==="ae";
  if(lesson===7)setStatus("ppo",teacher?"teacher bootstrap + PPO ready":"PPO ready",true);
  else setStatus("ppo","student policy ready",true);
  if(pt){
    if(liveTitle){liveTitle.textContent=pt.live[0];liveSub.textContent=pt.live[1];}
    if(latentTitle){latentTitle.textContent=pt.latent[0];latentSub.textContent=pt.latent[1];}
    if(latentImportant)latentImportant.innerHTML=pt.important;
  }else{
    if(liveTitle){liveTitle.textContent="Live tracker";liveSub.textContent="학습된 actor로 planner reference를 tracking";}
    if(latentTitle){latentTitle.textContent="④ Encoder → latent z → FSQ token q";latentSub.textContent="latent는 사람이 의미를 지정한 변수가 아니라, PPO + auxiliary loss가 학습한 내부 motion 좌표";}
    if(latentImportant)latentImportant.innerHTML="<b>왜 token인가?</b><br>Dynamic Decoder는 긴 future trajectory 전체가 아니라 compact motion representation과 현재 proprioception을 함께 받는다.";
  }

  const visible=new Set(cfg.panels);
  document.querySelectorAll("[data-panel]").forEach(el=>{
    const idx=cfg.panels.indexOf(el.dataset.panel);
    el.hidden=!visible.has(el.dataset.panel);
    el.style.order=idx>=0?String(idx):"";
  });
  document.querySelectorAll("[data-lesson]").forEach(b=>b.classList.toggle("active",Number(b.dataset.lesson)===lesson));
  $("lessonProgress").textContent=lesson+" / 8";
  const url=new URL(location.href);url.searchParams.set("lesson",String(lesson));history.replaceState(null,"",url);
}

async function setLesson(n){
  lesson=clamp(Math.floor(Number(n)||0),0,8);
  renderLesson();
  render();
  const recommended=LESSON_MODES[lesson];
  if(recommended && recommended!==mode && !busy){
    await changeMode(recommended);
    renderLesson();
    render();
  }
}

function attachUi(){
  document.querySelectorAll("[data-lesson]").forEach(b=>b.onclick=()=>{void setLesson(b.dataset.lesson);});
  $("goal").oninput=()=>setGoal($("goal").value);document.querySelectorAll("[data-goal]").forEach(b=>b.onclick=()=>setGoal(b.dataset.goal));
  $("resetCart").onclick=resetRobot;$("push").onclick=pushRobot;$("stepPolicy").onclick=policyStep;$("pushPolicy").onclick=()=>runIterations(10);$("iter1").onclick=()=>runIterations(1);$("iter10").onclick=()=>runIterations(10);
  $("runLive").onclick=()=>{live=!live;accumulator=0;render();};
  document.querySelectorAll("[data-mode]").forEach(b=>b.onclick=()=>changeMode(b.dataset.mode));$("simPreset").onchange=()=>changePreset($("simPreset").value);
}

async function registerWebMCP(){
  const mc=document.modelContext||navigator.modelContext;if(!mc||typeof mc.registerTool!=="function"){setStatus("mcp","WebMCP unavailable",false);return;}
  webmcpMode=typeof navigator.modelContextTesting?.listTools==="function"?"mcp-b-global":"native";
  const tools=[
    {name:"get_sonic_cartpole_state",description:"Read high-level goal, planner future reference, encoder latent, quantized token, dynamic-decoder action, MuJoCo state, and PPO metrics.",inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:true},execute:async()=>snapshot()},
    {name:"set_tracking_goal",description:"Set the high-level CartPole target position consumed by the planner.",inputSchema:{type:"object",properties:{x:{type:"number",minimum:-1.2,maximum:1.2}},required:["x"]},annotations:{readOnlyHint:false},execute:async({x})=>{setGoal(x);return snapshot();}},
    {name:"set_quantizer_mode",description:"Switch continuous, VQ, or FSQ and reset the SONIC-like toy policy.",inputSchema:{type:"object",properties:{mode:{type:"string",enum:["ae","vq","fsq"]}},required:["mode"]},annotations:{readOnlyHint:false},execute:async({mode:m})=>{await changeMode(m);return snapshot();}},
    {name:"run_ppo_iterations",description:"Run bounded PPO tracking iterations. Updates encoder, dynamic decoder, kinematic auxiliary decoder, and critic.",inputSchema:{type:"object",properties:{iterations:{type:"integer",minimum:1,maximum:20}},required:["iterations"]},annotations:{readOnlyHint:false},execute:async({iterations})=>{await runIterations(iterations);return snapshot();}},
    {name:"step_tracking_policy",description:"Run one deterministic policy action through native MuJoCo.",inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:false},execute:async()=>{policyStep();return snapshot();}},
    {name:"set_mujoco_model",description:"Switch the native MuJoCo WASM CartPole morphology/dynamics preset.",inputSchema:{type:"object",properties:{preset:{type:"string",enum:["playground","long","heavy"]}},required:["preset"]},annotations:{readOnlyHint:false},execute:async({preset:p})=>{await changePreset(p);return snapshot();}},
    {name:"set_learning_lesson",description:"Move the teaching UI to one of the structured lessons 0 through 8.",inputSchema:{type:"object",properties:{lesson:{type:"integer",minimum:0,maximum:8}},required:["lesson"]},annotations:{readOnlyHint:false},execute:async({lesson:n})=>{await setLesson(n);return snapshot();}},
    {name:"reset_sonic_cartpole",description:"Reset the live robot state without resetting learned policy weights.",inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:false},execute:async()=>{resetRobot();return snapshot();}}
  ];
  for(const t of tools)await mc.registerTool(t);webmcpTools=tools.map(t=>t.name);setStatus("mcp","WebMCP "+webmcpMode+" · "+tools.length+" tools",true);setBadge("mcpBadge","WebMCP · "+tools.length,true);
}

function snapshot(){
  const p=preview(),refs=referenceFramesPhysical(plannerContext,goal);
  return{lesson,goal,webgpu:webgpuStatus,planner:{context:{x:plannerContext[0],xDot:plannerContext[1]},frames:refs},policy:p?{mode,latent:p.z,token:p.q,actionMean:p.mu,force:p.force,kinematicReconstruction:p.kinRecon}:null,teacher:teacher?{metadata:teacher.metadata(),current:teacher.forward(state(),goal),studentBootstrap:trainer?.teacherBootstrap||null,bootstrapEval,heldoutEval}:null,robot:{backend:sim.backend,...sim.snapshot(lastForce)},training:trainer?{mode:trainer.mode,iter:trainer.iter,envSteps:trainer.envSteps,episodes:trainer.episodes,last:trainer.last,history:trainer.history.slice(-40)}:null,webmcp:{mode:webmcpMode,tools:webmcpTools}};
}

function loop(now){
  const dt=Math.min(.05,Math.max(0,(now-lastFrame)/1000));lastFrame=now;
  if(live&&!busy&&trainer&&physicsReady){
    accumulator+=dt;let n=0;while(accumulator>=.02&&n<4){const p=preview();sim.stepForce(p.force,2);plannerContext=advancePlannerContext(plannerContext,goal,.02);lastForce=p.force;accumulator-=.02;n++;}
    const s=state();if(Math.abs(s[0])>1.78||Math.abs(s[2])>.75)live=false;render();
  }
  requestAnimationFrame(loop);
}

async function init(){
  attachUi();await sim.init(preset);physicsReady=true;$("simPreset").value=preset;resetRobot();setBadge("physicsBadge",sim.backend,true);setStatus("physics",sim.backend+" · "+sim.snapshot().model,true);
  try{
    webgpuStatus=await probeWebGPUFSQ();
    setBadge("webgpuBadge",webgpuStatus.ok?"WebGPU FSQ ✓":webgpuStatus.available?"WebGPU fallback":"WebGPU unavailable",webgpuStatus.ok);
  }catch(err){
    webgpuStatus={available:false,ok:false,reason:err.message};
    setBadge("webgpuBadge","WebGPU unavailable",false);
  }
  try{teacher=await loadTeacherPolicy();}catch(err){console.warn("teacher unavailable",err);teacher=null;}
  await rebuildTrainer();await registerWebMCP();renderLesson();render();
  if(startupTrain>0) await runIterations(startupTrain);
  setInterval(()=>{try{fetch("/telemetry",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(snapshot()),keepalive:true}).catch(()=>{});}catch(_){}},750);
  requestAnimationFrame(loop);
}
init().catch(err=>{console.error(err);setStatus("physics","init error · "+err.message,false);setBadge("physicsBadge","init error",false);});
window.__sonicCartPole={snapshot,runIterations,setGoal,changeMode,changePreset,setLesson};
