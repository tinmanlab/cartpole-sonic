import { MuJoCoCartPole } from "./mujoco_sim.js";
import {
  SonicToyTrainer,
  planReference,
  advancePlannerContext,
  referenceFramesPhysical,
  SONIC_TOY_CONSTANTS,
} from "./sonic_toy.js";
import { loadTeacherPolicy } from "./teacher_policy.js";
import { probeWebGPUFSQ } from "./webgpu_fsq.js";
import {
  COURSE_VERSION,
  PRIMARY_PATH,
  OPTIONAL_BRANCHES,
  LESSONS,
  getLesson,
  getCourseOutline,
} from "./course.js";

const $ = id => document.getElementById(id);
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const fmt = (v,n=3) => Number.isFinite(v) ? Number(v).toFixed(n) : "—";

const LEGACY_LESSON_MAP = {
  "0":"ae","1":"ae","2":"vq","3":"vqvae","4":"fsq",
  "5":"motion-token","6":"dynamic-decoder","7":"ppo","8":"sonic",
};
const query = new URLSearchParams(location.search);
function resolveLesson(raw){
  if(!raw) return "ae";
  if(LESSONS[raw]) return raw;
  return LEGACY_LESSON_MAP[String(raw)] || "ae";
}

let lessonId = resolveLesson(query.get("lesson"));
let lesson = getLesson(lessonId);
let preset = ["playground","long","heavy"].includes(query.get("preset")) ? query.get("preset") : "playground";
let goal = query.has("goal") && Number.isFinite(Number(query.get("goal"))) ? clamp(Number(query.get("goal")),-1.2,1.2) : 0.8;

const sim = new MuJoCoCartPole();
let teacher = null;
let teacherPromise = null;
let currentTrainer = null;
const trainerCache = new Map();
let physicsReady = false;
let live = false;
let busy = false;
let plannerContext = [0,0];
let lastForce = 0;
let lastFrame = performance.now();
let accumulator = 0;
let webgpuStatus = {available:false,ok:false,reason:"not probed"};
let webmcpMode = "unavailable";
let webmcpTools = [];
let heldoutEval = null;
let ppoReferenceEvidence = null;
let ppoHeldoutHistory = [];

function state(){
  return physicsReady ? sim.getState() : [0,0,.1,0];
}
function currentReference(){
  return planReference(plannerContext,goal);
}
function preview(){
  return currentTrainer ? currentTrainer.preview(state(),currentReference(),goal) : null;
}
function setBadge(id,text,ok=true){
  const el=$(id); if(!el) return;
  el.textContent=text;
  el.className="badge "+(ok?"ok":"warn");
}

async function getTrainer(mode){
  if(trainerCache.has(mode)) return trainerCache.get(mode);
  busy=true; renderHeaderState();
  const t = new SonicToyTrainer(sim,{seed:20261003,mode,n:8,horizon:96,epochs:4,batch:128});
  let restored=false;
  try{
    const ckpt=await fetch("./assets/student_"+mode+"_bootstrap.json",{cache:"force-cache"}).then(r=>r.ok?r.json():Promise.reject(new Error("checkpoint "+r.status)));
    if(ckpt.schema!=="cartpole-sonic-student-bootstrap/v1"||ckpt.mode!==mode)throw new Error("checkpoint schema/mode mismatch");
    t.restorePolicy(ckpt.policy,{teacherBootstrap:ckpt.teacherBootstrap,rngState:ckpt.rng});
    t.bootstrapEval=ckpt.defaultPresetEval||null;
    restored=true;
  }catch(err){
    console.warn("student checkpoint unavailable; falling back to local teacher bootstrap",err);
  }
  if(!restored){
    if(!teacher && teacherPromise) teacher=await teacherPromise;
    if(teacher) t.bootstrapFromTeacher(teacher,{steps:500,batch:256,lr:.0015,auxCoef:.08});
  }
  trainerCache.set(mode,t);
  busy=false;
  return t;
}
function clearTrainers(){
  for(const t of trainerCache.values()) t.delete?.();
  trainerCache.clear();
  currentTrainer=null;
  heldoutEval=null;
  ppoHeldoutHistory=[];
}
async function ensureLessonTrainer(){
  currentTrainer=await getTrainer(lesson.mode);
  if(lesson.id==="ppo" && !heldoutEval){
    heldoutEval=currentTrainer.evaluate(12);
    ppoHeldoutHistory=[{iterations:currentTrainer.iter,mae:heldoutEval.trackingMae}];
  }
}

function resetRobot(){
  if(!physicsReady) return;
  sim.reset({x:0,xDot:0,theta:.1,thetaDot:0});
  plannerContext=[0,0];
  lastForce=0;
  accumulator=0;
  render();
}
function pushRobot(){
  if(!physicsReady) return;
  sim.stepForce(10,8);
  plannerContext=advancePlannerContext(plannerContext,goal,.08);
  lastForce=10;
  render();
}
function stepPolicy(steps=1){
  if(!currentTrainer||!physicsReady) return;
  const n=clamp(Math.floor(steps)||1,1,100);
  for(let i=0;i<n;i++){
    const p=preview();
    sim.stepForce(p.force,2);
    plannerContext=advancePlannerContext(plannerContext,goal,.02);
    lastForce=p.force;
  }
  render();
}
function setGoal(x){
  goal=clamp(Number(x),-1.2,1.2);
  $("goal").value=goal;
  updateUrl();
  render();
}
async function changePreset(next){
  if(!MuJoCoCartPole.presets[next]||busy) return;
  live=false; busy=true;
  clearTrainers();
  preset=next;
  await sim.loadPreset(preset);
  resetRobot();
  currentTrainer=await getTrainer(lesson.mode);
  busy=false;
  $("simPreset").value=preset;
  updateUrl();
  render();
}
async function runPPO(iterations){
  if(!currentTrainer||busy) return;
  busy=true; live=false; renderHeaderState();
  const n=clamp(Math.floor(iterations)||1,1,50);
  for(let i=0;i<n;i++){
    currentTrainer.iteration();
    if(i%2===1) await new Promise(r=>setTimeout(r,0));
  }
  heldoutEval=currentTrainer.evaluate(12);
  ppoHeldoutHistory.push({iterations:currentTrainer.iter,mae:heldoutEval.trackingMae});
  busy=false;
  render();
}

function updateUrl(){
  const u=new URL(location.href);
  u.searchParams.set("lesson",lesson.id);
  u.searchParams.set("goal",goal.toFixed(2));
  if(preset!=="playground") u.searchParams.set("preset",preset); else u.searchParams.delete("preset");
  history.replaceState(null,"",u);
}

function buildNav(){
  const host=$("primaryPath");
  host.innerHTML="";
  PRIMARY_PATH.forEach((id,i)=>{
    const b=document.createElement("button");
    b.textContent=LESSONS[id].nav;
    b.dataset.lesson=id;
    b.classList.toggle("active",id===lesson.id);
    b.onclick=()=>{void navigateLesson(id);};
    host.appendChild(b);
    if(i<PRIMARY_PATH.length-1){
      const a=document.createElement("span");
      a.className="arrow"; a.textContent="→"; host.appendChild(a);
    }
  });
  const branch=$("branchPath");
  branch.innerHTML='<span class="stem">↳ optional branch from Encoder / AE:</span>';
  for(const id of OPTIONAL_BRANCHES.ae||[]){
    const b=document.createElement("button");
    b.textContent=LESSONS[id].nav;
    b.classList.toggle("active",id===lesson.id);
    b.onclick=()=>{void navigateLesson(id);};
    branch.appendChild(b);
  }
  const ret=document.createElement("span");
  ret.className="stem"; ret.textContent="→ return to VQ";
  branch.appendChild(ret);
}

function prevNext(){
  if(lesson.optional) return {prev:"ae",next:lesson.returnsTo||"vq"};
  const i=PRIMARY_PATH.indexOf(lesson.id);
  return {
    prev:i>0?PRIMARY_PATH[i-1]:null,
    next:i>=0&&i<PRIMARY_PATH.length-1?PRIMARY_PATH[i+1]:null,
  };
}
async function navigateLesson(id){
  if(!LESSONS[id]||busy) return;
  live=false;
  lessonId=id; lesson=getLesson(id);
  currentTrainer=await getTrainer(lesson.mode);
  if(id==="ppo"){
    heldoutEval=currentTrainer.evaluate(12);
    if(!ppoHeldoutHistory.length || ppoHeldoutHistory.at(-1).iterations!==currentTrainer.iter){
      ppoHeldoutHistory.push({iterations:currentTrainer.iter,mae:heldoutEval.trackingMae});
    }
  }
  buildNav();
  updateUrl();
  render();
}

function renderSimulation(){
  const c=$("cart"),ctx=c.getContext("2d"),W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);
  const centerX=W/2,pivotY=225,railY=282,wheelY=272,scale=110,cartW=82,cartH=30;
  ctx.strokeStyle="#cbd5e1";ctx.lineWidth=6;ctx.lineCap="round";
  ctx.beginPath();ctx.moveTo(45,railY);ctx.lineTo(W-45,railY);ctx.stroke();

  const refs=referenceFramesPhysical(plannerContext,goal);
  refs.forEach((r,i)=>drawGhost(r.x,"#315dc9",.07+.045*i));
  const s=state();
  drawActual(s[0],s[2]);
  const gx=centerX+goal*scale;
  ctx.strokeStyle="#16805d";ctx.lineWidth=2;ctx.setLineDash([5,4]);
  ctx.beginPath();ctx.moveTo(gx,60);ctx.lineTo(gx,railY);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#16805d";ctx.font="600 10px system-ui";ctx.textAlign="center";ctx.fillText("goal",gx,53);

  function drawGhost(x,color,alpha){
    const cx=centerX+x*scale;
    ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=1.5;
    ctx.strokeRect(cx-cartW/2,pivotY,cartW,cartH);
    ctx.beginPath();ctx.moveTo(cx,pivotY);ctx.lineTo(cx,pivotY-96);ctx.stroke();ctx.restore();
  }
  function drawActual(x,theta){
    const cx=centerX+x*scale,L=142*(sim.spec?.poleLength||1);
    const tx=cx+Math.sin(theta)*L,ty=pivotY-Math.cos(theta)*L;
    ctx.strokeStyle="#d64f4f";ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(cx,pivotY);ctx.lineTo(tx,ty);ctx.stroke();
    ctx.fillStyle="#334155";ctx.fillRect(cx-cartW/2,pivotY,cartW,cartH);
    ctx.fillStyle="#1e293b";
    for(const dx of [-25,25]){ctx.beginPath();ctx.arc(cx+dx,wheelY,9,0,Math.PI*2);ctx.fill();}
  }

  $("goal").value=goal;
  $("goalVal").textContent=(goal>=0?"+":"")+fmt(goal,2)+" m";
  $("liveBtn").textContent="Live "+(live?"ON":"OFF");
  $("liveBtn").classList.toggle("active",live);
  $("vX").textContent=fmt(s[0],3)+" m";
  $("vXd").textContent=fmt(s[1],3);
  $("vTh").textContent=fmt(s[2]*180/Math.PI,1)+"°";
  $("vThd").textContent=fmt(s[3]*180/Math.PI,1);
  $("vForce").textContent=fmt(lastForce,2)+" N";
}

function clearViz(){
  const c=$("lessonViz"),ctx=c.getContext("2d");
  ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle="#fff";ctx.fillRect(0,0,c.width,c.height);
  $("vizHtml").classList.remove("visible");$("vizHtml").innerHTML="";
  $("lessonViz").style.display="block";
  $("vizMetrics").innerHTML="";
}
function setMetrics(items){
  $("vizMetrics").innerHTML=items.map(x=>'<span class="chip">'+x+'</span>').join("");
}
function drawAxes(ctx,c,{xmin=-1.25,xmax=1.25,ymin=-1.25,ymax=1.25}={}){
  const p=38;
  const X=x=>p+(x-xmin)/(xmax-xmin)*(c.width-2*p);
  const Y=y=>c.height-p-(y-ymin)/(ymax-ymin)*(c.height-2*p);
  ctx.strokeStyle="#e2e6ec";ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(p,Y(0));ctx.lineTo(c.width-p,Y(0));ctx.moveTo(X(0),p);ctx.lineTo(X(0),c.height-p);ctx.stroke();
  return {X,Y,p};
}
function reconstructionError(p,ref){
  if(!p) return NaN;
  let e=0;for(let i=0;i<ref.length;i++)e+=(p.kinRecon[i]-ref[i])**2;
  return e/ref.length;
}
function renderReconstructionViz(){
  const c=$("lessonViz"),ctx=c.getContext("2d"),p=preview(),ref=currentReference();
  const pad={l:48,r:22,t:32,b:48},ymin=-1.1,ymax=1.1;
  const X=i=>pad.l+i/(SONIC_TOY_CONSTANTS.REF_FRAMES-1)*(c.width-pad.l-pad.r);
  const Y=y=>pad.t+(ymax-y)/(ymax-ymin)*(c.height-pad.t-pad.b);
  ctx.strokeStyle="#e3e6eb";ctx.beginPath();ctx.moveTo(pad.l,Y(0));ctx.lineTo(c.width-pad.r,Y(0));ctx.stroke();

  const draw=(arr,color,dash=[])=>{
    ctx.strokeStyle=color;ctx.lineWidth=3;ctx.setLineDash(dash);ctx.beginPath();
    for(let i=0;i<SONIC_TOY_CONSTANTS.REF_FRAMES;i++){const x=X(i),y=Y(arr[i*2]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();ctx.setLineDash([]);
  };
  draw(ref,"#315dc9");
  if(p) draw(p.kinRecon,"#795fc5",[7,5]);

  ctx.fillStyle="#667085";ctx.font="12px system-ui";
  ctx.fillText("future x reference (normalized)",pad.l,pad.t-10);
  for(let i=0;i<8;i++){ctx.fillStyle="#7b8492";ctx.font="10px ui-monospace";ctx.fillText("t+"+fmt((i+1)*.08,2),X(i)-14,c.height-20);}
  $("vizCaption").innerHTML='<span style="color:#315dc9;font-weight:700">blue = original future reference</span> · <span style="color:#795fc5;font-weight:700">purple dashed = Decoder reconstruction</span>. Decoder가 복원할 수 있어야 작은 latent에 motion 정보가 남았다고 볼 수 있다.';
  if(p)setMetrics(["16D input → 2D latent → 16D recon","z=["+fmt(p.z[0],3)+", "+fmt(p.z[1],3)+"]","recon MSE="+fmt(reconstructionError(p,ref),4)]);
}
function renderLatentViz(){
  const c=$("lessonViz"),ctx=c.getContext("2d"),{X,Y}=drawAxes(ctx,c),cur=state(),codes=new Set();
  for(let g=-1.2;g<=1.2001;g+=.06){
    const ref=planReference(plannerContext,g),out=currentTrainer.preview(cur,ref,g);
    codes.add(out.q.map(v=>v.toFixed(2)).join(","));
    ctx.fillStyle="rgba(125,135,150,.28)";ctx.beginPath();ctx.arc(X(out.z[0]),Y(out.z[1]),2.4,0,Math.PI*2);ctx.fill();
  }
  if(lesson.mode==="vq"){
    for(let k=0;k<8;k++){const x=currentTrainer.policy.codebook[k*2],y=currentTrainer.policy.codebook[k*2+1];ctx.fillStyle="#7b8492";ctx.beginPath();ctx.arc(X(x),Y(y),7,0,Math.PI*2);ctx.fill();ctx.fillStyle="#4b5563";ctx.font="10px monospace";ctx.fillText(String(k),X(x)+9,Y(y)-5);}
  }else{
    for(const a of [-1,-.5,0,.5,1])for(const b of [-1,-.5,0,.5,1]){ctx.fillStyle="#c7ccd4";ctx.beginPath();ctx.arc(X(a),Y(b),3.5,0,Math.PI*2);ctx.fill();}
  }
  const p=preview();
  if(p){
    ctx.strokeStyle="#9aa3af";ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(X(p.z[0]),Y(p.z[1]));ctx.lineTo(X(p.q[0]),Y(p.q[1]));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle="#315dc9";ctx.beginPath();ctx.arc(X(p.z[0]),Y(p.z[1]),8,0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#d64f4f";ctx.beginPath();ctx.arc(X(p.q[0]),Y(p.q[1]),8,0,Math.PI*2);ctx.fill();
  }
  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.fillText("z₁",c.width-48,Y(0)-8);ctx.fillText("z₂",X(0)+7,22);
  if(lesson.mode==="vq"){
    $("vizCaption").innerHTML='회색 = goal sweep의 continuous latent · 회색 큰 점 = <b>learned codebook</b> · 파랑 z → 빨강 q = nearest-vector 선택.';
    setMetrics(["learned codebook: 8 vectors","active codes="+codes.size+"/8",p?"z→q distance="+fmt(Math.hypot(p.z[0]-p.q[0],p.z[1]-p.q[1]),3):""]);
  }else{
    $("vizCaption").innerHTML='회색 = goal sweep의 continuous latent · 작은 grid = <b>고정 FSQ finite levels</b> · 파랑 z → 빨강 q = bound + round.';
    setMetrics(["L=[5,5] → 25 implicit codes","active tokens="+codes.size+"/25",p?"q=["+fmt(p.q[0],2)+", "+fmt(p.q[1],2)+"]":""]);
  }
}
function showHtml(html,caption,metrics=[]){
  $("lessonViz").style.display="none";
  $("vizHtml").innerHTML=html;
  $("vizHtml").classList.add("visible");
  $("vizCaption").innerHTML=caption;
  setMetrics(metrics);
}
function renderVaeViz(){
  showHtml(
    '<div class="branch-map"><pre>Autoencoder latent\\n├── VAE  →  μ, σ → sample z + KL        (optional)\\n│\\n└── VQ   →  nearest learned vector q    (main path to FSQ)\\n                         ↓\\n                       VQ-VAE\\n                         ↓\\n                        FSQ</pre></div>',
    'VAE는 “확률적 continuous latent” 분기다. FSQ를 이해하기 위한 본선은 VQ → VQ-VAE → FSQ다.',
    ["VAE = optional background","VQ = direct predecessor to FSQ"]
  );
}
function renderVqvaeViz(){
  const p=preview();
  showHtml(
    '<div class="flow-diagram"><div class="flow-row">'+
    '<div class="flow-box"><strong>future reference</strong><span>16D target motion</span></div><div class="flow-arrow">→</div>'+
    '<div class="flow-box"><strong>Encoder</strong><span>continuous z</span></div><div class="flow-arrow">→</div>'+
    '<div class="flow-box accent"><strong>VQ</strong><span>nearest learned code q</span></div><div class="flow-arrow">→</div>'+
    '<div class="flow-box purple"><strong>Kinematic Decoder</strong><span>reconstruction loss</span></div>'+
    '</div><div style="text-align:center;margin-top:28px;font:12px system-ui;color:#667085">'+
    '<b>forward:</b> discrete q is used &nbsp;&nbsp; · &nbsp;&nbsp; <b>backward:</b> STE passes gradient to Encoder &nbsp;&nbsp; · &nbsp;&nbsp; commitment keeps z near q'+
    '</div></div>',
    'VQ-VAE는 discrete bottleneck을 “쓸 수 있게” 만드는 학습 구조다. nearest lookup 하나만으로는 Encoder/codebook을 안정적으로 함께 학습하기 어렵다.',
    [p?"z=["+fmt(p.z[0],2)+","+fmt(p.z[1],2)+"]":"",p?"q=["+fmt(p.q[0],2)+","+fmt(p.q[1],2)+"]":"","STE","commitment","codebook update"]
  );
}
function renderDecoderViz(){
  const p=preview(),s=state();
  const stateText='['+s.map(v=>fmt(v,2)).join(", ")+']';
  const token=p?'['+p.q.map(v=>fmt(v,2)).join(", ")+']':'—';
  const force=p?fmt(p.force,2)+" N":"—";
  showHtml(
    '<div class="flow-diagram"><div class="flow-row">'+
    '<div class="flow-box accent"><strong>motion token q</strong><span>'+token+'<br>what to do</span></div>'+
    '<div class="flow-arrow">+</div>'+
    '<div class="flow-box"><strong>proprioception</strong><span>'+stateText+'<br>where the robot is now</span></div>'+
    '<div class="flow-arrow">→</div>'+
    '<div class="flow-box purple"><strong>Dynamic Decoder</strong><span>state-conditioned motor policy</span></div>'+
    '<div class="flow-arrow">→</div>'+
    '<div class="flow-box"><strong>force</strong><span style="font-size:18px;font-weight:800;color:#172554">'+force+'</span></div>'+
    '</div></div>',
    'goal을 고정한 채 Push를 누르면 desired motion token보다 actual state가 크게 변한다. Dynamic Decoder가 그 차이를 보고 즉시 다른 action을 내는지 확인한다.',
    [p?"token="+token:"","state θ="+fmt(s[2]*180/Math.PI,1)+"°","action="+force]
  );
}
function renderTrainingViz(){
  const c=$("lessonViz"),ctx=c.getContext("2d");
  ctx.fillStyle="#fff";ctx.fillRect(0,0,c.width,c.height);
  const pad={l:62,r:28,t:32,b:50};
  ctx.strokeStyle="#e2e6ec";ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,c.height-pad.b);ctx.lineTo(c.width-pad.r,c.height-pad.b);ctx.stroke();

  const ref=(ppoReferenceEvidence?.checkpoints||[])
    .map(x=>({iterations:x.ppoIterations,mae:x.eval?.trackingMae}))
    .filter(x=>Number.isFinite(x.mae));
  const livePts=ppoHeldoutHistory.filter(x=>Number.isFinite(x.mae));
  const all=[...ref,...livePts];
  const maxIter=Math.max(50,...all.map(x=>x.iterations));
  const maxMae=Math.max(.25,...all.map(x=>x.mae));
  const minMae=Math.min(.10,...all.map(x=>x.mae));
  const X=i=>pad.l+i/maxIter*(c.width-pad.l-pad.r);
  const Y=v=>c.height-pad.b-(v-minMae)/(maxMae-minMae)*(c.height-pad.t-pad.b);

  for(let i=0;i<=5;i++){
    const v=minMae+(maxMae-minMae)*i/5,y=Y(v);
    ctx.strokeStyle="#f0f2f5";ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(c.width-pad.r,y);ctx.stroke();
    ctx.fillStyle="#7b8492";ctx.font="10px ui-monospace";ctx.textAlign="right";ctx.fillText(v.toFixed(2),pad.l-8,y+3);
  }

  const drawSeries=(pts,color,dash,width)=>{
    if(!pts.length)return;
    ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();
    pts.forEach((p,i)=>{const x=X(p.iterations),y=Y(p.mae);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    ctx.stroke();ctx.setLineDash([]);
    pts.forEach(p=>{ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(p.iterations),Y(p.mae),4,0,Math.PI*2);ctx.fill();});
  };
  drawSeries(ref,"#9aa3af",[6,5],2);
  drawSeries(livePts,"#315dc9",[],3);

  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.textAlign="left";
  ctx.fillText("held-out tracking MAE ↓",pad.l,18);
  ctx.textAlign="right";ctx.fillText("PPO iterations →",c.width-pad.r,c.height-15);

  $("vizCaption").innerHTML='회색 점선 = repo에 저장된 deterministic held-out sweep · 파랑 = 현재 세션. <b>낮을수록 좋다.</b> PPO는 reconstruction이 아니라 physics tracking error를 줄이는 본체다.';
  setMetrics([
    "PPO iter="+currentTrainer.iter,
    heldoutEval?"current held-out="+fmt(heldoutEval.trackingMae,3)+" m":"current held-out=—",
    ref.length?"verified best="+fmt(Math.min(...ref.map(x=>x.mae)),3)+" m":"verified sweep unavailable"
  ]);
}
function renderSonicMap(){
  showHtml(
    '<table class="compare-table"><thead><tr><th>CartPole lab</th><th>GEAR-SONIC role</th></tr></thead><tbody>'+
    '<tr><td>8 future [x, ẋ] frames</td><td>future whole-body motion reference</td></tr>'+
    '<tr><td>tiny reference Encoder</td><td>G1 / SMPL / teleop motion Encoder</td></tr>'+
    '<tr><td>2D latent z</td><td>learned compact motion representation</td></tr>'+
    '<tr><td>FSQ L=[5,5]</td><td>FSQ motion-token bottleneck</td></tr>'+
    '<tr><td>q + [x,ẋ,θ,θ̇]</td><td>motion token + robot proprioception</td></tr>'+
    '<tr><td>Dynamic Decoder → force</td><td>g1_dyn → whole-body joint action</td></tr>'+
    '<tr><td>Kinematic Decoder</td><td>future-motion reconstruction auxiliary path</td></tr>'+
    '<tr><td>MuJoCo rollout + PPO</td><td>large-scale physical tracking optimization</td></tr>'+
    '</tbody></table>',
    '이 장에서는 새 개념을 추가하지 않는다. 각 CartPole block을 SONIC의 같은 역할로 치환한다.',
    ["not reproduced: multi-contact","not reproduced: multi-encoder alignment","not reproduced: sim2real scale"]
  );
}
function renderLessonViz(){
  clearViz();
  $("vizTitle").textContent=lesson.title;
  $("vizSub").textContent="Main visualization · 모든 장에서 같은 위치와 크기";
  $("vizStatus").textContent=lesson.optional?"optional branch":"step "+lesson.step;
  switch(lesson.viz){
    case "reconstruction": renderReconstructionViz(); break;
    case "vae-branch": renderVaeViz(); break;
    case "latent": renderLatentViz(); break;
    case "vqvae": renderVqvaeViz(); break;
    case "decoder": renderDecoderViz(); break;
    case "training": renderTrainingViz(); break;
    case "sonic-map": renderSonicMap(); break;
  }
}

function renderGuide(){
  $("guideStep").textContent=lesson.optional?"OPTIONAL BRANCH":"STEP "+lesson.step;
  $("guideTitle").textContent=lesson.title;
  $("guideAnswer").textContent=lesson.answer;
  $("guideWhy").textContent=lesson.why;
  $("guideWatch").innerHTML=lesson.watch.map(x=>"<li>"+x+"</li>").join("");
  $("guideTry").textContent=lesson.try;
  $("guideTakeaway").textContent=lesson.takeaway;
  const {prev,next}=prevNext();
  $("prevLesson").disabled=!prev;
  $("nextLesson").disabled=!next;
  $("prevLesson").onclick=()=>prev&&navigateLesson(prev);
  $("nextLesson").onclick=()=>next&&navigateLesson(next);
  $("lessonAction").textContent=lesson.id==="ppo"?"+10 PPO iterations":lesson.id==="dynamic-decoder"?"Push + 1 Step":lesson.id==="motion-token"?"Push robot":lesson.id==="sonic"?"Run live tracking":"Change goal";
  $("lessonStatus").textContent=busy?"Preparing model…":"mode="+lesson.mode+" · preset="+preset;
}

function renderProcess(){
  document.querySelectorAll("[data-stage]").forEach(el=>{
    const on=lesson.highlights.includes(el.dataset.stage);
    el.classList.toggle("active",on);
  });
  $("sonicMapping").textContent="GEAR-SONIC: "+lesson.sonic;
}

function renderHeaderState(){
  if(busy) setBadge("mcpBadge","preparing…",false);
  else setBadge("mcpBadge",webmcpMode==="unavailable"?"WebMCP pending":"WebMCP · "+webmcpTools.length,webmcpMode!=="unavailable");
}

function renderCourseChrome(){
  buildNav();
  renderGuide();
  renderProcess();
  $("vizTitle").textContent=lesson.title;
  $("vizSub").textContent="Main visualization · same position and size in every lesson";
  $("vizStatus").textContent=lesson.optional?"optional branch":"step "+lesson.step;
  $("vizCaption").textContent=busy?"Preparing the lesson model…":"Ready";
  renderHeaderState();
}

function render(){
  if(!physicsReady||!currentTrainer) return;
  buildNav();
  renderSimulation();
  renderLessonViz();
  renderGuide();
  renderProcess();
  renderHeaderState();
}

async function runLessonAction(){
  switch(lesson.id){
    case "ae":
    case "vq":
    case "vqvae":
    case "fsq":
      setGoal(goal>0?-0.8:0.8);
      break;
    case "vae":
      await navigateLesson("vq");
      break;
    case "motion-token":
      pushRobot();
      break;
    case "dynamic-decoder":
      pushRobot(); stepPolicy(1);
      break;
    case "ppo":
      await runPPO(10);
      break;
    case "sonic":
      live=true; render();
      break;
  }
}

function courseSnapshot(){
  const p=preview(),ref=currentReference(),s=state();
  return {
    course:{
      version:COURSE_VERSION,
      currentLesson:lesson.id,
      lesson:{
        id:lesson.id,step:lesson.step,title:lesson.title,question:lesson.question,
        answer:lesson.answer,why:lesson.why,watch:lesson.watch,try:lesson.try,
        takeaway:lesson.takeaway,sonic:lesson.sonic,prerequisites:lesson.prerequisites,
        next:lesson.next||lesson.returnsTo||null,optional:Boolean(lesson.optional),
      },
      outline:getCourseOutline(),
    },
    experiment:{goal,live,preset,mode:lesson.mode},
    signals:{
      reference:Array.from(ref),
      latent:p?.z||null,
      token:p?.q||null,
      proprioception:s,
      actionMean:p?.mu??null,
      force:p?.force??null,
      kinematicReconstruction:p?.kinRecon||null,
    },
    training:currentTrainer?{
      ppoIterations:currentTrainer.iter,
      envSteps:currentTrainer.envSteps,
      episodes:currentTrainer.episodes,
      last:currentTrainer.last,
      heldoutEval,
      heldoutHistory:lesson.id==="ppo"?ppoHeldoutHistory:null,
      verifiedReference:lesson.id==="ppo"?ppoReferenceEvidence:null,
      teacherBootstrap:lesson.id==="ppo"?currentTrainer.teacherBootstrap||null:null,
    }:null,
    backends:{
      physics:physicsReady?sim.backend:null,
      webgpu:webgpuStatus,
      webmcp:{mode:webmcpMode,tools:webmcpTools},
    },
  };
}

async function registerWebMCP(){
  const mc=document.modelContext||navigator.modelContext;
  if(!mc||typeof mc.registerTool!=="function"){webmcpMode="unavailable";return;}
  webmcpMode=typeof navigator.modelContextTesting?.listTools==="function"?"mcp-b-global":"native";
  const lessonIds=Object.keys(LESSONS);
  const tools=[
    {
      name:"course_get_outline",
      description:"Read the canonical CartPole SONIC curriculum, including the primary path and optional VAE branch.",
      inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:true},
      execute:async()=>getCourseOutline()
    },
    {
      name:"course_get_state",
      description:"Read the current lesson, experiment, live SONIC-like signals, training evidence, and browser backends.",
      inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:true},
      execute:async()=>courseSnapshot()
    },
    {
      name:"course_navigate",
      description:"Navigate to one semantic lesson. The lesson controls the appropriate AE/VQ/FSQ representation mode.",
      inputSchema:{type:"object",properties:{lesson_id:{type:"string",enum:lessonIds}},required:["lesson_id"]},
      annotations:{readOnlyHint:false},execute:async({lesson_id})=>{await navigateLesson(lesson_id);return courseSnapshot();}
    },
    {
      name:"course_run_lesson_action",
      description:"Run the current lesson's canonical experiment action, such as changing the goal, applying a push, or running PPO.",
      inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:false},
      execute:async()=>{await runLessonAction();return courseSnapshot();}
    },
    {
      name:"simulation_control",
      description:"Control the shared CartPole simulation that remains fixed across all lessons.",
      inputSchema:{type:"object",properties:{
        action:{type:"string",enum:["step","live_on","live_off","push","reset"]},
        steps:{type:"integer",minimum:1,maximum:100}
      },required:["action"]},
      annotations:{readOnlyHint:false},
      execute:async({action,steps=1})=>{
        if(action==="step")stepPolicy(steps);
        else if(action==="live_on"){live=true;render();}
        else if(action==="live_off"){live=false;render();}
        else if(action==="push")pushRobot();
        else if(action==="reset")resetRobot();
        return courseSnapshot();
      }
    },
    {
      name:"experiment_set_goal",
      description:"Set the high-level target position used by the shared planner.",
      inputSchema:{type:"object",properties:{x:{type:"number",minimum:-1.2,maximum:1.2}},required:["x"]},
      annotations:{readOnlyHint:false},execute:async({x})=>{setGoal(x);return courseSnapshot();}
    },
    {
      name:"training_run",
      description:"Run bounded PPO tracking iterations for the current lesson's student policy.",
      inputSchema:{type:"object",properties:{iterations:{type:"integer",minimum:1,maximum:30}},required:["iterations"]},
      annotations:{readOnlyHint:false},execute:async({iterations})=>{await runPPO(iterations);return courseSnapshot();}
    },
    {
      name:"simulation_set_model",
      description:"Switch the shared native MuJoCo WASM CartPole model preset. This rebuilds lesson policies for the new dynamics.",
      inputSchema:{type:"object",properties:{preset:{type:"string",enum:["playground","long","heavy"]}},required:["preset"]},
      annotations:{readOnlyHint:false},execute:async({preset})=>{await changePreset(preset);return courseSnapshot();}
    }
  ];
  for(const t of tools) await mc.registerTool(t);
  webmcpTools=tools.map(t=>t.name);
  renderHeaderState();
}

function attachUI(){
  $("goal").oninput=()=>setGoal($("goal").value);
  $("stepBtn").onclick=()=>stepPolicy(1);
  $("liveBtn").onclick=()=>{live=!live;accumulator=0;render();};
  $("pushBtn").onclick=pushRobot;
  $("resetBtn").onclick=resetRobot;
  $("simPreset").onchange=()=>changePreset($("simPreset").value);
  $("lessonAction").onclick=()=>{void runLessonAction();};
}

function loop(now){
  const dt=Math.min(.05,Math.max(0,(now-lastFrame)/1000));lastFrame=now;
  if(live&&!busy&&currentTrainer&&physicsReady){
    accumulator+=dt;
    let n=0;
    while(accumulator>=.02&&n<4){
      const p=preview();
      sim.stepForce(p.force,2);
      plannerContext=advancePlannerContext(plannerContext,goal,.02);
      lastForce=p.force;
      accumulator-=.02;n++;
    }
    const s=state();
    if(Math.abs(s[0])>1.78||Math.abs(s[2])>.75)live=false;
    render();
  }
  requestAnimationFrame(loop);
}

async function init(){
  attachUI();
  busy=true;
  renderCourseChrome();

  teacherPromise=loadTeacherPolicy()
    .then(t=>(teacher=t,t))
    .catch(err=>{console.warn("teacher unavailable",err);return null;});
  fetch("./evidence/ppo_eval.json",{cache:"no-store"})
    .then(r=>r.ok?r.json():null)
    .then(ev=>{ppoReferenceEvidence=ev;if(lesson.id==="ppo"&&!busy)render();})
    .catch(()=>{ppoReferenceEvidence=null;});

  await sim.init(preset);
  physicsReady=true;
  $("simPreset").value=preset;
  resetRobot();
  setBadge("physicsBadge",sim.backend,true);

  await ensureLessonTrainer();
  busy=false;
  render();
  updateUrl();

  // These integrations are evidence/status surfaces, not prerequisites for the lesson UI.
  registerWebMCP().then(()=>renderHeaderState()).catch(()=>{});
  probeWebGPUFSQ().then(status=>{
    webgpuStatus=status;
    setBadge("webgpuBadge",status.ok?"WebGPU FSQ ✓":status.available?"WebGPU fallback":"WebGPU unavailable",status.ok);
  }).catch(err=>{
    webgpuStatus={available:false,ok:false,reason:err.message};
    setBadge("webgpuBadge","WebGPU unavailable",false);
  });

  if(location.hostname==="localhost"||location.hostname==="127.0.0.1"){
    setInterval(()=>{fetch("/telemetry",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(courseSnapshot()),keepalive:true}).catch(()=>{});},900);
  }
  requestAnimationFrame(loop);
}
init().catch(err=>{
  console.error(err);
  setBadge("physicsBadge","init error",false);
});

window.__cartpoleSonicCourse={
  getState:courseSnapshot,
  navigate:navigateLesson,
  runLessonAction,
  step:stepPolicy,
  setGoal,
  runPPO,
};
