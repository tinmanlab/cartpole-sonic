import { MuJoCoCartPole } from "./mujoco_sim.js";
import {
  SonicToyTrainer,
  planReference,
  advancePlannerContext,
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

const MAX_SIGNAL_HISTORY = 220;
let signalHistory = [];
let controlTick = 0;
let episodeIndex = 0;
let autoResetCount = 0;
let lastEpisodeEvent = "initial";

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

function beginCanvas(canvas){
  const rect=canvas.getBoundingClientRect();
  const w=Math.max(1,rect.width),h=Math.max(1,rect.height);
  const dpr=Math.min(window.devicePixelRatio||1,2);
  const bw=Math.max(1,Math.round(w*dpr)),bh=Math.max(1,Math.round(h*dpr));
  if(canvas.width!==bw||canvas.height!==bh){
    canvas.width=bw;canvas.height=bh;
  }
  const ctx=canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  return{ctx,w,h,dpr};
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

function resetSignalHistory(reason="reset"){
  signalHistory=[];
  controlTick=0;
  lastEpisodeEvent=reason;
}
function recordControlSample(p,s){
  if(!p) return;
  signalHistory.push({
    t:controlTick*0.02,
    episode:episodeIndex,
    state:Array.from(s),
    reference:Array.from(p.ref),
    z:Array.from(p.z),
    q:Array.from(p.q),
    force:p.force,
  });
  controlTick++;
  if(signalHistory.length>MAX_SIGNAL_HISTORY) signalHistory.shift();
}
function terminationReason(s){
  if(!s.every(Number.isFinite)) return "non-finite state";
  if(Math.abs(s[0])>1.78) return "track limit";
  if(Math.abs(s[2])>.75) return "pole angle";
  return null;
}
function resetRobot(reason="manual reset"){
  if(!physicsReady) return;
  sim.reset({x:0,xDot:0,theta:.1,thetaDot:0});
  plannerContext=[0,0];
  lastForce=0;
  accumulator=0;
  episodeIndex++;
  resetSignalHistory(reason);
  render();
}
function autoResetEpisode(reason){
  autoResetCount++;
  resetRobot("auto-reset: "+reason);
}
function pushRobot(){
  if(!physicsReady) return;
  sim.applyImpulse({xDotDelta:.75,thetaDotDelta:-1.25});
  lastEpisodeEvent="external impulse · reference unchanged";
  render();
}
function stepPolicy(steps=1){
  if(!currentTrainer||!physicsReady) return;
  const n=clamp(Math.floor(steps)||1,1,100);
  for(let i=0;i<n;i++){
    const p=preview();
    const s0=state();
    recordControlSample(p,s0);
    sim.stepForce(p.force,2);
    plannerContext=advancePlannerContext(plannerContext,goal,.02);
    lastForce=p.force;
    const reason=terminationReason(state());
    if(reason){
      lastEpisodeEvent="terminated: "+reason;
      break;
    }
  }
  render();
}
function setGoal(x){
  goal=clamp(Number(x),-1.2,1.2);
  $("goal").value=goal;
  resetSignalHistory("goal changed");
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
    render();
    await new Promise(r=>requestAnimationFrame(r));
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
  resetSignalHistory("lesson changed");
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
  const c=$("cart"),{ctx,w:W,h:H}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);

  const centerX=W/2;
  const railY=H*.82,pivotY=railY-54,wheelY=railY-11;
  const worldHalf=1.8;
  const scale=Math.min((W-74)/(worldHalf*2),118);
  const cartW=76,cartH=28;

  ctx.strokeStyle="#cbd5e1";ctx.lineWidth=5;ctx.lineCap="round";
  ctx.beginPath();ctx.moveTo(34,railY);ctx.lineTo(W-34,railY);ctx.stroke();

  const s=state();
  drawActual(s[0],s[2]);

  const gx=centerX+goal*scale;
  ctx.strokeStyle="#16805d";ctx.lineWidth=2;ctx.setLineDash([5,4]);
  ctx.beginPath();ctx.moveTo(gx,28);ctx.lineTo(gx,railY);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#16805d";ctx.font="600 10px system-ui";ctx.textAlign="center";ctx.fillText("goal",gx,22);

  ctx.fillStyle="#667085";ctx.font="10px system-ui";ctx.textAlign="left";
  ctx.fillText("actual MuJoCo robot",10,16);

  function drawActual(x,theta){
    const cx=centerX+x*scale,L=Math.min(H*.48,126)*(sim.spec?.poleLength||1);
    const tx=cx+Math.sin(theta)*L,ty=pivotY-Math.cos(theta)*L;
    ctx.strokeStyle="#d64f4f";ctx.lineWidth=7;
    ctx.beginPath();ctx.moveTo(cx,pivotY);ctx.lineTo(tx,ty);ctx.stroke();
    ctx.fillStyle="#334155";ctx.fillRect(cx-cartW/2,pivotY,cartW,cartH);
    ctx.fillStyle="#1e293b";
    for(const dx of [-23,23]){ctx.beginPath();ctx.arc(cx+dx,wheelY,8,0,Math.PI*2);ctx.fill();}
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
  $("episodeStatus").textContent="ep "+episodeIndex+" · t="+fmt(controlTick*.02,2)+"s"+(autoResetCount?" · ↻"+autoResetCount:"");
  $("episodeStatus").title=lastEpisodeEvent;
}

function clearViz(){
  const c=$("lessonViz");
  $("vizHtml").classList.remove("visible");$("vizHtml").innerHTML="";
  $("lessonViz").style.display="block";
  const {ctx,w,h}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  $("vizMetrics").innerHTML="";
}
function setMetrics(items){
  $("vizMetrics").innerHTML=items.map(x=>'<span class="chip">'+x+'</span>').join("");
}
function drawAxes(ctx,w,h,{xmin=-1.25,xmax=1.25,ymin=-1.25,ymax=1.25}={}){
  const margin=42;
  const sx=(w-2*margin)/(xmax-xmin),sy=(h-2*margin)/(ymax-ymin);
  const scale=Math.min(sx,sy);
  const plotW=(xmax-xmin)*scale,plotH=(ymax-ymin)*scale;
  const left=(w-plotW)/2,top=(h-plotH)/2;
  const X=x=>left+(x-xmin)*scale;
  const Y=y=>top+plotH-(y-ymin)*scale;
  ctx.strokeStyle="#e2e6ec";ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(left,Y(0));ctx.lineTo(left+plotW,Y(0));ctx.moveTo(X(0),top);ctx.lineTo(X(0),top+plotH);ctx.stroke();
  return{X,Y,left,top,plotW,plotH,scale};
}

function reconstructionError(p,ref){
  if(!p) return NaN;
  let e=0;for(let i=0;i<ref.length;i++)e+=(p.kinRecon[i]-ref[i])**2;
  return e/ref.length;
}
function renderReconstructionViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview(),ref=currentReference();
  const pad={l:50,r:22,t:28,b:34},gap=28;
  const panelH=(h-pad.t-pad.b-gap)/2;
  const X=i=>pad.l+i/(SONIC_TOY_CONSTANTS.REF_FRAMES-1)*(w-pad.l-pad.r);
  const Y=(v,top)=>top+panelH/2-v*(panelH*.43);

  const drawPanel=(component,top,label)=>{
    ctx.strokeStyle="#eef0f3";ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.l,Y(0,top));ctx.lineTo(w-pad.r,Y(0,top));ctx.stroke();
    ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.textAlign="left";ctx.fillText(label,pad.l,top+12);

    const draw=(arr,color,dash=[])=>{
      ctx.strokeStyle=color;ctx.lineWidth=2.6;ctx.setLineDash(dash);ctx.beginPath();
      for(let i=0;i<SONIC_TOY_CONSTANTS.REF_FRAMES;i++){
        const x=X(i),y=Y(arr[i*2+component],top);
        i?ctx.lineTo(x,y):ctx.moveTo(x,y);
      }
      ctx.stroke();ctx.setLineDash([]);
    };
    draw(ref,"#315dc9");
    if(p) draw(p.kinRecon,"#795fc5",[7,5]);
  };

  drawPanel(0,pad.t,"normalized future x");
  drawPanel(1,pad.t+panelH+gap,"normalized future ẋ");

  ctx.fillStyle="#7b8492";ctx.font="9.5px ui-monospace";ctx.textAlign="center";
  for(let i=0;i<8;i++) ctx.fillText("+"+fmt((i+1)*.08,2)+"s",X(i),h-12);

  $("vizCaption").innerHTML='<b>Reference world · pre-FSQ.</b> 위 = x, 아래 = ẋ. <span style="color:#315dc9;font-weight:700">blue = planner future reference</span> · <span style="color:#795fc5;font-weight:700">purple dashed = Kinematic Decoder reconstruction</span>. 16D=[8×(x,ẋ)] 전체를 두 plot으로 표시한다.';
  if(p)setMetrics(["16D → 2D latent → 16D recon","z=["+fmt(p.z[0],3)+", "+fmt(p.z[1],3)+"]","all-16D MSE="+fmt(reconstructionError(p,ref),4)]);
}

function renderLatentViz(){
  const c=$("lessonViz"),canvas=beginCanvas(c),ctx=canvas.ctx,w=canvas.w,h=canvas.h;
  const p=preview();
  const hist=signalHistory.filter(x=>x.episode===episodeIndex);
  const allZ=[...hist.map(x=>x.z),p?.z].filter(Boolean);
  let maxAbs=1.05;
  for(const z of allZ) maxAbs=Math.max(maxAbs,Math.abs(z[0]),Math.abs(z[1]));
  if(lesson.mode==="vq"){
    for(let k=0;k<8;k++) maxAbs=Math.max(maxAbs,Math.abs(currentTrainer.policy.codebook[k*2]),Math.abs(currentTrainer.policy.codebook[k*2+1]));
  }
  const lim=Math.max(1.25,Math.min(2.5,maxAbs*1.15));
  const {X,Y}=drawAxes(ctx,w,h,{xmin:-lim,xmax:lim,ymin:-lim,ymax:lim});

  if(lesson.mode==="vq"){
    for(let k=0;k<8;k++){
      const x=currentTrainer.policy.codebook[k*2],y=currentTrainer.policy.codebook[k*2+1];
      ctx.fillStyle="#9aa3af";ctx.beginPath();ctx.arc(X(x),Y(y),6,0,Math.PI*2);ctx.fill();
      ctx.fillStyle="#596273";ctx.font="10px ui-monospace";ctx.fillText(String(k),X(x)+8,Y(y)-5);
    }
  }else{
    for(const a of [-1,-.5,0,.5,1])for(const b of [-1,-.5,0,.5,1]){
      ctx.fillStyle="#c7ccd4";ctx.beginPath();ctx.arc(X(a),Y(b),3.3,0,Math.PI*2);ctx.fill();
    }
  }

  // Only the trajectory actually produced during this episode.
  if(hist.length>1){
    ctx.strokeStyle="rgba(49,93,201,.45)";ctx.lineWidth=2;ctx.beginPath();
    hist.forEach((sample,i)=>{
      const x=X(sample.z[0]),y=Y(sample.z[1]);
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    });
    ctx.stroke();
    hist.forEach((sample,i)=>{
      const alpha=.16+.54*(i+1)/hist.length;
      ctx.fillStyle="rgba(49,93,201,"+alpha.toFixed(3)+")";
      ctx.beginPath();ctx.arc(X(sample.z[0]),Y(sample.z[1]),2.3,0,Math.PI*2);ctx.fill();
    });
  }

  if(p){
    const zx=X(p.z[0]),zy=Y(p.z[1]),qx=X(p.q[0]),qy=Y(p.q[1]);
    // Current quantization displacement only. This is not a trajectory.
    ctx.strokeStyle="#7b8492";ctx.lineWidth=1.5;ctx.setLineDash([5,4]);
    ctx.beginPath();ctx.moveTo(zx,zy);ctx.lineTo(qx,qy);ctx.stroke();ctx.setLineDash([]);
    const mx=(zx+qx)/2,my=(zy+qy)/2;
    ctx.fillStyle="#667085";ctx.font="10px system-ui";ctx.fillText("quantize",mx+5,my-5);

    ctx.fillStyle="#315dc9";ctx.beginPath();ctx.arc(zx,zy,7,0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#d64f4f";ctx.beginPath();ctx.arc(qx,qy,7,0,Math.PI*2);ctx.fill();
  }

  ctx.fillStyle="#667085";ctx.font="11px system-ui";
  ctx.fillText("z₁",w-48,Y(0)-8);ctx.fillText("z₂",X(0)+7,22);

  const used=new Set(hist.map(x=>x.q.map(v=>v.toFixed(2)).join(",")));
  if(p) used.add(p.q.map(v=>v.toFixed(2)).join(","));
  if(lesson.mode==="vq"){
    $("vizCaption").innerHTML='<b>Reference/token world.</b> 회색 큰 점 = learned VQ codebook · 파란 trail = <b>이번 실제 episode에서 시간순으로 발생한 Encoder z</b> · 짧은 점선은 현재 z→q quantization 이동량이다. 가상 goal sweep은 표시하지 않는다.';
    const refX=p?p.ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0]:NaN,actualX=state()[0];
    setMetrics(["same tick t="+fmt(controlTick*.02,2)+"s","ref x(+80ms)="+fmt(refX,3)+"m","actual x="+fmt(actualX,3)+"m","episode codes="+used.size+"/8",p?"q=["+fmt(p.q[0],2)+", "+fmt(p.q[1],2)+"]":""]);
  }else{
    $("vizCaption").innerHTML='<b>Reference/token world.</b> 회색 grid = 고정 FSQ finite levels · 파란 trail = <b>이번 실제 episode의 Encoder z history</b> · 빨간 q = 현재 FSQ 결과. 짧은 점선은 quantization 이동이며 robot trajectory가 아니다.';
    const refX=p?p.ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0]:NaN,actualX=state()[0];
    setMetrics(["same tick t="+fmt(controlTick*.02,2)+"s","ref x(+80ms)="+fmt(refX,3)+"m","actual x="+fmt(actualX,3)+"m","episode tokens="+used.size+"/25",p?"q=["+fmt(p.q[0],2)+", "+fmt(p.q[1],2)+"]":""]);
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
function renderLearningGraphViz(){
  showHtml(
    '<div style="height:100%;display:grid;grid-template-rows:auto 1fr;gap:12px">'+
      '<div class="flow-row" style="padding-top:8px">'+
        '<div class="flow-box"><strong>Reference</strong><span>training data / command</span></div>'+
        '<div class="flow-arrow">→</div>'+
        '<div class="flow-box accent"><strong>Encoder θE</strong><span>LEARNED</span></div>'+
        '<div class="flow-arrow">→</div>'+
        '<div class="flow-box"><strong>FSQ</strong><span>FIXED levels<br>STE backward</span></div>'+
        '<div class="flow-arrow">→</div>'+
        '<div class="flow-box accent"><strong>Dynamic Decoder θD</strong><span>LEARNED by PPO</span></div>'+
      '</div>'+
      '<div style="overflow:auto">'+
        '<table class="compare-table">'+
          '<thead><tr><th>Module</th><th>Trainable?</th><th>What teaches it?</th><th>What changes?</th></tr></thead>'+
          '<tbody>'+
            '<tr><td><b>Encoder(s)</b></td><td>YES</td><td>PPO + reconstruction + latent alignment</td><td>weights mapping modality/reference → latent z</td></tr>'+
            '<tr><td><b>FSQ</b></td><td><b>NO</b></td><td>no parameter loss</td><td>nothing; finite levels are fixed. STE only passes gradient through round</td></tr>'+
            '<tr><td><b>Dynamic Decoder</b></td><td>YES</td><td>PPO tracking objective</td><td>weights mapping token + proprioception → action</td></tr>'+
            '<tr><td><b>Kinematic Decoder</b></td><td>YES</td><td>future-motion reconstruction auxiliary loss</td><td>weights mapping token → future motion</td></tr>'+
            '<tr><td><b>VQ codebook</b> (comparison)</td><td>YES</td><td>codebook/EMA style update</td><td>representative vectors move — this is what FSQ removes</td></tr>'+
          '</tbody>'+
        '</table>'+
      '</div>'+
    '</div>',
    '<b>핵심:</b> FSQ는 training graph 안에 있지만 자체 codebook/level parameter를 배우지 않는다. PPO·aux loss의 gradient가 STE를 통해 Encoder로 지나가고, Encoder/Decoder가 고정된 discrete grid를 잘 사용하도록 학습된다.',
    [
      "toy: 2 scalar dims × 5 fixed levels",
      "SONIC release: 32 scalar dims × 32 fixed levels × 2 tokens",
      "flattened motion token = 64 values",
      "g1_recon coef 0.01 · latent-alignment coefs 1.0",
      "Critic: learned separately by value loss"
    ]
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
    '<b>Push는 actual robot에만 impulse를 주고 planner/reference는 진행시키지 않는다.</b> 따라서 motion token q는 그대로이고 proprioception만 바뀐다. 그 결과 Dynamic Decoder action이 달라지는지 확인한다.',
    [p?"token="+token:"","state θ="+fmt(s[2]*180/Math.PI,1)+"°","action="+force]
  );
}
function renderTrainingViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  const pad={l:62,r:28,t:32,b:50};
  ctx.strokeStyle="#e2e6ec";ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();

  const ref=(ppoReferenceEvidence?.checkpoints||[])
    .map(x=>({iterations:x.ppoIterations,mae:x.eval?.trackingMae}))
    .filter(x=>Number.isFinite(x.mae));
  const rolloutPts=(currentTrainer.history||[])
    .map(x=>({iterations:x.iter,mae:x.tracking}))
    .filter(x=>Number.isFinite(x.mae));
  const heldoutPts=ppoHeldoutHistory.filter(x=>Number.isFinite(x.mae));
  const all=[...ref,...rolloutPts,...heldoutPts];
  const maxIter=Math.max(50,...all.map(x=>x.iterations));
  const maxMae=Math.max(.25,...all.map(x=>x.mae));
  const minMae=Math.min(.10,...all.map(x=>x.mae));
  const X=i=>pad.l+i/maxIter*(w-pad.l-pad.r);
  const Y=v=>h-pad.b-(v-minMae)/(maxMae-minMae)*(h-pad.t-pad.b);

  for(let i=0;i<=5;i++){
    const v=minMae+(maxMae-minMae)*i/5,y=Y(v);
    ctx.strokeStyle="#f0f2f5";ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
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
  drawSeries(rolloutPts,"#315dc9",[],2.5);
  heldoutPts.forEach(p=>{ctx.fillStyle="#16805d";ctx.beginPath();ctx.arc(X(p.iterations),Y(p.mae),5,0,Math.PI*2);ctx.fill();});

  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.textAlign="left";
  ctx.fillText("tracking MAE [m] ↓",pad.l,18);
  ctx.textAlign="right";ctx.fillText("PPO iterations →",w-pad.r,h-15);

  $("vizCaption").innerHTML='회색 점선 = deterministic held-out reference · <span style="color:#315dc9;font-weight:700">파랑 = 매 PPO iteration의 rollout tracking MAE (실시간)</span> · <span style="color:#16805d;font-weight:700">초록 = 현재 세션 held-out check</span>. 낮을수록 좋다.';
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
  const worldLabel={
    reference:"Reference world · pre-FSQ motion",
    concept:"Concept branch · no robot-state mixing",
    "reference-token":"Reference / token world · before and after quantization",
    bridge:"Bridge · token + actual robot proprioception",
    training:"Training evidence · physical tracking",
    mapping:"Role mapping · CartPole ↔ GEAR-SONIC",
  }[lesson.world]||"Main visualization";
  $("vizSub").textContent=worldLabel+" · same position/size every lesson";
  $("vizStatus").textContent=lesson.optional?"optional branch":"step "+lesson.step;
  switch(lesson.viz){
    case "reconstruction": renderReconstructionViz(); break;
    case "vae-branch": renderVaeViz(); break;
    case "latent": renderLatentViz(); break;
    case "vqvae": renderVqvaeViz(); break;
    case "learning-graph": renderLearningGraphViz(); break;
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
  $("lessonAction").textContent=lesson.id==="ppo"?"+10 PPO iterations":lesson.id==="dynamic-decoder"?"Push + 1 Step":lesson.id==="motion-token"?"Push robot":lesson.id==="learning-graph"?"Run 1 PPO iter":lesson.id==="sonic"?"Run live tracking":"Change goal";
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
    case "learning-graph":
      await runPPO(1);
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
    experiment:{goal,live,preset,mode:lesson.mode,episode:episodeIndex,autoResets:autoResetCount,lastEpisodeEvent},
    modelSemantics:{
      encoder:{trainable:true,signals:["PPO","reconstruction auxiliary","cross-encoder latent alignment"]},
      fsq:{trainable:false,levels:"fixed",gradient:"STE through rounding; no learned vector codebook"},
      dynamicDecoder:{trainable:true,signals:["PPO tracking objective"]},
      kinematicDecoder:{trainable:true,signals:["future-motion reconstruction auxiliary loss"]},
      critic:{trainable:true,signals:["value/return loss"]},
      releasedTokenShape:{numTokens:2,scalarDimsPerToken:32,fixedLevelsPerScalar:32,flattenedDim:64},
    },
    signals:{
      semantics:{
        reference:"pre-FSQ desired future motion from planner/reference source",
        latent:"continuous Encoder output before quantization",
        token:"post-FSQ/VQ compact motion representation; not a motor command",
        proprioception:"measured actual robot state; separate from reference",
        action:"Dynamic Decoder output applied to the actual robot",
      },
      reference:Array.from(ref),
      latent:p?.z||null,
      token:p?.q||null,
      proprioception:s,
      actionMean:p?.mu??null,
      force:p?.force??null,
      kinematicReconstruction:p?.kinRecon||null,
      liveHistory:signalHistory.slice(-80).map(x=>({t:x.t,episode:x.episode,z:x.z,q:x.q,force:x.force,state:x.state})),
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

function installCanvasResizeObserver(){
  if(!("ResizeObserver" in window))return;
  const ro=new ResizeObserver(()=>{if(physicsReady&&currentTrainer)render();});
  ro.observe($("cart"));
  ro.observe($("lessonViz"));
  window.__cartpoleSonicResizeObserver=ro;
}

function loop(now){
  const dt=Math.min(.05,Math.max(0,(now-lastFrame)/1000));lastFrame=now;
  if(live&&!busy&&currentTrainer&&physicsReady){
    accumulator+=dt;
    let n=0;
    while(accumulator>=.02&&n<4){
      const p=preview();
      const s0=state();
      recordControlSample(p,s0);
      sim.stepForce(p.force,2);
      plannerContext=advancePlannerContext(plannerContext,goal,.02);
      lastForce=p.force;
      accumulator-=.02;n++;
      const reason=terminationReason(state());
      if(reason){
        autoResetEpisode(reason);
        break;
      }
    }
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
  installCanvasResizeObserver();
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
