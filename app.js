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
  SONIC_FLOW,
  TRAINING_TOPICS,
  RUNTIME_DETAILS,
  TRAINING_DETAILS,
  CONCEPT_TEXT,
  getNode,
  getTrainingTopic,
  getSystemOutline,
} from "./course.js";

const $ = id => document.getElementById(id);
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const fmt = (v,n=3) => Number.isFinite(v) ? Number(v).toFixed(n) : "—";

const query = new URLSearchParams(location.search);
const LEGACY = {
  ae:{focus:"encoder",concept:"ae"},
  vae:{focus:"encoder",concept:"vae"},
  vq:{focus:"quantizer",concept:"vq"},
  vqvae:{focus:"quantizer",concept:"vqvae"},
  fsq:{focus:"quantizer",concept:"fsq"},
  "learning-graph":{training:"what-learns"},
  "motion-token":{focus:"token"},
  "dynamic-decoder":{focus:"control-decoder"},
  ppo:{training:"ppo"},
  sonic:{focus:"robot"},
};

const legacy = query.get("lesson") ? LEGACY[query.get("lesson")] : null;
let trainingMode = Boolean(query.get("training") || legacy?.training);
let focusId = query.get("focus") || legacy?.focus || "task";
let conceptId = query.get("concept") || legacy?.concept || null;
let trainingTopicId = query.get("training") || legacy?.training || "loss-flow";
let guideDepth = ["easy","mechanism","sonic"].includes(query.get("depth")) ? query.get("depth") : "easy";
let focus = getNode(focusId);
let trainingTopic = getTrainingTopic(trainingTopicId);
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
const MAX_SIGNAL_HISTORY = 260;
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
  if(canvas.width!==bw||canvas.height!==bh){canvas.width=bw;canvas.height=bh;}
  const ctx=canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  return {ctx,w,h,dpr};
}
function requiredMode(){
  if(trainingMode) return "fsq";
  if(focus.id==="encoder" && ["ae","vae"].includes(conceptId)) return "ae";
  if(focus.id==="quantizer" && ["vq","vqvae"].includes(conceptId)) return "vq";
  return "fsq";
}
async function getTrainer(mode){
  if(trainerCache.has(mode)) return trainerCache.get(mode);
  const t=new SonicToyTrainer(sim,{seed:20261003,mode,n:8,horizon:96,epochs:4,batch:128});
  let restored=false;
  try{
    const ckpt=await fetch("./assets/student_"+mode+"_bootstrap.json",{cache:"force-cache"}).then(r=>r.ok?r.json():Promise.reject(new Error("checkpoint "+r.status)));
    if(ckpt.schema!=="cartpole-sonic-student-bootstrap/v1"||ckpt.mode!==mode) throw new Error("checkpoint schema/mode mismatch");
    t.restorePolicy(ckpt.policy,{teacherBootstrap:ckpt.teacherBootstrap,rngState:ckpt.rng});
    t.bootstrapEval=ckpt.defaultPresetEval||null;
    restored=true;
  }catch(err){
    console.warn("student checkpoint unavailable",err);
  }
  if(!restored){
    if(!teacher&&teacherPromise) teacher=await teacherPromise;
    if(teacher) t.bootstrapFromTeacher(teacher,{steps:500,batch:256,lr:.0015,auxCoef:.08});
  }
  trainerCache.set(mode,t);
  return t;
}
function clearTrainers(){
  for(const t of trainerCache.values()) t.delete?.();
  trainerCache.clear();
  currentTrainer=null;
  heldoutEval=null;
  ppoHeldoutHistory=[];
}
async function ensureTrainer(){
  currentTrainer=await getTrainer(requiredMode());
  if(trainingMode&&trainingTopic.id==="ppo"&&!heldoutEval){
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
    t:controlTick*.02,episode:episodeIndex,
    state:Array.from(s),reference:Array.from(p.ref),z:Array.from(p.z),q:Array.from(p.q),force:p.force,
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
  lastForce=0;accumulator=0;episodeIndex++;
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
    const p=preview(),s0=state();
    recordControlSample(p,s0);
    sim.stepForce(p.force,2);
    plannerContext=advancePlannerContext(plannerContext,goal,.02);
    lastForce=p.force;
    const reason=terminationReason(state());
    if(reason){lastEpisodeEvent="terminated: "+reason;break;}
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
  live=false;busy=true;
  clearTrainers();preset=next;
  await sim.loadPreset(preset);
  resetRobot();
  currentTrainer=await getTrainer(requiredMode());
  busy=false;$("simPreset").value=preset;
  updateUrl();render();
}
async function runPPO(iterations){
  if(!currentTrainer||busy) return;
  busy=true;live=false;render();
  const n=clamp(Math.floor(iterations)||1,1,50);
  for(let i=0;i<n;i++){
    currentTrainer.iteration();
    render();
    await new Promise(r=>requestAnimationFrame(r));
  }
  heldoutEval=currentTrainer.evaluate(12);
  ppoHeldoutHistory.push({iterations:currentTrainer.iter,mae:heldoutEval.trackingMae});
  busy=false;render();
}

function normalizeConcept(){
  const ids=(focus.concepts||[]).map(x=>x.id);
  if(!ids.length){conceptId=null;return;}
  if(!conceptId||!ids.includes(conceptId)) conceptId=focus.id==="quantizer"?"fsq":"core";
}
function updateUrl(){
  const u=new URL(location.href);
  u.searchParams.delete("lesson");
  if(trainingMode){
    u.searchParams.delete("focus");u.searchParams.delete("concept");
    u.searchParams.set("training",trainingTopic.id);
  }else{
    u.searchParams.delete("training");
    u.searchParams.set("focus",focus.id);
    if(conceptId&&conceptId!=="core")u.searchParams.set("concept",conceptId);else u.searchParams.delete("concept");
  }
  if(guideDepth!=="easy")u.searchParams.set("depth",guideDepth);else u.searchParams.delete("depth");
  u.searchParams.set("goal",goal.toFixed(2));
  if(preset!=="playground")u.searchParams.set("preset",preset);else u.searchParams.delete("preset");
  history.replaceState(null,"",u);
}
async function switchTrainerIfNeeded(){
  const mode=requiredMode();
  if(currentTrainer?.mode===mode) return;
  currentTrainer=await getTrainer(mode);
  resetSignalHistory("representation changed");
}
async function focusNode(id,concept=null){
  if(busy) return;
  const next=getNode(id);
  trainingMode=false;focus=next;focusId=next.id;
  conceptId=concept;normalizeConcept();
  await switchTrainerIfNeeded();
  updateUrl();render();
}
async function setConcept(id){
  if(!(focus.concepts||[]).some(x=>x.id===id)||busy)return;
  conceptId=id;
  await switchTrainerIfNeeded();
  updateUrl();render();
}
async function openTraining(topicId="loss-flow"){
  if(busy)return;
  trainingMode=true;
  trainingTopic=getTrainingTopic(topicId);trainingTopicId=trainingTopic.id;
  await switchTrainerIfNeeded();
  if(trainingTopic.id==="ppo"&&!heldoutEval){
    heldoutEval=currentTrainer.evaluate(12);
    ppoHeldoutHistory=[{iterations:currentTrainer.iter,mae:heldoutEval.trackingMae}];
  }
  updateUrl();render();
}

function buildSystemMap(){
  const host=$("runtimeFlow");host.innerHTML="";
  const topIds=["task","generator","reference","encoder","quantizer","token","control-decoder","robot"];
  const trainingHits=trainingMode?(TRAINING_DETAILS[trainingTopic.id]?.highlights||[]):[];
  topIds.forEach((id,i)=>{
    const n=getNode(id);
    const b=document.createElement("button");b.className="map-node";
    b.classList.toggle("active",!trainingMode&&focus.id===id);
    b.classList.toggle("training-hit",trainingMode&&trainingHits.includes(id));
    b.innerHTML='<b>'+n.nav+'</b><span class="official">'+n.official+'</span><span class="toy">toy: '+n.toy+'</span>';
    b.onclick=()=>{void focusNode(id);};host.appendChild(b);
    if(i<topIds.length-1){const a=document.createElement("span");a.className="map-arrow";a.textContent="→";host.appendChild(a);}
  });
  const motion=getNode("motion-decoder");
  const mb=$("motionBranch");
  mb.classList.toggle("active",!trainingMode&&focus.id==="motion-decoder");
  mb.classList.toggle("training-hit",trainingMode&&trainingHits.includes("motion-decoder"));
  mb.innerHTML='<b>'+motion.nav+'</b><span>official: '+motion.official+' · toy: '+motion.toy+'</span>';
  mb.onclick=()=>{void focusNode("motion-decoder");};
  $("trainingButton").classList.toggle("active",trainingMode);
  $("trainingButton").onclick=()=>{void openTraining(trainingTopicId);};
}
function buildConceptTabs(){
  const host=$("conceptTabs");host.innerHTML="";
  const items=trainingMode?TRAINING_TOPICS:(focus.concepts||[]);
  for(const item of items){
    const b=document.createElement("button");
    const id=trainingMode?item.id:item.id;
    b.textContent=trainingMode?item.label:item.label;
    b.classList.toggle("active",trainingMode?trainingTopic.id===id:conceptId===id);
    b.onclick=()=>{trainingMode?void openTraining(id):void setConcept(id);};
    host.appendChild(b);
  }
}

function renderSimulation(){
  const c=$("cart"),{ctx,w:W,h:H}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);
  const centerX=W/2,railY=H*.82,pivotY=railY-54,wheelY=railY-11;
  const scale=Math.min((W-74)/(1.8*2),118),cartW=76,cartH=28;
  ctx.strokeStyle="#cbd5e1";ctx.lineWidth=5;ctx.lineCap="round";
  ctx.beginPath();ctx.moveTo(34,railY);ctx.lineTo(W-34,railY);ctx.stroke();
  const s=state();drawActual(s[0],s[2]);
  const gx=centerX+goal*scale;
  ctx.strokeStyle="#16805d";ctx.lineWidth=2;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(gx,28);ctx.lineTo(gx,railY);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#16805d";ctx.font="600 10px system-ui";ctx.textAlign="center";ctx.fillText("goal",gx,22);
  ctx.fillStyle="#667085";ctx.font="10px system-ui";ctx.textAlign="left";ctx.fillText("actual MuJoCo robot",10,16);
  function drawActual(x,theta){
    const cx=centerX+x*scale,L=Math.min(H*.48,126)*(sim.spec?.poleLength||1);
    const tx=cx+Math.sin(theta)*L,ty=pivotY-Math.cos(theta)*L;
    ctx.strokeStyle="#d64f4f";ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(cx,pivotY);ctx.lineTo(tx,ty);ctx.stroke();
    ctx.fillStyle="#334155";ctx.fillRect(cx-cartW/2,pivotY,cartW,cartH);ctx.fillStyle="#1e293b";
    for(const dx of [-23,23]){ctx.beginPath();ctx.arc(cx+dx,wheelY,8,0,Math.PI*2);ctx.fill();}
  }
  $("goal").value=goal;$("goalVal").textContent=(goal>=0?"+":"")+fmt(goal,2)+" m";
  $("liveBtn").textContent="Live "+(live?"ON":"OFF");$("liveBtn").classList.toggle("active",live);
  $("vX").textContent=fmt(s[0],3)+" m";$("vXd").textContent=fmt(s[1],3);
  $("vTh").textContent=fmt(s[2]*180/Math.PI,1)+"°";$("vThd").textContent=fmt(s[3]*180/Math.PI,1);
  $("vForce").textContent=fmt(lastForce,2)+" N";
  $("episodeStatus").textContent="ep "+episodeIndex+" · t="+fmt(controlTick*.02,2)+"s"+(autoResetCount?" · ↻"+autoResetCount:"");
  $("episodeStatus").title=lastEpisodeEvent;
}

function clearViz(){
  $("vizHtml").classList.remove("visible");$("vizHtml").innerHTML="";$("lessonViz").style.display="block";
  const {ctx,w,h}=beginCanvas($("lessonViz"));ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  $("vizMetrics").innerHTML="";
}
function showHtml(html,caption,metrics=[]){
  $("lessonViz").style.display="none";$("vizHtml").innerHTML=html;$("vizHtml").classList.add("visible");
  $("vizCaption").innerHTML=caption;setMetrics(metrics);
}
function setMetrics(items){$("vizMetrics").innerHTML=items.filter(Boolean).map(x=>'<span class="chip">'+x+'</span>').join("");}
function drawAxes(ctx,w,h,{xmin=-1.25,xmax=1.25,ymin=-1.25,ymax=1.25}={}){
  const margin=42,sx=(w-2*margin)/(xmax-xmin),sy=(h-2*margin)/(ymax-ymin),scale=Math.min(sx,sy);
  const plotW=(xmax-xmin)*scale,plotH=(ymax-ymin)*scale,left=(w-plotW)/2,top=(h-plotH)/2;
  const X=x=>left+(x-xmin)*scale,Y=y=>top+plotH-(y-ymin)*scale;
  ctx.strokeStyle="#e2e6ec";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,Y(0));ctx.lineTo(left+plotW,Y(0));ctx.moveTo(X(0),top);ctx.lineTo(X(0),top+plotH);ctx.stroke();
  return{X,Y,left,top,plotW,plotH,scale};
}
function reconstructionError(p,ref){
  if(!p)return NaN;let e=0;for(let i=0;i<ref.length;i++)e+=(p.kinRecon[i]-ref[i])**2;return e/ref.length;
}

function renderTaskViz(){
  showHtml(
    '<div class="modality-grid">'+
      '<div class="modality"><b>Interactive control</b><span>gamepad / high-level command</span></div>'+
      '<div class="modality"><b>VR / teleoperation</b><span>3-point or whole-body targets</span></div>'+
      '<div class="modality"><b>Video / VLA / multimodal</b><span>human or model-generated intent</span></div>'+
    '</div>',
    '공식 SONIC은 여러 task/interface를 같은 control policy 앞단으로 연결한다. CartPole toy에서는 이를 하나의 high-level goal x*로 축소한다.',
    ["toy goal="+fmt(goal,2)+" m","CONCEPT: official input modalities"]
  );
}
function renderReferenceCurveViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),ref=currentReference();
  const pad={l:54,r:22,t:28,b:38},gap=28,panelH=(h-pad.t-pad.b-gap)/2;
  const X=i=>pad.l+i/(SONIC_TOY_CONSTANTS.REF_FRAMES-1)*(w-pad.l-pad.r);
  const Y=(v,top)=>top+panelH/2-v*(panelH*.42);
  const labels=["future x [normalized]","future ẋ [normalized]"];
  for(let comp=0;comp<2;comp++){
    const top=pad.t+comp*(panelH+gap);
    ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,Y(0,top));ctx.lineTo(w-pad.r,Y(0,top));ctx.stroke();
    ctx.strokeStyle="#315dc9";ctx.lineWidth=2.7;ctx.beginPath();
    for(let i=0;i<8;i++){const x=X(i),y=Y(ref[i*2+comp],top);i?ctx.lineTo(x,y):ctx.moveTo(x,y);ctx.fillStyle="#315dc9";ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();}
    ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.textAlign="left";ctx.fillText(labels[comp],pad.l,top+12);
  }
  ctx.fillStyle="#7b8492";ctx.font="9.5px ui-monospace";ctx.textAlign="center";for(let i=0;i<8;i++)ctx.fillText("+"+fmt((i+1)*.08,2)+"s",X(i),h-12);
  $("vizCaption").innerHTML='<b>Reference world.</b> 이것은 actual robot trajectory가 아니라 planner가 만든 미래 target이다. Encoder는 이 시간축 reference를 읽는다.';
  setMetrics(["goal="+fmt(goal,2)+"m","planner context x="+fmt(plannerContext[0],3)+"m","8 future frames"]);
}
function renderReferenceVectorViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),ref=currentReference();
  const pad={l:32,r:20,t:28,b:50},n=ref.length,barW=(w-pad.l-pad.r)/n*.72;
  const Y0=h*.52,scale=(h-pad.t-pad.b)*.38;
  ctx.strokeStyle="#e3e6eb";ctx.beginPath();ctx.moveTo(pad.l,Y0);ctx.lineTo(w-pad.r,Y0);ctx.stroke();
  for(let i=0;i<n;i++){
    const x=pad.l+(i+.5)*(w-pad.l-pad.r)/n,v=ref[i],bh=Math.abs(v)*scale;
    ctx.fillStyle=i%2===0?"#315dc9":"#795fc5";ctx.fillRect(x-barW/2,v>=0?Y0-bh:Y0,barW,bh);
    ctx.fillStyle="#667085";ctx.font="8.5px ui-monospace";ctx.textAlign="center";ctx.fillText(i%2===0?"x":"ẋ",x,h-25);
    ctx.fillText(String(Math.floor(i/2)+1),x,h-10);
  }
  $("vizCaption").innerHTML='Encoder 입력을 실제 16개 숫자로 펼친 모습. frame 1~8마다 <b>x와 ẋ</b>가 한 쌍이며 actual proprioception은 여기에 포함되지 않는다.';
  setMetrics(["input dimension=16","8 frames × 2 features","pre-FSQ"]);
}
function renderEncoderCoreViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview(),ref=currentReference();
  const split=w*.58,pad=32,Y0=h*.54,scale=h*.31,n=ref.length,barW=(split-2*pad)/n*.7;
  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.fillText("16D future reference",pad,20);
  ctx.strokeStyle="#e3e6eb";ctx.beginPath();ctx.moveTo(pad,Y0);ctx.lineTo(split-pad,Y0);ctx.stroke();
  for(let i=0;i<n;i++){
    const x=pad+(i+.5)*(split-2*pad)/n,v=ref[i],bh=Math.abs(v)*scale;
    ctx.fillStyle=i%2===0?"#315dc9":"#795fc5";ctx.fillRect(x-barW/2,v>=0?Y0-bh:Y0,barW,bh);
  }
  ctx.strokeStyle="#d7dce4";ctx.beginPath();ctx.moveTo(split,25);ctx.lineTo(split,h-25);ctx.stroke();
  const box={x:split+38,y:80,w:w-split-76,h:h-160};
  ctx.strokeStyle="#dfe4ea";ctx.strokeRect(box.x,box.y,box.w,box.h);
  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.fillText("2D latent z",box.x,box.y-12);
  if(p){
    const lim=1.5,X=v=>box.x+(v+lim)/(2*lim)*box.w,Y=v=>box.y+box.h-(v+lim)/(2*lim)*box.h;
    ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(box.x,Y(0));ctx.lineTo(box.x+box.w,Y(0));ctx.moveTo(X(0),box.y);ctx.lineTo(X(0),box.y+box.h);ctx.stroke();
    ctx.fillStyle="#315dc9";ctx.beginPath();ctx.arc(X(p.z[0]),Y(p.z[1]),9,0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#172033";ctx.font="11px ui-monospace";ctx.fillText("z=["+fmt(p.z[0],2)+", "+fmt(p.z[1],2)+"]",box.x+8,box.y+18);
  }
  ctx.fillStyle="#9aa3af";ctx.font="22px system-ui";ctx.fillText("→",split-10,h/2);
  $("vizCaption").innerHTML='Encoder는 16D future reference를 2D latent로 압축한다. 이 toy의 2D는 이해를 위한 축소이며 실제 SONIC release token scalar dimension은 32/token이다.';
  setMetrics(["16D → 2D","trainable Encoder","not robot-state observer"]);
}
function renderAutoencoderViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview(),ref=currentReference();
  const pad={l:50,r:22,t:28,b:34},gap=28,panelH=(h-pad.t-pad.b-gap)/2;
  const X=i=>pad.l+i/7*(w-pad.l-pad.r),Y=(v,top)=>top+panelH/2-v*(panelH*.43);
  const drawPanel=(comp,top,label)=>{
    ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,Y(0,top));ctx.lineTo(w-pad.r,Y(0,top));ctx.stroke();
    const draw=(arr,color,dash=[])=>{ctx.strokeStyle=color;ctx.lineWidth=2.6;ctx.setLineDash(dash);ctx.beginPath();for(let i=0;i<8;i++){const x=X(i),y=Y(arr[i*2+comp],top);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();ctx.setLineDash([]);};
    draw(ref,"#315dc9");if(p)draw(p.kinRecon,"#795fc5",[7,5]);
    ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.fillText(label,pad.l,top+12);
  };
  drawPanel(0,pad.t,"future x");drawPanel(1,pad.t+panelH+gap,"future ẋ");
  $("vizCaption").innerHTML='<b>Autoencoder idea.</b> blue=input reference, purple=reconstruction. Decoder는 deployment 원본을 다시 쓰기 위해서가 아니라 작은 latent가 motion 정보를 보존하도록 학습 신호를 주기 위해 존재한다.';
  setMetrics(["16D → z(2D) → 16D","recon MSE="+fmt(reconstructionError(p,ref),4),"background concept for SONIC auxiliary decoder"]);
}
function renderVaeViz(){
  showHtml(
    '<div style="height:100%;display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:stretch">'+
      '<div style="display:grid;gap:10px;align-content:center">'+
        '<div class="flow-box accent"><strong>Deterministic Autoencoder</strong><span>input → Encoder → one z → Decoder → reconstruction</span></div>'+
        '<div class="flow-box purple"><strong>VAE · optional background</strong><span>input → Encoder → μ, σ → sample z<br>+ KL regularization</span></div>'+
        '<div class="flow-box"><strong>VQ / VQ-VAE</strong><span>input → Encoder → nearest discrete code q<br>direct conceptual path toward FSQ</span></div>'+
      '</div>'+
      '<div style="display:grid;align-content:center;gap:10px">'+
        '<table class="compare-table"><thead><tr><th></th><th>AE</th><th>VAE</th><th>VQ-VAE</th></tr></thead><tbody>'+
          '<tr><td>latent</td><td>continuous z</td><td>probabilistic z</td><td>discrete q</td></tr>'+
          '<tr><td>special idea</td><td>bottleneck</td><td>μ,σ + sampling + KL</td><td>codebook + STE</td></tr>'+
          '<tr><td>needed for SONIC FSQ?</td><td>useful foundation</td><td><b>optional</b></td><td><b>direct predecessor</b></td></tr>'+
        '</tbody></table>'+
        '<div class="concept-note"><b>Why is VAE shown at all?</b> 이름에 “VAE”가 들어간 VQ-VAE 때문에 계보를 혼동하기 쉽다. 하지만 FSQ를 이해하기 위한 본선은 AE → VQ → VQ-VAE → FSQ다.</div>'+
      '</div>'+
    '</div>',
    '<b>CONCEPT · not toy runtime data.</b> VAE는 continuous probabilistic latent를 배우는 별도 분기다. SONIC이 VAE를 runtime block으로 쓰는 것이 아니며, FSQ를 이해하기 위해 μ/σ/KL을 깊게 선행할 필요도 없다.',
    ["VAE = optional","VQ-VAE = direct FSQ predecessor","no fake VAE runtime data"]
  );
}

function renderLatentViz(mode){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview();
  const hist=signalHistory.filter(x=>x.episode===episodeIndex),allZ=[...hist.map(x=>x.z),p?.z].filter(Boolean);
  let maxAbs=1.05;for(const z of allZ)maxAbs=Math.max(maxAbs,Math.abs(z[0]),Math.abs(z[1]));
  if(mode==="vq")for(let k=0;k<8;k++)maxAbs=Math.max(maxAbs,Math.abs(currentTrainer.policy.codebook[k*2]),Math.abs(currentTrainer.policy.codebook[k*2+1]));
  const lim=Math.max(1.25,Math.min(2.5,maxAbs*1.15)),{X,Y}=drawAxes(ctx,w,h,{xmin:-lim,xmax:lim,ymin:-lim,ymax:lim});
  if(mode==="vq"){
    for(let k=0;k<8;k++){const x=currentTrainer.policy.codebook[k*2],y=currentTrainer.policy.codebook[k*2+1];ctx.fillStyle="#9aa3af";ctx.beginPath();ctx.arc(X(x),Y(y),6,0,Math.PI*2);ctx.fill();ctx.fillStyle="#596273";ctx.font="10px ui-monospace";ctx.fillText(String(k),X(x)+8,Y(y)-5);}
  }else{
    for(const a of [-1,-.5,0,.5,1])for(const b of [-1,-.5,0,.5,1]){ctx.fillStyle="#c7ccd4";ctx.beginPath();ctx.arc(X(a),Y(b),3.3,0,Math.PI*2);ctx.fill();}
  }
  if(hist.length>1){
    ctx.strokeStyle="rgba(49,93,201,.45)";ctx.lineWidth=2;ctx.beginPath();
    hist.forEach((s,i)=>{const x=X(s.z[0]),y=Y(s.z[1]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  }
  if(p){
    const zx=X(p.z[0]),zy=Y(p.z[1]),qx=X(p.q[0]),qy=Y(p.q[1]);
    ctx.strokeStyle="#7b8492";ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(zx,zy);ctx.lineTo(qx,qy);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle="#315dc9";ctx.beginPath();ctx.arc(zx,zy,8,0,Math.PI*2);ctx.fill();ctx.fillStyle="#d64f4f";ctx.beginPath();ctx.arc(qx,qy,8,0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#667085";ctx.font="10px system-ui";ctx.fillText("quantize",(zx+qx)/2+5,(zy+qy)/2-5);
  }
  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.fillText("z₁",w-48,Y(0)-8);ctx.fillText("z₂",X(0)+7,22);
  const used=new Set(hist.map(x=>x.q.map(v=>v.toFixed(2)).join(",")));if(p)used.add(p.q.map(v=>v.toFixed(2)).join(","));
  const refX=p?p.ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0]:NaN;
  if(mode==="vq"){
    $("vizCaption").innerHTML='<b>LIVE VQ.</b> 회색 큰 점=learned codebook, 파란 trail=actual episode z history, 빨간 q=nearest learned vector. 짧은 점선은 현재 quantization 이동이다.';
    setMetrics(["codebook=8 learned vectors","episode codes="+used.size+"/8","ref x(+80ms)="+fmt(refX,3)+"m"]);
  }else{
    $("vizCaption").innerHTML='<b>LIVE FSQ.</b> 회색 grid=fixed finite levels, 파란 trail=actual episode z history, 빨간 q=현재 FSQ output. FSQ grid는 학습되어 움직이지 않는다.';
    setMetrics(["toy L=[5,5]","episode tokens="+used.size+"/25","q=["+fmt(p?.q[0],2)+","+fmt(p?.q[1],2)+"]"]);
  }
}
function renderVqvaeViz(){
  const p=preview(),ref=currentReference(),recon=fmt(reconstructionError(p,ref),4);
  const code=p?.qIndex??"—";
  showHtml(
    '<div style="height:100%;display:grid;grid-template-rows:1fr auto;gap:12px">'+
      '<div class="flow-row">'+
        '<div class="flow-box"><strong>Reference</strong><span>16D future motion</span></div><div class="flow-arrow">→</div>'+
        '<div class="flow-box"><strong>Encoder</strong><span>live z=['+fmt(p?.z[0],2)+', '+fmt(p?.z[1],2)+']</span></div><div class="flow-arrow">→</div>'+
        '<div class="flow-box accent"><strong>Nearest VQ code</strong><span>index '+code+'<br>q=['+fmt(p?.q[0],2)+', '+fmt(p?.q[1],2)+']</span></div><div class="flow-arrow">→</div>'+
        '<div class="flow-box purple"><strong>Decoder</strong><span>live recon MSE '+recon+'</span></div>'+
      '</div>'+
      '<table class="compare-table"><thead><tr><th>Training problem</th><th>Why it exists</th><th>Mechanism</th></tr></thead><tbody>'+
        '<tr><td>nearest lookup is non-differentiable</td><td>Encoder still needs gradient</td><td><b>STE</b>: backward treats quantization approximately like identity</td></tr>'+
        '<tr><td>Encoder can drift far from selected code</td><td>z should commit to usable codes</td><td><b>commitment loss</b></td></tr>'+
        '<tr><td>codebook must represent data</td><td>representative vectors need to move</td><td><b>codebook / EMA-style update</b></td></tr>'+
        '<tr><td>some codes may never be selected</td><td>capacity is wasted</td><td>dead-code / collapse management</td></tr>'+
      '</tbody></table>'+
    '</div>',
    '<b>LIVE values + training explanation.</b> 현재 z/q/reconstruction 값은 실제 toy model에서 오지만, Live 버튼은 reference만 움직이고 weight를 재학습하지 않는다. 아래 표가 VQ-VAE training 시 필요한 STE · commitment · codebook update · dead-code 문제를 설명한다. FSQ는 이 learned-codebook machinery를 제거한다.',
    ["current code="+code,"recon MSE="+recon,"STE","commitment","codebook update"]
  );
}
function renderTokenViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview(),hist=signalHistory.filter(x=>x.episode===episodeIndex);
  const q=p?.q||[0,0];

  // Current toy token — always visible, even before Live starts.
  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.textAlign="left";
  ctx.fillText("Current CartPole motion token (post-FSQ)",36,24);
  const cellY=38,cellW=135,cellH=64;
  for(let i=0;i<2;i++){
    const x=36+i*(cellW+12);
    ctx.strokeStyle=i===0?"#315dc9":"#795fc5";ctx.lineWidth=2;ctx.strokeRect(x,cellY,cellW,cellH);
    ctx.fillStyle="#172033";ctx.font="700 12px system-ui";ctx.fillText("q"+(i+1),x+10,cellY+21);
    ctx.fillStyle=i===0?"#315dc9":"#795fc5";ctx.font="700 22px ui-monospace";ctx.fillText(fmt(q[i],2),x+10,cellY+50);
  }
  ctx.fillStyle="#667085";ctx.font="10px system-ui";
  ctx.fillText("These are quantized numeric values — not one vocabulary integer ID.",330,72);

  // Schematic of actual released token shape: 2 tokens x 32 scalar dims.
  const gridTop=128,gridLeft=36,gridRight=w-36,cols=32,gap=2;
  const cw=(gridRight-gridLeft-(cols-1)*gap)/cols,ch=20;
  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.fillText("Released SONIC token shape (schematic)",gridLeft,gridTop-10);
  for(let row=0;row<2;row++){
    ctx.fillStyle="#596273";ctx.font="9px ui-monospace";ctx.fillText("token "+(row+1),gridLeft,gridTop+row*34+14);
    for(let col=0;col<cols;col++){
      const x=gridLeft+58+col*(cw*.82+gap);
      const y=gridTop+row*34;
      ctx.fillStyle=row===0?"rgba(49,93,201,.55)":"rgba(121,95,197,.55)";
      ctx.fillRect(x,y,Math.max(4,cw*.78),ch);
    }
  }
  ctx.fillStyle="#667085";ctx.font="9.5px system-ui";ctx.fillText("2 temporal tokens × 32 scalar dimensions = 64 flattened numeric values; each scalar uses fixed finite FSQ levels.",gridLeft,gridTop+77);

  // Live toy token history in lower half.
  const pts=hist.length?hist:[{t:0,q}],chartTop=235,pad={l:50,r:24,b:42},panelGap=18;
  const chartH=h-chartTop-pad.b,panelH=(chartH-panelGap)/2,tMax=Math.max(.2,...pts.map(x=>x.t));
  const X=t=>pad.l+t/tMax*(w-pad.l-pad.r),Y=(v,top)=>top+panelH/2-v*(panelH*.42);
  const levels=[-1,-.5,0,.5,1];
  for(let k=0;k<2;k++){
    const top=chartTop+k*(panelH+panelGap);
    for(const lv of levels){ctx.strokeStyle="#f0f2f5";ctx.beginPath();ctx.moveTo(pad.l,Y(lv,top));ctx.lineTo(w-pad.r,Y(lv,top));ctx.stroke();}
    ctx.strokeStyle=k===0?"#315dc9":"#795fc5";ctx.lineWidth=2.5;ctx.beginPath();
    pts.forEach((s,i)=>{const x=X(s.t),y=Y(s.q[k],top);if(i===0)ctx.moveTo(x,y);else{const py=Y(pts[i-1].q[k],top);ctx.lineTo(x,py);ctx.lineTo(x,y);}});ctx.stroke();
    const last=pts.at(-1);ctx.fillStyle=k===0?"#315dc9":"#795fc5";ctx.beginPath();ctx.arc(X(last.t),Y(last.q[k],top),4,0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#667085";ctx.font="10px system-ui";ctx.fillText("q"+(k+1)+" live discrete history",pad.l,top+12);
  }
  $("vizCaption").innerHTML='<b>Universal Token.</b> 위는 지금 decoder가 받는 toy q, 가운데는 실제 SONIC release의 token shape 개념, 아래는 Live에서 q가 finite level 사이를 step-wise 이동하는 history다.';
  setMetrics(["toy q=["+fmt(q[0],2)+","+fmt(q[1],2)+"]","SONIC: 2×32=64","32 fixed levels/scalar","token = numeric vector(s)"]);
}

function renderMotionDecoderViz(){
  renderAutoencoderViz();
  $("vizCaption").innerHTML='<b>LIVE Robot Motion Decoder / Kinematic Decoder.</b> token에서 future motion을 복원한다. 이 경로는 motion 정보를 token 안에 유지시키는 auxiliary reconstruction 역할이며 physical motor command가 아니다.';
}
function renderControlDecoderViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview(),s=state(),hist=signalHistory.filter(x=>x.episode===episodeIndex);
  const topH=175;
  const boxes=[
    {x:45,w:150,title:"motion token q",text:p?"["+p.q.map(v=>fmt(v,2)).join(", ")+"]":"—",color:"#315dc9"},
    {x:240,w:190,title:"actual proprioception",text:"["+s.map(v=>fmt(v,2)).join(", ")+"]",color:"#667085"},
    {x:480,w:170,title:"Dynamic Decoder",text:"token + state",color:"#795fc5"},
    {x:700,w:120,title:"force",text:p?fmt(p.force,2)+" N":"—",color:"#16805d"},
  ];
  for(const b of boxes){ctx.strokeStyle=b.color;ctx.lineWidth=2;ctx.strokeRect(b.x,45,b.w,78);ctx.fillStyle="#172033";ctx.font="700 11px system-ui";ctx.fillText(b.title,b.x+8,66);ctx.fillStyle="#596273";ctx.font="10px ui-monospace";ctx.fillText(b.text,b.x+8,92);}
  ctx.fillStyle="#9aa3af";ctx.font="20px system-ui";ctx.fillText("+",218,88);ctx.fillText("→",446,88);ctx.fillText("→",672,88);
  const pad={l:52,r:25,t:topH+25,b:40},pts=hist.length?hist:[{t:0,force:p?.force||0}],tMax=Math.max(.2,...pts.map(x=>x.t)),maxF=Math.max(10,...pts.map(x=>Math.abs(x.force)));
  const X=t=>pad.l+t/tMax*(w-pad.l-pad.r),Y=v=>h-pad.b-(v+maxF)/(2*maxF)*(h-pad.t-pad.b);
  ctx.strokeStyle="#e9ecf1";ctx.beginPath();ctx.moveTo(pad.l,Y(0));ctx.lineTo(w-pad.r,Y(0));ctx.stroke();
  ctx.strokeStyle="#16805d";ctx.lineWidth=2.5;ctx.beginPath();pts.forEach((x,i)=>{const px=X(x.t),py=Y(x.force);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();
  const lastPt=pts.at(-1);ctx.fillStyle="#16805d";ctx.beginPath();ctx.arc(X(lastPt.t),Y(lastPt.force),4,0,Math.PI*2);ctx.fill();
  ctx.fillStyle="#667085";ctx.font="10px system-ui";ctx.fillText("actual action / force history",pad.l,pad.t-8);
  $("vizCaption").innerHTML='<b>LIVE Robot Control Decoder.</b> q는 motor command가 아니다. Push는 reference/token을 유지한 채 actual proprioception만 바꾸므로 action이 어떻게 달라지는지 바로 확인할 수 있다.';
  setMetrics(["token="+(p?"["+p.q.map(v=>fmt(v,2)).join(",")+"]":"—"),"θ="+fmt(s[2]*180/Math.PI,1)+"°","force="+fmt(p?.force,2)+"N"]);
}
function renderRobotTrackingViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),hist=signalHistory.filter(x=>x.episode===episodeIndex),p=preview();
  const pts=hist.length?hist:[{t:0,state:state(),reference:Array.from(currentReference()),force:p?.force||0}];
  const pad={l:55,r:24,t:34,b:45},gap=30,panelH=(h-pad.t-pad.b-gap)/2,tMax=Math.max(.2,...pts.map(x=>x.t));
  const X=t=>pad.l+t/tMax*(w-pad.l-pad.r);
  const top1=pad.t,Y1=v=>top1+panelH/2-v/1.8*(panelH*.45);
  ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,Y1(0));ctx.lineTo(w-pad.r,Y1(0));ctx.stroke();
  const draw=(get,color,dash=[])=>{ctx.strokeStyle=color;ctx.lineWidth=2.3;ctx.setLineDash(dash);ctx.beginPath();pts.forEach((s,i)=>{const x=X(s.t),y=Y1(get(s));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);};
  draw(s=>s.state[0],"#d64f4f");draw(s=>s.reference[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],"#315dc9",[6,4]);
  ctx.fillStyle="#667085";ctx.font="10px system-ui";ctx.fillText("x tracking: red actual · blue dashed reference(+80ms)",pad.l,top1+12);
  const top2=pad.t+panelH+gap,maxF=10,Y2=v=>top2+panelH/2-v/maxF*(panelH*.44);
  ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,Y2(0));ctx.lineTo(w-pad.r,Y2(0));ctx.stroke();
  ctx.strokeStyle="#16805d";ctx.lineWidth=2.3;ctx.beginPath();pts.forEach((s,i)=>{const x=X(s.t),y=Y2(s.force);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  ctx.fillStyle="#667085";ctx.fillText("control force",pad.l,top2+12);
  const err=p?Math.abs(state()[0]-p.ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0]):NaN;
  $("vizCaption").innerHTML='<b>LIVE closed-loop result.</b> actual robot과 desired reference는 같은 것이 아니다. Dynamic Decoder action이 physics를 바꾸고 measured state가 다시 feedback된다.';
  setMetrics(["current x error="+fmt(err,3)+"m","episode="+episodeIndex,"auto resets="+autoResetCount]);
}
function renderTrainingFlowViz(){
  showHtml(
    '<div class="training-grid"><div class="training-flow">'+
      '<div class="training-box main"><b>Physics rollout</b><br>reference + current state → action → MuJoCo</div>'+
      '<div class="training-box main"><b>PPO loss</b><br>tracking reward / advantage → Dynamic Decoder + Encoder</div>'+
      '<div class="training-box aux"><b>Reconstruction aux</b><br>token → Kinematic Decoder → future motion</div>'+
      '<div class="training-box aux"><b>Latent alignment aux</b><br>G1 ↔ SMPL ↔ teleop encoders share semantics</div>'+
      '<div class="training-box"><b>Critic</b><br>value / return loss for PPO</div>'+
      '<div class="training-box"><b>FSQ</b><br>fixed levels; STE passes gradient, no learned codebook</div>'+
    '</div><div class="flow-diagram"><div class="flow-row"><div class="flow-box accent"><strong>Total update</strong><span>PPO + weighted auxiliary losses</span></div></div></div></div>',
    'SONIC training은 reconstruction 하나가 아니라 <b>physical PPO + representation auxiliary losses</b>를 함께 사용한다. deployment runtime graph와 training graph를 섞어 보지 않는 것이 중요하다.',
    ["PPO = physical tracking","aux = representation","FSQ = fixed bottleneck"]
  );
}
function renderLearningGraphViz(){
  showHtml(
    '<table class="compare-table"><thead><tr><th>Module</th><th>Trainable?</th><th>Main signal</th><th>What changes?</th></tr></thead><tbody>'+
    '<tr><td><b>Encoder(s)</b></td><td>YES</td><td>PPO + reconstruction + latent alignment</td><td>reference/modality → latent mapping</td></tr>'+
    '<tr><td><b>FSQ</b></td><td><b>NO</b></td><td>STE backward only</td><td>fixed levels do not move</td></tr>'+
    '<tr><td><b>Dynamic Decoder</b></td><td>YES</td><td>PPO tracking</td><td>token + state → action mapping</td></tr>'+
    '<tr><td><b>Kinematic Decoder</b></td><td>YES</td><td>reconstruction aux</td><td>token → future motion mapping</td></tr>'+
    '<tr><td><b>VQ codebook</b> (comparison)</td><td>YES</td><td>codebook / EMA update</td><td>representative vectors move</td></tr>'+
    '</tbody></table>',
    '<b>FSQ participates in training, but FSQ itself is not parameter-learned.</b> Encoder/Decoder가 fixed discrete bottleneck을 유용하게 쓰는 법을 배운다.',
    ["toy: 2 dims × 5 levels","SONIC: 2 tokens × 32 dims","32 fixed levels/scalar","flattened=64"]
  );
}
function renderAlignmentViz(){
  showHtml(
    '<div style="height:100%;display:grid;align-items:center"><div class="flow-row">'+
      '<div class="flow-box"><strong>G1 Encoder</strong><span>robot joint motion</span></div>'+
      '<div class="flow-box"><strong>SMPL Encoder</strong><span>human body motion</span></div>'+
      '<div class="flow-box"><strong>Teleop Encoder</strong><span>VR targets</span></div>'+
      '<div class="flow-arrow">→</div><div class="flow-box accent"><strong>Shared latent meaning</strong><span>alignment losses</span></div>'+
      '<div class="flow-arrow">→</div><div class="flow-box purple"><strong>Same FSQ / Universal Token</strong><span>shared decoder interface</span></div>'+
    '</div></div>',
    '<b>CONCEPT · not toy runtime data.</b> 이 CartPole toy는 Encoder가 하나라 실제 multi-encoder alignment를 재현하지 않는다. 공식 SONIC은 여러 latent-alignment auxiliary loss로 서로 다른 modality Encoder를 같은 의미 공간에 맞춘다.',
    ["G1↔SMPL","G1↔teleop","teleop↔SMPL","FSQ alone is not enough"]
  );
}
function renderTrainingViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);const pad={l:62,r:28,t:32,b:50};
  ctx.strokeStyle="#e2e6ec";ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();
  const ref=(ppoReferenceEvidence?.checkpoints||[]).map(x=>({iterations:x.ppoIterations,mae:x.eval?.trackingMae})).filter(x=>Number.isFinite(x.mae));
  const rollout=(currentTrainer.history||[]).map(x=>({iterations:x.iter,mae:x.tracking})).filter(x=>Number.isFinite(x.mae));
  const held=ppoHeldoutHistory.filter(x=>Number.isFinite(x.mae)),all=[...ref,...rollout,...held];
  const maxIter=Math.max(50,...all.map(x=>x.iterations)),maxMae=Math.max(.25,...all.map(x=>x.mae)),minMae=Math.min(.10,...all.map(x=>x.mae));
  const X=i=>pad.l+i/maxIter*(w-pad.l-pad.r),Y=v=>h-pad.b-(v-minMae)/(maxMae-minMae)*(h-pad.t-pad.b);
  for(let i=0;i<=5;i++){const v=minMae+(maxMae-minMae)*i/5,y=Y(v);ctx.strokeStyle="#f0f2f5";ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillStyle="#7b8492";ctx.font="10px ui-monospace";ctx.textAlign="right";ctx.fillText(v.toFixed(2),pad.l-8,y+3);}
  const draw=(pts,color,dash,width)=>{if(!pts.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();pts.forEach((p,i)=>{const x=X(p.iterations),y=Y(p.mae);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);};
  draw(ref,"#9aa3af",[6,5],2);draw(rollout,"#315dc9",[],2.5);held.forEach(p=>{ctx.fillStyle="#16805d";ctx.beginPath();ctx.arc(X(p.iterations),Y(p.mae),5,0,Math.PI*2);ctx.fill();});
  ctx.fillStyle="#667085";ctx.font="11px system-ui";ctx.textAlign="left";ctx.fillText("tracking MAE [m] ↓",pad.l,18);ctx.textAlign="right";ctx.fillText("PPO iterations →",w-pad.r,h-15);
  $("vizCaption").innerHTML='회색 점선=deterministic held-out reference · 파랑=현재 PPO rollout MAE · 초록=현재 held-out check. <b>PPO는 physical tracking을 학습한다.</b>';
  setMetrics(["PPO iter="+currentTrainer.iter,heldoutEval?"held-out="+fmt(heldoutEval.trackingMae,3)+"m":"held-out=—"]);
}

function visualizationKind(){
  if(trainingMode) return trainingTopic.id==="ppo"?"live":"concept";
  if(focus.id==="task") return "concept";
  if(focus.id==="encoder"&&conceptId==="vae") return "concept";
  return "live";
}
function renderVisualization(){
  clearViz();
  buildConceptTabs();
  const kind=visualizationKind(),vm=$("vizMode");
  vm.textContent=kind==="live"?"LIVE · current toy state":"CONCEPT · explanatory, not toy runtime data";
  vm.className="viz-mode "+(kind==="live"?"live":"concept");
  if(trainingMode){
    $("vizTitle").textContent=trainingTopic.title;
    $("vizSub").textContent="Training view · separate from deployment runtime";
    switch(trainingTopic.viz){
      case "training-flow":renderTrainingFlowViz();break;
      case "learning-graph":renderLearningGraphViz();break;
      case "alignment":renderAlignmentViz();break;
      case "training":renderTrainingViz();break;
    }
    return;
  }
  const activeConcept=(conceptId&&conceptId!=="core")?CONCEPT_TEXT[conceptId]:null;
  $("vizTitle").textContent=activeConcept?activeConcept.title:focus.title;
  $("vizSub").textContent=activeConcept?("Background for SONIC block: "+focus.nav+" · "+activeConcept.short):("Official role: "+focus.official+" · toy: "+focus.toy);
  switch(focus.viz){
    case "task":renderTaskViz();break;
    case "reference":renderReferenceCurveViz();break;
    case "reference-vector":renderReferenceVectorViz();break;
    case "encoder":
      if(conceptId==="ae")renderAutoencoderViz();
      else if(conceptId==="vae")renderVaeViz();
      else renderEncoderCoreViz();
      break;
    case "quantizer":
      if(conceptId==="vq")renderLatentViz("vq");
      else if(conceptId==="vqvae")renderVqvaeViz();
      else renderLatentViz("fsq");
      break;
    case "token":renderTokenViz();break;
    case "motion-decoder":renderMotionDecoderViz();break;
    case "control-decoder":renderControlDecoderViz();break;
    case "tracking":renderRobotTrackingViz();break;
  }
}

function activeConcept(){
  if(trainingMode)return null;
  if(!conceptId||conceptId==="core")return null;
  if((focus.id==="encoder"||focus.id==="quantizer")&&CONCEPT_TEXT[conceptId])return CONCEPT_TEXT[conceptId];
  return null;
}
function guideData(){
  if(trainingMode){
    return {
      kicker:"TRAINING · "+trainingTopic.label,
      title:trainingTopic.title,
      map:"Training-only view · runtime map의 보라색 block이 이 topic의 주요 update 대상",
      input:trainingTopic.input,output:trainingTopic.output,
      question:trainingTopic.question,concept:null,
      details:TRAINING_DETAILS[trainingTopic.id]||{}
    };
  }
  const concept=activeConcept();
  return {
    kicker:concept?("CONCEPT INSIDE · "+focus.nav):("SONIC SYSTEM BLOCK · "+focus.nav),
    title:concept?concept.title:focus.title,
    map:"Official block: "+focus.official+" · CartPole mapping: "+focus.toy,
    input:focus.input,output:focus.output,
    question:concept?.question||focus.question,
    concept,
    details:RUNTIME_DETAILS[focus.id]||{}
  };
}
function guideLiveValues(){
  const p=preview(),s=state(),ref=currentReference();
  const cell=(k,v)=>({k,v:String(v)});
  if(trainingMode){
    return [
      cell("PPO iter",currentTrainer?.iter??0),
      cell("rollout MAE",fmt(currentTrainer?.last?.tracking,3)),
      cell("held-out",heldoutEval?fmt(heldoutEval.trackingMae,3)+" m":"—"),
      cell("representation",requiredMode())
    ];
  }
  switch(focus.id){
    case "task":
      return [cell("goal x*",fmt(goal,2)+" m"),cell("Live",live?"ON":"OFF"),cell("episode",episodeIndex),cell("preset",preset)];
    case "generator":
      return [cell("planner x",fmt(plannerContext[0],3)+" m"),cell("planner ẋ",fmt(plannerContext[1],3)),cell("ref +80ms",fmt(ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)+" m"),cell("horizon","0.64 s")];
    case "reference":
      return [cell("input dim","16"),cell("frame 1 x",fmt(ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)),cell("frame 1 ẋ",fmt(ref[1]*SONIC_TOY_CONSTANTS.STATE_SCALE[1],3)),cell("actual x",fmt(s[0],3))];
    case "encoder":
      return [cell("z₁",fmt(p?.z?.[0],3)),cell("z₂",fmt(p?.z?.[1],3)),cell("recon MSE",fmt(reconstructionError(p,ref),4)),cell("mode",requiredMode())];
    case "quantizer":
      return [cell("z",p?"["+p.z.map(v=>fmt(v,2)).join(",")+"]":"—"),cell("q",p?"["+p.q.map(v=>fmt(v,2)).join(",")+"]":"—"),cell("mode",requiredMode()),cell("history",signalHistory.length+" samples")];
    case "token":
      return [cell("q₁",fmt(p?.q?.[0],2)),cell("q₂",fmt(p?.q?.[1],2)),cell("toy token","2 values"),cell("SONIC release","64 flattened")];
    case "motion-decoder":
      return [cell("recon MSE",fmt(reconstructionError(p,ref),4)),cell("recon x₁",fmt((p?.kinRecon?.[0]??NaN)*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)),cell("target x₁",fmt(ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)),cell("token",p?"["+p.q.map(v=>fmt(v,2)).join(",")+"]":"—")];
    case "control-decoder":
      return [cell("token",p?"["+p.q.map(v=>fmt(v,2)).join(",")+"]":"—"),cell("θ",fmt(s[2]*180/Math.PI,1)+"°"),cell("ẋ",fmt(s[1],3)),cell("force",fmt(p?.force,2)+" N")];
    case "robot":
      const rx=p?p.ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0]:NaN;
      return [cell("actual x",fmt(s[0],3)+" m"),cell("ref +80ms",fmt(rx,3)+" m"),cell("|error|",fmt(Math.abs(s[0]-rx),3)+" m"),cell("force",fmt(lastForce,2)+" N")];
    default:return [];
  }
}
function setGuideDepth(depth){
  if(!["easy","mechanism","sonic"].includes(depth))return;
  guideDepth=depth;updateUrl();renderGuide();
}

function guideActionSpec(){
  if(trainingMode){
    if(trainingTopic.id==="ppo")return{label:"+10 PPO iterations",disabled:false};
    if(trainingTopic.id==="loss-flow"||trainingTopic.id==="what-learns")return{label:"Run 1 PPO iteration",disabled:false};
    return{label:"Concept only · no fake runtime action",disabled:true};
  }
  if(focus.id==="control-decoder")return{label:"Push robot + 1 Step",disabled:false};
  if(focus.id==="quantizer"){
    if(conceptId==="vqvae")return{label:live?"Stop live input":"Start live input · weights stay frozen",disabled:false};
    return{label:live?"Stop Live":"Start Live",disabled:false};
  }
  if(["token","robot"].includes(focus.id))return{label:live?"Stop Live":"Start Live",disabled:false};
  if(focus.id==="encoder"&&conceptId==="vae")return{label:"Optional background · no runtime action",disabled:true};
  return{label:"Change goal",disabled:false};
}
function renderGuide(){
  const g=guideData(),d=g.details||{},c=g.concept;
  $("guideKicker").textContent=g.kicker;
  $("guideTitle").textContent=g.title;
  $("guideMap").textContent=g.map;
  $("guideInput").textContent=g.input;
  $("guideOutput").textContent=g.output;

  document.querySelectorAll("#guideDepthTabs button").forEach(b=>{
    b.classList.toggle("active",b.dataset.depth===guideDepth);
    b.onclick=()=>setGuideDepth(b.dataset.depth);
  });

  let s1Title="",s1="",s2Title="",s2="";
  if(guideDepth==="easy"){
    s1Title="한 줄 이해";
    s1=c?.short||d.easy||"—";
    s2Title="없으면 / 다음";
    const miss=c?.ifMissing||d.ifMissing||"—";
    const next=(!trainingMode&&!c&&d.next)?(" 다음: "+d.next):"";
    s2="없으면: "+miss+next;
  }else if(guideDepth==="mechanism"){
    s1Title="내부 동작";
    s1=c?.mechanism||d.mechanism||"—";
    s2Title="왜 필요한가";
    s2=c?.why||(!trainingMode?focus.why:trainingTopic.why)||"—";
  }else{
    s1Title="SONIC 실제 구조";
    s1=c?.sonic||d.sonic||"—";
    s2Title="Toy ↔ SONIC / 주의";
    const mapping=trainingMode?"training-only topic":("toy: "+focus.toy+" · official: "+focus.official);
    const warning=c?.key||(!trainingMode?focus.misconception:trainingTopic.misconception)||"—";
    s2=mapping+" · "+warning;
  }
  $("guideSection1Title").textContent=s1Title;
  $("guideSection1").textContent=s1;
  $("guideSection2Title").textContent=s2Title;
  $("guideSection2").textContent=s2;

  const live=guideLiveValues();
  const hideLive=(guideDepth==="sonic")||(trainingMode&&trainingTopic.id!=="ppo")||(focus.id==="encoder"&&conceptId==="vae")||(focus.id==="quantizer"&&conceptId==="vqvae"&&guideDepth==="mechanism");
  $("guideLiveBlock").hidden=hideLive;
  $("guideLive").innerHTML=hideLive?"":live.map(x=>'<div class="live-kv"><span>'+x.k+'</span><b>'+x.v+'</b></div>').join("");

  const toyShape=c?.toyShape||d.toyShape||(!trainingMode?focus.toy:"—");
  const sonicShape=c?.sonicShape||d.sonicShape||(!trainingMode?focus.official:"—");
  $("guideShapeBlock").hidden=(guideDepth==="easy");
  $("guideShape").textContent="toy: "+toyShape+"\nSONIC: "+sonicShape;
  $("guideQuestion").textContent=g.question;

  const note=$("conceptNote");
  note.hidden=true;note.innerHTML="";

  const seq=SONIC_FLOW.map(x=>x.id),idx=seq.indexOf(focus.id);
  $("prevNode").disabled=trainingMode||idx<=0;
  $("nextNode").disabled=trainingMode||idx<0||idx>=seq.length-1;
  $("prevNode").onclick=()=>idx>0&&focusNode(seq[idx-1]);
  $("nextNode").onclick=()=>idx>=0&&idx<seq.length-1&&focusNode(seq[idx+1]);

  const action=guideActionSpec();
  $("focusAction").textContent=action.label;
  $("focusAction").disabled=action.disabled;
  $("guideStatus").textContent=(busy?"preparing · ":"")+(trainingMode?"training="+trainingTopic.id:"focus="+focus.id+(conceptId?" · concept="+conceptId:""))+" · depth="+guideDepth+" · mode="+requiredMode()+" · preset="+preset;
}

async function runFocusAction(){
  if(trainingMode){
    if(trainingTopic.id==="ppo")await runPPO(10);
    else if(trainingTopic.id==="loss-flow"||trainingTopic.id==="what-learns")await runPPO(1);
    return;
  }
  if(focus.id==="control-decoder"){pushRobot();stepPolicy(1);return;}
  if(["quantizer","token","robot"].includes(focus.id)){live=!live;accumulator=0;render();return;}
  if(focus.id==="encoder"&&conceptId==="vae")return;
  setGoal(goal>0?-0.8:0.8);
}
function renderHeaderState(){
  setBadge("mcpBadge",webmcpMode==="unavailable"?"WebMCP pending":"WebMCP · "+webmcpTools.length,webmcpMode!=="unavailable");
}
function renderPreload(){
  buildSystemMap();
  buildConceptTabs();
  renderSimulation();
  renderGuide();
  renderHeaderState();

  const kind=visualizationKind(),vm=$("vizMode");
  vm.textContent=kind==="live"?"LIVE · waiting for MuJoCo/model":"CONCEPT · explanatory, not toy runtime data";
  vm.className="viz-mode "+(kind==="live"?"live":"concept");

  // Concept-only screens do not need the physics model and can be useful immediately.
  if(kind==="concept"){
    clearViz();
    if(trainingMode){
      $("vizTitle").textContent=trainingTopic.title;
      $("vizSub").textContent="Training view · separate from deployment runtime";
      if(trainingTopic.viz==="training-flow")renderTrainingFlowViz();
      else if(trainingTopic.viz==="learning-graph")renderLearningGraphViz();
      else if(trainingTopic.viz==="alignment")renderAlignmentViz();
    }else if(focus.id==="task"){
      $("vizTitle").textContent=focus.title;
      $("vizSub").textContent="Official role: "+focus.official+" · toy: "+focus.toy;
      renderTaskViz();
    }else if(focus.id==="encoder"&&conceptId==="vae"){
      $("vizTitle").textContent=CONCEPT_TEXT.vae.title;
      $("vizSub").textContent="Background for SONIC block: Encoder(s)";
      renderVaeViz();
    }
  }else{
    clearViz();
    $("vizTitle").textContent=(conceptId&&conceptId!=="core"&&CONCEPT_TEXT[conceptId])?CONCEPT_TEXT[conceptId].title:focus.title;
    $("vizSub").textContent="Loading native MuJoCo WASM + student checkpoint…";
    showHtml(
      '<div style="height:100%;display:grid;place-items:center"><div style="text-align:center"><b style="font-size:15px">Preparing live visualization</b><div style="margin-top:8px;color:#667085;font-size:11px">MuJoCo physics and the precomputed student checkpoint are loading.<br>The SONIC block explanation is already available on the right.</div></div></div>',
      'LIVE visualization will appear as soon as the actual model state is available.',
      ["loading physics","loading student model"]
    );
  }
}
function render(){
  if(!physicsReady||!currentTrainer)return;
  buildSystemMap();renderSimulation();renderVisualization();renderGuide();renderHeaderState();
}

function systemSnapshot(){
  const p=preview(),ref=currentReference(),s=state();
  return {
    system:{
      version:COURSE_VERSION,
      mode:trainingMode?"training":"runtime",
      explanationDepth:guideDepth,
      focus:trainingMode?null:{node:focus.id,concept:conceptId},
      focusExplanation:trainingMode?(TRAINING_DETAILS[trainingTopic.id]||null):(RUNTIME_DETAILS[focus.id]||null),
      conceptExplanation:(!trainingMode&&conceptId&&conceptId!=="core")?(CONCEPT_TEXT[conceptId]||null):null,
      trainingTopic:trainingMode?trainingTopic.id:null,
      outline:getSystemOutline(),
    },
    experiment:{goal,live,preset,representationMode:requiredMode(),episode:episodeIndex,autoResets:autoResetCount,lastEpisodeEvent},
    semantics:{
      reference:"desired future motion upstream of Encoder/FSQ; not measured robot state",
      latent:"continuous Encoder output",
      token:"post-VQ/FSQ quantized numeric motion representation; not motor command",
      proprioception:"measured actual robot state",
      action:"Robot Control/Dynamic Decoder output applied to physics",
      motionDecoder:"token → reconstructed future motion; auxiliary/kinematic role",
    },
    signals:{
      reference:Array.from(ref),latent:p?.z||null,token:p?.q||null,proprioception:s,actionMean:p?.mu??null,force:p?.force??null,
      kinematicReconstruction:p?.kinRecon||null,
      liveHistory:signalHistory.slice(-80).map(x=>({t:x.t,episode:x.episode,z:x.z,q:x.q,force:x.force,state:x.state,reference:x.reference}))
    },
    modelSemantics:{
      encoder:{trainable:true,signals:["PPO","reconstruction auxiliary","cross-encoder latent alignment"]},
      fsq:{trainable:false,levels:"fixed",gradient:"STE through rounding; no learned vector codebook"},
      dynamicDecoder:{trainable:true,signals:["PPO tracking objective"]},
      kinematicDecoder:{trainable:true,signals:["future-motion reconstruction auxiliary loss"]},
      critic:{trainable:true,signals:["value/return loss"]},
      releasedTokenShape:{numTokens:2,scalarDimsPerToken:32,fixedLevelsPerScalar:32,flattenedDim:64},
    },
    training:currentTrainer?{ppoIterations:currentTrainer.iter,envSteps:currentTrainer.envSteps,episodes:currentTrainer.episodes,last:currentTrainer.last,heldoutEval,heldoutHistory:ppoHeldoutHistory,verifiedReference:ppoReferenceEvidence}:null,
    backends:{physics:physicsReady?sim.backend:null,webgpu:webgpuStatus,webmcp:{mode:webmcpMode,tools:webmcpTools}}
  };
}
async function registerWebMCP(){
  const mc=document.modelContext||navigator.modelContext;if(!mc||typeof mc.registerTool!=="function"){webmcpMode="unavailable";return;}
  webmcpMode=typeof navigator.modelContextTesting?.listTools==="function"?"mcp-b-global":"native";
  const nodeIds=SONIC_FLOW.map(x=>x.id),trainingIds=TRAINING_TOPICS.map(x=>x.id);
  const tools=[
    {name:"sonic_get_map",description:"Read the canonical SONIC runtime map and separate training topics.",inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:true},execute:async()=>getSystemOutline()},
    {name:"sonic_get_state",description:"Read current SONIC focus, live reference/latent/token/robot/action signals, training evidence, and backend state.",inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:true},execute:async()=>systemSnapshot()},
    {name:"sonic_focus",description:"Focus one runtime SONIC block and optionally one contextual concept such as AE, VAE, VQ, VQ-VAE, or FSQ.",inputSchema:{type:"object",properties:{node_id:{type:"string",enum:nodeIds},concept_id:{type:"string"}},required:["node_id"]},annotations:{readOnlyHint:false},execute:async({node_id,concept_id=null})=>{await focusNode(node_id,concept_id);return systemSnapshot();}},
    {name:"sonic_open_training",description:"Open a training-only topic without pretending it is part of the deployment runtime graph.",inputSchema:{type:"object",properties:{topic_id:{type:"string",enum:trainingIds}},required:["topic_id"]},annotations:{readOnlyHint:false},execute:async({topic_id})=>{await openTraining(topic_id);return systemSnapshot();}},
    {name:"sonic_set_explanation_depth",description:"Switch the right-side explanation between easy intuition, internal mechanism, and actual SONIC structure.",inputSchema:{type:"object",properties:{depth:{type:"string",enum:["easy","mechanism","sonic"]}},required:["depth"]},annotations:{readOnlyHint:false},execute:async({depth})=>{setGuideDepth(depth);return systemSnapshot();}},
    {name:"sonic_run_focus_action",description:"Run the canonical experiment for the currently focused block.",inputSchema:{type:"object",properties:{}},annotations:{readOnlyHint:false},execute:async()=>{await runFocusAction();return systemSnapshot();}},
    {name:"simulation_control",description:"Control the shared actual MuJoCo robot.",inputSchema:{type:"object",properties:{action:{type:"string",enum:["step","live_on","live_off","push","reset"]},steps:{type:"integer",minimum:1,maximum:100}},required:["action"]},annotations:{readOnlyHint:false},execute:async({action,steps=1})=>{if(action==="step")stepPolicy(steps);else if(action==="live_on"){live=true;render();}else if(action==="live_off"){live=false;render();}else if(action==="push")pushRobot();else if(action==="reset")resetRobot();return systemSnapshot();}},
    {name:"experiment_set_goal",description:"Set the high-level task goal used by the motion generator.",inputSchema:{type:"object",properties:{x:{type:"number",minimum:-1.2,maximum:1.2}},required:["x"]},annotations:{readOnlyHint:false},execute:async({x})=>{setGoal(x);return systemSnapshot();}},
    {name:"training_run",description:"Run bounded PPO iterations on the current student policy.",inputSchema:{type:"object",properties:{iterations:{type:"integer",minimum:1,maximum:30}},required:["iterations"]},annotations:{readOnlyHint:false},execute:async({iterations})=>{await runPPO(iterations);return systemSnapshot();}},
    {name:"simulation_set_model",description:"Switch native MuJoCo CartPole dynamics/morphology preset.",inputSchema:{type:"object",properties:{preset:{type:"string",enum:["playground","long","heavy"]}},required:["preset"]},annotations:{readOnlyHint:false},execute:async({preset})=>{await changePreset(preset);return systemSnapshot();}}
  ];
  for(const t of tools)await mc.registerTool(t);webmcpTools=tools.map(t=>t.name);renderHeaderState();
}
function attachUI(){
  $("goal").oninput=()=>setGoal($("goal").value);$("stepBtn").onclick=()=>stepPolicy(1);$("liveBtn").onclick=()=>{live=!live;accumulator=0;render();};
  $("pushBtn").onclick=pushRobot;$("resetBtn").onclick=()=>resetRobot();$("simPreset").onchange=()=>changePreset($("simPreset").value);
  $("focusAction").onclick=()=>{void runFocusAction();};
}
function installCanvasResizeObserver(){
  if(!("ResizeObserver" in window))return;
  const ro=new ResizeObserver(()=>{if(physicsReady&&currentTrainer)render();});ro.observe($("cart"));ro.observe($("lessonViz"));window.__cartpoleSonicResizeObserver=ro;
}
function loop(now){
  const dt=Math.min(.05,Math.max(0,(now-lastFrame)/1000));lastFrame=now;
  if(live&&!busy&&currentTrainer&&physicsReady){
    accumulator+=dt;let n=0;
    while(accumulator>=.02&&n<4){
      const p=preview(),s0=state();recordControlSample(p,s0);sim.stepForce(p.force,2);plannerContext=advancePlannerContext(plannerContext,goal,.02);lastForce=p.force;accumulator-=.02;n++;
      const reason=terminationReason(state());if(reason){autoResetEpisode(reason);break;}
    }
    render();
  }
  requestAnimationFrame(loop);
}
async function init(){
  attachUI();busy=true;normalizeConcept();renderPreload();
  teacherPromise=loadTeacherPolicy().then(t=>(teacher=t,t)).catch(err=>{console.warn("teacher unavailable",err);return null;});
  fetch("./evidence/ppo_eval.json",{cache:"no-store"}).then(r=>r.ok?r.json():null).then(x=>{ppoReferenceEvidence=x;if(trainingMode&&trainingTopic.id==="ppo"&&!busy)render();}).catch(()=>{});
  await sim.init(preset);physicsReady=true;$("simPreset").value=preset;resetRobot();
  setBadge("physicsBadge",sim.backend,true);
  currentTrainer=await getTrainer(requiredMode());busy=false;
  if(trainingMode&&trainingTopic.id==="ppo"){heldoutEval=currentTrainer.evaluate(12);ppoHeldoutHistory=[{iterations:currentTrainer.iter,mae:heldoutEval.trackingMae}];}
  updateUrl();render();
  registerWebMCP().then(()=>renderHeaderState()).catch(()=>{});
  probeWebGPUFSQ().then(status=>{webgpuStatus=status;setBadge("webgpuBadge",status.ok?"WebGPU FSQ ✓":status.available?"WebGPU fallback":"WebGPU unavailable",status.ok);}).catch(err=>{webgpuStatus={available:false,ok:false,reason:err.message};setBadge("webgpuBadge","WebGPU unavailable",false);});
  if(location.hostname==="localhost"||location.hostname==="127.0.0.1"){setInterval(()=>{fetch("/telemetry",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(systemSnapshot()),keepalive:true}).catch(()=>{});},900);}
  installCanvasResizeObserver();requestAnimationFrame(loop);
}
init().catch(err=>{console.error(err);setBadge("physicsBadge","init error",false);});

window.__cartpoleSonic={
  getState:systemSnapshot,
  focus:focusNode,
  concept:setConcept,
  openTraining,
  runFocusAction,
  step:stepPolicy,
  setGoal,
  runPPO,
};
