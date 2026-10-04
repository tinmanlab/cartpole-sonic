import {mountNativeLesson} from "./native_lesson.js";
import {physicalFailureReason,finiteNumber,integerCount,CONTROL_DT,CONTROL_SUBSTEPS} from "./control_contract.js";
import { controlSample, controllerIdentity, controllerSummary, blockShape, decoderLayout, decoderConnections, robotGeometry, canvasLines, canvasTicks } from "./presentation.js";
import { MuJoCoCartPole } from "./mujoco_sim.js";
import {
  SonicToyTrainer,
  planReference,
  advancePlannerContext,
  SONIC_TOY_CONSTANTS,
} from "./sonic_toy.js";
import { loadTeacherPolicy } from "./teacher_policy.js";
import { probeWebGPUFSQ } from "./webgpu_fsq.js";
import { AlignmentLab } from "./alignment_lab.js";
import { TemporalTokenLab } from "./temporal_token_lab.js";
import { TemporalControlLab } from "./temporal_control_lab.js";
import { renderOptimizerEvidencePanel, optimizerViewState } from "./optimizer_evidence_view.js";
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

let nativeLesson=null;
let labAnimation=0;
const query = new URLSearchParams(location.search);
const startsInLesson=query.get("lesson")==="future"||(!query.has("focus")&&!query.has("training")&&!query.has("lesson"));
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
let executionState="ready", executionReason="", episodeTerminated=false;
let lastTrainingTargets=[];
const busyDisabledControls=new Map();
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
let alignmentLab = null;
let temporalTokenLab = null;
let temporalControlLab = null;
let temporalControlLoadPromise = null;
let temporalControlBootstrapSnapshot = null;
let temporalControlSelected = "one";
let temporalControlEvidence = null;
let optimizerEvidence = null;
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
function temporalControlModeRequested(){
  return !trainingMode&&focus.id==="token"&&conceptId==="temporal-control";
}
function temporalControlActive(){
  return temporalControlModeRequested()&&temporalControlLab;
}
function resolveActiveController(){
  const temporal=Boolean(temporalControlActive());
  const trainer=temporal?(temporalControlSelected==="two"?temporalControlLab.two:temporalControlLab.one):currentTrainer;
  return {trainer,id:temporal?"temporal-"+temporalControlSelected:trainer?"student-"+trainer.mode:null,
    trainingTargets:temporal?[temporalControlLab.one,temporalControlLab.two]:trainer?[trainer]:[],
    trainingIds:temporal?["temporal-one","temporal-two"]:trainer?["student-"+trainer.mode]:[],
    demo:trainingMode&&trainingTopic.id==="alignment"?"sparse-keypoint alignment":!trainingMode&&conceptId==="temporal"?"temporal reconstruction":null};
}
function preview(){
  const {trainer}=resolveActiveController();
  const out=trainer?.preview(state(),currentReference(),goal);
  return out?{...out,ref:Array.from(currentReference())}:null;
}
function activeController(p=preview()){
  const {trainer,id,demo}=resolveActiveController();
  return {...controllerIdentity(p,trainer?.mode||requiredMode(),temporalControlActive()?trainer?.policy:null),id,demo};
}
function executionSnapshot(){return{state:busy?"busy":live?"running":executionState,reason:executionReason,resetRequired:episodeTerminated};}
function executionText(){
  const labels={ready:"준비",running:"실행 중",paused:"일시정지",terminated:"실패로 멈춤",error:"확인 필요",busy:"작업 중"};
  const names={"student-ae":"AE 모델","student-vq":"VQ 모델","student-fsq":"FSQ 모델","temporal-one":"1-token","temporal-two":"2-token","alignment-secondary":"입력 표현 정렬","temporal-reconstruction-one/two":"1·2-token 복원 비교"};
  const reason=executionReason.replace("track limit (1.78 m)","카트 이동 한계(1.78m)").replace("pole angle (0.65 rad)","막대 각도 한계(0.65rad)");
  return labels[executionSnapshot().state]+" · "+(reason||"한 단계 또는 계속 실행")+" · 학습 대상: "+(lastTrainingTargets.map(id=>names[id]||id).join(" · ")||"아직 없음");
}
function assertIdle(){if(nativeLesson?.active)throw new Error("실험 수업 기록 재생 중에는 구조 탐색 도구를 실행할 수 없습니다. 구조 탐색으로 전환하세요.");if(busy)throw new Error("작업 중입니다. 완료 후 다시 시도하세요.");}
function assertEpisodeReady(){
  assertIdle();if(!physicsReady||!resolveActiveController().trainer)throw new Error("시뮬레이션/모델 준비가 필요합니다.");
  if(episodeTerminated||executionState==="error")throw new Error("Reset 필요: "+executionReason);
}
function showError(error){live=false;executionState="error";executionReason=error.message||String(error);}
async function withOperation(label,work){
  assertIdle();const priorReason=executionReason;busy=true;live=false;executionReason=label;
  try{render();const result=await work();executionState=episodeTerminated?"terminated":"ready";executionReason=episodeTerminated?priorReason:"";return result;}
  catch(error){showError(error);throw error;}
  finally{busy=false;render();}
}
function uiAction(work){try{Promise.resolve(work()).catch(()=>{});}catch(error){showError(error);render();}}
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
  ctx.font="13px system-ui";ctx.textAlign="left";ctx.textBaseline="alphabetic";
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
  try{
    const ckpt=await fetch("./assets/student_"+mode+"_bootstrap.json",{cache:"force-cache"}).then(r=>r.ok?r.json():Promise.reject(new Error("checkpoint "+r.status)));
    if(ckpt.schema!=="cartpole-sonic-student-bootstrap/v1"||ckpt.mode!==mode) throw new Error("checkpoint schema/mode mismatch");
    t.restorePolicy(ckpt.policy,{teacherBootstrap:ckpt.teacherBootstrap,rngState:ckpt.rng});
    t.bootstrapEval=ckpt.defaultPresetEval||null;
  }catch(err){
    t.delete();throw new Error("학생 체크포인트 로드 실패 · 다시 시도: "+err.message);
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
  alignmentLab=null;
  if(temporalControlLab)temporalControlLab.delete();
  temporalControlLab=null;
  temporalControlLoadPromise=null;
  temporalControlBootstrapSnapshot=null;
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
  signalHistory.push({...controlSample(controlTick*CONTROL_DT,s,plannerContext[0],p.ref,p,lastForce,SONIC_TOY_CONSTANTS.STATE_SCALE[0]),episode:episodeIndex});
  controlTick++;
  if(signalHistory.length>MAX_SIGNAL_HISTORY) signalHistory.shift();
}
function terminationReason(s){return physicalFailureReason(s);}
function resetRobotState(reason="로봇 상태 초기화"){
  if(!physicsReady)throw new Error("physics not ready");
  sim.reset({x:0,xDot:0,theta:.1,thetaDot:0});
  plannerContext=[0,0];lastForce=0;accumulator=0;episodeIndex++;live=false;
  episodeTerminated=false;executionState="ready";executionReason="";
  resetSignalHistory(reason);render();
}
function resetRobot(){assertIdle();resetRobotState();}
function latchTermination(reason){
  episodeTerminated=true;live=false;executionState="terminated";executionReason=reason+" · 로봇 상태 초기화 후 다시 실행";
  lastEpisodeEvent="terminated: "+reason;accumulator=0;
}
function pushRobot(){
  assertEpisodeReady();sim.applyImpulse({xDotDelta:.75,thetaDotDelta:-1.25});
  lastEpisodeEvent="external impulse · reference unchanged";
  const reason=terminationReason(state());if(reason)latchTermination(reason);render();
}
function advanceOneControlStep(){
  const before=terminationReason(state());if(before){latchTermination(before);return false;}
  const p=preview(),s0=state();recordControlSample(p,s0);
  sim.stepForce(p.force,CONTROL_SUBSTEPS);plannerContext=advancePlannerContext(plannerContext,goal,CONTROL_DT);lastForce=sim.data.ctrl[0]*SONIC_TOY_CONSTANTS.ACTION_FORCE;
  const reason=terminationReason(state());if(reason)latchTermination(reason);
  return !reason;
}
function stepPolicy(steps=1){
  integerCount(steps,"control steps",1,100);assertEpisodeReady();
  try{for(let i=0;i<steps;i++)if(!advanceOneControlStep())break;}
  catch(error){showError(error);throw error;}finally{render();}
}
function setLive(enabled){
  if(typeof enabled!=="boolean")throw new Error("live must be boolean");
  if(enabled)assertEpisodeReady();else assertIdle();
  live=enabled;accumulator=0;if(!episodeTerminated&&executionState!=="error")executionState=enabled?"running":"paused";render();
}
function simulationControl({action,steps=1}){
  if(!["step","live_on","live_off","push","reset"].includes(action))throw new Error("unknown simulation action: "+action);
  integerCount(steps,"control steps",1,100);assertIdle();
  if(action==="step")stepPolicy(steps);else if(action==="live_on")setLive(true);else if(action==="live_off")setLive(false);else if(action==="push")pushRobot();else resetRobot();
}
function setGoal(x){
  finiteNumber(x,"goal x",-1.2,1.2);assertIdle();goal=x;
  $("goal").value=goal;
  resetSignalHistory("goal changed");
  updateUrl();
  render();
}
async function changePresetInternal(next){
  if(!Object.hasOwn(MuJoCoCartPole.presets,next))throw new Error("unknown preset: "+next);
  if(temporalControlModeRequested()&&next!=="playground")throw new Error("temporal comparison requires playground");
  clearTrainers();preset=next;await sim.loadPreset(preset);resetRobotState();
  currentTrainer=await getTrainer(requiredMode());$("simPreset").value=preset;updateUrl();
}
async function changePreset(next){
  assertIdle();if(!Object.hasOwn(MuJoCoCartPole.presets,next))throw new Error("unknown preset: "+next);
  if(temporalControlModeRequested()&&next!=="playground")throw new Error("temporal comparison requires playground");
  return withOperation("모델 로드",()=>changePresetInternal(next));
}
async function trainControllers(targets,ids,n,lab=null){
  lastTrainingTargets=ids.slice();
  for(let i=0;i<n;i++){
    for(const target of targets)target.iteration();render();await new Promise(r=>requestAnimationFrame(r));
  }
  if(lab)lab.evaluate();else{
    heldoutEval=targets[0].evaluate(12);ppoHeldoutHistory.push({iterations:targets[0].iter,mae:heldoutEval.trackingMae});alignmentLab=null;
  }
}
async function runPPO(iterations){
  integerCount(iterations,"PPO iterations",1,30);assertIdle();
  const resolved=resolveActiveController(),lab=temporalControlActive()?temporalControlLab:null;
  if(!resolved.trainingTargets.length)throw new Error("controller not ready");
  return withOperation("PPO 학습",()=>trainControllers(resolved.trainingTargets,resolved.trainingIds,iterations,lab));
}
function ensureAlignmentLab(){
  if(!currentTrainer)return null;
  if(!alignmentLab||alignmentLab.policy!==currentTrainer.policy)alignmentLab=new AlignmentLab(currentTrainer.policy);
  return alignmentLab;
}
async function runAlignment(steps=50){
  integerCount(steps,"training steps",1,200);assertIdle();
  const lab=ensureAlignmentLab();if(!lab)throw new Error("lab not ready");
  return withOperation("표현 학습",async()=>{
    lastTrainingTargets=["alignment-secondary"];
    for(let i=0;i<steps;i++){lab.trainStep();if(i%2===1||i===steps-1){render();await new Promise(r=>requestAnimationFrame(r));}}
  });
}

function resetAlignment(){
  assertIdle();
  const lab=ensureAlignmentLab();if(!lab)return;
  lab.reset();render();
}
function ensureTemporalTokenLab(){
  if(!temporalTokenLab)temporalTokenLab=new TemporalTokenLab();
  return temporalTokenLab;
}
async function runTemporalTokenTraining(steps=50){
  integerCount(steps,"training steps",1,250);assertIdle();
  const lab=ensureTemporalTokenLab();if(!lab)throw new Error("lab not ready");
  return withOperation("표현 학습",async()=>{
    lastTrainingTargets=["temporal-reconstruction-one/two"];
    for(let i=0;i<steps;i++){lab.trainStep();if(i%2===1||i===steps-1){render();await new Promise(r=>requestAnimationFrame(r));}}
  });
}

function resetTemporalTokenLab(){
  assertIdle();
  ensureTemporalTokenLab().reset();
  render();
}
async function ensureTemporalControlLabReady(){
  if(temporalControlLab)return temporalControlLab;
  if(temporalControlLoadPromise)return temporalControlLoadPromise;
  temporalControlLoadPromise=(async()=>{
    const snap=await fetch("./assets/temporal_control_bootstrap.json",{cache:"force-cache"}).then(r=>r.ok?r.json():Promise.reject(new Error("temporal control checkpoint "+r.status)));
    const lab=new TemporalControlLab(sim);
    try{lab.restore(snap);}catch(error){lab.delete();throw error;}
    temporalControlLab=lab;
    temporalControlBootstrapSnapshot=snap;
    return lab;
  })().finally(()=>{temporalControlLoadPromise=null;});
  return temporalControlLoadPromise;
}
async function runTemporalControlPPO(steps=5){
  integerCount(steps,"PPO iterations",1,30);assertIdle();
  return withOperation("두 컨트롤러 PPO 학습",async()=>{
    const lab=await ensureTemporalControlLabReady();
    return trainControllers([lab.one,lab.two],["temporal-one","temporal-two"],steps,lab);
  });
}
function resetTemporalControlLab(){
  assertIdle();
  if(!temporalControlLab||!temporalControlBootstrapSnapshot)return;
  temporalControlLab.restore(temporalControlBootstrapSnapshot);
  temporalControlSelected="one";
  resetSignalHistory("temporal control reset");
  render();
}
function selectTemporalController(which){
  assertIdle();if(!["one","two"].includes(which))throw new Error("unknown controller: "+which);
  if(!temporalControlLab)throw new Error("temporal controller not ready");
  temporalControlSelected=which;
  resetSignalHistory("controller switched");
  render();
}

function normalizeConcept(){
  const ids=(focus.concepts||[]).map(x=>x.id);
  if(!ids.length){conceptId=null;return;}
  if(!conceptId||!ids.includes(conceptId)) conceptId=focus.id==="quantizer"?"fsq":"core";
}
function updateUrl(){
  if(nativeLesson?.active)return;
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
  if(!SONIC_FLOW.some(n=>n.id===id))throw new Error("unknown node: "+id);
  if(concept!==null&&!(getNode(id).concepts||[]).some(c=>c.id===concept))throw new Error("unknown concept: "+concept);
  leaveRecordedLessonForNavigation();
  return withOperation("화면/표현 로드",()=>focusNodeInternal(id,concept));
}
async function focusNodeInternal(id,concept){
  const next=getNode(id);
  trainingMode=false;focus=next;focusId=next.id;
  conceptId=concept;normalizeConcept();
  if(focus.id==="token"&&conceptId==="temporal-control"&&preset!=="playground")await changePresetInternal("playground");
  await switchTrainerIfNeeded();
  if(focus.id==="token"&&conceptId==="temporal-control")await ensureTemporalControlLabReady();
  updateUrl();render();
}
async function setConcept(id){
  if(!(focus.concepts||[]).some(x=>x.id===id))throw new Error("unknown concept: "+id);
  return withOperation("표현 로드",()=>setConceptInternal(id));
}
async function setConceptInternal(id){
  conceptId=id;
  if(focus.id==="token"&&conceptId==="temporal-control"&&preset!=="playground")await changePresetInternal("playground");
  await switchTrainerIfNeeded();
  if(focus.id==="token"&&conceptId==="temporal-control")await ensureTemporalControlLabReady();
  updateUrl();render();
}
async function openTraining(topicId="loss-flow"){
  if(!TRAINING_TOPICS.some(t=>t.id===topicId))throw new Error("unknown training topic: "+topicId);
  leaveRecordedLessonForNavigation();
  return withOperation("학습 화면 로드",()=>openTrainingInternal(topicId));
}
async function openTrainingInternal(topicId){
  trainingMode=true;
  trainingTopic=getTrainingTopic(topicId);trainingTopicId=trainingTopic.id;
  await switchTrainerIfNeeded();
  if(trainingTopic.id==="ppo"&&!heldoutEval){
    heldoutEval=currentTrainer.evaluate(12);
    ppoHeldoutHistory=[{iterations:currentTrainer.iter,mae:heldoutEval.trackingMae}];
  }
  if(trainingTopic.id==="alignment")ensureAlignmentLab();
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
    b.innerHTML='<b>'+n.nav+'</b><span class="official">'+n.official+'</span><span class="toy">browser: '+blockShape(id,activeController(),n.toy)+'</span>';
    b.onclick=()=>{uiAction(()=>focusNode(id));};host.appendChild(b);
    if(i<topIds.length-1){const a=document.createElement("span");a.className="map-arrow";a.textContent="→";host.appendChild(a);}
  });
  const motion=getNode("motion-decoder");
  const mb=$("motionBranch");
  mb.classList.toggle("active",!trainingMode&&focus.id==="motion-decoder");
  mb.classList.toggle("training-hit",trainingMode&&trainingHits.includes("motion-decoder"));
  mb.innerHTML='<b>'+motion.nav+'</b><span>official: '+motion.official+' · toy: '+motion.toy+'</span>';
  mb.onclick=()=>{uiAction(()=>focusNode("motion-decoder"));};
  $("trainingButton").classList.toggle("active",trainingMode);
  $("trainingButton").onclick=()=>{uiAction(()=>openTraining(trainingTopicId));};
}
function buildConceptTabs(){
  const host=$("conceptTabs");host.innerHTML="";
  const items=trainingMode?TRAINING_TOPICS:(focus.concepts||[]);
  for(const item of items){
    const b=document.createElement("button");
    const id=trainingMode?item.id:item.id;
    b.textContent=trainingMode?item.label:item.label;
    b.classList.toggle("active",trainingMode?trainingTopic.id===id:conceptId===id);
    b.onclick=()=>{uiAction(()=>trainingMode?openTraining(id):setConcept(id));};
    host.appendChild(b);
  }
}

function renderSimulation(){
  const c=$("cart");
  if(c.clientWidth<2||c.clientHeight<42)return; // Hidden/reflowing canvas: no negative drawing radius.
  const {ctx,w:W,h:H}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);
  const s=state(),geometry=robotGeometry(W,H,sim.spec?.poleLength||1,s[0],s[2]);
  const {scale,railY,pivotY,wheelY,cartW,cartH}=geometry,centerX=W/2;
  ctx.strokeStyle="#cbd5e1";ctx.lineWidth=5;ctx.lineCap="round";
  ctx.beginPath();ctx.moveTo(centerX-1.8*scale,railY);ctx.lineTo(centerX+1.8*scale,railY);ctx.stroke();
  drawActual();
  const gx=centerX+goal*scale;
  ctx.strokeStyle="#16805d";ctx.lineWidth=2;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(gx,28);ctx.lineTo(gx,railY);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#16805d";ctx.font="600 12px system-ui";ctx.textAlign="center";ctx.fillText("goal",gx,22);
  // Runtime source is already named in the visible panel header.
  function drawActual(){
    const {cx,tx,ty}=geometry;
    ctx.strokeStyle="#d64f4f";ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(cx,pivotY);ctx.lineTo(tx,ty);ctx.stroke();
    ctx.fillStyle="#334155";ctx.fillRect(cx-cartW/2,pivotY,cartW,cartH);ctx.fillStyle="#1e293b";
    for(const dx of [-.19*scale,.19*scale]){ctx.beginPath();ctx.arc(cx+dx,wheelY,.07*scale,0,Math.PI*2);ctx.fill();}
  }
  $("goal").value=goal;$("goalVal").textContent=(goal>=0?"+":"")+fmt(goal,2)+" m";
  $("simPreset").disabled=busy||temporalControlModeRequested();
  $("liveBtn").textContent=live?"실행 멈추기":"계속 실행";$("liveBtn").classList.toggle("active",live);
  $("vX").textContent=fmt(s[0],3)+" m";$("vXd").textContent=fmt(s[1],3)+" m/s";
  $("vTh").textContent=fmt(s[2]*180/Math.PI,1)+"°";$("vThd").textContent=fmt(s[3]*180/Math.PI,1)+" °/s";
  $("vForce").textContent=fmt(lastForce,2)+" N";
  $("episodeStatus").textContent=executionText();
  $("episodeStatus").title=lastEpisodeEvent;
  const identity=activeController();
  $("activeController").textContent="현재 제어기 · "+controllerSummary(identity);
  if($("simNote")){
    $("simNote").innerHTML=temporalControlActive()
      ?("<b>Closed-loop ablation:</b> 브라우저 MuJoCo WASM 시뮬레이션은 현재 <b>"+(temporalControlSelected==="two"?"2-token":"1-token")+"</b> controller로 구동된다. 두 제어기는 같은 목표 궤적과 측정 상태를 사용한다.")
      :"LIVE 브라우저 계산: 현재 수업에 따라 AE·VQ·FSQ 제어기가 바뀐다. temporal 표현·alignment 학습은 별도 실험이며 왼쪽 제어기를 갱신하지 않는다.";
    $("simNote").innerHTML+="<br>상태를 그린 2D 도식이다. 화면 배율과 바퀴 그림은 물리 asset 자체가 아니다.";
    if(preset!=="playground"||Math.abs(goal)>.8)$("simNote").innerHTML+="<br><b>범위 주의:</b> 기본 모델·목표 범위 밖에서는 저장된 평가 성능을 보장하지 않는다.";
  }
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
    ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.textAlign="left";ctx.fillText(labels[comp],pad.l,top+12);
  }
  ctx.fillStyle="#7b8492";ctx.font="12px ui-monospace";ctx.textAlign="center";canvasTicks(ctx,Array.from({length:8},(_,i)=>"+"+fmt((i+1)*.08,2)+"s"),Array.from({length:8},(_,i)=>X(i)),h-12);
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
    ctx.fillStyle="#667085";ctx.font="12px ui-monospace";ctx.textAlign="center";ctx.fillText(i%2===0?"x":"ẋ",x,h-25);
    ctx.fillText(String(Math.floor(i/2)+1),x,h-10);
  }
  $("vizCaption").innerHTML='Encoder 입력을 실제 16개 숫자로 펼친 모습. frame 1~8마다 <b>x와 ẋ</b>가 한 쌍이며 actual proprioception은 여기에 포함되지 않는다.';
  setMetrics(["input dimension=16","8 frames × 2 features","pre-FSQ"]);
}
function renderEncoderCoreViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview(),ref=currentReference();
  const split=w*.58,pad=32,Y0=h*.54,scale=h*.31,n=ref.length,barW=(split-2*pad)/n*.7;
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.fillText("16D future reference",pad,20);
  ctx.strokeStyle="#e3e6eb";ctx.beginPath();ctx.moveTo(pad,Y0);ctx.lineTo(split-pad,Y0);ctx.stroke();
  for(let i=0;i<n;i++){
    const x=pad+(i+.5)*(split-2*pad)/n,v=ref[i],bh=Math.abs(v)*scale;
    ctx.fillStyle=i%2===0?"#315dc9":"#795fc5";ctx.fillRect(x-barW/2,v>=0?Y0-bh:Y0,barW,bh);
  }
  ctx.strokeStyle="#d7dce4";ctx.beginPath();ctx.moveTo(split,25);ctx.lineTo(split,h-25);ctx.stroke();
  const box={x:split+38,y:80,w:w-split-76,h:h-160};
  ctx.strokeStyle="#dfe4ea";ctx.strokeRect(box.x,box.y,box.w,box.h);
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.fillText("2D latent z",box.x,box.y-12);
  if(p){
    const lim=1.5,X=v=>box.x+(v+lim)/(2*lim)*box.w,Y=v=>box.y+box.h-(v+lim)/(2*lim)*box.h;
    ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(box.x,Y(0));ctx.lineTo(box.x+box.w,Y(0));ctx.moveTo(X(0),box.y);ctx.lineTo(X(0),box.y+box.h);ctx.stroke();
    ctx.fillStyle="#315dc9";ctx.beginPath();ctx.arc(X(p.z[0]),Y(p.z[1]),9,0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#172033";ctx.font="12px ui-monospace";canvasLines(ctx,"z=["+fmt(p.z[0],2)+", "+fmt(p.z[1],2)+"]",box.x+8,box.y+18,box.w-16);
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
    ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.fillText(label,pad.l,top+12);
  };
  drawPanel(0,pad.t,"future x");drawPanel(1,pad.t+panelH+gap,"future ẋ");
  $("vizCaption").innerHTML='<b>Autoencoder idea.</b> blue=input reference, purple=reconstruction. Decoder는 deployment 원본을 다시 쓰기 위해서가 아니라 작은 latent가 motion 정보를 보존하도록 학습 신호를 주기 위해 존재한다.';
  setMetrics(["16D → z(2D) → 16D","정규화 복원 MSE (무차원)="+fmt(reconstructionError(p,ref),4),"background concept for SONIC auxiliary decoder"]);
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
    for(let k=0;k<8;k++){const x=currentTrainer.policy.codebook[k*2],y=currentTrainer.policy.codebook[k*2+1];ctx.fillStyle="#9aa3af";ctx.beginPath();ctx.arc(X(x),Y(y),6,0,Math.PI*2);ctx.fill();ctx.fillStyle="#596273";ctx.font="12px ui-monospace";ctx.fillText(String(k),X(x)+8,Y(y)-5);}
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
    ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.fillText("quantize",(zx+qx)/2+5,(zy+qy)/2-5);
  }
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.fillText("z₁",w-48,Y(0)-8);ctx.fillText("z₂",X(0)+7,22);
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
        '<div class="flow-box purple"><strong>Decoder</strong><span>정규화 MSE (무차원) '+recon+'</span></div>'+
      '</div>'+
      '<table class="compare-table"><thead><tr><th>Training problem</th><th>Why it exists</th><th>Mechanism</th></tr></thead><tbody>'+
        '<tr><td>nearest lookup is non-differentiable</td><td>Encoder still needs gradient</td><td><b>STE</b>: backward treats quantization approximately like identity</td></tr>'+
        '<tr><td>Encoder can drift far from selected code</td><td>z should commit to usable codes</td><td><b>commitment loss</b></td></tr>'+
        '<tr><td>codebook must represent data</td><td>representative vectors need to move</td><td><b>codebook / EMA-style update</b></td></tr>'+
        '<tr><td>some codes may never be selected</td><td>capacity is wasted</td><td>개념: 미사용 코드 관리 (브라우저의 별도 갱신 경로 아님)</td></tr>'+
      '</tbody></table>'+
    '</div>',
    '<b>LIVE values + training explanation.</b> 현재 z/q/reconstruction 값은 실제 toy model에서 오지만, Live 버튼은 선택한 브라우저 제어기로 물리 시뮬레이션과 목표 궤적을 진행하며 weight를 재학습하지 않는다. 표는 VQ-VAE의 학습 개념이다. 브라우저 VQ는 STE·z 근접 항·코드 평균 갱신만 실행하며, 별도의 미사용 코드 관리는 구현하지 않는다. FSQ는 학습 코드북을 사용하지 않는다.',
    ["current code="+code,"정규화 복원 MSE (무차원)="+recon,"STE","commitment","codebook update"]
  );
}
function renderTokenViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview(),hist=signalHistory.filter(x=>x.episode===episodeIndex);
  const q=p?.q||[0,0];

  // Current toy token — always visible, even before Live starts.
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.textAlign="left";
  ctx.fillText("Current CartPole motion token (post-FSQ)",36,24);
  const cellY=38,cellW=Math.min(135,(w-84)/2),cellH=64;
  for(let i=0;i<2;i++){
    const x=36+i*(cellW+12);
    ctx.strokeStyle=i===0?"#315dc9":"#795fc5";ctx.lineWidth=2;ctx.strokeRect(x,cellY,cellW,cellH);
    ctx.fillStyle="#172033";ctx.font="700 12px system-ui";ctx.fillText("q"+(i+1),x+10,cellY+21);
    ctx.fillStyle=i===0?"#315dc9":"#795fc5";ctx.font="700 22px ui-monospace";ctx.fillText(fmt(q[i],2),x+10,cellY+50);
  }
  ctx.fillStyle="#667085";ctx.font="12px system-ui";
  ctx.fillText("양자화된 수치 벡터",36,105);

  // Schematic of actual released token shape: 2 tokens x 32 scalar dims.
  const gridTop=128,gridLeft=36,gridRight=w-36,cols=32,gap=2;
  const cw=(gridRight-gridLeft-(cols-1)*gap)/cols,ch=20;
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.fillText("Released SONIC token shape (schematic)",gridLeft,gridTop-10);
  for(let row=0;row<2;row++){
    ctx.fillStyle="#596273";ctx.font="12px ui-monospace";ctx.fillText("token "+(row+1),gridLeft,gridTop+row*34+14);
    for(let col=0;col<cols;col++){
      const x=gridLeft+58+col*(cw*.82+gap);
      const y=gridTop+row*34;
      ctx.fillStyle=row===0?"rgba(49,93,201,.55)":"rgba(121,95,197,.55)";
      ctx.fillRect(x,y,Math.max(4,cw*.78),ch);
    }
  }
  ctx.fillStyle="#667085";ctx.font="12px system-ui";canvasLines(ctx,"2 표현 칸 × 32 성분 = 64개 값 · 고정 수준",gridLeft,gridTop+77,w-gridLeft-24);

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
    ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.fillText("q"+(k+1)+" live discrete history",pad.l,top+12);
  }
  $("vizCaption").innerHTML='<b>Universal Token.</b> 위는 지금 decoder가 받는 toy q, 가운데는 실제 SONIC release의 token shape 개념, 아래는 Live에서 q가 finite level 사이를 step-wise 이동하는 history다.';
  setMetrics(["toy q=["+fmt(q[0],2)+","+fmt(q[1],2)+"]","SONIC: 2×32=64","32 fixed levels/scalar","token = numeric vector(s)"]);
}

function drawTemporalRecon(lab,ref){
  const c=$("temporalRecon");if(!c)return;
  const {ctx,w,h}=beginCanvas(c),cmp=lab.compare(ref);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  const pad={l:43,r:18,t:24,b:30},gap=22,panelH=(h-pad.t-pad.b-gap)/2;
  const X=i=>pad.l+i/7*(w-pad.l-pad.r);
  const Y=(v,top)=>top+panelH/2-v*(panelH*.42);
  const draw=(arr,comp,top,color,dash=[])=>{
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash);ctx.beginPath();
    for(let i=0;i<8;i++){const x=X(i),y=Y(arr[i*2+comp],top);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();ctx.setLineDash([]);
  };
  for(let comp=0;comp<2;comp++){
    const top=pad.t+comp*(panelH+gap);
    ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,Y(0,top));ctx.lineTo(w-pad.r,Y(0,top));ctx.stroke();
    draw(ref,comp,top,"#172033");
    draw(cmp.one.recon,comp,top,"#c97b20",[5,4]);
    draw(cmp.two.recon,comp,top,"#315dc9");
    ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.textAlign="left";ctx.fillText(comp===0?"future x":"future ẋ",pad.l,top+10);
  }
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.textAlign="right";ctx.fillText("black target · orange 1-token · blue 2-token",w-pad.r,14);
}
function drawTemporalCurve(lab){
  const c=$("temporalCurve");if(!c)return;
  const {ctx,w,h}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  const pts=[{step:0,...lab.baseline},...lab.history],pad={l:43,r:17,t:25,b:28},maxStep=Math.max(200,...pts.map(x=>x.step));
  const logs=pts.flatMap(x=>[Math.log10(Math.max(x.mseOne,1e-6)),Math.log10(Math.max(x.mseTwo,1e-6))]);
  const lo=Math.min(-3,...logs),hi=Math.max(-.7,...logs);
  const X=x=>pad.l+x/maxStep*(w-pad.l-pad.r),Y=v=>pad.t+(hi-v)/(hi-lo)*(h-pad.t-pad.b);
  ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,h-pad.b);ctx.lineTo(w-pad.r,h-pad.b);ctx.stroke();
  const draw=(key,color)=>{
    ctx.strokeStyle=color;ctx.lineWidth=2.2;ctx.beginPath();
    pts.forEach((p,i)=>{const x=X(p.step),y=Y(Math.log10(Math.max(p[key],1e-6)));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    pts.forEach(p=>{ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(p.step),Y(Math.log10(Math.max(p[key],1e-6))),2.5,0,Math.PI*2);ctx.fill();});
  };
  draw("mseOne","#c97b20");draw("mseTwo","#315dc9");
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.textAlign="left";ctx.fillText("log₁₀ reconstruction MSE ↓",pad.l,13);
  ctx.textAlign="right";ctx.fillText("training steps →",w-pad.r,h-8);
}
function renderTemporalTokenViz(){
  const lab=ensureTemporalTokenLab(),ref=currentReference(),cmp=lab.compare(ref),sens=lab.sensitivity(ref),snap=lab.snapshot(),m=snap.current,b=snap.baseline;
  const pct=v=>(v*100).toFixed(1)+"%";
  const sensCell=v=>'<span class="sens-cell">'+fmt(v,3)+'</span>';
  showHtml(
    '<div class="temporal-wrap">'+
      '<div class="temporal-pipelines">'+
        '<div class="temporal-pipe">'+
          '<div class="temporal-step"><b>Whole future window</b><span>8×[x,ẋ] = 16D</span></div><div class="temporal-arrow">→</div>'+
          '<div class="temporal-step"><b>Encoder</b><span>reads all 8 frames jointly</span></div><div class="temporal-arrow">→</div>'+
          '<div class="temporal-step"><b>1 token × 2 scalars</b><span>q=['+cmp.one.q[0].map(v=>fmt(v,2)).join(',')+']</span></div>'+
        '</div>'+
        '<div class="temporal-pipe two">'+
          '<div class="temporal-step"><b>Same whole window</b><span>NOT pre-split near/far</span></div><div class="temporal-arrow">→</div>'+
          '<div class="temporal-step"><b>Encoder</b><span>joint output → reshape</span></div><div class="temporal-arrow">→</div>'+
          '<div class="temporal-step"><b>2 tokens × 2 scalars</b><span>q₁=['+cmp.two.q[0].map(v=>fmt(v,2)).join(',')+']<br>q₂=['+cmp.two.q[1].map(v=>fmt(v,2)).join(',')+']</span></div>'+
        '</div>'+
      '</div>'+
      '<div class="temporal-metrics">'+
        '<div class="temporal-metric"><span>training step</span><b>'+snap.step+'</b><small>both models same batches</small></div>'+
        '<div class="temporal-metric"><span>1-token recon MSE</span><b>'+fmt(m.mseOne,5)+'</b><small>baseline '+fmt(b.mseOne,4)+'</small></div>'+
        '<div class="temporal-metric"><span>2-token recon MSE</span><b>'+fmt(m.mseTwo,5)+'</b><small>baseline '+fmt(b.mseTwo,4)+'</small></div>'+
        '<div class="temporal-metric"><span>2-token / 1-token MSE</span><b>'+pct(m.mseRatio)+'</b><small>lower = better reconstruction</small></div>'+
      '</div>'+
      '<div class="temporal-bottom">'+
        '<div class="temporal-left">'+
          '<div class="temporal-recon"><canvas id="temporalRecon"></canvas></div>'+
          '<div class="temporal-actions"><button id="temporalTrain50" class="primary">Train +50</button><button id="temporalTrain200">Train to 200</button><button id="temporalReset">학습 기준 복원</button></div>'+
        '</div>'+
        '<div class="temporal-right">'+
          '<div class="temporal-curve"><canvas id="temporalCurve"></canvas></div>'+
          '<div class="sensitivity"><b>Token-slot sensitivity diagnostic</b>'+
            '<table><thead><tr><th>input perturbation</th><th>slot 1 Δz</th><th>slot 2 Δz</th></tr></thead><tbody>'+
              '<tr><td>early frames 1–4</td><td>'+sensCell(sens.earlyLatentDelta[0])+'</td><td>'+sensCell(sens.earlyLatentDelta[1])+'</td></tr>'+
              '<tr><td>late frames 5–8</td><td>'+sensCell(sens.lateLatentDelta[0])+'</td><td>'+sensCell(sens.lateLatentDelta[1])+'</td></tr>'+
            '</tbody></table>'+
            '<div style="font-size:13px;line-height:1.3;color:#667085;margin-top:5px"><b>Do not label slot 1=near and slot 2=far.</b> Both slots read the entire window. This matrix only probes learned sensitivity after training.</div>'+
          '</div>'+
        '</div>'+
      '</div>'+
    '</div>',
    '<b>LIVE 1-token vs 2-token capacity lab.</b> 두 모델 모두 같은 whole future window를 읽는다. 차이는 output token slot 수뿐이다. reconstruction과 used code combinations를 비교하고, sensitivity matrix로 slot 역할이 사전 지정되지 않았음을 확인한다.',
    ["1-token codes="+m.uniqueCodesOne,"2-token codes="+m.uniqueCodesTwo,"official: max_num_tokens=2","SONIC token_dim=32"]
  );
  requestAnimationFrame(()=>{
    drawTemporalRecon(lab,ref);
    drawTemporalCurve(lab);
    const b50=$("temporalTrain50"),b200=$("temporalTrain200"),reset=$("temporalReset");
    if(b50)b50.onclick=()=>{uiAction(()=>runTemporalTokenTraining(50));};
    if(b200){b200.disabled=lab.step>=200;b200.onclick=()=>{if(lab.step<200)void runTemporalTokenTraining(200-lab.step);};}
    if(reset)reset.onclick=()=>resetTemporalTokenLab();
  });
}
function drawTemporalControlCurve(lab){
  const c=$("temporalControlCurve");if(!c)return;
  const {ctx,w,h}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  const pts=lab.evalHistory||[];
  const ref=temporalControlEvidence?.checkpoints||[];
  const all=[...pts,...ref];
  const pad={l:46,r:18,t:24,b:30},maxIter=Math.max(10,...all.map(x=>x.ppoIterations||0));
  const vals=[];
  for(const x of all){
    if(Number.isFinite(x?.one?.clean?.trackingMae))vals.push(x.one.clean.trackingMae);
    if(Number.isFinite(x?.two?.clean?.trackingMae))vals.push(x.two.clean.trackingMae);
  }
  const ymin=Math.min(.15,...vals),ymax=Math.max(.6,...vals);
  const X=i=>pad.l+(i/maxIter)*(w-pad.l-pad.r),Y=v=>h-pad.b-(v-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(let i=0;i<=4;i++){
    const v=ymin+(ymax-ymin)*i/4,y=Y(v);
    ctx.strokeStyle="#f0f2f5";ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillStyle="#7b8492";ctx.font="12px ui-monospace";ctx.textAlign="right";ctx.fillText(v.toFixed(2),pad.l-5,y+3);
  }
  const draw=(arr,get,color,dash=[])=>{
    const q=arr.filter(x=>Number.isFinite(get(x)));if(!q.length)return;
    ctx.strokeStyle=color;ctx.lineWidth=2.2;ctx.setLineDash(dash);ctx.beginPath();
    q.forEach((x,i)=>{const px=X(x.ppoIterations),py=Y(get(x));i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();ctx.setLineDash([]);
    q.forEach(x=>{ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(x.ppoIterations),Y(get(x)),3,0,Math.PI*2);ctx.fill();});
  };
  draw(ref,x=>x.one?.clean?.trackingMae,"rgba(201,123,32,.45)",[5,4]);
  draw(ref,x=>x.two?.clean?.trackingMae,"rgba(49,93,201,.40)",[5,4]);
  draw(pts,x=>x.one?.clean?.trackingMae,"#c97b20");
  draw(pts,x=>x.two?.clean?.trackingMae,"#315dc9");
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.textAlign="left";ctx.fillText("clean tracking MAE ↓",pad.l,13);
  ctx.textAlign="right";ctx.fillText("PPO iterations →",w-pad.r,h-8);
}
function renderTemporalControlViz(){
  const lab=temporalControlLab;
  if(!lab){
    showHtml('<div style="height:100%;display:grid;place-items:center"><div style="text-align:center"><b>Loading closed-loop controller lab…</b><div style="font-size:13px;color:#667085;margin-top:6px">precomputed 1-token / 2-token bootstrap checkpoint</div></div></div>','Closed-loop controller comparison is loading.');
    return;
  }
  const ref=currentReference(),s=state();
  const a=lab.one.preview(s,ref,goal),b=lab.two.preview(s,ref,goal),snap=lab.snapshot();
  const e=snap.evalHistory.at(-1);
  const baseline=snap.evalHistory[0];
  const verified10=temporalControlEvidence?.checkpoints?.find(x=>x.ppoIterations===10);
  const mae=x=>Number.isFinite(x)?fmt(x,3)+" m":"—";
  const pct=(n,d)=>d?((n/d)*100).toFixed(0)+"%":"—";
  showHtml(
    '<div class="control-ablation">'+
      '<div class="control-pipes">'+
        '<div class="control-pipe '+(temporalControlSelected==="one"?"selected":"")+'">'+
          '<div class="control-step"><b>16D future</b><span>whole window</span></div><div class="control-arrow">→</div>'+
          '<div class="control-step"><b>1 token FSQ</b><span>q2=['+a.q.map(v=>fmt(v,2)).join(',')+']</span></div><div class="control-arrow">+</div>'+
          '<div class="control-step"><b>state4 → Decoder</b><span>same proprioception</span></div><div class="control-arrow">→</div>'+
          '<div class="control-step"><b>'+fmt(a.force,2)+' N</b><span>MuJoCo</span></div>'+
        '</div>'+
        '<div class="control-pipe two '+(temporalControlSelected==="two"?"selected":"")+'">'+
          '<div class="control-step"><b>same 16D future</b><span>whole window</span></div><div class="control-arrow">→</div>'+
          '<div class="control-step"><b>2 token FSQ</b><span>q4=['+b.q.map(v=>fmt(v,2)).join(',')+']</span></div><div class="control-arrow">+</div>'+
          '<div class="control-step"><b>state4 → Decoder</b><span>same proprioception</span></div><div class="control-arrow">→</div>'+
          '<div class="control-step"><b>'+fmt(b.force,2)+' N</b><span>MuJoCo</span></div>'+
        '</div>'+
      '</div>'+
      '<div class="control-metrics">'+
        '<div class="control-metric"><span>matched PPO iter</span><b>'+lab.one.iter+'</b><small>same iteration budget</small></div>'+
        '<div class="control-metric"><span>1-token clean MAE</span><b>'+mae(e?.one?.clean?.trackingMae)+'</b><small>bootstrap '+mae(baseline?.one?.clean?.trackingMae)+'</small></div>'+
        '<div class="control-metric"><span>2-token clean MAE</span><b>'+mae(e?.two?.clean?.trackingMae)+'</b><small>bootstrap '+mae(baseline?.two?.clean?.trackingMae)+'</small></div>'+
        '<div class="control-metric"><span>2 / 1 clean MAE</span><b>'+pct(e?.two?.clean?.trackingMae,e?.one?.clean?.trackingMae)+'</b><small>lower is better</small></div>'+
      '</div>'+
      '<div class="control-bottom">'+
        '<div class="control-chart"><canvas id="temporalControlCurve"></canvas></div>'+
        '<div class="control-side">'+
          '<table><thead><tr><th>metric</th><th>1 token</th><th>2 tokens</th></tr></thead><tbody>'+
            '<tr><td>clean survival</td><td>'+(e?.one?.clean?.successes??"—")+'/12</td><td>'+(e?.two?.clean?.successes??"—")+'/12</td></tr>'+
            '<tr><td>clean tracking MAE</td><td>'+mae(e?.one?.clean?.trackingMae)+'</td><td>'+mae(e?.two?.clean?.trackingMae)+'</td></tr>'+
            '<tr><td>push post-MAE</td><td>'+mae(e?.one?.push?.disturbance?.postPushTrackingMae)+'</td><td>'+mae(e?.two?.push?.disturbance?.postPushTrackingMae)+'</td></tr>'+
            '<tr><td>verified +10 PPO</td><td>'+mae(verified10?.one?.clean?.trackingMae)+'</td><td>'+mae(verified10?.two?.clean?.trackingMae)+'</td></tr>'+
          '</tbody></table>'+
          '<div class="control-actions">'+
            '<button id="driveOneBtn" class="'+(temporalControlSelected==="one"?"primary":"")+'">Drive robot: 1 token</button>'+
            '<button id="driveTwoBtn" class="'+(temporalControlSelected==="two"?"primary":"")+'">Drive robot: 2 tokens</button>'+
            '<button id="controlPpo5" class="primary">PPO both +5</button><button id="controlPpo10">PPO both +10</button>'+
            '<button id="controlReset" style="grid-column:1/-1">학습 기준 복원</button>'+
          '</div>'+
          '<div class="control-note"><b>Interpretation:</b> the 2-token reconstruction lab had more capacity, but this matched-control ablation asks a different question. Extra token slots also enlarge the policy interface and can make PPO optimization harder. Same hyperparameters are intentionally used; neither controller is individually tuned.</div>'+
        '</div>'+
      '</div>'+
    '</div>',
    '<b>주황 = 1-token, 파랑 = 2-token. 실선·점 = 현재 브라우저 평가, 옅은 점선 = 저장된 평가.</b><br>왼쪽 MuJoCo WASM 시뮬레이션은 선택한 교육용 제어기가 구동한다. 두 제어기의 reference·현재 상태·teacher bootstrap·PPO 예산을 맞춰 비교한다. 복원이 좋아져도 제어가 자동으로 좋아지는 것은 아니다.',
    ["selected="+temporalControlSelected+"-token","1-token q dim=2","2-token q dim=4","matched 300-step teacher bootstrap"]
  );
  requestAnimationFrame(()=>{
    drawTemporalControlCurve(lab);
    const one=$("driveOneBtn"),two=$("driveTwoBtn"),p5=$("controlPpo5"),p10=$("controlPpo10"),reset=$("controlReset");
    if(one)one.onclick=()=>selectTemporalController("one");
    if(two)two.onclick=()=>selectTemporalController("two");
    if(p5)p5.onclick=()=>{uiAction(()=>runTemporalControlPPO(5));};
    if(p10)p10.onclick=()=>{uiAction(()=>runTemporalControlPPO(10));};
    if(reset)reset.onclick=()=>resetTemporalControlLab();
  });
}
function renderMotionDecoderViz(){
  renderAutoencoderViz();
  $("vizCaption").innerHTML='<b>LIVE Robot Motion Decoder / Kinematic Decoder.</b> token에서 future motion을 복원한다. 이 경로는 motion 정보를 token 안에 유지시키는 auxiliary reconstruction 역할이며 physical motor command가 아니다.';
}
function renderControlDecoderViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),p=preview(),s=state(),hist=signalHistory.filter(x=>x.episode===episodeIndex);
  const topH=220,layout=decoderLayout(w);
  const boxes=[
    {title:"motion token q",text:p?"["+p.q.map(v=>fmt(v,2)).join(", ")+"]":"—",color:"#315dc9"},
    {title:"measured state",text:"x="+fmt(s[0],2)+" m · ẋ="+fmt(s[1],2)+" m/s",text2:"θ="+fmt(s[2],2)+" rad · θ̇="+fmt(s[3],2)+" rad/s",color:"#667085"},
    {title:"Dynamic Decoder",text:"token + measured state",color:"#795fc5"},
    {title:"다음 힘 · next force",text:p?fmt(p.force,2)+" N":"—",color:"#16805d"},
  ];
  boxes.forEach((b,i)=>{const r=layout[i];ctx.strokeStyle=b.color;ctx.lineWidth=2;ctx.strokeRect(r.x,r.y,r.w,r.h);ctx.fillStyle="#172033";ctx.font="700 12px system-ui";ctx.fillText(b.title,r.x+8,r.y+22);ctx.fillStyle="#596273";ctx.font="12px system-ui";canvasLines(ctx,b.text,r.x+8,r.y+44,r.w-16);if(b.text2)ctx.fillText(b.text2,r.x+8,r.y+62);});
  ctx.strokeStyle="#7b8492";ctx.fillStyle="#7b8492";ctx.lineWidth=1.5;
  for(const edge of decoderConnections(w)){
    ctx.beginPath();edge.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();
    const [x,y]=edge.points.at(-1);
    if(edge.arrow){ctx.beginPath();ctx.moveTo(x,y);if(edge.arrow==="down"){ctx.lineTo(x-4,y-6);ctx.lineTo(x+4,y-6);}else{ctx.lineTo(x-6,y-4);ctx.lineTo(x-6,y+4);}ctx.closePath();ctx.fill();}
  }
  const pad={l:52,r:25,t:topH+25,b:40},pts=hist.length?hist:[{t:0,force:lastForce}],tMax=Math.max(.2,...pts.map(x=>x.t)),maxF=Math.max(10,...pts.map(x=>Math.abs(x.force)));
  const X=t=>pad.l+t/tMax*(w-pad.l-pad.r),Y=v=>h-pad.b-(v+maxF)/(2*maxF)*(h-pad.t-pad.b);
  ctx.strokeStyle="#e9ecf1";ctx.beginPath();ctx.moveTo(pad.l,Y(0));ctx.lineTo(w-pad.r,Y(0));ctx.stroke();
  ctx.strokeStyle="#16805d";ctx.lineWidth=2.5;ctx.beginPath();pts.forEach((x,i)=>{const px=X(x.t),py=Y(x.force);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();
  const lastPt=pts.at(-1);ctx.fillStyle="#16805d";ctx.beginPath();ctx.arc(X(lastPt.t),Y(lastPt.force),4,0,Math.PI*2);ctx.fill();
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.fillText("applied command history (N), sample time (s)",pad.l,pad.t-8);
  $("vizCaption").innerHTML='<b>LIVE Robot Control Decoder.</b> q는 motor command가 아니다. Push는 reference/token을 유지한 채 actual proprioception만 바꾸므로 action이 어떻게 달라지는지 바로 확인할 수 있다.';
  setMetrics(["token="+(p?"["+p.q.map(v=>fmt(v,2)).join(",")+"]":"—"),"θ="+fmt(s[2]*180/Math.PI,1)+"°","planned next="+fmt(p?.force,2)+" N","last applied="+fmt(lastForce,2)+" N"]);
}
function renderRobotTrackingViz(){
  const c=$("lessonViz"),{ctx,w,h}=beginCanvas(c),hist=signalHistory.filter(x=>x.episode===episodeIndex),p=preview();
  const pts=hist.length?hist:[{t:0,state:state(),reference:Array.from(currentReference()),referenceNow:plannerContext[0],force:lastForce}];
  const pad={l:55,r:24,t:34,b:45},gap=30,panelH=(h-pad.t-pad.b-gap)/2,tMax=Math.max(.2,...pts.map(x=>x.t));
  const X=t=>pad.l+t/tMax*(w-pad.l-pad.r);
  const top1=pad.t,Y1=v=>top1+panelH/2-v/1.8*(panelH*.45);
  ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,Y1(0));ctx.lineTo(w-pad.r,Y1(0));ctx.stroke();
  const draw=(get,color,dash=[])=>{ctx.strokeStyle=color;ctx.lineWidth=2.3;ctx.setLineDash(dash);ctx.beginPath();pts.forEach((s,i)=>{const x=X(s.t),y=Y1(get(s));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);};
  draw(s=>s.state[0],"#d64f4f");draw(s=>s.referenceNow,"#315dc9",[6,4]);draw(s=>s.reference[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],"#b76a12",[2,5]);
  ctx.fillStyle="#667085";ctx.font="12px system-ui";canvasLines(ctx,"x (m): 빨강 실제 · 파랑 같은 시각 목표 · 황색 +80ms",pad.l,top1+12,w-pad.l-pad.r);
  const top2=pad.t+panelH+gap,maxF=10,Y2=v=>top2+panelH/2-v/maxF*(panelH*.44);
  ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,Y2(0));ctx.lineTo(w-pad.r,Y2(0));ctx.stroke();
  ctx.strokeStyle="#16805d";ctx.lineWidth=2.3;ctx.beginPath();pts.forEach((s,i)=>{const x=X(s.t),y=Y2(s.force);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  ctx.fillStyle="#667085";ctx.fillText("applied command (N) · sample time (s)",pad.l,top2+12);
  const err=p?Math.abs(state()[0]-plannerContext[0]):NaN;
  $("vizCaption").innerHTML='<b>LIVE closed-loop result.</b> actual robot과 desired reference는 같은 것이 아니다. Dynamic Decoder action이 physics를 바꾸고 measured state가 다시 feedback된다.';
  setMetrics(["current x error="+fmt(err,3)+"m","episode="+episodeIndex,"상태="+executionSnapshot().state]);
}
function renderTrainingFlowViz(){
  showHtml(
    '<div class="training-grid"><div class="training-flow">'+
      '<div class="training-box main"><b>Physics rollout</b><br>reference + current state → action → MuJoCo</div>'+
      '<div class="training-box main"><b>PPO loss</b><br>tracking reward / advantage → Dynamic Decoder + Encoder</div>'+
      '<div class="training-box aux"><b>Reconstruction aux</b><br>token → Kinematic Decoder → future motion</div>'+
      '<div class="training-box aux"><b>Concept-only: official alignment</b><br>G1 robot / SMPL human body / teleoperation encoders; separate browser demo</div>'+
      '<div class="training-box"><b>Critic</b><br>value / return loss for PPO</div>'+
      '<div class="training-box"><b>FSQ</b><br>fixed levels; STE passes gradient, no learned codebook</div>'+
    '</div><div class="flow-diagram"><div class="flow-row"><div class="flow-box accent"><strong>LIVE browser update</strong><span>PPO + reconstruction; alignment demo is separate</span></div></div></div></div>',
    'Concept-only 공식 구조와 LIVE 브라우저 학습을 구분한다. 브라우저 PPO(Proximal Policy Optimization, 정책 개선)와 reconstruction(입력 복원)만 왼쪽 제어기를 갱신한다. alignment(표현 정렬)는 별도 교육 실험이다. SONIC training은 reconstruction 하나가 아니라 <b>physical PPO + representation auxiliary losses</b>를 함께 사용한다. deployment runtime graph와 training graph를 섞어 보지 않는 것이 중요하다.',
    ["PPO = physical tracking","aux = representation","FSQ = fixed bottleneck"]
  );
}
function renderLearningGraphViz(){
  showHtml(
    '<table class="compare-table"><thead><tr><th>Module</th><th>Trainable?</th><th>Main signal</th><th>What changes?</th></tr></thead><tbody>'+
    '<tr><td><b>Encoder(s)</b></td><td>YES</td><td>Browser: PPO + reconstruction; alignment demo separately</td><td>reference/modality → latent mapping</td></tr>'+
    '<tr><td><b>FSQ</b></td><td><b>NO</b></td><td>STE backward only</td><td>fixed levels do not move</td></tr>'+
    '<tr><td><b>Dynamic Decoder</b></td><td>YES</td><td>PPO tracking</td><td>token + state → action mapping</td></tr>'+
    '<tr><td><b>Kinematic Decoder</b></td><td>YES</td><td>reconstruction aux</td><td>token → future motion mapping</td></tr>'+
    '<tr><td><b>VQ codebook</b> (comparison)</td><td>YES</td><td>codebook / EMA update</td><td>representative vectors move</td></tr>'+
    '</tbody></table>',
    '<b>FSQ participates in training, but FSQ itself is not parameter-learned.</b> Encoder/Decoder가 fixed discrete bottleneck을 유용하게 쓰는 법을 배운다.',
    ["toy: 2 dims × 5 levels","SONIC: 2 tokens × 32 dims","32 fixed levels/scalar","flattened=64"]
  );
}
function drawAlignmentCurve(lab){
  const c=$("alignmentCurve");if(!c)return;
  const {ctx,w,h}=beginCanvas(c);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);
  const pts=[{step:0,...lab.baseline},...lab.history];
  const pad={l:48,r:18,t:24,b:28},gap=24,panelH=(h-pad.t-pad.b-gap)/2,maxStep=Math.max(50,...pts.map(x=>x.step));

  // Top: log10 latent MSE.
  const logs=pts.map(x=>Math.log10(Math.max(x.latentMse,1e-6)));
  const lo=Math.min(-4,...logs),hi=Math.max(-.5,...logs);
  const X=x=>pad.l+x/maxStep*(w-pad.l-pad.r);
  const Ylog=v=>pad.t+(hi-v)/(hi-lo)*panelH;
  ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,pad.t+panelH);ctx.lineTo(w-pad.r,pad.t+panelH);ctx.stroke();
  ctx.strokeStyle="#315dc9";ctx.lineWidth=2.3;ctx.beginPath();
  pts.forEach((p,i)=>{const x=X(p.step),y=Ylog(Math.log10(Math.max(p.latentMse,1e-6)));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  pts.forEach(p=>{ctx.fillStyle="#315dc9";ctx.beginPath();ctx.arc(X(p.step),Ylog(Math.log10(Math.max(p.latentMse,1e-6))),3.2,0,Math.PI*2);ctx.fill();});
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.textAlign="left";ctx.fillText("log₁₀ latent MSE ↓",pad.l,14);

  // Bottom: token agreement.
  const top2=pad.t+panelH+gap,Yagree=v=>top2+panelH-(v*panelH);
  ctx.strokeStyle="#eef0f3";ctx.beginPath();ctx.moveTo(pad.l,top2+panelH);ctx.lineTo(w-pad.r,top2+panelH);ctx.stroke();
  ctx.strokeStyle="#795fc5";ctx.lineWidth=2.3;ctx.beginPath();
  pts.forEach((p,i)=>{const x=X(p.step),y=Yagree(p.tokenAgreement);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  pts.forEach(p=>{ctx.fillStyle="#795fc5";ctx.beginPath();ctx.arc(X(p.step),Yagree(p.tokenAgreement),3.2,0,Math.PI*2);ctx.fill();});
  ctx.fillStyle="#667085";ctx.fillText("token agreement ↑",pad.l,top2-6);
  ctx.textAlign="right";ctx.fillText("alignment steps →",w-pad.r,h-8);
}
function renderAlignmentViz(){
  const lab=ensureAlignmentLab();
  if(!lab){
    showHtml('<div style="height:100%;display:grid;place-items:center">Alignment lab unavailable.</div>','Alignment lab requires the FSQ student policy.');
    return;
  }
  const ref=currentReference(),s=state(),cmp=lab.compare(ref,s,goal),snap=lab.snapshot();
  const b=snap.baseline,c=snap.current;
  const pct=v=>(v*100).toFixed(1)+"%";
  showHtml(
    '<div class="alignment-wrap">'+
      '<div class="align-lanes">'+
        '<div class="align-lane anchor">'+
          '<div class="align-step"><b>Full trajectory</b><span>8×[x,ẋ] = 16D</span></div><div class="align-arrow">→</div>'+
          '<div class="align-step"><b>Encoder A</b><span>primary anchor · frozen</span></div><div class="align-arrow">→</div>'+
          '<div class="align-step"><b>zA → FSQ → qA</b><span>z=['+cmp.primary.z.map(v=>fmt(v,2)).join(',')+']<br>q=['+cmp.primary.q.map(v=>fmt(v,2)).join(',')+']</span></div>'+
        '</div>'+
        '<div class="align-arrow">↔</div>'+
        '<div class="align-lane trainable">'+
          '<div class="align-step"><b>Sparse keypoints</b><span>frames 1,3,6,8 = 8D</span></div><div class="align-arrow">→</div>'+
          '<div class="align-step"><b>Encoder B</b><span>secondary · trainable</span></div><div class="align-arrow">→</div>'+
          '<div class="align-step"><b>zB → same FSQ → qB</b><span>z=['+cmp.secondary.z.map(v=>fmt(v,2)).join(',')+']<br>q=['+cmp.secondary.q.map(v=>fmt(v,2)).join(',')+']</span></div>'+
        '</div>'+
      '</div>'+
      '<div class="align-metrics">'+
        '<div class="align-metric"><span>alignment step</span><b>'+snap.step+'</b><small>secondary Encoder updates</small></div>'+
        '<div class="align-metric"><span>validation latent MSE ↓</span><b>'+fmt(c.latentMse,5)+'</b><small>before '+fmt(b.latentMse,4)+'</small></div>'+
        '<div class="align-metric"><span>token agreement ↑</span><b>'+pct(c.tokenAgreement)+'</b><small>before '+pct(b.tokenAgreement)+'</small></div>'+
        '<div class="align-metric"><span>same-state action MAE ↓</span><b>'+fmt(c.actionMae,4)+' N</b><small>before '+fmt(b.actionMae,3)+' N</small></div>'+
      '</div>'+
      '<div class="align-bottom">'+
        '<div class="align-chart"><canvas id="alignmentCurve"></canvas></div>'+
        '<div class="align-table">'+
          '<table><thead><tr><th>current same motion</th><th>Encoder A</th><th>Encoder B</th></tr></thead><tbody>'+
            '<tr><td>z</td><td>['+cmp.primary.z.map(v=>fmt(v,2)).join(',')+']</td><td>['+cmp.secondary.z.map(v=>fmt(v,2)).join(',')+']</td></tr>'+
            '<tr><td>FSQ q</td><td>['+cmp.primary.q.map(v=>fmt(v,2)).join(',')+']</td><td>['+cmp.secondary.q.map(v=>fmt(v,2)).join(',')+']</td></tr>'+
            '<tr><td>force, same state</td><td>'+fmt(cmp.primary.force,3)+' N</td><td>'+fmt(cmp.secondary.force,3)+' N</td></tr>'+
          '</tbody></table>'+
          '<div class="align-actions"><button id="alignmentTrainBtn" class="primary">Train +50</button><button id="alignmentResetBtn">학습 기준 복원</button></div>'+
          '<div style="font-size:13px;line-height:1.35;color:#667085;margin-top:6px">Toy simplification: Encoder A is a frozen anchor. SONIC aligns multiple modality Encoders jointly; the toy reproduces the alignment mechanism, not the real G1/SMPL/teleop modalities.</div>'+
        '</div>'+
      '</div>'+
    '</div>',
    '<b>LIVE multi-encoder alignment.</b> 같은 future motion을 16D full trajectory와 8D sparse keypoints로 다르게 표현한다. alignment loss가 Encoder B의 z를 Encoder A에 맞추면 같은 FSQ q와 같은-state action 의미로 수렴한다.',
    ["alignment step="+snap.step,"latent distance(now)="+fmt(cmp.latentDistance,4),"q same(now)="+(cmp.tokenSame?"YES":"NO"),"force Δ(now)="+fmt(cmp.actionDifference,4)+"N"]
  );
  requestAnimationFrame(()=>{
    drawAlignmentCurve(lab);
    const train=$("alignmentTrainBtn"),reset=$("alignmentResetBtn");
    if(train)train.onclick=()=>{uiAction(()=>runAlignment(50));};
    if(reset)reset.onclick=()=>resetAlignment();
  });
}

function renderOptimizerEvidenceViz(){
  renderOptimizerEvidencePanel({evidence:optimizerEvidence,showHtml,beginCanvas,rerender:render});
  $("vizCaption").innerHTML+='<br><b>Saved evidence:</b> 저장된 결과는 LIVE 계산과 별개다. Python native matched-cue 결과는 이 애니메이션에서 실행하지 않는다. <a href="https://github.com/tinmanlab/cartpole-sonic/blob/main/native/MATCHED_CUES.md">별도 native 검증 문서</a>';
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
  for(let i=0;i<=5;i++){const v=minMae+(maxMae-minMae)*i/5,y=Y(v);ctx.strokeStyle="#f0f2f5";ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillStyle="#7b8492";ctx.font="12px ui-monospace";ctx.textAlign="right";ctx.fillText(v.toFixed(2),pad.l-8,y+3);}
  const draw=(pts,color,dash,width)=>{if(!pts.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();pts.forEach((p,i)=>{const x=X(p.iterations),y=Y(p.mae);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);};
  draw(ref,"#9aa3af",[6,5],2);draw(rollout,"#315dc9",[],2.5);held.forEach(p=>{ctx.fillStyle="#16805d";ctx.beginPath();ctx.arc(X(p.iterations),Y(p.mae),5,0,Math.PI*2);ctx.fill();});
  ctx.fillStyle="#667085";ctx.font="12px system-ui";ctx.textAlign="left";ctx.fillText("tracking MAE [m] ↓",pad.l,18);ctx.textAlign="right";ctx.fillText("PPO iterations →",w-pad.r,h-15);
  $("vizCaption").innerHTML='회색 점선=deterministic held-out reference · 파랑=현재 PPO rollout MAE · 초록=현재 held-out check. <b>PPO는 physical tracking을 학습한다.</b>';
  setMetrics(["PPO iter="+currentTrainer.iter,heldoutEval?"held-out="+fmt(heldoutEval.trackingMae,3)+"m":"held-out=—"]);
}

function visualizationKind(){
  if(trainingMode){
    if(trainingTopic.id==="optimizer-sensitivity")return "evidence";
    return ["ppo","alignment"].includes(trainingTopic.id)?"live":"concept";
  }
  if(focus.id==="task") return "concept";
  if(focus.id==="encoder"&&conceptId==="vae") return "concept";
  return "live";
}
function renderVisualization(){
  clearViz();
  buildConceptTabs();
  const kind=visualizationKind(),vm=$("vizMode");
  vm.textContent=kind==="live"?"LIVE browser calculation":kind==="evidence"?"Saved evidence":"Concept-only";
  vm.title=kind==="evidence"?"EVIDENCE · deterministic repo ablation":kind==="live"?"Values calculated by the browser teaching implementation":"Explanatory architecture concept";
  vm.className="viz-mode "+kind;
  if(trainingMode){
    $("vizTitle").textContent=trainingTopic.title;
    $("vizSub").textContent="Training view · separate from deployment runtime";
    switch(trainingTopic.viz){
      case "training-flow":renderTrainingFlowViz();break;
      case "learning-graph":renderLearningGraphViz();break;
      case "alignment":renderAlignmentViz();break;
      case "optimizer-evidence":renderOptimizerEvidenceViz();break;
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
    case "token":
      if(conceptId==="temporal")renderTemporalTokenViz();
      else if(conceptId==="temporal-control")renderTemporalControlViz();
      else renderTokenViz();
      break;
    case "motion-decoder":renderMotionDecoderViz();break;
    case "control-decoder":renderControlDecoderViz();break;
    case "tracking":renderRobotTrackingViz();break;
  }
}

function activeConcept(){
  if(trainingMode)return null;
  if(!conceptId||conceptId==="core")return null;
  if((focus.concepts||[]).some(x=>x.id===conceptId)&&CONCEPT_TEXT[conceptId])return CONCEPT_TEXT[conceptId];
  return null;
}
function guideData(){
  if(trainingMode){
    return {
      kicker:"TRAINING · "+trainingTopic.label,
      title:trainingTopic.title,
      map:"Training-only view · 보라색 block = 이 topic의 gradient/representation 경로 (모두 trainable이라는 뜻은 아님)",
      input:trainingTopic.input,output:trainingTopic.output,
      question:trainingTopic.question,concept:null,
      details:TRAINING_DETAILS[trainingTopic.id]||{}
    };
  }
  const concept=activeConcept();
  return {
    kicker:concept?("CONCEPT INSIDE · "+focus.nav):("SONIC SYSTEM BLOCK · "+focus.nav),
    title:concept?concept.title:focus.title,
    map:"Official architecture concept: "+focus.official+" · Browser baseline: "+(conceptId==="temporal-control"?"1-token q2 / 2-token q4 comparison":focus.toy),
    input:concept?.input||focus.input,output:concept?.output||focus.output,
    question:concept?.question||focus.question,
    concept,
    details:conceptId==="temporal-control"?{...RUNTIME_DETAILS[focus.id],toyShape:CONCEPT_TEXT["temporal-control"].toyShape}:RUNTIME_DETAILS[focus.id]||{}
  };
}
function guideLiveValues(){
  const p=preview(),s=state(),ref=currentReference();
  const cell=(k,v)=>({k,v:String(v)});
  if(trainingMode){
    if(trainingTopic.id==="alignment"){
      const lab=ensureAlignmentLab(),snap=lab?.snapshot();
      return [
        cell("align step",snap?.step??0),
        cell("latent MSE",fmt(snap?.current?.latentMse,5)),
        cell("token agree",snap?((snap.current.tokenAgreement*100).toFixed(1)+"%"):"—"),
        cell("action MAE",snap?(fmt(snap.current.actionMae,4)+" N"):"—")
      ];
    }
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
      return [cell("planner x",fmt(plannerContext[0],3)+" m"),cell("planner ẋ",fmt(plannerContext[1],3)+" m/s"),cell("ref +80ms",fmt(ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)+" m"),cell("horizon","0.64 s")];
    case "reference":
      return [cell("input dim","16"),cell("frame 1 x",fmt(ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)+" m"),cell("frame 1 ẋ",fmt(ref[1]*SONIC_TOY_CONSTANTS.STATE_SCALE[1],3)+" m/s"),cell("measured x",fmt(s[0],3)+" m")];
    case "encoder":
      return [cell("z₁",fmt(p?.z?.[0],3)),cell("z₂",fmt(p?.z?.[1],3)),cell("recon MSE",fmt(reconstructionError(p,ref),4)),cell("mode",requiredMode())];
    case "quantizer":
      return [cell("z",p?"["+p.z.map(v=>fmt(v,2)).join(",")+"]":"—"),cell("q",p?"["+p.q.map(v=>fmt(v,2)).join(",")+"]":"—"),cell("mode",requiredMode()),cell("history",signalHistory.length+" samples")];
    case "token":
      if(conceptId==="temporal"){
        const lab=ensureTemporalTokenLab(),snap=lab.snapshot();
        return [cell("train step",snap.step),cell("1-token MSE",fmt(snap.current.mseOne,5)),cell("2-token MSE",fmt(snap.current.mseTwo,5)),cell("MSE ratio",(snap.current.mseRatio*100).toFixed(1)+"%")];
      }
      if(conceptId==="temporal-control"&&temporalControlLab){
        const e=temporalControlLab.evalHistory.at(-1);
        return [cell("selected",temporalControlSelected+"-token"),cell("PPO iter",temporalControlLab.one.iter),cell("1-token MAE",fmt(e?.one?.clean?.trackingMae,3)+" m"),cell("2-token MAE",fmt(e?.two?.clean?.trackingMae,3)+" m")];
      }
      return [cell("q₁",fmt(p?.q?.[0],2)),cell("q₂",fmt(p?.q?.[1],2)),cell("toy token","2 values"),cell("SONIC release","64 flattened")];
    case "motion-decoder":
      return [cell("recon MSE",fmt(reconstructionError(p,ref),4)),cell("recon x₁",fmt((p?.kinRecon?.[0]??NaN)*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)+" m"),cell("target x₁",fmt(ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)+" m"),cell("token",p?"["+p.q.map(v=>fmt(v,2)).join(",")+"]":"—")];
    case "control-decoder":
      return [cell("token",p?"["+p.q.map(v=>fmt(v,2)).join(",")+"]":"—"),cell("θ",fmt(s[2]*180/Math.PI,1)+"°"),cell("ẋ",fmt(s[1],3)+" m/s"),cell("planned next",fmt(p?.force,2)+" N")];
    case "robot":
      const rx=plannerContext[0];
      return [cell("actual x",fmt(s[0],3)+" m"),cell("target now",fmt(rx,3)+" m"),cell("preview +80ms",fmt(p?.ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],3)+" m"),cell("|error|",fmt(Math.abs(s[0]-rx),3)+" m"),cell("last applied",fmt(lastForce,2)+" N")];
    default:return [];
  }
}
function setGuideDepth(depth){
  assertIdle();if(!["easy","mechanism","sonic"].includes(depth))throw new Error("unknown explanation depth: "+depth);
  guideDepth=depth;updateUrl();renderGuide();
}

function guideActionSpec(){
  if(trainingMode){
    if(trainingTopic.id==="ppo")return{label:"+10 PPO iterations",disabled:false};
    if(trainingTopic.id==="alignment")return{label:"Train alignment +50",disabled:false};
    if(trainingTopic.id==="optimizer-sensitivity")return{label:"Open live closed-loop ablation",disabled:false};
    if(trainingTopic.id==="loss-flow"||trainingTopic.id==="what-learns")return{label:"Run 1 PPO iteration",disabled:false};
    return{label:"Concept only · no fake runtime action",disabled:true};
  }
  if(focus.id==="control-decoder")return{label:"Push robot + 1 Step",disabled:false};
  if(focus.id==="quantizer"){
    if(conceptId==="vqvae")return{label:live?"Stop live input":"Start live input · weights stay frozen",disabled:false};
    return{label:live?"Stop Live":"Start Live",disabled:false};
  }
  if(focus.id==="token"&&conceptId==="temporal")return{label:"Train both +50",disabled:false};
  if(focus.id==="token"&&conceptId==="temporal-control")return{label:"PPO both +5",disabled:false};
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
    s1Title="무엇을 계산하나";
    s1=c?.short||d.easy||"—";
    s2Title="한계 · 다음 단계";
    const miss=c?.ifMissing||d.ifMissing||"—";
    const next=(!trainingMode&&!c&&d.next)?(" 다음: "+d.next):"";
    s2="없으면: "+miss+next;
  }else if(guideDepth==="mechanism"){
    s1Title="무엇을 계산하나";
    s1=c?.mechanism||d.mechanism||"—";
    s2Title="왜 필요한가";
    s2=c?.why||(!trainingMode?focus.why:trainingTopic.why)||"—";
  }else{
    s1Title="고정된 원본 설정과 비교";
    s1=c?.sonic||d.sonic||"—";
    s2Title="교육용 구현과 원본의 차이";
    const mapping=trainingMode?"training-only topic":("toy: "+focus.toy+" · official: "+focus.official);
    const warning=c?.key||(!trainingMode?focus.misconception:trainingTopic.misconception)||"—";
    s2=mapping+" · "+warning;
  }
  $("guideSection1Title").textContent=s1Title;
  $("guideSection1").textContent=s1;
  $("guideSection2Title").textContent=s2Title;
  $("guideSection2").textContent=s2;

  const live=guideLiveValues();
  const hideLive=(guideDepth==="sonic")||(trainingMode&&!["ppo","alignment"].includes(trainingTopic.id))||(focus.id==="encoder"&&conceptId==="vae")||(focus.id==="quantizer"&&conceptId==="vqvae"&&guideDepth==="mechanism")||(focus.id==="token"&&["temporal","temporal-control"].includes(conceptId)&&guideDepth==="mechanism");
  $("guideLiveBlock").hidden=hideLive;
  $("guideLive").innerHTML=hideLive?"":live.map(x=>'<div class="live-kv"><span>'+x.k+'</span><b>'+x.v+'</b></div>').join("");

  const toyShape=c?.toyShape||d.toyShape||(!trainingMode?focus.toy:"—");
  const sonicShape=c?.sonicShape||d.sonicShape||(!trainingMode?focus.official:"—");
  $("guideShapeBlock").hidden=false;
  $("guideShape").textContent=((c?.formula||d.formula)?(c?.formula||d.formula)+"\n\n":"")+"브라우저: "+toyShape+"\nSONIC: "+sonicShape;
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
  assertIdle();
  if(trainingMode){
    if(trainingTopic.id==="optimizer-sensitivity")await focusNode("token","temporal-control");
    else if(trainingTopic.id==="ppo")await runPPO(10);
    else if(trainingTopic.id==="alignment")await runAlignment(50);
    else if(trainingTopic.id==="loss-flow"||trainingTopic.id==="what-learns")await runPPO(1);
    return;
  }
  if(focus.id==="control-decoder"){pushRobot();stepPolicy(1);return;}
  if(focus.id==="token"&&conceptId==="temporal"){await runTemporalTokenTraining(50);return;}
  if(focus.id==="token"&&conceptId==="temporal-control"){await runTemporalControlPPO(5);return;}
  if(["quantizer","token","robot"].includes(focus.id)){setLive(!live);return;}
  if(focus.id==="encoder"&&conceptId==="vae")return;
  setGoal(goal>0?-0.8:0.8);
}
function renderHeaderState(){
  setBadge("mcpBadge",webmcpMode==="unavailable"?"WebMCP pending":"WebMCP · "+webmcpTools.length,webmcpMode!=="unavailable");
}
function renderPreload(){
  if(nativeLesson?.active){renderHeaderState();return;}
  buildSystemMap();
  buildConceptTabs();
  renderSimulation();
  renderGuide();
  renderHeaderState();

  const kind=visualizationKind(),vm=$("vizMode");
  vm.textContent=kind==="live"?"LIVE · waiting for MuJoCo/model":kind==="evidence"?"EVIDENCE · loading deterministic repo ablation":"CONCEPT · explanatory, not toy runtime data";
  vm.title=kind==="evidence"?"EVIDENCE · deterministic repo ablation":kind==="live"?"Values calculated by the browser teaching implementation":"Explanatory architecture concept";
  vm.className="viz-mode "+kind;

  if(kind==="evidence"){
    clearViz();
    $("vizTitle").textContent=trainingTopic.title;
    $("vizSub").textContent="Deterministic repository evidence · separate from live browser training";
    renderOptimizerEvidenceViz();
  }
  // Concept-only screens do not need the physics model and can be useful immediately.
  else if(kind==="concept"){
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
    $("vizSub").textContent="Loading browser MuJoCo WASM + student checkpoint…";
    showHtml(
      '<div style="height:100%;display:grid;place-items:center"><div style="text-align:center"><b style="font-size:15px">Preparing live visualization</b><div style="margin-top:8px;color:#667085;font-size:13px">MuJoCo physics and the precomputed student checkpoint are loading.<br>The SONIC block explanation is already available on the right.</div></div></div>',
      'LIVE visualization will appear as soon as the actual model state is available.',
      ["loading physics","loading student model"]
    );
  }
}
function render(){
  if(nativeLesson?.active){renderHeaderState();return;}
  for(const [el,wasDisabled] of busyDisabledControls)el.disabled=wasDisabled;
  busyDisabledControls.clear();
  if(physicsReady&&currentTrainer){buildSystemMap();renderSimulation();renderVisualization();renderGuide();renderHeaderState();}
  const status=$("episodeStatus");
  if(status)status.textContent=executionText();
  // Recreated lesson buttons retain their own bounded/lesson-specific disabled rules.
  if(busy)for(const el of document.querySelectorAll(".system-map button,.shell button,.shell input,.shell select")){busyDisabledControls.set(el,el.disabled);el.disabled=true;}
  for(const id of ["stepBtn","liveBtn","pushBtn"])$(id).disabled=busy||episodeTerminated||executionState==="error"||!currentTrainer;
  $("resetBtn").disabled=busy||!physicsReady;$("goal").disabled=busy;
  $("simPreset").disabled=busy||temporalControlModeRequested();
  if(!busy)for(const el of document.querySelectorAll("[data-depth]"))el.disabled=false;
}

function liveSignals(p=preview(),ref=currentReference(),s=state()){
  const sampleTime=controlTick*CONTROL_DT;
  return {
    timeBase:"seconds since signal trace reset",simulationTime:sim.data?.time??null,
    sampleTime,targetTime:sampleTime,referenceNow:plannerContext[0],
    previewTargetTime:sampleTime+.08,referencePreview:ref[0]*SONIC_TOY_CONSTANTS.STATE_SCALE[0],
    plannedForce:p?.force??null,lastAppliedForce:lastForce,activeController:activeController(p),
    reference:Array.from(ref),latent:p?.z||null,token:p?.q||null,proprioception:s,
    actionMean:p?.mu??null,force:p?.force??null,
  };
}
function systemSnapshot(){
  const p=preview(),ref=currentReference(),s=state();
  return {
    ...nativeLesson?.snapshot(),
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
    activeController:activeController(p),lastTrainingTargets:lastTrainingTargets.slice(),execution:executionSnapshot(),
    experiment:{goal,live,preset,busy,representationMode:requiredMode(),episode:episodeIndex,autoResets:autoResetCount,lastEpisodeEvent},
    semantics:{
      reference:"desired future motion upstream of Encoder/FSQ; not measured robot state",
      latent:"continuous Encoder output",
      token:"post-VQ/FSQ quantized numeric motion representation; not motor command",
      proprioception:"measured actual robot state",
      action:"force is planned next command; lastAppliedForce is the last actuator command (N)",
      motionDecoder:"token → reconstructed future motion; auxiliary/kinematic role",
    },
    signals:{
      ...liveSignals(p,ref,s),
      drawing:robotGeometry($("cart").clientWidth,$("cart").clientHeight,sim.spec?.poleLength||1,s[0],s[2]),
      kinematicReconstruction:p?.kinRecon||null,
      liveHistory:signalHistory.slice(-80).map(x=>({...x}))
    },
    modelSemantics:{
      encoder:{trainable:true,signals:["PPO","reconstruction auxiliary"],separateDemo:"cross-encoder latent alignment"},
      fsq:{trainable:false,levels:"fixed",gradient:"STE through rounding; no learned vector codebook"},
      dynamicDecoder:{trainable:true,signals:["PPO tracking objective"]},
      kinematicDecoder:{trainable:true,signals:["future-motion reconstruction auxiliary loss"]},
      critic:{trainable:true,signals:["value/return loss"]},
      releasedTokenShape:{numTokens:2,scalarDimsPerToken:32,fixedLevelsPerScalar:32,flattenedDim:64},
    },
    training:currentTrainer?{
      ppoIterations:currentTrainer.iter,envSteps:currentTrainer.envSteps,episodes:currentTrainer.episodes,last:currentTrainer.last,
      heldoutEval,heldoutHistory:ppoHeldoutHistory,verifiedReference:ppoReferenceEvidence,
      alignment:alignmentLab?{...alignmentLab.snapshot(),live:alignmentLab.compare(ref,s,goal)}:null,
      optimizerEvidence:optimizerEvidence,optimizerEvidenceView:optimizerViewState()
    }:null,
    representationLabs:{
      temporalTokens:temporalTokenLab?{
        ...temporalTokenLab.snapshot(),
        live:temporalTokenLab.compare(ref),
        sensitivity:temporalTokenLab.sensitivity(ref)
      }:null,
      temporalControl:temporalControlLab?{
        ...temporalControlLab.snapshot(),
        selectedController:temporalControlSelected,
        verifiedReference:temporalControlEvidence
      }:null
    },
    backends:{physics:physicsReady?sim.backend:null,webgpu:webgpuStatus,webmcp:{mode:webmcpMode,tools:webmcpTools}}
  };
}
// Local periodic telemetry is bounded. Full weights and static evidence are returned only on explicit reads.
function telemetrySnapshot(){
  const p=preview(),brief=t=>t?{iter:t.iter,envSteps:t.envSteps,last:t.last,parameters:t.policy.parameterBreakdown()}:null;
  return {
    schema:"cartpole-sonic-telemetry/v2",
    ...nativeLesson?.snapshot(),
    system:{version:COURSE_VERSION,mode:trainingMode?"training":"runtime",focus:trainingMode?null:{node:focus.id,concept:conceptId},trainingTopic:trainingMode?trainingTopic.id:null},
    activeController:activeController(p),lastTrainingTargets:lastTrainingTargets.slice(),execution:executionSnapshot(),
    experiment:{goal,live,preset,busy,episode:episodeIndex,autoResets:autoResetCount},
    signals:liveSignals(p),
    training:currentTrainer?{ppoIterations:currentTrainer.iter,envSteps:currentTrainer.envSteps,last:currentTrainer.last}:null,
    representationLabs:{temporalControl:temporalControlLab?{selectedController:temporalControlSelected,one:brief(temporalControlLab.one),two:brief(temporalControlLab.two)}:null},
    optimizerEvidence:{revision:optimizerEvidence?.revision??null,uri:"./evidence/control_optimization_eval.json",view:optimizerViewState()},
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
    {name:"simulation_control",description:"Control the shared browser MuJoCo WASM robot.",inputSchema:{type:"object",properties:{action:{type:"string",enum:["step","live_on","live_off","push","reset"]},steps:{type:"integer",minimum:1,maximum:100}},required:["action"]},annotations:{readOnlyHint:false},execute:async(args)=>{simulationControl(args);return systemSnapshot();}},
    {name:"experiment_set_goal",description:"Set the high-level task goal used by the motion generator.",inputSchema:{type:"object",properties:{x:{type:"number",minimum:-1.2,maximum:1.2}},required:["x"]},annotations:{readOnlyHint:false},execute:async({x})=>{setGoal(x);return systemSnapshot();}},
    {name:"training_run",description:"Train the displayed student policy; in the 1/2-token comparison train both controllers equally. Read lastTrainingTargets and execution status for the actual target and outcome.",inputSchema:{type:"object",properties:{iterations:{type:"integer",minimum:1,maximum:30}},required:["iterations"]},annotations:{readOnlyHint:false},execute:async({iterations})=>{await runPPO(iterations);return systemSnapshot();}},
    {name:"alignment_control",description:"Train or reset the live CartPole multi-encoder alignment lab. The tool first switches to the alignment training view/FSQ student, then aligns a sparse-keypoint Encoder to the frozen full-trajectory Encoder and reports latent/token/action agreement.",inputSchema:{type:"object",properties:{action:{type:"string",enum:["train","reset"]},steps:{type:"integer",minimum:1,maximum:200}},required:["action"]},annotations:{readOnlyHint:false},execute:async({action,steps=50})=>{assertIdle();if(!["train","reset"].includes(action))throw new Error("unknown alignment action");integerCount(steps,"alignment steps",1,200);if(!trainingMode||trainingTopic.id!=="alignment")await openTraining("alignment");if(action==="reset")resetAlignment();else await runAlignment(steps);return systemSnapshot();}},
    {name:"temporal_token_control",description:"Train or reset the live 1-token vs 2-token temporal-slot lab. The tool switches to Universal Token → 1 vs 2 token slots and compares reconstruction capacity without assigning near/far semantics to token indices.",inputSchema:{type:"object",properties:{action:{type:"string",enum:["train","reset"]},steps:{type:"integer",minimum:1,maximum:250}},required:["action"]},annotations:{readOnlyHint:false},execute:async({action,steps=50})=>{assertIdle();if(!["train","reset"].includes(action))throw new Error("unknown token action");integerCount(steps,"token steps",1,250);if(trainingMode||focus.id!=="token"||conceptId!=="temporal")await focusNode("token","temporal");if(action==="reset")resetTemporalTokenLab();else await runTemporalTokenTraining(steps);return systemSnapshot();}},
    {name:"temporal_control_control",description:"Operate the matched 1-token vs 2-token closed-loop control ablation. Switches to Universal Token → Closed-loop 1 vs 2 on the default Playground model, can select which controller drives the shared MuJoCo robot, train both with the same PPO budget, or reset to the matched teacher-bootstrap checkpoint.",inputSchema:{type:"object",properties:{action:{type:"string",enum:["train","reset","select"]},steps:{type:"integer",minimum:1,maximum:30},controller:{type:"string",enum:["one","two"]}},required:["action"]},annotations:{readOnlyHint:false},execute:async({action,steps=5,controller="one"})=>{assertIdle();if(!["train","reset","select"].includes(action))throw new Error("unknown temporal action");integerCount(steps,"PPO steps",1,30);if(!["one","two"].includes(controller))throw new Error("unknown controller");if(!temporalControlModeRequested())await focusNode("token","temporal-control");await ensureTemporalControlLabReady();if(action==="reset")resetTemporalControlLab();else if(action==="select")selectTemporalController(controller);else await runTemporalControlPPO(steps);return systemSnapshot();}},
    {name:"simulation_set_model",description:"Switch browser MuJoCo WASM CartPole dynamics/morphology preset.",inputSchema:{type:"object",properties:{preset:{type:"string",enum:["playground","long","heavy"]}},required:["preset"]},annotations:{readOnlyHint:false},execute:async({preset})=>{await changePreset(preset);return systemSnapshot();}}
  ];
  for(const t of tools)await mc.registerTool(t);webmcpTools=tools.map(t=>t.name);renderHeaderState();
}
function attachUI(){
  $("goal").oninput=()=>uiAction(()=>setGoal(Number($("goal").value)));$("stepBtn").onclick=()=>uiAction(()=>simulationControl({action:"step"}));$("liveBtn").onclick=()=>uiAction(()=>simulationControl({action:live?"live_off":"live_on"}));
  $("pushBtn").onclick=()=>uiAction(()=>simulationControl({action:"push"}));$("resetBtn").onclick=()=>uiAction(()=>simulationControl({action:"reset"}));$("simPreset").onchange=()=>uiAction(()=>changePreset($("simPreset").value));
  $("focusAction").onclick=()=>{uiAction(()=>runFocusAction());};
}
function installCanvasResizeObserver(){
  if(!("ResizeObserver" in window))return;
  const ro=new ResizeObserver(()=>{if(physicsReady&&currentTrainer)render();});ro.observe($("cart"));ro.observe($("lessonViz"));window.__cartpoleSonicResizeObserver=ro;
}
function loop(now){
  labAnimation=0;
  if(nativeLesson?.active)return;
  const dt=Math.min(.05,Math.max(0,(now-lastFrame)/1000));lastFrame=now;
  if(live&&!busy&&currentTrainer&&physicsReady){
    accumulator+=dt;let n=0;
    while(accumulator>=CONTROL_DT&&n<4){
      accumulator-=CONTROL_DT;n++;
      try{if(!advanceOneControlStep())break;}catch(error){showError(error);break;}

    }
    render();
  }
  labAnimation=requestAnimationFrame(loop);
}
async function init(){
  attachUI();normalizeConcept();
  nativeLesson=mountNativeLesson({root:$("nativeLesson"),onEnter(){setLive(false);cancelAnimationFrame(labAnimation);labAnimation=0;$("nativeLessonTab").classList.add("active");$("explorationTab").classList.remove("active");},onLeave(){updateUrl();render();lastFrame=performance.now();if(!labAnimation)labAnimation=requestAnimationFrame(loop);$("nativeLessonTab").classList.remove("active");$("explorationTab").classList.add("active");}});
  $("nativeLessonTab").onclick=()=>{if(!nativeLesson.active){nativeLesson.enter();const u=new URL(location.href);u.searchParams.set("lesson","future");history.replaceState(null,"",u);}};
  $("explorationTab").onclick=()=>{if(nativeLesson.active)nativeLesson.leave();};
  if(startsInLesson)nativeLesson.enter();else $("explorationTab").classList.add("active");
  busy=true;
  if(focus.id==="token"&&conceptId==="temporal-control")preset="playground";
  renderPreload();
  teacherPromise=loadTeacherPolicy().then(t=>(teacher=t,t)).catch(err=>{console.warn("teacher unavailable",err);return null;});
  fetch("./evidence/ppo_eval.json",{cache:"no-store"}).then(r=>r.ok?r.json():null).then(x=>{ppoReferenceEvidence=x;if(trainingMode&&trainingTopic.id==="ppo"&&!busy)render();}).catch(()=>{});
  fetch("./evidence/temporal_control_eval.json",{cache:"no-store"}).then(r=>r.ok?r.json():null).then(x=>{temporalControlEvidence=x;if(!trainingMode&&focus.id==="token"&&conceptId==="temporal-control"&&!busy)render();}).catch(()=>{});
  fetch("./evidence/control_optimization_eval.json",{cache:"no-store"}).then(r=>r.ok?r.json():null).then(x=>{optimizerEvidence=x;if(trainingMode&&trainingTopic.id==="optimizer-sensitivity"&&!busy)render();}).catch(()=>{});
  await sim.init(preset);physicsReady=true;$("simPreset").value=preset;resetRobotState();
  setBadge("physicsBadge",sim.backend,true);
  currentTrainer=await getTrainer(requiredMode());
  if(!trainingMode&&focus.id==="token"&&conceptId==="temporal-control")await ensureTemporalControlLabReady();
  busy=false;
  if(trainingMode&&trainingTopic.id==="ppo"){heldoutEval=currentTrainer.evaluate(12);ppoHeldoutHistory=[{iterations:currentTrainer.iter,mae:heldoutEval.trackingMae}];}
  if(trainingMode&&trainingTopic.id==="alignment")ensureAlignmentLab();
  updateUrl();render();
  registerWebMCP().then(()=>renderHeaderState()).catch(()=>{});
  probeWebGPUFSQ().then(status=>{webgpuStatus=status;setBadge("webgpuBadge",status.ok?"WebGPU FSQ ✓":status.available?"WebGPU fallback":"WebGPU unavailable",status.ok);}).catch(err=>{webgpuStatus={available:false,ok:false,reason:err.message};setBadge("webgpuBadge","WebGPU unavailable",false);});
  if(location.hostname==="localhost"||location.hostname==="127.0.0.1"){setInterval(()=>{fetch("/telemetry",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(telemetrySnapshot()),keepalive:true}).catch(()=>{});},900);}
  installCanvasResizeObserver();if(!nativeLesson?.active)labAnimation=requestAnimationFrame(loop);
}
init().catch(err=>{busy=false;showError(err);render();setBadge("physicsBadge","model load error",false);if(!nativeLesson?.active)labAnimation=requestAnimationFrame(loop);});

window.__cartpoleSonic={
  getState:systemSnapshot,
  getTelemetry:telemetrySnapshot,
  focus:focusNode,
  concept:setConcept,
  openTraining,
  runFocusAction,
  step:stepPolicy,
  simulationControl,
  setLive,
  reset:resetRobot,
  push:pushRobot,
  changePreset,
  setGoal,
  runPPO,
  runAlignment,
  resetAlignment,
  runTemporalTokenTraining,
  resetTemporalTokenLab,
  runTemporalControlPPO,
  resetTemporalControlLab,
  selectTemporalController,
};

// Explicit navigation may leave read-only replay; simulation and training never do so implicitly.
function leaveRecordedLessonForNavigation(){
  if(!nativeLesson?.active)return;
  if(busy)throw new Error("모델 준비 중입니다. 준비가 끝나면 구조 탐색으로 전환하세요.");
  nativeLesson.leave();
}
