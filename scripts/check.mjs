import fs from "node:fs";

const html=fs.readFileSync(new URL("../index.html",import.meta.url),"utf8");
const app=fs.readFileSync(new URL("../app.js",import.meta.url),"utf8");
const course=fs.readFileSync(new URL("../course.js",import.meta.url),"utf8");
const toy=fs.readFileSync(new URL("../sonic_toy.js",import.meta.url),"utf8");
const sim=fs.readFileSync(new URL("../mujoco_sim.js",import.meta.url),"utf8");
const gpu=fs.readFileSync(new URL("../webgpu_fsq.js",import.meta.url),"utf8");
const alignment=fs.readFileSync(new URL("../alignment_lab.js",import.meta.url),"utf8");
const temporal=fs.readFileSync(new URL("../temporal_token_lab.js",import.meta.url),"utf8");
const temporalControl=fs.readFileSync(new URL("../temporal_control_lab.js",import.meta.url),"utf8");

for(const name of ["student_ae_bootstrap.json","student_vq_bootstrap.json","student_fsq_bootstrap.json"]){
  const u=new URL("../assets/"+name,import.meta.url);
  if(!fs.existsSync(u)) throw new Error("missing checkpoint asset: "+name);
  const j=JSON.parse(fs.readFileSync(u,"utf8"));
  if(j.schema!=="cartpole-sonic-student-bootstrap/v1") throw new Error("invalid checkpoint schema: "+name);
}

for(const [name,schema] of [["temporal_control_bootstrap.json","cartpole-sonic-temporal-control-lab/v1"]]){
  const u=new URL("../assets/"+name,import.meta.url);
  if(!fs.existsSync(u))throw new Error("missing temporal-control asset: "+name);
  const j=JSON.parse(fs.readFileSync(u,"utf8"));
  if(j.schema!==schema)throw new Error("invalid temporal-control asset schema: "+name);
}
const controlEvidenceUrl=new URL("../evidence/temporal_control_eval.json",import.meta.url);
if(!fs.existsSync(controlEvidenceUrl))throw new Error("missing temporal-control evidence");
if(JSON.parse(fs.readFileSync(controlEvidenceUrl,"utf8")).schema!=="cartpole-sonic-temporal-control-eval/v1")throw new Error("invalid temporal-control evidence schema");
const optimizerEvidenceUrl=new URL("../evidence/control_optimization_eval.json",import.meta.url);
if(!fs.existsSync(optimizerEvidenceUrl))throw new Error("missing optimizer evidence");
if(JSON.parse(fs.readFileSync(optimizerEvidenceUrl,"utf8")).schema!=="cartpole-sonic-control-optimization-ablation/v1")throw new Error("invalid optimizer evidence schema");

const required=[
  [html,'system-map',"fixed SONIC system map"],
  [html,'id="runtimeFlow"',"clickable SONIC runtime flow"],
  [html,'id="motionBranch"',"Robot Motion Decoder branch"],
  [html,'id="trainingButton"',"separate training view"],
  [html,'sim-card',"shared actual-robot simulation"],
  [html,'viz-card',"single fixed visualization panel"],
  [html,'guide-card',"single fixed explanation panel"],
  [html,'id="conceptTabs"',"contextual concept tabs"],
  [html,'id="guideDepthTabs"',"three-level explanation depth control"],
  [html,'feedback-loop',"explicit closed-loop proprioception feedback"],
  [course,'RUNTIME_DETAILS',"systematic runtime explanation contract"],
  [course,'TRAINING_DETAILS',"systematic training explanation contract"],
  [app,'guideDepth',"shared explanation-depth state"],
  [app,'training-hit',"training topics highlight affected runtime blocks"],
  [app,'sonic_set_explanation_depth',"WebMCP controls the same explanation depth"],
  [app,'focusExplanation:',"WebMCP exposes focused explanation contract"],
  [html,'id="stepBtn"',"shared one-step control"],
  [html,'id="liveBtn"',"shared live control"],
  [course,'SONIC_FLOW = [',"canonical SONIC runtime map"],
  [course,'id:"task"',"Task block"],
  [course,'id:"generator"',"Motion Generator block"],
  [course,'id:"reference"',"Motion Reference block"],
  [course,'id:"encoder"',"Encoder block"],
  [course,'id:"quantizer"',"FSQ Quantizer block"],
  [course,'id:"token"',"Universal Token block"],
  [course,'id:"motion-decoder"',"Robot Motion Decoder block"],
  [course,'id:"control-decoder"',"Robot Control Decoder block"],
  [course,'id:"robot"',"Robot feedback block"],
  [course,'TRAINING_TOPICS',"training topics are separate from runtime map"],
  [course,'{id:"vae",label:"VAE · optional"}',"VAE is contextual optional background"],
  [course,'{id:"vqvae",label:"VQ-VAE"}',"VQ-VAE is contextual to Quantizer"],
  [course,'id:"what-learns"',"What-learns is training topic, not runtime block"],
  [app,'function buildSystemMap()',"top SONIC map rendering"],
  [app,'function buildConceptTabs()',"contextual concept navigation"],
  [app,'function renderVqvaeViz()',"VQ-VAE has its own detailed visualization"],
  [app,'function renderVaeViz()',"VAE has explicit optional explanation"],
  [app,'function renderTokenViz()',"motion token has live visualization"],
  [app,'function renderControlDecoderViz()',"control decoder has live visualization"],
  [app,'function renderRobotTrackingViz()',"robot tracking has live visualization"],
  [app,'function renderAlignmentViz()',"alignment has a dedicated live visualization"],
  [alignment,'class AlignmentLab',"live secondary-encoder alignment lab"],
  [alignment,'sparseKeypointReference',"second motion representation"],
  [alignment,'tokenAgreement',"alignment validates token agreement"],
  [alignment,'actionMae',"alignment validates same-state action agreement"],
  [app,'function runAlignment(steps=50)',"bounded live alignment training"],
  [app,'alignment_control',"WebMCP alignment control tool"],
  [course,'{id:"temporal",label:"1 vs 2 token slots"}',"temporal-token experiment lives under Universal Token"],
  [app,'function renderTemporalTokenViz()',"live 1-token vs 2-token visualization"],
  [temporal,'class TemporalTokenLab',"live temporal-token capacity lab"],
  [temporal,'tokens=2',"two-token model"],
  [temporal,'earlyLatentDelta',"token-slot sensitivity diagnostic"],
  [app,'temporal_token_control',"WebMCP temporal-token control tool"],
  [course,'{id:"temporal-control",label:"Closed-loop 1 vs 2"}',"closed-loop token-count ablation lives under Universal Token"],
  [app,'function renderTemporalControlViz()',"live 1-token vs 2-token closed-loop visualization"],
  [temporalControl,'class TemporalControlLab',"closed-loop temporal-token control lab"],
  [temporalControl,'class TemporalControlTrainer',"matched controller trainer"],
  [temporalControl,'disturbance:true',"disturbance evaluation path"],
  [app,'temporal_control_control',"WebMCP closed-loop temporal control tool"],
  [app,'temporalControlActive()',"shared MuJoCo can be driven by selected ablation controller"],
  [course,'id:"optimizer-sensitivity"',"optimizer-sensitivity training topic"],
  [app,'function renderOptimizerEvidenceViz()',"deterministic optimizer evidence visualization"],
  [app,'EVIDENCE · deterministic repo ablation',"evidence is explicitly distinct from LIVE and CONCEPT"],
  [app,'optimizerEvidence',"optimizer evidence is exposed in state"],
  [app,'visualizationKind()',"live vs concept visualization is explicit"],
  [app,'sonic_get_map',"semantic WebMCP system-map tool"],
  [app,'sonic_get_state',"semantic WebMCP state tool"],
  [app,'sonic_focus',"semantic WebMCP runtime focus"],
  [app,'sonic_open_training',"semantic WebMCP training focus"],
  [app,'simulation_control',"shared robot control tool"],
  [app,'experiment_set_goal',"goal-setting tool"],
  [app,'training_run',"PPO training tool"],
  [app,'systemSnapshot()',"canonical WebMCP/readback state"],
  [app,'fsq:{trainable:false',"FSQ trainability semantics"],
  [app,'releasedTokenShape:{numTokens:2,scalarDimsPerToken:32,fixedLevelsPerScalar:32,flattenedDim:64}',"released token-shape semantics"],
  [app,'signalHistory',"live graphs use actual episode history"],
  [app,'latchTermination',"manual and Live failure latch"],
  [sim,'applyImpulse(',"robot-only push"],
  [toy,'[quant.q[0],quant.q[1],...sNorm]',"Dynamic Decoder gets token + proprioception"],
  [toy,'this.kinematicDecoder=new MLP',"kinematic auxiliary decoder"],
  [sim,'MjModel.from_xml_string',"native MuJoCo model"],
  [gpu,'round(tanh(src[i]) * 1.998) / 2.0',"WebGPU FSQ kernel"],
];
for(const [src,token,label] of required){
  if(!src.includes(token)) throw new Error("missing contract: "+label);
}

const forbidden=[
  [course,'PRIMARY_PATH',"old concept curriculum must not drive top runtime map"],
  [course,'OPTIONAL_BRANCHES',"old curriculum branch structure"],
  [app,'navigateLesson',"old lesson navigation"],
  [app,'course_get_outline',"old course-level WebMCP API"],
  [app,'course_navigate',"old course-level WebMCP navigation"],
  [html,'process-card',"duplicate bottom runtime diagram"],
  [html,'lesson-card',"old lesson dashboard"],
  [app,'drawGhost(',"reference ghost must not be drawn in robot simulation"],
  [app,'for(let g=-1.2',"hypothetical goal sweep must not be mixed into live latent plot"],
];
for(const [src,token,label] of forbidden){
  if(src.includes(token)) throw new Error("forbidden legacy pattern: "+label);
}

console.log("SONIC_SYSTEM_MAP_CONTRACT_OK");
