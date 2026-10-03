import fs from "node:fs";

const html=fs.readFileSync(new URL("../index.html",import.meta.url),"utf8");
const app=fs.readFileSync(new URL("../app.js",import.meta.url),"utf8");
const course=fs.readFileSync(new URL("../course.js",import.meta.url),"utf8");
const toy=fs.readFileSync(new URL("../sonic_toy.js",import.meta.url),"utf8");
const sim=fs.readFileSync(new URL("../mujoco_sim.js",import.meta.url),"utf8");
const gpu=fs.readFileSync(new URL("../webgpu_fsq.js",import.meta.url),"utf8");

for(const name of ["student_ae_bootstrap.json","student_vq_bootstrap.json","student_fsq_bootstrap.json"]){
  const u=new URL("../assets/"+name,import.meta.url);
  if(!fs.existsSync(u)) throw new Error("missing checkpoint asset: "+name);
  const j=JSON.parse(fs.readFileSync(u,"utf8"));
  if(j.schema!=="cartpole-sonic-student-bootstrap/v1") throw new Error("invalid checkpoint schema: "+name);
}

const required=[
  [html,'system-map',"fixed SONIC system map"],
  [html,'id="runtimeFlow"',"clickable SONIC runtime flow"],
  [html,'id="motionBranch"',"Robot Motion Decoder branch"],
  [html,'id="trainingButton"',"separate training view"],
  [html,'sim-card',"shared actual-robot simulation"],
  [html,'viz-card',"single fixed visualization panel"],
  [html,'guide-card',"single fixed explanation panel"],
  [html,'id="conceptTabs"',"contextual concept tabs"],
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
  [app,'function renderAlignmentViz()',"alignment concept is explicitly visualized"],
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
  [app,'autoResetEpisode',"Live auto-reset"],
  [sim,'applyImpulse({xDotDelta=0.75, thetaDotDelta=-1.25}',"robot-only push"],
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
