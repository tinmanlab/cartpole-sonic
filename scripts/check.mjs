import fs from "node:fs";
import {SONIC_FLOW} from "../course.js";

const read=name=>fs.readFileSync(new URL("../"+name,import.meta.url),"utf8");
const html=read("index.html"),app=read("app.js"),course=read("course.js");
const toy=read("sonic_toy.js"),sim=read("mujoco_sim.js"),gpu=read("webgpu_fsq.js");
const alignment=read("alignment_lab.js"),temporal=read("temporal_token_lab.js"),temporalControl=read("temporal_control_lab.js");

for(const name of ["student_ae_bootstrap.json","student_vq_bootstrap.json","student_fsq_bootstrap.json"]){
  const j=JSON.parse(read("assets/"+name));
  if(j.schema!=="cartpole-sonic-student-bootstrap/v1")throw new Error("invalid checkpoint schema: "+name);
}
for(const [file,schema] of [
  ["assets/temporal_control_bootstrap.json","cartpole-sonic-temporal-control-lab/v1"],
  ["evidence/temporal_control_eval.json","cartpole-sonic-temporal-control-eval/v1"],
  ["evidence/control_optimization_eval.json","cartpole-sonic-control-optimization-ablation/v1"]
])if(JSON.parse(read(file)).schema!==schema)throw new Error("invalid evidence/checkpoint schema: "+file);

// Public concepts are identified by stable IDs, not one language's display label.
for(const [node,concept] of [["encoder","vae"],["quantizer","vqvae"],["token","temporal"],["token","temporal-control"]]){
  if(!SONIC_FLOW.find(n=>n.id===node)?.concepts?.some(c=>c.id===concept))throw new Error("missing contextual concept: "+node+"/"+concept);
}
const required=[
  [html,'system-map',"fixed SONIC system map"],
  [html,'id="runtimeFlow"',"clickable SONIC runtime flow"],
  [html,'id="motionBranch"',"Robot Motion Decoder branch"],
  [html,'id="trainingButton"',"separate training view"],
  [html,'sim-card',"shared actual-robot simulation"],
  [html,'viz-card',"single visualization panel"],
  [html,'guide-card',"single explanation panel"],
  [html,'id="conceptTabs"',"contextual concept tabs"],
  [html,'id="guideDepthTabs"',"three-level explanation depth control"],
  [html,'feedback-loop',"explicit closed-loop proprioception feedback"],
  [course,'RUNTIME_DETAILS',"runtime explanation contract"],
  [course,'TRAINING_DETAILS',"training explanation contract"],
  [app,'guideDepth',"shared explanation-depth state"],
  [app,'training-hit',"training topics highlight affected runtime blocks"],
  [app,'sonic_set_explanation_depth',"shared explanation depth tool"],
  [app,'focusExplanation:',"focused explanation contract"],
  [html,'id="stepBtn"',"shared one-step control"],
  [html,'id="liveBtn"',"shared live control"],
  [course,'SONIC_FLOW = [',"canonical runtime map"],
  ...["task","generator","reference","encoder","quantizer","token","motion-decoder","control-decoder","robot"].map(id=>[course,'id:"'+id+'"',id+' block']),
  [course,'TRAINING_TOPICS',"separate training topics"],
  [course,'id:"what-learns"',"what-learns training topic"],
  ...["buildSystemMap","buildConceptTabs","renderVqvaeViz","renderVaeViz","renderTokenViz","renderControlDecoderViz","renderRobotTrackingViz","renderAlignmentViz","renderTemporalTokenViz","renderTemporalControlViz","renderOptimizerEvidenceViz"].map(name=>[app,'function '+name+'()',name]),
  [alignment,'class AlignmentLab',"secondary-encoder alignment lab"],
  [alignment,'sparseKeypointReference',"second motion representation"],
  [alignment,'tokenAgreement',"token agreement"],
  [alignment,'actionMae',"same-state action agreement"],
  [app,'function runAlignment(steps=50)',"bounded alignment training"],
  [temporal,'class TemporalTokenLab',"temporal-token capacity lab"],
  [temporal,'tokens=2',"two-token model"],
  [temporal,'earlyLatentDelta',"token-slot sensitivity diagnostic"],
  [temporalControl,'class TemporalControlLab',"closed-loop temporal-token lab"],
  [temporalControl,'class TemporalControlTrainer',"matched controller trainer"],
  [temporalControl,'disturbance:true',"disturbance evaluation"],
  [app,'temporalControlActive()',"selected ablation controller"],
  [course,'id:"optimizer-sensitivity"',"optimizer evidence topic"],
  [app,'EVIDENCE · deterministic repo ablation',"stored versus live evidence"],
  [app,'optimizerEvidence',"optimizer evidence state"],
  [app,'visualizationKind()',"live versus concept classification"],
  ...["alignment_control","temporal_token_control","temporal_control_control","sonic_get_map","sonic_get_state","sonic_focus","sonic_open_training","simulation_control","experiment_set_goal","training_run"].map(name=>[app,name,name+' tool']),
  [app,'systemSnapshot()',"canonical state"],
  [app,'fsq:{trainable:false',"fixed FSQ levels"],
  [app,'releasedTokenShape:{numTokens:2,scalarDimsPerToken:32,fixedLevelsPerScalar:32,flattenedDim:64}',"released configuration shape"],
  [app,'signalHistory',"actual episode history"],
  [app,'latchTermination',"manual and live failure latch"],
  [sim,'applyImpulse(',"robot-only push"],
  [toy,'[quant.q[0],quant.q[1],...sNorm]',"token plus proprioception"],
  [toy,'this.kinematicDecoder=new MLP',"auxiliary reconstruction decoder"],
  [sim,'MjModel.from_xml_string',"native physics model"],
  [gpu,'round(tanh(src[i]) * 1.998) / 2.0',"WebGPU FSQ kernel"]
];
for(const [src,token,label] of required)if(!src.includes(token))throw new Error("missing contract: "+label);
for(const [src,token,label] of [
  [course,'PRIMARY_PATH',"old curriculum runtime map"],
  [course,'OPTIONAL_BRANCHES',"old curriculum branch structure"],
  [app,'navigateLesson',"old lesson navigation"],
  [app,'course_get_outline',"old course-level API"],
  [app,'course_navigate',"old course-level navigation"],
  [html,'process-card',"duplicate bottom runtime diagram"],
  [html,'lesson-card',"old dashboard"],
  [app,'drawGhost(',"reference ghost mixed with actual robot"],
  [app,'for(let g=-1.2',"hypothetical goal sweep mixed with live latent"]
])if(src.includes(token))throw new Error("forbidden legacy pattern: "+label);
console.log("SONIC_SYSTEM_MAP_CONTRACT_OK");
