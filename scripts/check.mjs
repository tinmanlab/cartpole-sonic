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
  [html,'sim-card',"shared simulation panel"],
  [html,'viz-card',"single fixed main visualization panel"],
  [html,'guide-card',"single fixed lesson guide panel"],
  [html,'process-card',"stable information-flow strip"],
  [html,'id="lessonViz"',"single main visualization canvas"],
  [html,'id="stepBtn"',"shared one-step control"],
  [html,'id="liveBtn"',"shared live control"],
  [html,'data-stage="reference"',"stable reference stage"],
  [html,'data-stage="dynamic"',"stable dynamic decoder stage"],
  [course,'PRIMARY_PATH = [',"canonical course primary path"],
  [course,'"ae",\n  "vq",\n  "vqvae",\n  "fsq",',"AE→VQ→VQ-VAE→FSQ ordering"],
  [course,'OPTIONAL_BRANCHES',"optional branch map"],
  [course,'ae: ["vae"]',"VAE is an optional AE branch"],
  [course,'title: "왜 차원을 줄였다가 다시 복원할까?"',"dimension-bottleneck question is lesson 1"],
  [course,'Decoder는 deployment에서 원본을 다시 쓰기 위한 장치가 아니다',"decoder role is explicit"],
  [app,'import {\n  COURSE_VERSION',"app consumes the course SSOT"],
  [app,'switch(lesson.viz)',"one fixed visualization slot changes by lesson"],
  [app,'course_get_outline',"semantic WebMCP outline tool"],
  [app,'course_get_state',"semantic WebMCP state tool"],
  [app,'course_navigate',"semantic WebMCP navigation"],
  [app,'course_run_lesson_action',"semantic WebMCP lesson action"],
  [app,'simulation_control',"shared simulation control tool"],
  [app,'experiment_set_goal',"semantic goal setter"],
  [app,'training_run',"semantic PPO training tool"],
  [app,'courseSnapshot()',"canonical WebMCP/readback state"],
  [html,'Robot world · actual only',"simulation/reference separation"],
  [html,'aspect-ratio:16/9',"simulation aspect is preserved"],
  [html,'aspect-ratio:15/8',"main visualization aspect is preserved"],
  [app,'function beginCanvas(canvas)',"DPR-aware canvas backing-store sync"],
  [app,'Reference world · pre-FSQ',"reference world explicitly precedes FSQ"],
  [app,'token:"post-FSQ/VQ compact motion representation; not a motor command"',"WebMCP token semantics"],
  [app,'render();\n    await new Promise(r=>requestAnimationFrame(r));',"PPO graph redraws each iteration"],
  [app,'student_"+mode+"_bootstrap.json',"precomputed student checkpoint loading"],
  [toy,'restorePolicy(snapshot',"checkpoint restore path"],
  [app,'plannerContext=[0,0]',"runtime reference state independent from robot state"],
  [toy,'planReference(context,goal',"planner implementation"],
  [toy,'[quant.q[0],quant.q[1],...sNorm]',"dynamic decoder receives token plus proprioception"],
  [toy,'this.kinematicDecoder=new MLP',"kinematic auxiliary decoder"],
  [toy,'ratio=Math.exp',"PPO optimization"],
  [sim,'MjModel.from_xml_string',"native MuJoCo model"],
  [sim,'mj_step',"native MuJoCo stepping"],
  [gpu,'round(tanh(src[i]) * 1.998) / 2.0',"real WebGPU FSQ parity kernel"],
];
for(const [src,token,label] of required){
  if(!src.includes(token)) throw new Error("missing contract: "+label);
}

const forbidden=[
  [app,'const LESSONS = [',"lesson content duplicated in app instead of course.js"],
  [app,'panelText',"separate per-panel lesson text"],
  [html,'data-panel=',"lesson-specific panel shuffling"],
  [html,'lesson-card',"old variable lesson-card dashboard"],
  [app,'set_quantizer_mode',"old low-level WebMCP API"],
  [app,'set_learning_lesson',"old numeric lesson API"],
  [app,'get_sonic_cartpole_state',"old non-course state API"],
  [html,'height:610px',"artificial lesson stretching"],
  [html,'height:470px',"artificial lesson stretching"],
  [app,'drawGhost(',"reference ghost must not be drawn in robot simulation"],
];
for(const [src,token,label] of forbidden){
  if(src.includes(token)) throw new Error("forbidden legacy pattern: "+label);
}

console.log("COURSE_WEBMCP_CONTRACT_OK");
