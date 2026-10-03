import fs from "node:fs";

const html=fs.readFileSync(new URL("../index.html",import.meta.url),"utf8");
const app=fs.readFileSync(new URL("../app.js",import.meta.url),"utf8");
const toy=fs.readFileSync(new URL("../sonic_toy.js",import.meta.url),"utf8");
const sim=fs.readFileSync(new URL("../mujoco_sim.js",import.meta.url),"utf8");
const teacher=fs.readFileSync(new URL("../teacher_policy.js",import.meta.url),"utf8");

const required=[
 [html,"High-level 목표 → Planner","goal-to-planner surface"],
 [html,"lesson-nav","progressive lesson navigation"],
 [app,"왜 굳이 CartPole로 축소하는가?","reduction rationale lesson"],
 [app,"const LESSONS = [","lesson content map"],
 [app,"setLesson(n)","lesson routing"],
 [app,"set_learning_lesson","WebMCP lesson routing"],
 [html,"mini-legend","local visual legends"],
 [app,"LESSON_MODES","lesson-to-representation routing"],
 [app,'panels:["latent","reference"]',"VQ-VAE lesson pairs latent with reconstruction evidence"],
 [app,'panels:["goal","runtime","live"]',"Dynamic Decoder lesson puts runtime in the wide center"],
 [app,'panels:["train","live"]',"PPO lesson removes redundant runtime panel and pairs training with result"],
 [app,"liveTitle","context-sensitive panel headings"],
 [html,"SONIC-like runtime path","SONIC-like runtime surface"],
 [html,"Training only · Kinematic Decoder","kinematic auxiliary path"],
 [html,"Training only · Critic","critic training-only path"],
 [app,"referenceFramesPhysical","planner reference rendering"],
 [app,"trainer.preview(state(),currentReference(),goal)","runtime policy uses current state plus planned motion"],
 [toy,"planReference(context,goal","planner implementation"],
 [toy,"refContext=[0,0]","training reference state independent from robot state"],
 [app,"plannerContext = [0,0]","runtime planner state independent from robot state"],
 [toy,"this.encoder=new MLP","motion encoder"],
 [toy,"this.dynamicDecoder=new MLP","dynamic decoder"],
 [toy,"this.kinematicDecoder=new MLP","kinematic decoder"],
 [toy,"this.critic=new MLP","critic"],
 [toy,"fn",""],
 [toy,"fsqScalar","FSQ bottleneck"],
 [toy,"[quant.q[0],quant.q[1],...sNorm]","dynamic decoder receives token plus proprioception"],
 [toy,"rewardFor","tracking reward"],
 [toy,"gae(data","GAE"],
 [toy,"ratio=Math.exp","PPO ratio"],
 [toy,"auxLoss","kinematic reconstruction auxiliary loss"],
 [toy,"this.sim.mujoco.mj_step(this.sim.model,this.data);\n    this.sim.mujoco.mj_step(this.sim.model,this.data);","50 Hz policy/control parity via two 10 ms MuJoCo steps"],
 [toy,"advancePlannerContext(this.refContext,this.goal,.02)","planner advances at the 50 Hz control period"],
 [sim,"MjModel.from_xml_string","native MuJoCo model compilation"],
 [sim,"mj_step","native MuJoCo step"],
 [html,"Frozen teacher bootstrap","teacher is explicitly lab-only"],
 [app,"bootstrapFromTeacher","trained teacher is actually used"],
 [teacher,"class FrozenCartPoleTeacher","frozen teacher adapter"]
].filter(x=>x[1]!=="fn");
for(const [src,token,label] of required)if(!src.includes(token))throw new Error("missing: "+label);

const forbidden=[
 [app,"normalizedMse","state reconstruction must not be the main runtime objective"],
 [app,"WebGPUTrainer","old autoencoder WebGPU trainer must not drive the SONIC-like path"],
 [html,"Decoder reconstruction","old observer-centric framing"],
 [html,"state → 압축","old state-autoencoder framing"],
 [html,"height:610px","lesson cards must not be vertically stretched"],
 [html,"height:420px","lesson runtime must not be artificially stretched"],
 [html,"height:470px","lesson latent panel must not be artificially stretched"],
 [html,"grid-template-rows:180px 592px","lesson layout must use natural row heights"]
];
for(const [src,token,label] of forbidden)if(src.includes(token))throw new Error("forbidden: "+label);

console.log("SONIC_STRUCTURE_CONTRACT_OK");
