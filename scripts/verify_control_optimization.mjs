import fs from "node:fs";
import {MuJoCoCartPole} from "../mujoco_sim.js";
import {FrozenCartPoleTeacher} from "../teacher_policy.js";
import {TemporalControlTrainer} from "../temporal_control_lab.js";

const evidence=JSON.parse(fs.readFileSync(new URL("../evidence/control_optimization_eval.json",import.meta.url),"utf8"));
const bootstrap=JSON.parse(fs.readFileSync(new URL("../assets/temporal_control_bootstrap.json",import.meta.url),"utf8"));
const teacher=new FrozenCartPoleTeacher(JSON.parse(fs.readFileSync(new URL("../assets/teacher_cartpole_ppo.json",import.meta.url),"utf8")));

if(evidence?.schema!=="cartpole-sonic-control-optimization-ablation/v1")throw new Error("invalid optimization evidence schema");

const sim=new MuJoCoCartPole();
await sim.init("playground");
const close=(a,b,tol=1e-12)=>Math.abs(a-b)<=tol;
const expected=id=>evidence.variants.find(x=>x.id===id);

// 1) Exact matched parameter count does not remove the gap.
const matched=new TemporalControlTrainer(sim,{tokens:2,seed:9118,widths:{encoder:18,dynamic:26,kinematic:20,critic:20}});
matched.bootstrapFromTeacher(teacher,{steps:300,batch:256,lr:.0015,auxCoef:.08,dataSeed:0xC011AB});
if(matched.policy.parameterCount()!==1220)throw new Error("matched-capacity controller is not 1220 parameters");
for(let i=0;i<10;i++)matched.iteration();
const matchedEval=matched.evaluate({episodes:12,seed:8181});
if(!close(matchedEval.trackingMae,expected("two-matched-capacity").after.cleanMae))throw new Error("matched-capacity evidence mismatch");
matched.delete();

// 2) Encoder freeze only partially improves the default 2-token result.
const frozen=new TemporalControlTrainer(sim,{tokens:2,seed:9118,ppoModuleScales:{encoder:0,dynamic:1,kinematic:1,critic:1}});
frozen.restore(bootstrap.two);
for(let i=0;i<10;i++)frozen.iteration();
const frozenEval=frozen.evaluate({episodes:12,seed:8181});
if(!close(frozenEval.trackingMae,expected("two-freeze-encoder").after.cleanMae))throw new Error("freeze-Encoder evidence mismatch");
frozen.delete();

// 3) Small coupled actor steps + longer budget recover the 2-token controller.
const tuned=new TemporalControlTrainer(sim,{tokens:2,seed:9118,ppoModuleScales:{encoder:.05,dynamic:.05,kinematic:1,critic:1}});
tuned.restore(bootstrap.two);
for(let i=0;i<50;i++)tuned.iteration();
const tunedEval=tuned.evaluate({episodes:12,seed:8181});
const expected50=evidence.tunedTrajectory.find(x=>x.ppoIterations===50);
if(!close(tunedEval.trackingMae,expected50.cleanMae))throw new Error("tuned long-horizon evidence mismatch");
if(!(tunedEval.trackingMae<.26))throw new Error("tuned 2-token controller failed to recover clean tracking below 0.26 m");
tuned.delete();

const defaultTwo=expected("two-default").after.cleanMae;
const matchedTwo=expected("two-matched-capacity").after.cleanMae;
const frozenTwo=expected("two-freeze-encoder").after.cleanMae;
if(!(matchedTwo>defaultTwo))throw new Error("matched-capacity result no longer demonstrates that parameter count alone is insufficient");
if(!(frozenTwo<defaultTwo && frozenTwo>.30))throw new Error("freeze-Encoder result no longer shows partial-only recovery");

console.log(JSON.stringify({
  schema:"control-optimization-verification/v1",
  matchedCapacity:{params:1220,cleanMae:matchedEval.trackingMae},
  freezeEncoder:{cleanMae:frozenEval.trackingMae},
  tunedActor005At50:{cleanMae:tunedEval.trackingMae},
  interpretation:"In this deterministic CartPole ablation, token count is not isolated by parameter count alone. Coupled actor update scale is a major recoverable optimization factor."
},null,2));

sim.dispose();
