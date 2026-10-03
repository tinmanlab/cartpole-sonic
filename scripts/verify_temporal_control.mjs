import fs from "node:fs";
import {MuJoCoCartPole} from "../mujoco_sim.js";
import {TemporalControlLab} from "../temporal_control_lab.js";

const bootstrap=JSON.parse(fs.readFileSync(new URL("../assets/temporal_control_bootstrap.json",import.meta.url),"utf8"));
const evidence=JSON.parse(fs.readFileSync(new URL("../evidence/temporal_control_eval.json",import.meta.url),"utf8"));

if(bootstrap?.schema!=="cartpole-sonic-temporal-control-lab/v1")throw new Error("invalid temporal control bootstrap");
if(evidence?.schema!=="cartpole-sonic-temporal-control-eval/v1")throw new Error("invalid temporal control evidence");

const sim=new MuJoCoCartPole();
await sim.init("playground");
const lab=new TemporalControlLab(sim);
lab.restore(bootstrap);

const expected0=evidence.checkpoints.find(x=>x.ppoIterations===0);
const expected10=evidence.checkpoints.find(x=>x.ppoIterations===10);
if(!expected0||!expected10)throw new Error("missing temporal-control evidence checkpoints");

const close=(a,b,tol=1e-12)=>Math.abs(a-b)<=tol;
const checkEval=(actual,expected,label)=>{
  for(const side of ["one","two"]){
    if(actual[side].clean.successes!==expected[side].clean.successes)throw new Error(label+" "+side+" clean survival mismatch");
    if(!close(actual[side].clean.trackingMae,expected[side].clean.trackingMae))throw new Error(label+" "+side+" clean MAE mismatch");
    if(!close(actual[side].push.disturbance.postPushTrackingMae,expected[side].push.disturbance.postPushTrackingMae))throw new Error(label+" "+side+" push MAE mismatch");
  }
};

const actual0=lab.evalHistory.at(-1);
checkEval(actual0,expected0,"bootstrap");

for(let i=0;i<10;i++){lab.one.iteration();lab.two.iteration();}
const actual10=lab.evaluate();
checkEval(actual10,expected10,"ppo10");

if(actual0.one.clean.successes<12||actual0.two.clean.successes<12)throw new Error("bootstrap controllers must survive all clean episodes");
if(actual10.one.clean.successes<12||actual10.two.clean.successes<12)throw new Error("ppo10 controllers must survive all clean episodes");

console.log(JSON.stringify({
  schema:"temporal-closed-loop-control/v1",
  bootstrap:actual0,
  after10:actual10,
  interpretation:"Matched-budget closed-loop ablation. Reconstruction capacity and control optimization are separate questions; no requirement assumes that more token slots must outperform.",
},null,2));

lab.delete();
sim.dispose();
