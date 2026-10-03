import fs from "node:fs";
import {MuJoCoCartPole} from "../mujoco_sim.js";
import {SonicToyTrainer} from "../sonic_toy.js";

const root=new URL("../",import.meta.url);
const ckpt=JSON.parse(fs.readFileSync(new URL("assets/student_fsq_bootstrap.json",root),"utf8"));
const evidence=JSON.parse(fs.readFileSync(new URL("evidence/ppo_eval.json",root),"utf8"));

if(ckpt.schema!=="cartpole-sonic-student-bootstrap/v1"||ckpt.mode!=="fsq"){
  throw new Error("invalid FSQ checkpoint schema");
}

const sim=new MuJoCoCartPole();
await sim.init("playground");
const t=new SonicToyTrainer(sim,{mode:"fsq",n:8,horizon:96,epochs:4,batch:128});
t.restorePolicy(ckpt.policy,{teacherBootstrap:ckpt.teacherBootstrap,rngState:ckpt.rng});

const eval0=t.evaluate(12);
const expected0=evidence.checkpoints.find(x=>x.ppoIterations===0)?.eval;
if(!expected0) throw new Error("missing 0-iter evidence");

for(let i=0;i<20;i++) t.iteration();
const eval20=t.evaluate(12);
const expected20=evidence.checkpoints.find(x=>x.ppoIterations===20)?.eval;
if(!expected20) throw new Error("missing 20-iter evidence");

const close=(a,b,tol=1e-12)=>Math.abs(a-b)<=tol;
if(eval0.successes!==expected0.successes || !close(eval0.trackingMae,expected0.trackingMae)){
  throw new Error("checkpoint baseline diverges from deterministic evidence");
}
if(eval20.successes!==expected20.successes || !close(eval20.trackingMae,expected20.trackingMae)){
  throw new Error("checkpoint PPO path diverges from deterministic evidence");
}

console.log(JSON.stringify({
  schema:"checkpoint-parity/v1",
  baseline:eval0,
  after20:eval20,
},null,2));

t.delete();
sim.dispose();
