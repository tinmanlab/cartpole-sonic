import fs from "node:fs";
import {MuJoCoCartPole} from "../mujoco_sim.js";
import {SonicToyTrainer, planReference} from "../sonic_toy.js";

const ckpt=JSON.parse(fs.readFileSync(new URL("../assets/student_fsq_bootstrap.json",import.meta.url),"utf8"));
const sim=new MuJoCoCartPole();
await sim.init("playground");

const trainer=new SonicToyTrainer(sim,{mode:"fsq",n:8,horizon:96,epochs:4,batch:128});
trainer.restorePolicy(ckpt.policy,{teacherBootstrap:ckpt.teacherBootstrap,rngState:ckpt.rng});

const goal=.8;
const plannerContext=[0,0];
const ref=planReference(plannerContext,goal);
const beforeState=sim.getState();
const before=trainer.preview(beforeState,ref,goal);

sim.applyImpulse({xDotDelta:.75,thetaDotDelta:-1.25});
const afterState=sim.getState();
const after=trainer.preview(afterState,ref,goal);

const same=(a,b,tol=1e-12)=>a.length===b.length&&a.every((v,i)=>Math.abs(v-b[i])<=tol);
if(!same(before.ref,after.ref)) throw new Error("Push changed reference: robot/reference worlds are coupled");
if(!same(before.q,after.q)) throw new Error("Push changed token despite unchanged reference");
if(same(beforeState,afterState)) throw new Error("Push failed to perturb actual robot state");
if(Math.abs(before.force-after.force)<1e-6) throw new Error("Dynamic Decoder action did not react to changed proprioception");

console.log(JSON.stringify({
  schema:"reference-robot-separation/v1",
  referenceSame:true,
  tokenSame:true,
  stateChanged:true,
  actionChanged:true,
  before:{state:beforeState,token:before.q,force:before.force},
  after:{state:afterState,token:after.q,force:after.force},
},null,2));

trainer.delete();
sim.dispose();
