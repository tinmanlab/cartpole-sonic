import fs from "node:fs";
import {MuJoCoCartPole} from "../mujoco_sim.js";
import {SonicToyTrainer} from "../sonic_toy.js";
import {AlignmentLab} from "../alignment_lab.js";

const ckpt=JSON.parse(fs.readFileSync(new URL("../assets/student_fsq_bootstrap.json",import.meta.url),"utf8"));
const sim=new MuJoCoCartPole();
await sim.init("playground");

const trainer=new SonicToyTrainer(sim,{mode:"fsq",n:8,horizon:96,epochs:4,batch:128});
trainer.restorePolicy(ckpt.policy,{teacherBootstrap:ckpt.teacherBootstrap,rngState:ckpt.rng});

const lab=new AlignmentLab(trainer.policy);
const before=lab.snapshot().baseline;
lab.train(50);
const after=lab.snapshot().current;

if(!(after.latentMse < before.latentMse*.20)){
  throw new Error("alignment latent MSE did not improve by at least 80%");
}
if(!(after.tokenAgreement >= .85)){
  throw new Error("alignment token agreement below 85%");
}
if(!(after.actionMae <= .05)){
  throw new Error("alignment same-state action MAE above 0.05 N");
}

console.log(JSON.stringify({
  schema:"multi-encoder-alignment/v1",
  before,
  after,
  improvement:{
    latentMseRatio:after.latentMse/before.latentMse,
    tokenAgreementGain:after.tokenAgreement-before.tokenAgreement,
    actionMaeRatio:after.actionMae/before.actionMae,
  }
},null,2));

trainer.delete();
sim.dispose();
