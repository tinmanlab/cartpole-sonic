import fs from "node:fs";
import { MuJoCoCartPole } from "../mujoco_sim.js";
import { SonicToyTrainer } from "../sonic_toy.js";
import { FrozenCartPoleTeacher } from "../teacher_policy.js";

const teacher=new FrozenCartPoleTeacher(JSON.parse(fs.readFileSync(new URL("../assets/teacher_cartpole_ppo.json",import.meta.url),"utf8")));
const sim=new MuJoCoCartPole();await sim.init("playground");
const trainer=new SonicToyTrainer(sim,{mode:"fsq",n:8,horizon:96,epochs:4,batch:128});
trainer.bootstrapFromTeacher(teacher,{steps:500,batch:256,lr:.0015,auxCoef:.08});
const checkpoints=[];
for(let i=0;i<=50;i++){
  if([0,10,20,30,40,50].includes(i))checkpoints.push({ppoIterations:i,eval:trainer.evaluate(12)});
  if(i<50)trainer.iteration();
}
const result={schema:"sonic-cartpole-ppo-eval/v1",physics:sim.backend,mode:"fsq",checkpoints};
fs.mkdirSync(new URL("../evidence/",import.meta.url),{recursive:true});
fs.writeFileSync(new URL("../evidence/ppo_eval.json",import.meta.url),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
trainer.delete();sim.dispose();
