import fs from "node:fs";
import { MuJoCoCartPole } from "../mujoco_sim.js";
import { FrozenCartPoleTeacher } from "../teacher_policy.js";

const teacher = new FrozenCartPoleTeacher(JSON.parse(fs.readFileSync(new URL("../assets/teacher_cartpole_ppo.json", import.meta.url),"utf8")));
const sim = new MuJoCoCartPole();
await sim.init("playground");

const rows=[];
for(const goal of [-.8,0,.8]){
  for(let ep=0;ep<5;ep++){
    sim.reset({x:(ep-2)*.03,xDot:0,theta:.05*(ep%2?1:-1),thetaDot:0});
    let sumErr=0,failed=false,executed=0;
    while(executed<1000){
      const s=sim.getState(),f=teacher.forward(s,goal).forceN;
      sim.stepForce(f,1);
      executed++;
      const ns=sim.getState();
      sumErr+=Math.abs(ns[0]-goal);
      if(Math.abs(ns[0])>1.78||Math.abs(ns[2])>.65){failed=true;break;}
    }
    rows.push({goal,steps:executed,failed,mae:sumErr/executed});
  }
}
const result={
  schema:"teacher-mujoco-eval/v1",
  physics:sim.backend,
  preset:"playground",
  horizon_seconds:10,
  teacher:teacher.metadata(),
  episodes:rows.length,
  survived:rows.filter(r=>!r.failed).length,
  mean_steps:rows.reduce((s,r)=>s+r.steps,0)/rows.length,
  mean_mae:rows.reduce((s,r)=>s+r.mae,0)/rows.length,
  by_goal:[-.8,0,.8].map(g=>{
    const a=rows.filter(r=>r.goal===g);
    return{goal:g,survived:a.filter(r=>!r.failed).length,episodes:a.length,mean_steps:a.reduce((s,r)=>s+r.steps,0)/a.length,mae:a.reduce((s,r)=>s+r.mae,0)/a.length};
  })
};
fs.mkdirSync(new URL("../evidence/", import.meta.url),{recursive:true});
fs.writeFileSync(new URL("../evidence/teacher_mujoco.json", import.meta.url),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
sim.dispose();
