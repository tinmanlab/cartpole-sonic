import fs from "node:fs";
import {MuJoCoCartPole} from "../mujoco_sim.js";
import {SonicToyTrainer} from "../sonic_toy.js";
import {FrozenCartPoleTeacher} from "../teacher_policy.js";

const teacher=new FrozenCartPoleTeacher(JSON.parse(fs.readFileSync(new URL("../assets/teacher_cartpole_ppo.json",import.meta.url),"utf8")));
const sim=new MuJoCoCartPole();
await sim.init("playground");

for(const mode of ["ae","vq","fsq"]){
  const t=new SonicToyTrainer(sim,{mode,n:8,horizon:96,epochs:4,batch:128});
  const bootstrapRaw=t.bootstrapFromTeacher(teacher,{steps:500,batch:256,lr:.0015,auxCoef:.08});
  const {ms: _elapsedMs, ...bootstrap}=bootstrapRaw;
  const evalResult=t.evaluate(12);
  const payload={
    schema:"cartpole-sonic-student-bootstrap/v1",
    mode,
    seed:20261003,
    teacherBootstrap:bootstrap,
    defaultPresetEval:evalResult,
    rng:t.snapshot().rng,
    policy:t.snapshot().policy
  };
  fs.writeFileSync(new URL("../assets/student_"+mode+"_bootstrap.json",import.meta.url),JSON.stringify(payload));
  console.log(mode,evalResult);
  t.delete();
}
sim.dispose();
