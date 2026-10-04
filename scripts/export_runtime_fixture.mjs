// Bounded differential-test adapter, not an additional runtime/control layer.
import fs from 'node:fs';
import {MuJoCoCartPole} from '../mujoco_sim.js';
import {SonicToyTrainer} from '../sonic_toy.js';
import {TemporalControlTrainer} from '../temporal_control_lab.js';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
if(!Array.isArray(input.cases)||input.cases.length>20)throw new Error('bounded fixture cases required');
const sim=new MuJoCoCartPole();await sim.init('playground');
const result=[];
try{
  for(const item of input.cases){
    if(!Array.isArray(item.state)||item.state.length!==4||!Array.isArray(item.forces)||item.forces.length>100)throw new Error('invalid fixture');
    sim.setState(item.state);
    const rows=[];
    for(const force of item.forces){
      const snapshot=sim.stepForce(force,2);
      rows.push({state:sim.getState(),time:sim.data.time,
        requestedForceN:snapshot.requestedForceN,appliedForceN:snapshot.appliedForceN});
    }
    result.push({initial:item.state,rows});
  }
  const trainerContracts=[];
  for(const kind of ['standard','temporal']){
    const trainer=kind==='standard'?new SonicToyTrainer(sim,{n:1,horizon:1,epochs:1,batch:1}):new TemporalControlTrainer(sim,{tokens:2,n:1,horizon:1,epochs:1,batch:1});
    try{
      const env=trainer.envs[0];env.data.qpos[0]=0;env.data.qpos[1]=.7;env.data.qvel.fill(0);
      sim.mujoco.mj_forward(sim.model,env.data);
      const step=env.step(0);
      trainerContracts.push({kind,done:step.done,terminated:step.terminated});
    }finally{trainer.delete();}
  }
  console.log(JSON.stringify({backend:sim.backend,timestep:sim.model.opt.timestep,controlSteps:2,cases:result,trainerContracts}));
}finally{sim.dispose();}
