import test from 'node:test';
import assert from 'node:assert/strict';
import {MuJoCoCartPole} from '../mujoco_sim.js';
import {SonicToyTrainer} from '../sonic_toy.js';
import {TemporalControlLab} from '../temporal_control_lab.js';
const sim=new MuJoCoCartPole(); await sim.init();
test.after(()=>sim.dispose());
test('invalid force, counts, reset and impulse are atomic',()=>{
  sim.reset(); const before=sim.snapshot();
  for(const f of [NaN,Infinity,'3'])assert.throws(()=>sim.stepForce(f));
  for(const n of [0,1.5,NaN,'2',1001])assert.throws(()=>sim.stepForce(0,n));
  assert.throws(()=>sim.reset({theta:NaN}));
  assert.throws(()=>sim.reset({x:undefined}));
  assert.throws(()=>sim.applyImpulse({xDotDelta:undefined}));
  assert.throws(()=>sim.applyImpulse({thetaDotDelta:Infinity}));
  assert.deepEqual(sim.snapshot(),before);
});
test('reset clears external forces and warmstart and reproduces fresh dynamics',()=>{
  sim.data.qfrc_applied.fill(9);sim.data.xfrc_applied.fill(9);sim.data.qacc_warmstart.fill(9);
  sim.reset();
  for(const key of ['qfrc_applied','xfrc_applied','qacc_warmstart'])assert.ok(sim.data[key].every(x=>x===0),key);
  const fresh=sim.makeData();sim.mujoco.mj_resetData(sim.model,fresh);fresh.qpos[1]=.1;sim.mujoco.mj_forward(sim.model,fresh);
  for(let i=0;i<20;i++){sim.stepForce(2,2);fresh.ctrl[0]=.2;sim.mujoco.mj_step(sim.model,fresh);sim.mujoco.mj_step(sim.model,fresh);}
  assert.deepEqual(sim.getState(),sim.getState(fresh));fresh.delete();
});
test('sonic later invalid network does not mutate earlier networks',()=>{
  const t=new SonicToyTrainer(sim,{n:1});try{
    const before=t.snapshot().policy, bad=structuredClone(before);bad.encoder.p[0]+=1;bad.critic.p[0]=NaN;
    assert.throws(()=>t.restorePolicy(bad));assert.deepEqual(t.snapshot().policy,before);
  }finally{t.delete();}
});
test('invalid second temporal controller cannot mutate first',()=>{
  const lab=new TemporalControlLab(sim);try{
    const before=lab.snapshot(),bad=structuredClone(before);bad.one.policy.encoder.p[0]+=1;bad.two.policy.critic.p[0]=NaN;
    assert.throws(()=>lab.restore(bad));assert.deepEqual(lab.snapshot(),before);
  }finally{lab.delete();}
});

test('shared physical rule and JS timeout are distinct',async()=>{
  const {physicalFailureReason}=await import('../control_contract.js');
  assert.equal(physicalFailureReason([0,0,.65,0]),null);
  assert.match(physicalFailureReason([0,0,.70,0]),/pole angle/);
  const t=new SonicToyTrainer(sim,{n:1});try{
    const e=t.envs[0];sim.mujoco.mj_resetData(sim.model,e.data);sim.mujoco.mj_forward(sim.model,e.data);e.steps=499;
    const timeout=e.step(0);assert.equal(timeout.done,true);assert.equal(timeout.terminated,false);
    e.data.qpos[1]=.70;sim.mujoco.mj_forward(sim.model,e.data);assert.equal(e.step(0).terminated,true);
  }finally{t.delete();}
});
test('checkpoint optimizer, architecture, codebook and RNG validation are atomic',()=>{
  const t=new SonicToyTrainer(sim,{n:1});try{
    const before=t.snapshot().policy;
    for(const corrupt of [p=>{p.critic.m[0]=Infinity;},p=>{p.critic.v[0]=-1;},p=>{p.critic.t=1.5;},p=>{p.critic.n=99;},p=>{p.critic.m=null;},p=>{p.codebook[0]='1';}]){
      const bad=structuredClone(before);bad.encoder.p[0]+=1;corrupt(bad);assert.throws(()=>t.restorePolicy(bad));assert.deepEqual(t.snapshot().policy,before);
    }
    const bad=structuredClone(before);bad.encoder.p[0]+=1;
    for(const rngState of [{s:-1,spare:null},{s:1.5,spare:null},{s:1,spare:NaN}]){assert.throws(()=>t.restorePolicy(bad,{rngState}));assert.deepEqual(t.snapshot().policy,before);}
    const minimal=structuredClone(before);for(const name of ['encoder','dynamicDecoder','kinematicDecoder','critic'])for(const key of ['m','v','t'])delete minimal[name][key];
    t.restorePolicy(minimal);assert.ok(t.policy.critic.m.every(x=>x===0));assert.equal(t.policy.critic.t,0);
  }finally{t.delete();}
});
test('both training resets clear MuJoCo force and warmstart buffers',async()=>{
  const {TemporalControlTrainer}=await import('../temporal_control_lab.js');
  for(const make of [()=>new SonicToyTrainer(sim,{n:1}),()=>new TemporalControlTrainer(sim,{n:1})]){
    const t=make();try{
      const e=t.envs[0];for(const key of ['qfrc_applied','xfrc_applied','qacc_warmstart'])e.data[key].fill(9);e.reset();
      for(const key of ['qfrc_applied','xfrc_applied','qacc_warmstart'])assert.ok(e.data[key].every(x=>x===0),key);
    }finally{t.delete();}
  }
});
test('force telemetry separates requested and physical actuator force',()=>{
  sim.reset();const result=sim.stepForce(25,2);assert.equal(result.forceN,25);assert.equal(result.requestedForceN,25);assert.equal(result.appliedForceN,10);assert.equal(sim.data.ctrl[0],1);
});
test('all morphology presets use a full reset with identical control contract',async()=>{
  for(const preset of ['long','heavy','playground']){
    await sim.loadPreset(preset);sim.data.qfrc_applied.fill(9);sim.reset();
    assert.ok(sim.data.qfrc_applied.every(x=>x===0));assert.equal(sim.snapshot().timestep,.01);
    assert.equal(sim.stepForce(-25,2).appliedForceN,-10);
  }
});
