import test from 'node:test';
import assert from 'node:assert/strict';
import {MuJoCoCartPole} from '../mujoco_sim.js';
import {SonicCartPolePolicy, SonicToyTrainer, MLP, RNG, SONIC_TOY_CONSTANTS} from '../sonic_toy.js';
import {TemporalControlTrainer} from '../temporal_control_lab.js';

const sim = new MuJoCoCartPole();
await sim.init('playground');
test.after(() => sim.dispose());
const ref = Array(16).fill(0), state = [0,0,0,0];
const std = SONIC_TOY_CONSTANTS.FIXED_STD;
const logp = (a,mu) => -.5*((a-mu)/std)**2-Math.log(std)-.9189385332046727;
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-10, `${a} != ${b}`);

for (const kind of ['sonic','temporal']) {
  const make = () => kind==='sonic'
    ? new SonicToyTrainer(sim,{n:1,horizon:1,epochs:1,batch:1})
    : new TemporalControlTrainer(sim,{tokens:2,n:1,horizon:1,epochs:1,batch:1});
  test(`${kind}: Gaussian sample survives physical clipping`, () => {
    const t=make();
    try {
      const out=t.policy.forward(state,ref,0,{sample:true,rng:{normal:()=>20}});
      assert.ok(out.rawAction>1, 'keep the original Gaussian sample for PPO');
      assert.equal(out.action,1);
      assert.equal(out.force,10);
      close(out.rawAction, out.mu+20*std);
    } finally {t.delete();}
  });
  test(`${kind}: rollout log probability uses the raw sample`, () => {
    const t=make();
    try {
      t.rng.normal=()=>20;
      const q=t.collect().data[0];
      assert.ok(q.rawAction>1, 'rollout must store rawAction');
      assert.equal(q.action,1);
      close(q.oldLogp,logp(q.rawAction,q.oldMu));
    } finally {t.delete();}
  });
  test(`${kind}: score gradient uses the raw sample, not actuator input`, () => {
    const t=make();
    try {
      const out=t.policy.forward(state,ref,0);
      const original=t.policy.dynamicDecoder.backward.bind(t.policy.dynamicDecoder);
      let gradient;
      t.policy.dynamicDecoder.backward=(cache,dy,...rest)=>{gradient=dy[0];return original(cache,dy,...rest);};
      t.optimize([{state,ref,goal:0,rawAction:1.4,action:1,oldMu:out.mu,oldQ:out.q,oldLogp:logp(1.4,out.mu),oldV:0,ret:0,adv:1}]);
      close(gradient,-(1.4-out.mu)/(std*std));
    } finally {t.delete();}
  });
  test(`${kind}: ratio outside clip range is counted even when gradient remains active`, () => {
    const t=make();
    try {
      const out=t.policy.forward(state,ref,0);
      const result=t.optimize([{state,ref,goal:0,rawAction:0,action:0,oldMu:out.mu,oldQ:out.q,oldLogp:logp(0,out.mu)-Math.log(.7),oldV:0,ret:0,adv:1}]);
      assert.equal(result.clipFraction,1);
      assert.equal(result.objectiveClipFraction,0);
    } finally {t.delete();}
  });
  test(`${kind}: physical failure at the time limit is still terminal`, () => {
    const t=make();
    try {
      const e=t.envs[0]; e.steps=499; e.data.qpos[1]=.8;
      sim.mujoco.mj_forward(sim.model,e.data);
      const result=e.step(0);
      assert.equal(result.done,true);
      assert.equal(result.terminated,true);
    } finally {t.delete();}
  });
}

test('Adam diagnostics measure the actual parameter delta after clipping and moments', () => {
  const net=new MLP(new RNG(4),1,1,1);
  const before=Array.from(net.p), G=Float64Array.from(net.p,()=>10);
  net.adam(G,.002,.7);
  assert.ok(net.lastUpdate, 'Adam must expose measured update diagnostics');
  const actual=Math.hypot(...net.p.map((x,i)=>x-before[i]));
  close(net.lastUpdate.updateNorm,actual);
  close(net.lastUpdate.postClipNorm,.7);
  assert.ok(Math.abs(actual-.002*net.lastUpdate.preClipNorm)>.001, 'LR times gradient is not an Adam step');
  net.adam(G,0,.7);
  assert.equal(net.lastUpdate.updateNorm,0);
});

test('temporal: total parameter matching does not imply actor parameter matching', () => {
  const one=new TemporalControlTrainer(sim,{tokens:1,n:1});
  const two=new TemporalControlTrainer(sim,{tokens:2,n:1,widths:{encoder:18,dynamic:26,kinematic:20,critic:20}});
  try {
    assert.equal(typeof one.policy.parameterBreakdown,'function');
    const a=one.policy.parameterBreakdown(),b=two.policy.parameterBreakdown();
    assert.equal(a.total,1220); assert.equal(b.total,1220);
    assert.equal(a.actor,607); assert.equal(b.actor,643);
  } finally {one.delete();two.delete();}
});

test('temporal: frozen weights produce zero measured KL, token changes and Adam displacement', () => {
  const t=new TemporalControlTrainer(sim,{tokens:2,n:2,horizon:8,epochs:1,batch:8,ppoLrScale:0});
  try {
    const result=t.iteration();
    assert.equal(result.postUpdateKl,0);
    assert.equal(result.tokenChangeFraction,0);
    assert.equal(result.updates.encoder.updateNorm,0);
    assert.equal(result.updates.dynamic.updateNorm,0);
    assert.ok(Number.isFinite(result.sampleClipFraction));
  } finally {t.delete();}
});

test('temporal: failure before push is missing push data, never zero tracking error', () => {
  const t=new TemporalControlTrainer(sim,{tokens:2,n:1});
  try {
    t.policy.dynamicDecoder.p.fill(0);
    t.policy.dynamicDecoder.p[t.policy.dynamicDecoder.b2]=20;
    const result=t.evaluate({episodes:2,disturbance:true});
    assert.equal(result.successes,0);
    assert.equal(result.disturbance.pushReachedEpisodes,0);
    assert.equal(result.disturbance.postPushTrackingMae,null);
  } finally {t.delete();}
});
