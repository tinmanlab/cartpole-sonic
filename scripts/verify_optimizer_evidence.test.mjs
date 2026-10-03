import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const evidence=JSON.parse(fs.readFileSync(new URL('../evidence/control_optimization_eval.json',import.meta.url),'utf8'));

test('optimizer evidence reports measured Adam displacement, not an LR-gradient proxy',()=>{
  for(const v of evidence.variants){
    assert.ok(v.parameters?.actor>0,'separate actor and total parameter counts');
    assert.ok(Number.isFinite(v.diagnostics.updates?.dynamic?.updateNorm),'measure actual Adam delta');
    assert.equal(v.diagnostics.dynamicUpdateProxy,undefined);
  }
});
test('optimizer evidence compares both token counts and learning rates at equal budgets',()=>{
  assert.ok(evidence.pairedSeedStudy,'paired PPO seed study is present');
  const study=evidence.pairedSeedStudy;
  assert.deepEqual(study.seeds,[17011,27011,37011]);
  assert.equal(study.modelInitializationReplicates,1,'do not call rollout seeds independent model initialization');
  for(const run of study.runs){
    assert.equal(run.arms.length,4);
    for(const arm of run.arms){
      assert.deepEqual(arm.checkpoints.map(c=>c.ppoIterations),[0,10,50]);
      assert.deepEqual(arm.checkpoints.map(c=>c.envSteps),[0,7680,38400]);
      for(const c of arm.checkpoints){
        assert.ok(c.cleanSuccesses>=0&&c.cleanSuccesses<=12);
        assert.ok(c.pushSuccesses>=0&&c.pushSuccesses<=12);
      }
    }
  }
  for(const s of study.summary){
    assert.equal(s.cleanMae.n,3);
    assert.ok(Number.isFinite(s.cleanMae.mean));
    assert.ok(Number.isFinite(s.cleanMae.sampleStd));
  }
});
