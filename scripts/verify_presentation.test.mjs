import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const app=fs.readFileSync(new URL('../app.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../index.html',import.meta.url),'utf8');
test('tracking display uses same-time planner target',()=>{
  assert.match(app,/referenceNow:plannerContext\[0\]/);
  assert.doesNotMatch(app,/Math\.abs\(state\(\)\[0\]-p\.ref\[0\]/);
});
test('planned and last applied commands are separately named',()=>{
  assert.match(app,/plannedForce/);assert.match(app,/lastAppliedForce/);
  assert.match(html,/last applied/);
});
test('browser identity and responsive decoder are explicit',()=>{
  assert.match(html,/브라우저 교육용 모델 · MuJoCo WASM 시뮬레이션/);
  assert.match(app,/activeController/);
  assert.doesNotMatch(app,/\{x:700,w:120/);
});

import {controlSample,controllerIdentity,decoderLayout,robotGeometry} from '../presentation.js';
test('sample timestamps and force identities come from inputs',()=>{
  const p={z:[.2,.3],q:[0,.5],force:7};
  const sample=controlSample(.4,[.12,0,0,0],.1,[.3],p,-2,2);
  assert.equal(sample.referencePreview,.6);assert.equal(sample.targetTime,.4);assert.equal(sample.referenceNow,.1);
  assert.equal(sample.previewTargetTime,.4+.08);assert.equal(sample.plannedForce,7);assert.equal(sample.lastAppliedForce,-2);
});
test('identity follows actual preview and selected policy',()=>{
  assert.deepEqual(controllerIdentity({q:[0,1]},'ae'),{source:'LIVE browser calculation',mode:'ae',tokens:0,scalarDimsPerToken:0,flattenedDim:2});
  assert.equal(controllerIdentity({q:[0,1,2,3]},'fsq',{tokens:2,tokenDim:2}).flattenedDim,4);
});
test('decoder and long pole fit supported panel sizes',()=>{
  for(const w of [480,500,820])for(const r of decoderLayout(w))assert.ok(r.x>=0&&r.x+r.w<=w&&r.y+r.h<=220);
  for(const length of [1,1.35,2])for(const x of [-1.78,0,1.78])for(const theta of [-.75,0,.75]){
    const g=robotGeometry(360,202,length,x,theta);assert.ok(g.tx>=0&&g.tx<=360&&g.ty>=0&&g.ty<=202);
  }
});

import {controllerSummary,blockShape} from '../presentation.js';
test('AE is continuous, and the system-map dimensions follow the selected control policy',()=>{
  const continuous=controllerIdentity({q:[.2,.3]},'ae');
  assert.equal(continuous.tokens,0);
  assert.match(controllerSummary(continuous),/양자화 없음/);
  assert.match(blockShape('token',continuous,''),/연속 latent 2D/);
  const two=controllerIdentity({q:[0,.5,1,0]},'fsq',{tokens:2,tokenDim:2});
  assert.equal(blockShape('encoder',two,''),'16D reference → 4D latent');
  assert.equal(blockShape('token',two,''),'2 token × 2 scalars');
});
