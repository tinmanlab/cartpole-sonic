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
  assert.match(html,/마지막 적용 힘/);
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

test('system-map height grows with wrapped labels instead of clipping its motion branch',()=>{
  assert.match(html,/\.system-map\{[^}]*height:auto/);
});

import {decoderConnections} from '../presentation.js';
test('both input boxes connect to the actual decoder, which alone connects to force',()=>{
  for(const width of [480,700,900]){
    const b=decoderLayout(width),e=decoderConnections(width);
    assert.deepEqual(e[0].points[0],[b[0].x+b[0].w/2,b[0].y+b[0].h]);
    assert.deepEqual(e[1].points[0],[b[1].x+b[1].w/2,b[1].y+b[1].h]);
    assert.deepEqual(e[0].points.at(-1),[b[2].x+b[2].w/2,b[2].y]);
    assert.deepEqual(e[2].points.at(-1),[b[3].x,b[3].y+b[3].h/2]);
  }
});

import {renderOptimizerEvidencePanel} from '../optimizer_evidence_view.js';
test('evidence controls are interactive immediately, before the next paint frame',()=>{
  const originalDocument=globalThis.document,originalFrame=globalThis.requestAnimationFrame;
  const elements=Object.fromEntries(['optBudget','optDiagnostics','opt10','opt50','optVariant'].map(k=>[k,{}]));
  globalThis.document={getElementById:id=>elements[id]||null};
  globalThis.requestAnimationFrame=()=>0;
  try{
    const evidence=JSON.parse(fs.readFileSync(new URL('../evidence/control_optimization_eval.json',import.meta.url)));
    renderOptimizerEvidencePanel({evidence,showHtml:()=>{},beginCanvas:()=>{},rerender:()=>{}});
    assert.equal(typeof elements.optBudget.onclick,'function');
    assert.equal(typeof elements.optVariant.onchange,'function');
  }finally{globalThis.document=originalDocument;globalThis.requestAnimationFrame=originalFrame;}
});

import {createHash} from 'node:crypto';
test('captured browser evidence is bound to the current rendering source',()=>{
  const report=JSON.parse(fs.readFileSync(new URL('../evidence/browser_audit.json',import.meta.url)));
  assert.equal(report.passed,true);assert.deepEqual(report.errors,[]);
  assert.ok(report.checks.length>=150);
  for(const file of ['app.js','course.js','index.html','presentation.js','optimizer_evidence_view.js','control_contract.js','mujoco_sim.js','sonic_toy.js','temporal_control_lab.js','native_lesson.js','native_lesson.css','evidence/guided_lesson/traces.json','scripts/browser_smoke.mjs']){
    const actual=createHash('sha256').update(fs.readFileSync(new URL('../'+file,import.meta.url))).digest('hex');
    assert.equal(actual,report.source_sha256[file],file+' changed after the captured browser audit');
  }
  assert.ok(report.checks.filter(c=>Object.hasOwn(c,'clippedMap')).every(c=>c.clippedMap.length===0));
});

test('terminal-latched UI does not advertise automatic episode restarts',()=>{
  assert.doesNotMatch(app,/"auto resets="/);
});

test('execution recovery evidence uses the same current sources as the visual audit',()=>{
  const visual=JSON.parse(fs.readFileSync(new URL('../evidence/browser_audit.json',import.meta.url)));
  const execution=JSON.parse(fs.readFileSync(new URL('../evidence/browser_execution_audit.json',import.meta.url)));
  assert.equal(execution.passed,true);assert.deepEqual(execution.errors,[]);
  assert.equal(execution.executionOnly,true);
  assert.deepEqual(execution.source_sha256,visual.source_sha256);
  assert.ok(execution.checks.some(x=>x.label.includes('corrupt checkpoint recovery')));
});
