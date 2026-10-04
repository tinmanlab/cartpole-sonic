import test from 'node:test';
import assert from 'node:assert/strict';
import {validateLessonData, selectLessonFrame} from '../native_lesson.js';
// Synthetic fixture only: deliberately not a native execution claim.
export function fixture(){
 const reference=Array.from({length:2},(_,b)=>Array.from({length:29},(_,i)=>[b*Math.sin(i*Math.PI/28)*.001,0,0,0]));
 return {schema:'cartpole-sonic-guided-lesson/v1',source:{kind:'native-recorded-reexecution',encoderSeed:52101,modality:'joints',training:'SYNTHETIC TEST FIXTURE ONLY',scope:'synthetic validator test',upstreamCommit:'0'.repeat(40),files:{'synthetic-test-only':'0'.repeat(64)}},protocol:{controlDt:.02,horizonSteps:28,referenceOffsets:[0,4,8,12,16,20,24,28],stateOrder:['x','xdot','theta','thetadot'],stateUnits:['m','m/s','rad','rad/s'],conditions:['full_future','endpoints_only'],offsets:{nominal:[0,0,0,0],perturbed:[0,.02,.001,-.02]}},cases:Array.from({length:3},(_,c)=>({id:`test-00${c}`,reference:structuredClone(reference),endpointMaxError:0,runs:['full_future','endpoints_only'].flatMap(condition=>['nominal','perturbed'].flatMap(offset=>[0,1].map(branch=>({condition,offset,branch,frames:reference[branch].map((r,tick)=>({tick,time:tick*.02,state:r.map((v,i)=>v+(offset==='perturbed'?[0,.02,.001,-.02][i]:0)),referenceNow:[...r],errorX:0,decision:tick===28?null:{referenceWindow:Array.from({length:8},()=>[0,0,0,0]),latent:[0,0,0,0],token:[0,0,0,0],requestedForceN:1,appliedForceN:1}})),metrics:{steps:28,completed:true,cartMae:0}}))))}))};
}
test('complete aligned fixture and pure selection',()=>{const d=fixture();assert.equal(validateLessonData(d),d);const s=selectLessonFrame(d,{caseId:'test-001',branch:1,offset:'perturbed',tick:28});assert.equal(s.runs.length,2);assert.equal(s.frames[0].decision,null);assert.equal(s.reference.length,29);});
for(const [name,mutate] of Object.entries({schema:d=>d.schema='wrong',provenance:d=>d.source.files['synthetic-test-only']='bad',missing:d=>d.cases[0].runs.pop(),clock:d=>d.cases[0].runs[0].frames[1].time=.03,finite:d=>d.cases[0].runs[0].frames[1].decision.token[0]=NaN,final:d=>d.cases[0].runs[0].frames[28].decision={},reference:d=>d.cases[0].runs[0].frames[2].referenceNow[0]=1,initial:d=>d.cases[0].runs[0].frames[0].state[0]=1,order:d=>d.cases.reverse(),metrics:d=>d.cases[0].runs[0].metrics.cartMae=1})){test(`reject ${name}`,()=>{const d=fixture();mutate(d);assert.throws(()=>validateLessonData(d));});}
test('reject invalid selection',()=>assert.throws(()=>selectLessonFrame(fixture(),{caseId:'test-000',branch:0,offset:'nominal',tick:29})));

test('reject physically inconsistent requested/applied force metadata',()=>{
 const d=fixture();d.cases[0].runs[0].frames[0].decision.appliedForceN=11;
 assert.throws(()=>validateLessonData(d));
});

import fs from 'node:fs';
test('canvas layout never treats mutable backing-store height as CSS height',()=>{
 const source=fs.readFileSync(new URL('../native_lesson.js',import.meta.url),'utf8');
 assert.equal(source.includes("Number(canvas.getAttribute('height'))"),false,'DPR>1 redraws must not multiply height on every frame');
});

import {createHash} from 'node:crypto';
test('captured native lesson is tied to current rendering and trace files',()=>{
 const report=JSON.parse(fs.readFileSync(new URL('../evidence/guided_lesson/browser_audit.json',import.meta.url)));
 assert.equal(report.passed,true);assert.deepEqual(report.errors,[]);assert.ok(report.checks.length>=4);
 for(const [file,hash] of Object.entries(report.source_sha256)){
  assert.equal(createHash('sha256').update(fs.readFileSync(new URL('../'+file,import.meta.url))).digest('hex'),hash,file);
 }
});
