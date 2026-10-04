import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateLessonData} from '../native_lesson.js';
import {CONCEPT_TEXT,TRAINING_DETAILS} from '../course.js';
test('endpoint evidence means final minus initial within each branch',()=>{
 const d=JSON.parse(fs.readFileSync('evidence/guided_lesson/traces.json'));
 const c=d.cases[0],residual=5e-8;
 for(const ref of c.reference)ref[28][0]=ref[0][0]+residual;
 for(const r of c.runs){const f=r.frames[28];f.referenceNow=[...c.reference[r.branch][28]];f.errorX=f.state[0]-f.referenceNow[0];r.metrics.cartMae=r.frames.slice(1).reduce((s,f)=>s+Math.abs(f.errorX),0)/28;}
 c.endpointMaxError=Math.max(...c.reference.flatMap(r=>r[28].map((v,i)=>Math.abs(v-r[0][i]))));
 assert.ok(c.endpointMaxError>0);assert.equal(validateLessonData(d),d);
 c.endpointMaxError=0;assert.throws(()=>validateLessonData(d),/endpoint/);
});
test('equations carry symbol keys and scope',()=>{
 for(const id of ['ae','vq','vqvae','fsq'])assert.match(CONCEPT_TEXT[id].formula,/\n/);
 assert.match(CONCEPT_TEXT.fsq.formula,/1\.998/);assert.match(CONCEPT_TEXT.ae.formula,/무차원/);
 assert.match(TRAINING_DETAILS.ppo.formula,/A/);
});

import {canvasTicks,canvasLines} from '../presentation.js';
test('measured ticks never overlap and captions wrap without shrinking',()=>{
 const calls=[],ctx={measureText:t=>({width:t.length*8}),fillText:(text,x,y)=>calls.push({text,x,y})};
 canvasTicks(ctx,['0.00s','0.08s','0.16s','0.24s'],[40,65,90,115],100);
 assert.equal(calls.length,2);
 assert.ok(calls[1].x-calls[1].text.length*4>=calls[0].x+calls[0].text.length*4+8);
 calls.length=0;assert.equal(canvasLines(ctx,'abcdefghij',10,20,32),3);
 assert.ok(calls.every(c=>ctx.measureText(c.text).width<=32));
});
test('full-state endpoint matching remains bounded at 1e-7',()=>{
 const d=JSON.parse(fs.readFileSync('evidence/guided_lesson/traces.json'));
 d.cases[0].reference[1][28][2]+=2e-7;
 assert.throws(()=>validateLessonData(d),/full-state endpoint matching/);
});

test('VQ-VAE has one explicit non-wrapping signal row',()=>{
  const source=fs.readFileSync(new URL('../app.js',import.meta.url),'utf8');
  const body=source.slice(source.indexOf('function renderVqvaeViz()'),source.indexOf('function renderTokenViz()'));
  assert.equal(body.includes('class="vq-pipeline"'),true,'do not let the final arrow wrap away from the reconstruction decoder');
});
