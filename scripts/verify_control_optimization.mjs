import fs from 'node:fs';
import assert from 'node:assert/strict';
import {runOptimizationExperiment,describe} from './control_optimization_experiment.mjs';
const expected=JSON.parse(fs.readFileSync(new URL('../evidence/control_optimization_eval.json',import.meta.url),'utf8'));
assert.equal(expected.revision,2);
assert.deepEqual(describe([1,2,3,null]),{n:3,mean:2,sampleStd:1,min:1,max:3});
assert.deepEqual(describe([null]),{n:0,mean:null,sampleStd:null,min:null,max:null});
const actual=await runOptimizationExperiment({onProgress:label=>console.log('replayed:',label)});
let numbers=0;
function compare(a,b,path='root'){
  if(typeof b==='number'){
    assert.equal(typeof a,'number',path);
    assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(b)),`${path}: ${a} != ${b}`);numbers++;return;
  }
  if(b===null||typeof b!=='object'){assert.deepEqual(a,b,path);return;}
  assert.deepEqual(Object.keys(a),Object.keys(b),path+' keys');
  for(const k of Object.keys(b))compare(a[k],b[k],path+'.'+k);
}
compare(actual,expected);
// Scientific verification checks reproducibility and measurement contracts, not a preferred winner.
console.log(JSON.stringify({schema:'control-optimization-verification/v2',numericMeasurementsCompared:numbers,variants:actual.variants.length,pairedPpoSeeds:actual.pairedSeedStudy.seeds,independentModelInitializationsPerArchitecture:1,allMeasurementsMatch:true,winnerAssertion:false},null,2));
