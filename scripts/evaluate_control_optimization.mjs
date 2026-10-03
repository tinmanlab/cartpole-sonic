import fs from 'node:fs';
import {runOptimizationExperiment} from './control_optimization_experiment.mjs';
const output=await runOptimizationExperiment({onProgress:label=>console.log('measured:',label)});
fs.writeFileSync(new URL('../evidence/control_optimization_eval.json',import.meta.url),JSON.stringify(output,null,2));
console.log(JSON.stringify({variants:output.variants.map(v=>({id:v.id,parameters:v.parameters,cleanMae:v.after.cleanMae,pushMae:v.after.pushMae,updateNorm:v.diagnostics.updates.dynamic.updateNorm,kl:v.diagnostics.postUpdateKl,sampleClip:v.diagnostics.meanSampleClipAcrossIterations})),pairedSeedSummary:output.pairedSeedStudy.summary},null,2));
