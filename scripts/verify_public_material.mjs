import fs from 'node:fs';
import path from 'node:path';
import {gunzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
// Public at the time this material was audited. The second source is retained
// only for the bundled teacher's legitimate provenance, not as a dependency.
const PUBLIC_FIRST_PARTY=new Set(['cartpole-sonic','cartpole-ppo']);
const PROSE=new Set(['.md','.html']);
const TEXT=new Set(['.md','.html','.js','.mjs','.py','.json','.yml','.yaml','.gz']);

export function contextualReferences(text,{owner='tinmanlab',allowed=PUBLIC_FIRST_PARTY}={}){
  const safeOwner=owner.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const pattern=new RegExp('\\b'+safeOwner+'/([a-zA-Z0-9_.-]+)','g');
  return [...text.matchAll(pattern)].map(m=>m[1].replace(/\.git$/,'')).filter(name=>!allowed.has(name));
}
export function machineLocations(text){
  return /\/home\/[a-zA-Z0-9_.-]+\/|[A-Za-z]:\\Users\\[a-zA-Z0-9_.-]+\\/.test(text);
}
function walk(dir){
  return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>{
    if(['.git','node_modules','vendor','.venv-native','.upstream-sonic','__pycache__'].includes(e.name))return [];
    const p=path.join(dir,e.name);return e.isDirectory()?walk(p):[p];
  });
}
export function auditPublicMaterial(root=ROOT){
  const problems=[];let scanned=0,prose=0;
  for(const file of walk(root)){
    const relative=path.relative(root,file).replaceAll(path.sep,'/');
    if(relative==='scripts/verify_public_material.mjs'||relative==='scripts/verify_public_material.test.mjs')continue;
    if(!TEXT.has(path.extname(file)))continue;
    const raw=fs.readFileSync(file);const text=path.extname(file)==='.gz'?gunzipSync(raw).toString('utf8'):raw.toString('utf8');
    scanned++;
    for(const name of new Set(contextualReferences(text)))problems.push(relative+': unreviewed first-party repository reference');
    if(PROSE.has(path.extname(file))){
      prose++;
      if(machineLocations(text))problems.push(relative+': machine-specific path in public prose');
      if(relative==='README.md'||relative.startsWith('native/')){
        for(const match of text.matchAll(/\]\(([^\s)]+)\)/g)){
          const link=match[1];if(/^(?:https?:|mailto:|#)/.test(link))continue;
          const target=decodeURIComponent(link.split('#')[0]);
          if(target&&!fs.existsSync(path.resolve(path.dirname(file),target)))problems.push(relative+': broken relative link '+target);
        }
      }
    }
  }
  const notice=fs.readFileSync(path.join(root,'THIRD_PARTY_NOTICES.md'),'utf8');
  assert.match(notice,/MuJoCo/);assert.match(notice,/SONIC/);
  const readme=fs.readFileSync(path.join(root,'README.md'),'utf8');
  assert.match(readme,/JavaScript/);assert.match(readme,/Python/);
  assert.match(readme,/NVlabs\/GR00T-WholeBodyControl/);
  return {scannedTextFiles:scanned,publicProseFiles:prose,problems,
    scope:'current public files and compressed evidence, not destructive Git-history rewriting'};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const result=auditPublicMaterial();console.log(JSON.stringify(result,null,2));
  if(result.problems.length)process.exitCode=1;
}
