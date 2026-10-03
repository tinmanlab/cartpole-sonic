import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const spec=process.env.PLAYWRIGHT_MODULE;
const {chromium}=await import(spec?pathToFileURL(spec).href:'playwright');
// A static HTTP server owned by this test process: no persistent shell, desktop or process to kill.
let server,latest={status:'no-browser'};
let base=process.argv[2];
if(!base){
  const root=process.cwd();
  server=http.createServer(async(req,res)=>{
    const pathname=new URL(req.url,'http://localhost').pathname;
    if(pathname==='/telemetry'){
      if(req.method==='POST'){
        const chunks=[];let bytes=0;
        for await(const chunk of req){bytes+=chunk.length;if(bytes<=65536)chunks.push(chunk);}
        if(bytes>65536){res.writeHead(413);res.end();return;}
        try{latest=JSON.parse(Buffer.concat(chunks).toString());}catch{res.writeHead(400);res.end();return;}
      }
      res.setHeader('Content-Type','application/json');res.end(JSON.stringify(latest));return;
    }
    const file=path.resolve(root,'.'+decodeURIComponent(pathname==='/'?'/index.html':pathname));
    if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end();return;}
    const types={'.html':'text/html','.js':'text/javascript','.json':'application/json','.wasm':'application/wasm','.png':'image/png','.svg':'image/svg+xml'};
    res.setHeader('Content-Type',types[path.extname(file)]||'application/octet-stream');res.setHeader('Cache-Control','no-store');fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));base='http://127.0.0.1:'+server.address().port+'/';
}
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined,args:['--enable-experimental-web-platform-features']});
const page=await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:1});
const errors=[],checks=[];
page.on('pageerror',e=>errors.push(e.message));
const report={schema:'cartpole-sonic-browser-audit/v1',base,browser:browser.version(),checks,errors};
const state=()=>page.evaluate(()=>window.__cartpoleSonic.getState());
const waitReady=async()=>{
  await page.waitForFunction(()=>window.__cartpoleSonic?.getState().backends.physics?.includes('MuJoCo'));
  await page.waitForFunction(()=>document.querySelector('.optimizer-audit canvas'));
};
const bounds=async label=>{
  await page.waitForTimeout(35);
  const result=await page.evaluate(()=>{
    const guide=document.querySelector('.guide-card');
    return {documentXOverflow:document.documentElement.scrollWidth>innerWidth+1,guideScrollNeeded:guide.scrollHeight>guide.clientHeight+2,guideOverflow:guide.scrollHeight>guide.clientHeight+2&&!['auto','scroll'].includes(getComputedStyle(guide).overflowY),canvasSizes:[...document.querySelectorAll('canvas')].filter(c=>c.getClientRects().length&&getComputedStyle(c).display!=='none').map(c=>[c.id,c.width,c.height])};
  });
  checks.push({label,...result});
  assert.equal(result.documentXOverflow,false,label+' page horizontal overflow');
  assert.equal(result.guideOverflow,false,label+' hidden guide content');
  for(const [id,w,h] of result.canvasSizes)assert.ok(w>0&&h>0,label+' empty canvas '+id);
};
try{
  await page.goto(base+'?training=optimizer-sensitivity&depth=mechanism');await waitReady();
  assert.equal((await state()).system.version,'2.6');
  await page.screenshot({path:'media/optimizer-budget-50.png',fullPage:true});
  assert.equal(await page.locator('.opt-data tbody tr').count(),16);
  await bounds('optimizer +50');
  await page.click('#opt10');
  assert.equal((await state()).training.optimizerEvidenceView.budget,10);
  await page.screenshot({path:'media/optimizer-budget-10.png',fullPage:true});await bounds('optimizer +10');
  await page.click('#optDiagnostics');await bounds('optimizer diagnostics');
  await page.selectOption('#optVariant','two-matched-capacity');
  assert.ok((await page.locator('.opt-cards').innerText()).includes('1220 / 643'));
  await page.selectOption('#optVariant','two-matched-actor');
  assert.ok((await page.locator('.opt-cards').innerText()).includes('1268 / 607'));
  await page.selectOption('#optVariant','two-default');
  await page.screenshot({path:'media/optimizer-diagnostics.png',fullPage:true});
  for(const width of [1440,1920]){await page.setViewportSize({width,height:1000});await bounds('optimizer '+width+'px');}
  await page.setViewportSize({width:1600,height:1000});
  const map=await page.evaluate(async()=>{const c=await import('./course.js');return {nodes:c.SONIC_FLOW.map(n=>({id:n.id,concepts:n.concepts})),training:c.TRAINING_TOPICS.map(t=>t.id)};});
  for(const node of map.nodes){
    await page.evaluate(id=>window.__cartpoleSonic.focus(id),node.id);await bounds(node.id);
    for(const c of node.concepts||[]){await page.evaluate(({id,c})=>window.__cartpoleSonic.focus(id,c.id),{id:node.id,c});await bounds(node.id+'/'+c.id);}
  }
  for(const id of map.training){await page.evaluate(id=>window.__cartpoleSonic.openTraining(id),id);await bounds('training/'+id);}
  await page.evaluate(()=>window.__cartpoleSonic.focus('token','temporal-control'));
  await page.click('#driveTwoBtn');const before=(await state()).signals.proprioception;
  await page.click('#liveBtn');
  await page.waitForFunction(s=>window.__cartpoleSonic.getState().signals.proprioception.some((v,i)=>Math.abs(v-s[i])>.005),before);
  await page.click('#liveBtn');
  const live=await state();
  assert.equal(live.representationLabs.temporalControl.selectedController,'two');
  assert.ok(Number.isFinite(live.signals.force));
  assert.equal(await page.locator('#simPreset').isDisabled(),true);
  checks.push({label:'actual native MuJoCo two-token control',before,after:live.signals.proprioception,force:live.signals.force});
  await page.screenshot({path:'media/optimizer-token-live.png',fullPage:true});
  await page.evaluate(()=>window.__cartpoleSonic.runTemporalControlPPO(1));
  const trained=await state(),lab=trained.representationLabs.temporalControl;
  assert.equal(lab.one.iter,1);assert.equal(lab.two.iter,1);
  assert.ok(Number.isFinite(lab.two.last.postUpdateKl));
  assert.ok(lab.two.last.updates.dynamic.updateNorm>0);
  assert.equal(trained.backends.webmcp.tools.length,13);
  checks.push({label:'matched live PPO / WebMCP state',oneIterations:lab.one.iter,twoIterations:lab.two.iter,kl:lab.two.last.postUpdateKl,adamDelta:lab.two.last.updates.dynamic.updateNorm,webmcp:trained.backends.webmcp});
  await page.evaluate(()=>window.__cartpoleSonic.resetTemporalControlLab());
  assert.equal((await state()).representationLabs.temporalControl.one.iter,0);
  if(server){
    await page.waitForTimeout(1100);
    const telemetry=await page.evaluate(async()=>{const r=await fetch('/telemetry');return r.json();});
    checks.push({label:'bounded local telemetry',bytes:JSON.stringify(telemetry).length,status:telemetry.status});
    assert.notEqual(telemetry.status,'no-browser','local telemetry must actually reach the server');
    assert.ok(JSON.stringify(telemetry).length<65536,'keepalive telemetry must stay within its byte budget');
  }
  assert.deepEqual(errors,[]);report.passed=true;
}catch(error){report.passed=false;report.failure=error.stack;throw error;
}finally{
  fs.writeFileSync('evidence/browser_audit.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));await browser.close();
  if(server)await new Promise(resolve=>server.close(resolve));
}
