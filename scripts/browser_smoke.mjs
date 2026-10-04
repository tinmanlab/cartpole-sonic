import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const executionOnly=process.env.EXECUTION_ONLY==='1'||process.argv.includes('--execution');
const output=path.join(process.env.UI_TEST_OUTPUT||'/tmp/sonic-public-ui-test',executionOnly?'execution':'visual');
fs.mkdirSync(output,{recursive:true});
const screenshot=name=>page.screenshot({path:path.join(output,name),fullPage:true});
const spec=process.env.PLAYWRIGHT_MODULE;
const {chromium}=await import(spec?pathToFileURL(spec).href:'playwright');
// A static HTTP server owned by this test process: no persistent shell, desktop or process to kill.
let server,latest={status:'no-browser'};
let base=process.argv.slice(2).find(x=>!x.startsWith('--'));
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
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);}).catch(error=>{fs.writeFileSync(path.join(output,'browser_audit.json'),JSON.stringify({passed:false,stage:'local HTTP server startup',failure:error.message,screenshots:[]},null,2));throw error;});base='http://127.0.0.1:'+server.address().port+'/';
}
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined,args:['--enable-experimental-web-platform-features']});
const page=await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:1});
const errors=[],checks=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const sourceFiles=['app.js','course.js','index.html','presentation.js','optimizer_evidence_view.js','control_contract.js','mujoco_sim.js','sonic_toy.js','temporal_control_lab.js','native_lesson.js','native_lesson.css','evidence/guided_lesson/traces.json','scripts/browser_smoke.mjs'];
const sourceHashes=()=>Object.fromEntries(sourceFiles.map(file=>[file,createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
const report={schema:'cartpole-sonic-browser-audit/v1',base,browser:browser.version(),executionOnly,source_sha256:sourceHashes(),checks,errors};
const state=()=>page.evaluate(()=>window.__cartpoleSonic.getState());
const waitReady=async()=>{
  await page.waitForFunction(()=>window.__cartpoleSonic?.getState().backends.physics?.includes('MuJoCo'));
  await page.waitForFunction(()=>window.__cartpoleSonic?.getState().signals.token?.length>0);
  if(new URL(page.url()).searchParams.get('training')==='optimizer-sensitivity')await page.waitForFunction(()=>document.querySelector('.optimizer-audit canvas'));
};
const bounds=async label=>{
  await page.waitForTimeout(35);
  const result=await page.evaluate(()=>{
    const guide=document.querySelector('.guide-card');
    return {documentXOverflow:document.documentElement.scrollWidth>innerWidth+1,guideScrollNeeded:guide.scrollHeight>guide.clientHeight+2,guideOverflow:guide.scrollHeight>guide.clientHeight+2&&!['auto','scroll'].includes(getComputedStyle(guide).overflowY),canvasSizes:[...document.querySelectorAll('canvas')].filter(c=>c.getClientRects().length&&getComputedStyle(c).display!=='none').map(c=>[c.id,c.width,c.height])};
  });
  const clippedMap=await page.evaluate(()=>{
    const bottom=document.querySelector('.system-map').getBoundingClientRect().bottom;
    return [...document.querySelectorAll('.system-map .map-node,.system-map .map-lower,.system-map #motionBranch')].filter(e=>e.getBoundingClientRect().bottom>bottom+1).map(e=>e.id||e.className);
  });
  assert.deepEqual(clippedMap,[],label+' clipped system-map content');
  checks.push({label,...result,clippedMap});
  assert.ok(result.canvasSizes.some(([id])=>id==='cart'),label+' missing visible simulation canvas');
  if((await state()).system.focus?.node==='control-decoder'){
    const layout=await page.evaluate(async()=>{const c=document.querySelector('#lessonViz'),r=c.getBoundingClientRect();return {width:r.width,height:r.height,boxes:(await import('./presentation.js')).decoderLayout(r.width)};});
    for(const r of layout.boxes)assert.ok(r.x>=0&&r.x+r.w<=layout.width&&r.y+r.h<=layout.height,label+' decoder box clipped');
  }
  assert.equal(result.documentXOverflow,false,label+' page horizontal overflow');
  assert.equal(result.guideOverflow,false,label+' hidden guide content');
  for(const [id,w,h] of result.canvasSizes)assert.ok(w>0&&h>0,label+' empty canvas '+id);
};
try{
  if(executionOnly){
    await page.goto(base+'?focus=task');await waitReady();
    const api=code=>page.evaluate(code);
    const reject=async work=>{const result=await page.evaluate(async source=>{try{await Function('return ('+source+')')()();return null;}catch(e){return e.message;}},work.toString());assert.ok(result,'expected API rejection');};
    const before=await state();
    const telemetryBefore=await api(()=>window.__cartpoleSonic.getTelemetry());
    for(const key of ['sampleTime','targetTime','referenceNow','referencePreview','plannedForce','lastAppliedForce'])assert.equal(telemetryBefore.signals[key],before.signals[key],key+' must use the same signal definition');
    for(const work of [()=>window.__cartpoleSonic.step(NaN),()=>window.__cartpoleSonic.step(1.5),()=>window.__cartpoleSonic.setGoal('1'),()=>window.__cartpoleSonic.setGoal(Infinity),()=>window.__cartpoleSonic.simulationControl({action:'unknown'}),()=>window.__cartpoleSonic.runPPO(0)])await reject(work);
    assert.deepEqual((await state()).signals.proprioception,before.signals.proprioception);
    assert.equal((await state()).experiment.goal,before.experiment.goal);
    await api(()=>window.__cartpoleSonic.step(2));
    assert.equal((await state()).signals.simulationTime,.04);
    await api(()=>window.__cartpoleSonic.setLive(true));await page.waitForTimeout(150);await api(()=>window.__cartpoleSonic.setLive(false));
    assert.equal((await state()).execution.state,'paused');
    await api(()=>window.__cartpoleSonic.reset());
    await api(()=>{for(let i=0;i<40&&!window.__cartpoleSonic.getState().execution.resetRequired;i++){window.__cartpoleSonic.push();window.__cartpoleSonic.step(1);}});
    assert.equal((await state()).execution.state,'terminated');
    const failed=(await state()).signals.proprioception;
    for(const work of [()=>window.__cartpoleSonic.step(),()=>window.__cartpoleSonic.setLive(true),()=>window.__cartpoleSonic.push()])await reject(work);
    assert.deepEqual((await state()).signals.proprioception,failed);
    await api(()=>window.__cartpoleSonic.reset());await api(()=>window.__cartpoleSonic.step());
    assert.equal((await state()).execution.resetRequired,false);
    checks.push({label:'invalid inputs preserve state; manual/live termination latch and reset'});
    await screenshot('execution-reset.png');
    const ordinary=(await state()).training.ppoIterations;
    await api(()=>window.__cartpoleSonic.focus('token','temporal-control'));
    const pair=(await state()).representationLabs.temporalControl;
    const concurrent=await api(async()=>{
      const running=window.__cartpoleSonic.runPPO(1);
      const disabled=document.querySelector('#stepBtn').disabled;
      let rejection;try{await window.__cartpoleSonic.focus('robot');}catch(e){rejection=e.message;}
      await running;return{disabled,rejection};
    });
    assert.equal(concurrent.disabled,true);assert.ok(concurrent.rejection);
    const trained=await state();assert.equal(trained.representationLabs.temporalControl.one.iter,pair.one.iter+1);assert.equal(trained.representationLabs.temporalControl.two.iter,pair.two.iter+1);
    assert.equal(trained.training.ppoIterations,ordinary);
    assert.deepEqual(trained.lastTrainingTargets,['temporal-one','temporal-two']);
    assert.equal(trained.execution.state,'ready');
    checks.push({label:'generic training targets both comparison policies; cached ordinary policy unchanged; busy calls rejected'});
    await api(()=>window.__cartpoleSonic.focus('robot'));
    await page.route('**/assets/student_ae_bootstrap.json',route=>route.fulfill({status:200,contentType:'application/json',body:'{"schema":"cartpole-sonic-student-bootstrap/v1","mode":"ae","policy":{}}'}));
    await reject(()=>window.__cartpoleSonic.focus('encoder','ae'));
    assert.equal((await state()).experiment.busy,false);assert.equal((await state()).experiment.live,false);assert.equal((await state()).execution.state,'error');
    assert.ok((await page.locator('#episodeStatus').innerText()).includes('체크포인트'));
    await page.unroute('**/assets/student_ae_bootstrap.json');await api(()=>window.__cartpoleSonic.focus('encoder','ae'));
    assert.equal((await state()).activeController.mode,'ae');assert.equal((await state()).execution.state,'ready');
    await api(()=>window.__cartpoleSonic.openTraining('optimizer-sensitivity'));
    await api(()=>window.__cartpoleSonic.runFocusAction());
    assert.equal((await state()).system.focus?.concept,'temporal-control','evidence action must open the advertised live comparison');
    checks.push({label:'execution boundaries, latch, matched generic training, busy rejection, corrupt checkpoint recovery'});
    await screenshot('execution-recovered.png');
    assert.deepEqual(errors,[]);report.passed=true;
  }else{
  await page.goto(base+'?training=optimizer-sensitivity&depth=mechanism');await waitReady();
  assert.equal((await state()).system.version,'2.9');
  await screenshot('optimizer-budget-50.png');
  assert.equal(await page.locator('.opt-data tbody tr').count(),16);
  await bounds('optimizer +50');
  await page.click('#opt10');
  assert.equal((await state()).training.optimizerEvidenceView.budget,10);
  await screenshot('optimizer-budget-10.png');await bounds('optimizer +10');
  await page.click('#optDiagnostics');await bounds('optimizer diagnostics');
  await page.selectOption('#optVariant','two-matched-capacity');
  assert.ok((await page.locator('.opt-cards').innerText()).includes('1220 / 643'));
  await page.selectOption('#optVariant','two-matched-actor');
  assert.ok((await page.locator('.opt-cards').innerText()).includes('1268 / 607'));
  await page.selectOption('#optVariant','two-default');
  await screenshot('optimizer-diagnostics.png');
  for(const width of [1440,1920]){await page.setViewportSize({width,height:1000});await bounds('optimizer '+width+'px');}
  await page.setViewportSize({width:1600,height:1000});
  const map=await page.evaluate(async()=>{const c=await import('./course.js');return {nodes:c.SONIC_FLOW.map(n=>({id:n.id,concepts:n.concepts})),training:c.TRAINING_TOPICS.map(t=>t.id)};});
  for(const [width,height] of [[1440,900],[1920,1080]]){
    await page.setViewportSize({width,height});
    for(const depth of ['easy','mechanism','sonic']){
      await page.click('[data-depth="'+depth+'"]');
      for(const node of map.nodes){
        await page.evaluate(id=>window.__cartpoleSonic.focus(id),node.id);await bounds(width+'/'+depth+'/'+node.id);
        for(const c of node.concepts||[]){await page.evaluate(({id,c})=>window.__cartpoleSonic.focus(id,c.id),{id:node.id,c});await bounds(width+'/'+depth+'/'+node.id+'/'+c.id);}
      }
      for(const id of map.training){await page.evaluate(id=>window.__cartpoleSonic.openTraining(id),id);await bounds(width+'/'+depth+'/training/'+id);}
    }
    await page.evaluate(()=>window.__cartpoleSonic.focus('task'));
    await page.click('[data-depth="easy"]');
    await screenshot('overview-'+width+'.png');
    for(const preset of ['playground','long','heavy']){
      await page.selectOption('#simPreset',preset);
      await page.waitForFunction(p=>window.__cartpoleSonic.getState().experiment.preset===p&&!window.__cartpoleSonic.getState().experiment.busy,preset);
      await page.waitForFunction(()=>!document.querySelector('#stepBtn').disabled);
      await bounds(width+'/preset/'+preset);
      const g=(await state()).signals.drawing;
      const rect=await page.locator('#cart').boundingBox();
      assert.ok(g.ty>=0&&g.ty<=rect.height&&g.tx>=0&&g.tx<=rect.width,'pole fits '+preset);
      checks.push({label:'preset drawing '+width+'/'+preset,geometry:g,rect});
    }
    await page.selectOption('#simPreset','playground');
    await page.waitForFunction(()=>!window.__cartpoleSonic.getState().experiment.busy);
  }
  await page.click('[data-depth="easy"]');
  await page.evaluate(()=>window.__cartpoleSonic.focus('control-decoder'));
  await screenshot('control-decoder.png');
  const pushed=await page.evaluate(()=>{
    const before=window.__cartpoleSonic.getState();document.querySelector('#pushBtn').click();
    return {before,after:window.__cartpoleSonic.getState()};
  });
  assert.deepEqual(pushed.before.signals.reference,pushed.after.signals.reference);
  assert.deepEqual(pushed.before.signals.token,pushed.after.signals.token);
  assert.equal(pushed.before.signals.sampleTime,pushed.after.signals.sampleTime);
  assert.notDeepEqual(pushed.before.signals.proprioception,pushed.after.signals.proprioception);
  assert.equal(pushed.before.signals.lastAppliedForce,pushed.after.signals.lastAppliedForce);
  assert.notEqual(pushed.before.signals.plannedForce,pushed.after.signals.plannedForce);
  await page.evaluate(()=>window.__cartpoleSonic.step(3));
  const sampled=(await state()).signals;
  assert.equal(sampled.targetTime,sampled.sampleTime);
  assert.equal(await page.locator('#vForce').innerText(),sampled.lastAppliedForce.toFixed(2)+' N');
  for(const h of sampled.liveHistory){assert.equal(h.targetTime,h.t);assert.equal(h.previewTargetTime,h.t+.08);assert.equal(h.plannedForce,h.force);}
  await page.evaluate(()=>window.__cartpoleSonic.focus('robot'));
  const robot=(await state()).signals;
  assert.ok((await page.locator('#guideLive').innerText()).includes(Math.abs(robot.proprioception[0]-robot.referenceNow).toFixed(3)+' m'));
  assert.ok(!(await page.locator('body').innerText()).includes('native MuJoCo'));
  assert.ok((await page.locator('.identity-strip').innerText()).includes('브라우저 교육용 모델 · MuJoCo WASM 시뮬레이션'));
  const scale=await page.evaluate(async()=> (await import('./sonic_toy.js')).SONIC_TOY_CONSTANTS.STATE_SCALE[0]);
  for(const h of sampled.liveHistory)assert.equal(h.referencePreview,h.reference[0]*scale);
  await page.evaluate(()=>window.__cartpoleSonic.focus('token','temporal-control'));
  assert.equal((await state()).signals.activeController.flattenedDim,2);
  await page.click('#driveTwoBtn');
  assert.equal((await state()).signals.activeController.flattenedDim,4);
  assert.equal((await state()).signals.activeController.tokens,2);
  await screenshot('two-token.png');
  const before=(await state()).signals.proprioception;
  await page.click('#liveBtn');
  await page.waitForFunction(s=>window.__cartpoleSonic.getState().signals.proprioception.some((v,i)=>Math.abs(v-s[i])>.005),before);
  await page.click('#liveBtn');
  const live=await state();
  assert.equal(live.representationLabs.temporalControl.selectedController,'two');
  assert.ok(Number.isFinite(live.signals.force));
  assert.equal(await page.locator('#simPreset').isDisabled(),true);
  checks.push({label:'LIVE browser MuJoCo WASM two-token control',before,after:live.signals.proprioception,force:live.signals.force});
  await screenshot('optimizer-token-live.png');
  await page.evaluate(()=>window.__cartpoleSonic.focus('encoder','ae'));
  assert.equal((await state()).signals.activeController.mode,'ae');
  assert.equal((await state()).signals.activeController.flattenedDim,2);
  assert.ok((await page.locator('#activeController').innerText()).includes('AE'));
  await page.evaluate(()=>window.__cartpoleSonic.focus('quantizer','vq'));
  assert.equal((await state()).signals.activeController.mode,'vq');
  await page.evaluate(()=>window.__cartpoleSonic.focus('token','temporal-control'));
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
  }
}catch(error){report.passed=false;report.failure=error.stack;throw error;
}finally{
  assert.deepEqual(sourceHashes(),report.source_sha256,'rendering source changed during browser audit');
  fs.writeFileSync(path.join(output,'browser_audit.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({passed:report.passed,checks:checks.length,errors,failure:report.failure},null,2));await browser.close();
  if(server)await new Promise(resolve=>server.close(resolve));
}
