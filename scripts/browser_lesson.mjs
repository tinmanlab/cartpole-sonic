import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {validateLessonData} from '../native_lesson.js';
const output=path.join(process.env.UI_TEST_OUTPUT||'/tmp/sonic-lesson-ui','lesson');fs.mkdirSync(output,{recursive:true});
const data=validateLessonData(JSON.parse(fs.readFileSync('evidence/guided_lesson/traces.json','utf8')));
const root=process.cwd();
const server=http.createServer((req,res)=>{
 const pathname=new URL(req.url,'http://localhost').pathname;
 if(pathname==='/telemetry'){res.setHeader('Content-Type','application/json');res.end('{}');return;}
 const file=path.resolve(root,'.'+(pathname==='/'?'/index.html':decodeURIComponent(pathname)));
 if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end();return;}
 res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.wasm':'application/wasm'})[path.extname(file)]||'application/octet-stream');fs.createReadStream(file).pipe(res);
});
let browser,passed=false,failure=null,browserVersion=null,negativeCase=false;
const checks=[],screenshots=[],errors=[];
const sourceFiles=['app.js','index.html','native_lesson.js','native_lesson.css','presentation.js','scripts/browser_lesson.mjs','evidence/guided_lesson/traces.json'];
const hashes=()=>Object.fromEntries(sourceFiles.map(f=>[f,createHash('sha256').update(fs.readFileSync(f)).digest('hex')]));
const source_sha256=hashes();
try{
 await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
 browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined});browserVersion=browser.version();
 const page=await browser.newPage({viewport:{width:1440,height:900},deviceScaleFactor:2,reducedMotion:'reduce'});
 const base=process.argv[2]||`http://127.0.0.1:${server.address().port}/`;
 page.on('pageerror',e=>errors.push(e.message));
 page.on('console',m=>{if(m.type()==='error'&&!negativeCase)errors.push(m.text());});
 const snap=()=>page.evaluate(()=>window.__cartpoleSonic.getState());
 const shot=async name=>{await page.screenshot({path:path.join(output,name),fullPage:true});screenshots.push(name);};
 const seek=async tick=>page.locator('#lessonSeek').evaluate((el,t)=>{el.value=t;el.dispatchEvent(new Event('input',{bubbles:true}));},tick);
 const fit=async()=>{
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  assert.equal(await page.locator('#nativeLesson').evaluate(el=>[...el.querySelectorAll('button,select,table,canvas')].filter(x=>x.getClientRects().length).some(x=>x.getBoundingClientRect().right>innerWidth)),false);
 };
 await page.goto(base);await page.locator('#lessonBody').waitFor({state:'visible'});
 assert.equal((await snap()).activeSurface,'native-lesson');assert.equal((await snap()).experiment.live,false);
 await page.waitForFunction(()=>{const s=window.__cartpoleSonic?.getState();return s?.backends.physics?.includes('MuJoCo')&&!s.experiment.busy;});
 assert.notEqual((await snap()).execution.state,'error','hidden exploration initializes without canvas drawing');
 await fit();await shot('lesson-default.png');
 await page.getByRole('button',{name:'구별할 수 있다',exact:true}).click();
 assert.match(await page.locator('#lessonFeedback').textContent(),/추가 정보/);
 await page.getByRole('button',{name:'구별할 수 없다',exact:true}).click();
 assert.match(await page.locator('#lessonFeedback').textContent(),/구별할 수 없습니다/);
 await page.locator('#lessonReveal').click();await seek(7);
 assert.equal((await snap()).lesson.recorded.time,.14,'API must identify the recorded lesson sample, not the paused live plant');
 for(const condition of data.protocol.conditions){
  const f=data.cases[0].runs.find(r=>r.condition===condition&&r.offset==='nominal'&&r.branch===0).frames[7];
  const values=await page.locator(`[data-condition="${condition}"] td`).allTextContents();
  for(let j=0;j<4;j++)assert.equal(Number(values[j]),Number(f.state[j].toFixed(4)));
  assert.equal(values[4],f.decision.appliedForceN.toFixed(3));
  assert.equal(values[5],f.decision.token.join(', '));
 }
 const chartHeight=await page.locator('#lessonChart').evaluate(c=>c.clientHeight);
 for(const t of [0,1,7,14,28,7])await seek(t);
 assert.equal(await page.locator('#lessonChart').evaluate(c=>c.clientHeight),chartHeight,'DPR2 chart height is stable');
 await shot('lesson-comparison.png');await seek(28);
 assert.match(await page.locator('#lessonDecision').textContent(),/기록 종료: 새 행동 없음/);
 assert.match(await page.locator('#lessonValues').textContent(),/—/);
 for(const [id,value] of [['lessonCase',data.cases[1].id],['lessonBranch','1'],['lessonOffset','perturbed']]){
  await seek(10);await page.locator('#'+id).selectOption(value);
  assert.equal((await snap()).lesson.tick,0);assert.equal((await snap()).lesson.playing,false);
 }
 await page.locator('#lessonInterpret').click();assert.match(await page.locator('#lessonStats').textContent(),/사례 2/);
 await shot('lesson-interpretation.png');await fit();
 const before=await snap();
 await assert.rejects(()=>page.evaluate(()=>window.__cartpoleSonic.step()),/구조 탐색/);
 await assert.rejects(()=>page.evaluate(()=>window.__cartpoleSonic.runPPO(1)),/구조 탐색/);
 await page.waitForTimeout(200);const after=await snap();
 assert.deepEqual(after.signals.proprioception,before.signals.proprioception);
 assert.equal(after.training?.ppoIterations,before.training?.ppoIterations);
 await page.evaluate(()=>window.__cartpoleSonic.focus('robot'));
 assert.equal((await snap()).activeSurface,'exploration','explicit tool navigation must provide an exit from record mode');
 await page.locator('#nativeLessonTab').click();
 await page.locator('#lessonPlay').click();await page.waitForTimeout(100);await page.locator('#explorationTab').click();
 assert.equal((await snap()).lesson.playing,false);assert.equal((await snap()).activeSurface,'exploration');
 await page.locator('main.shell').waitFor({state:'visible'});
 await page.locator('#nativeLessonTab').click();await seek(27);await page.locator('#lessonPlay').click();
 await page.waitForTimeout(250);assert.equal((await snap()).lesson.tick,28);assert.equal((await snap()).lesson.playing,false);
 await page.getByRole('button',{name:'1 예상',exact:true}).click();await page.locator('#lessonSkip').click();
 assert.equal((await snap()).lesson.stage,2);
 checks.push('prediction/skip, raw state/force/token alignment, end null, selection reset, playback end, surfaces and mutation rejection');
 await page.setViewportSize({width:1920,height:1080});await fit();await shot('lesson-wide.png');
 checks.push('1440x900 and 1920x1080 horizontal bounds');
 await page.goto(base+'?focus=token');await page.waitForFunction(()=>window.__cartpoleSonic?.getState().activeSurface==='exploration');
 assert.equal(await page.locator('#nativeLesson').isVisible(),false);await page.locator('main.shell').waitFor({state:'visible'});
 await page.goto(base+'?training=ppo');await page.waitForFunction(()=>window.__cartpoleSonic?.getState().activeSurface==='exploration');
 assert.equal((await snap()).system.trainingTopic,'ppo');checks.push('explicit focus/training deep links');
 negativeCase=true;
 for(const invalid of [false,true]){
  await page.route('**/evidence/guided_lesson/traces.json',route=>route.fulfill({status:invalid?200:404,contentType:'application/json',body:invalid?'{}':'missing'}));
  await page.goto(base+'?lesson=future');await page.locator('#lessonRetry').waitFor({state:'visible'});
  assert.equal(await page.locator('#lessonBody').isVisible(),false);assert.match(await page.locator('#lessonStatus').textContent(),/표시할 수 없습니다/);
  await page.unroute('**/evidence/guided_lesson/traces.json');await page.locator('#lessonRetry').click();await page.locator('#lessonBody').waitFor({state:'visible'});
 }
 negativeCase=false;checks.push('missing/invalid evidence error and real-record retry');
 assert.deepEqual(errors,[]);assert.deepEqual(hashes(),source_sha256);passed=true;
 console.log(JSON.stringify({passed,checks,errors,screenshots},null,2));
}catch(error){failure=error.stack;console.error(JSON.stringify({passed:false,checks,errors,error:error.message},null,2));process.exitCode=1;
}finally{
 fs.writeFileSync(path.join(output,'browser_audit.json'),JSON.stringify({passed,checks,errors,failure,browser:browserVersion,source_sha256,screenshots},null,2));
 await browser?.close();await new Promise(resolve=>server.close(resolve));
}
