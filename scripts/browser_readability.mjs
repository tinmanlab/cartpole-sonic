import {chromium} from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {createHash} from 'node:crypto';
import {SONIC_FLOW,TRAINING_TOPICS} from '../course.js';
const output=path.join(process.env.UI_TEST_OUTPUT||'/tmp/sonic-readability-agent','readability');fs.mkdirSync(output,{recursive:true});
const root=process.cwd(),findings=[],views=[],errors=[],screenshots=[];
const sources=['app.js','course.js','index.html','native_lesson.js','native_lesson.css','presentation.js','optimizer_evidence_view.js','scripts/browser_readability.mjs','scripts/verify_readability.test.mjs'];
const source_sha256=Object.fromEntries(sources.map(f=>[f,createHash('sha256').update(fs.readFileSync(f)).digest('hex')]));
const server=http.createServer((req,res)=>{const u=new URL(req.url,'http://localhost');if(u.pathname==='/telemetry'){res.end('{}');return;}const f=path.resolve(root,'.'+(u.pathname==='/'?'/index.html':u.pathname));if(!f.startsWith(root+path.sep)||!fs.existsSync(f)){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',({'.js':'text/javascript','.html':'text/html','.css':'text/css','.json':'application/json','.wasm':'application/wasm'})[path.extname(f)]||'application/octet-stream');fs.createReadStream(f).pipe(res);});
let browser;
try{
 await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||undefined});
 for(const [width,height,dpr] of [[1440,900,1],[1440,900,2],[1920,1080,1],[1920,1080,2]]){
 const page=await browser.newPage({viewport:{width,height},deviceScaleFactor:dpr});
 await page.route('https://**',r=>r.abort());page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>{window.__textAudit=[];for(const dimension of ['width','height']){const d=Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype,dimension);Object.defineProperty(HTMLCanvasElement.prototype,dimension,{...d,set(v){window.__textAudit=window.__textAudit.filter(t=>t.canvas!==this.id);return d.set.call(this,v);}});}const clear=CanvasRenderingContext2D.prototype.clearRect;CanvasRenderingContext2D.prototype.clearRect=function(...args){window.__textAudit=window.__textAudit.filter(t=>t.canvas!==this.canvas.id);return clear.apply(this,args);};for(const name of ['fillText','strokeText']){const original=CanvasRenderingContext2D.prototype[name];CanvasRenderingContext2D.prototype[name]=function(text,x,y,...rest){const m=this.measureText(String(text)),scale=this.getTransform().a||1,w=this.canvas.width/scale;let left=x;if(this.textAlign==='center')left-=m.width/2;else if(['right','end'].includes(this.textAlign))left-=m.width;window.__textAudit.push({canvas:this.canvas.id,text:String(text),font:this.font,left,right:left+m.width,w,y});return original.call(this,text,x,y,...rest);};}});
 await page.goto(`http://127.0.0.1:${server.address().port}/?focus=task`);await page.waitForFunction(()=>window.__cartpoleSonic&&!window.__cartpoleSonic.getState().experiment.busy&&window.__cartpoleSonic.getState().backends.physics);await page.evaluate(()=>window.__cartpoleSonic.setLive(false));
 await page.evaluate(()=>{const el=document.querySelector('.sim');el.style.display='none';window.__cartpoleSonic.setGoal(.8);el.style.display='';window.__cartpoleSonic.setGoal(.8);});
 const audit=async(name,formula=null)=>{try{
 await page.waitForTimeout(35);
 const result=await page.evaluate(()=>{const faults=[];if(document.documentElement.scrollWidth>innerWidth+2)faults.push({kind:'page-horizontal-overflow',width:document.documentElement.scrollWidth,viewport:innerWidth});const visible=e=>e&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden'&&e.checkVisibility();
 for(const el of document.querySelectorAll('body *')){if(!visible(el)||!Array.from(el.childNodes).some(n=>n.nodeType===3&&n.textContent.trim())||['SCRIPT','STYLE','OPTION'].includes(el.tagName))continue;const s=getComputedStyle(el),size=parseFloat(s.fontSize),floor=el.matches('p,.viz-caption,.sim-note,.guide-core,.guide-warn')?13:12;if(size<floor)faults.push({kind:'font',tag:el.tagName,class:el.className,text:el.textContent.slice(0,65),size,floor});if(el.scrollWidth>el.clientWidth+2&&s.overflowX==='hidden')faults.push({kind:'clipped',text:el.textContent.slice(0,65)});if(/undefined|NaN|\\(?:frac|argmin|tanh)/.test(el.textContent)&&el.children.length===0)faults.push({kind:'placeholder',text:el.textContent.slice(0,65)});}
 for(const t of window.__textAudit){const c=document.getElementById(t.canvas);if(!c||!visible(c))continue;const size=parseFloat(t.font.match(/([\d.]+)px/)?.[1]||0);if(size<12||t.left<-.5||t.right>t.w+.5)faults.push({kind:'canvas',...t});}
 const texts=[];
 for(const el of document.querySelectorAll('button,th,td,.io-box,.live-kv,.guide-block p')){
 if(!visible(el))continue;const range=document.createRange();range.selectNodeContents(el);
 for(const r of range.getClientRects())texts.push({el,left:r.left,right:r.right,top:r.top,bottom:r.bottom,text:el.textContent.slice(0,50)});
 }
 for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){const a=texts[i],b=texts[j];if(a.el===b.el||a.el.contains(b.el)||b.el.contains(a.el))continue;if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>2&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>2)faults.push({kind:'DOM-text-overlap',a:a.text,b:b.text});}
 const captions=[...new Map(window.__textAudit.map(t=>[JSON.stringify(t),t])).values()];
 for(let i=0;i<captions.length;i++)for(let j=i+1;j<captions.length;j++){const a=captions[i],b=captions[j];if(a.canvas!==b.canvas||!visible(document.getElementById(a.canvas)))continue;const ah=parseFloat(a.font.match(/([\d.]+)px/)?.[1]||0),bh=parseFloat(b.font.match(/([\d.]+)px/)?.[1]||0);if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>2&&Math.min(a.y,b.y)-Math.max(a.y-ah,b.y-bh)>2)faults.push({kind:'canvas-text-overlap',a:a.text,b:b.text});}
 window.__textAudit=[];return faults;});
 if(formula){const shape=await page.locator('#guideShape').textContent();if(!shape.includes(formula)||!shape.includes('\n')||await page.locator('#guideShape').evaluate(e=>getComputedStyle(e).whiteSpace)==='normal')result.push({kind:'formula',formula,shape});}
 if(name==='lesson/2'){
 const caption=await page.locator('.lesson-values caption').textContent();const axis=await page.locator('.lesson-axis-note').textContent();
 if(!caption.includes('동일한 시각')||!axis.includes('mm')||!axis.includes('같은 축'))result.push({kind:'units-or-time',caption,axis});
 }
 if(name.startsWith('training/optimizer-sensitivity')&&!(await page.locator('#vizHtml').textContent()).includes('저장된 평가 기록'))result.push({kind:'source-identity'});
 views.push({name,width,height,dpr,findings:result.length});for(const f of result.slice(0,40))findings.push({view:name,width,...f});
 if(width===1440&&dpr===1&&['quantizer/fsq/mechanism','quantizer/vqvae/mechanism','training/ppo/mechanism','training/optimizer-sensitivity/mechanism','lesson/3'].includes(name)){const file=name.replaceAll('/','-')+'.png';await page.screenshot({path:path.join(output,file),fullPage:true});screenshots.push(file);}
 }catch(error){errors.push({view:name,width,error:error.message});}
 };
 for(const n of SONIC_FLOW)for(const concept of [null,...n.concepts.map(c=>typeof c==='string'?c:c.id)]){try{await page.evaluate(async({id,concept})=>{window.__textAudit=[];await window.__cartpoleSonic.focus(id,concept);},{id:n.id,concept});for(const depth of ['easy','mechanism','sonic']){await page.locator(`[data-depth="${depth}"]`).click();await audit(`${n.id}/${concept||'core'}/${depth}`,depth!=='easy'?({ae:'MSE',vq:'k*',vqvae:'k*',fsq:'1.998'})[concept]:null);}}catch(error){errors.push({view:n.id+'/'+concept,width,error:error.message});}}
 for(const t of TRAINING_TOPICS){try{await page.evaluate(async id=>{window.__textAudit=[];await window.__cartpoleSonic.openTraining(id);},t.id);for(const depth of ['easy','mechanism','sonic']){await page.locator(`[data-depth="${depth}"]`).click();await audit(`training/${t.id}/${depth}`,t.id==='ppo'&&depth!=='easy'?'A':null);}if(t.id==='optimizer-sensitivity'){await page.locator('#optDiagnostics').click();await audit('optimizer/diagnostics');}}catch(error){errors.push({view:'training/'+t.id,width,error:error.message});}}
 await page.locator('#nativeLessonTab').click();await page.locator('#lessonBody').waitFor({state:'visible'});for(const stage of [1,2,3]){await page.locator(`button[data-stage="${stage}"]`).click();await audit('lesson/'+stage);}
 // 200% desktop zoom has half the CSS layout width. Actual browser zoom needs parent verification.
 if(width===1440){await page.setViewportSize({width:720,height:450});await audit('lesson/200-percent-layout');await page.locator('#explorationTab').click();await audit('exploration/200-percent-layout');}
 await page.close();
 }
}catch(e){errors.push(e.stack);}finally{const stable=sources.every(f=>source_sha256[f]===createHash('sha256').update(fs.readFileSync(f)).digest('hex'));if(!stable)errors.push('source changed during audit');const report={passed:findings.length===0&&errors.length===0,scope:'isolated layout audit: external HTTPS unavailable; full browser suite checks live dependency behavior',views,findings,errors,screenshots,source_sha256};fs.writeFileSync(path.join(output,'readability.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,views:views.length,findings:findings.length,errors,output}));if(!report.passed)process.exitCode=1;await browser?.close();if(server.listening)await new Promise(r=>server.close(r));}
