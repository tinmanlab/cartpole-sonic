import {canvasTicks} from './presentation.js';
const CONDITIONS=['full_future','endpoints_only'];
const NAMES=['미래 구간 학습','현재·도착점 학습'];
const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const close=(a,b)=>Math.abs(a-b)<1e-9;
function requireValue(ok,message){if(!ok)throw new Error('기록 검증 실패: '+message);}
function vector(v,n=4){return Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);}
export function validateLessonData(d){
 requireValue(d?.schema==='cartpole-sonic-guided-lesson/v1'&&d.source?.kind==='native-recorded-reexecution','schema/source');
 const source=d.source;
 requireValue(source.encoderSeed===52101&&source.modality==='joints'&&typeof source.training==='string'&&typeof source.scope==='string'&&/^[a-f0-9]{40}$/.test(source.upstreamCommit)&&source.files&&Object.keys(source.files).length>0&&Object.entries(source.files).every(([name,hash])=>!name.startsWith('/')&&!name.split('/').includes('..')&&/^[a-f0-9]{64}$/.test(hash)),'source provenance');
 const p=d.protocol;
 requireValue(p&&p.controlDt===.02&&p.horizonSteps===28&&equal(p.referenceOffsets,[0,4,8,12,16,20,24,28])&&equal(p.stateOrder,['x','xdot','theta','thetadot'])&&equal(p.stateUnits,['m','m/s','rad','rad/s'])&&equal(p.conditions,CONDITIONS)&&equal(p.offsets,{nominal:[0,0,0,0],perturbed:[0,.02,.001,-.02]}),'protocol');
 requireValue(Array.isArray(d.cases)&&d.cases.length===3,'three cases');
 let previous='';
 for(const c of d.cases){
  requireValue(typeof c.id==='string'&&/^test-\d{3,}$/.test(c.id)&&c.id>previous,'case order/ID');previous=c.id;
  requireValue(Array.isArray(c.reference)&&c.reference.length===2&&c.reference.every(r=>Array.isArray(r)&&r.length===29&&r.every(v=>vector(v)))&&Number.isFinite(c.endpointMaxError)&&c.endpointMaxError>=0,'reference');
  const endpointError=Math.max(...c.reference.flatMap(r=>r[28].map((v,i)=>Math.abs(v-r[0][i]))));
  requireValue(endpointError<=1e-7&&[0,28].every(t=>c.reference[0][t].every((v,i)=>Math.abs(v-c.reference[1][t][i])<=1e-7)),'full-state endpoint matching');
  requireValue(close(endpointError,c.endpointMaxError),'endpoint evidence');
  requireValue(Array.isArray(c.runs)&&c.runs.length===8,'eight runs');
  const keys=new Set();
  for(const r of c.runs){
   const key=[r.condition,r.offset,r.branch].join('/');
   requireValue(CONDITIONS.includes(r.condition)&&['nominal','perturbed'].includes(r.offset)&&[0,1].includes(r.branch)&&!keys.has(key),'run identity');keys.add(key);
   requireValue(Array.isArray(r.frames)&&r.frames.length===29,'29 samples');
   for(const [i,f] of r.frames.entries()){
    requireValue(f.tick===i&&Number.isFinite(f.time)&&close(f.time,i*.02)&&vector(f.state)&&vector(f.referenceNow)&&equal(f.referenceNow,c.reference[r.branch][i])&&Number.isFinite(f.errorX)&&close(f.errorX,f.state[0]-f.referenceNow[0]),'frame alignment');
    const a=f.decision;
    requireValue(i===28?a===null:a&&Array.isArray(a.referenceWindow)&&a.referenceWindow.length===8&&a.referenceWindow.every(v=>vector(v))&&vector(a.latent)&&vector(a.token)&&Number.isFinite(a.requestedForceN)&&Number.isFinite(a.appliedForceN),'decision/final null');
    if(a)requireValue(Math.abs(a.appliedForceN)<=10&&close(a.appliedForceN,Math.max(-10,Math.min(10,a.requestedForceN))),'force request/application');
   }
   const expected=c.reference[r.branch][0].map((v,i)=>v+p.offsets[r.offset][i]);
   requireValue(r.frames[0].state.every((v,i)=>close(v,expected[i])),'equal initial state');
   const mae=r.frames.slice(1).reduce((sum,f)=>sum+Math.abs(f.errorX),0)/28;
   requireValue(r.metrics?.steps===28&&typeof r.metrics.completed==='boolean'&&Number.isFinite(r.metrics.cartMae)&&r.metrics.cartMae>=0&&close(r.metrics.cartMae,mae),'metrics');
  }
 }
 return d;
}
export function selectLessonFrame(data,{caseId,branch,offset,tick}){
 const c=data.cases.find(c=>c.id===caseId);
 requireValue(c&&[0,1].includes(branch)&&['nominal','perturbed'].includes(offset)&&Number.isInteger(tick)&&tick>=0&&tick<=28,'selection');
 const runs=CONDITIONS.map(condition=>c.runs.find(r=>r.condition===condition&&r.branch===branch&&r.offset===offset));
 requireValue(runs.every(Boolean),'selected runs');
 return {case:c,reference:c.reference[branch],runs,frames:runs.map(r=>r.frames[tick])};
}
export function mountNativeLesson({root,onEnter,onLeave}){
 let data=null,active=false,playing=false,timer=0,last=0,elapsed=0,tick=0,step=1,caseId='',branch=0,offset='nominal';
 root.innerHTML=`<div class="lesson-top"><p class="lesson-kicker">하나의 실험 · 두 경로 · 같은 출발과 도착</p><h2>출발과 도착이 같으면, 첫 힘도 같을까요?</h2><p>native 실행 기록 재생 — 이 화면에서는 모델을 학습하거나 물리를 재계산하지 않습니다</p><nav aria-label="수업 단계"><button data-stage="1">1 예상</button><button data-stage="2">2 비교</button><button data-stage="3">3 해석</button></nav></div><p id="lessonStatus" role="status" aria-live="polite">native 실행 기록을 불러오는 중…</p><button id="lessonRetry" hidden>기록 다시 불러오기</button><div id="lessonBody" hidden><section id="lessonPrediction"><h3>현재 상태와 도착점만으로 경로 A와 B를 구별할 수 있을까요?</h3><p>두 경로는 같은 곳에서 출발해 같은 곳에 도착합니다. 그 사이의 움직임은 다릅니다.</p><div class="lesson-actions"><button data-predict="yes">구별할 수 있다</button><button data-predict="no">구별할 수 없다</button><button id="lessonSkip">예상 건너뛰기</button></div><p id="lessonFeedback" role="status" aria-live="polite"></p><button id="lessonReveal">기록 비교하기 →</button><canvas id="lessonPaths" height="190" aria-label="같은 출발과 도착을 가진 목표 경로 A와 B"></canvas></section><section id="lessonCompare" hidden><div class="lesson-selectors"><label>사례 <select id="lessonCase"></select></label><label>목표 경로 <select id="lessonBranch"><option value="0">A</option><option value="1">B</option></select></label><label>초기 상태 <select id="lessonOffset"><option value="nominal">기준 상태 (nominal)</option><option value="perturbed">작은 초기 편차 (perturbed)</option></select></label><span>별도로 기록된 실행을 선택합니다.</span></div><div class="lesson-legend"><span>┄ 검정: 목표 reference(t)</span><span style="color:#315dc9">━ 미래 구간 학습: 실제 x(t)</span><span style="color:#b05b19">━ 현재·도착점 학습: 실제 x(t)</span></div><canvas id="lessonChart" height="280" aria-label="목표와 두 학습 조건의 카트 변위 비교"></canvas><p class="lesson-axis-note">출발 위치 대비 변위 (mm) · 모든 사례와 조건에 같은 축 범위 · 선은 기록 표본을 연결하며 새 물리 표본을 만들지 않습니다.</p><div class="lesson-play"><button id="lessonPrev" aria-label="이전 표본">← 이전 표본</button><button id="lessonPlay">재생</button><button id="lessonNext" aria-label="다음 표본">다음 표본 →</button><label>기록 시각 <input id="lessonSeek" aria-label="기록 표본" type="range" min="0" max="28" step="1" value="0"></label><output id="lessonTime"></output><span>0.25× · 기록 0.56s → 표시 2.24s</span></div><table class="lesson-values"><caption>현재 상태와 다음 20ms 힘 — 동일한 시각 · 표시만 반올림</caption><thead><tr><th>학습 조건</th><th>x (m)</th><th>ẋ (m/s)</th><th>θ (rad)</th><th>θ̇ (rad/s)</th><th>적용 힘 (N)</th><th>token 성분 4개</th></tr></thead><tbody id="lessonValues"></tbody></table><p id="lessonDecision"></p><p class="lesson-readout-note">reference(t)는 목표 상태, x(t)는 기록된 실제 상태입니다. 힘은 다음 20ms에 적용됩니다. token은 힘이 아닙니다. 아래 입력 단서는 Encoder가 받은 값이며 목표 궤적과 구분합니다.</p><details><summary>원시 값·단위·재현 정보 (반올림 전)</summary><pre id="lessonCue"></pre></details><button id="lessonInterpret">선택한 사례 해석하기 →</button></section><section id="lessonInterpretation" hidden><h3>둘 다 완주하면 제어 품질도 같은가?</h3><p id="lessonStats"></p><p id="lessonDiagnosis"></p><p>완주는 종료 여부이고, MAE는 목표에서 얼마나 벗어났는지입니다. 같은 완주라도 추적 오차는 다를 수 있습니다. 현재·도착점 입력만 같으면 두 중간 경로를 구별할 단서가 없습니다. 미래 구간 입력에는 그 단서가 있지만, 그것이 모든 실행의 더 작은 오차를 보장하지는 않습니다.</p><p>현재·도착점만 받은 정책과 미래 구간을 받은 정책은 각각 별도로 학습했습니다. 선택한 목표 궤적은 두 정책에 동일합니다. 한계: 세 사례의 0.56초 직립 근처 기록으로 전체 성능을 일반화할 수 없습니다.</p></section><footer class="lesson-source"><a href="https://github.com/tinmanlab/cartpole-sonic/blob/main/native/GUIDED_LESSON.md">원시 기록 출처 · 재현 방법</a> · 같은 Decoder · 2 token × 2성분 / 성분당 32단계 · seed 52101</footer></div>`;
 const $=id=>root.querySelector('#'+id);
 const stop=()=>{playing=false;cancelAnimationFrame(timer);timer=0;$('lessonPlay').textContent='재생';};
 function stage(n){stop();step=n;root.dataset.stage=n;$('lessonPrediction').hidden=n!==1;$('lessonCompare').hidden=n===1;$('lessonInterpretation').hidden=n!==3;root.querySelectorAll('[data-stage]').forEach(b=>{b.classList.toggle('active',Number(b.dataset.stage)===n);b.setAttribute('aria-current',Number(b.dataset.stage)===n?'step':'false');});draw();}
 function plot(canvas,series,cursor,range){
  if(!canvas.clientWidth||!canvas.clientHeight)return;const w=canvas.clientWidth,h=canvas.clientHeight;const ratio=devicePixelRatio||1;canvas.width=w*ratio;canvas.height=h*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const left=65,right=w-170,top=20,bottom=h-35;const x=i=>left+i/28*(right-left),y=v=>bottom-(v+range)/(range*2)*(bottom-top);
  ctx.font='12px system-ui';ctx.fillStyle='#667085';for(let i=0;i<5;i++){const v=-range+i*range/2;ctx.strokeStyle='#e3e7ee';ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(right,y(v));ctx.stroke();ctx.fillText(v.toFixed(2)+' mm',4,y(v)+4);}ctx.textAlign='center';canvasTicks(ctx,Array.from({length:8},(_,i)=>(i*.08).toFixed(2)+'s'),Array.from({length:8},(_,i)=>x(i*4)),h-10);ctx.textAlign='left';
  for(const s of series){ctx.strokeStyle=s.color;ctx.setLineDash(s.dash?[6,4]:[]);ctx.lineWidth=2;ctx.beginPath();s.values.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=s.color;ctx.fillText(s.name,right+10,top+18+series.indexOf(s)*22);if(cursor!==null){ctx.beginPath();ctx.arc(x(cursor),y(s.values[cursor]),4,0,Math.PI*2);ctx.fill();}}
  if(cursor!==null){ctx.strokeStyle='#795fc5';ctx.beginPath();ctx.moveTo(x(cursor),top);ctx.lineTo(x(cursor),bottom);ctx.stroke();}
 }
 let range=1;
 function draw(){if(!data)return;const s=selectLessonFrame(data,{caseId,branch,offset,tick});
  if(step===1)plot($('lessonPaths'),s.case.reference.map((r,b)=>({name:'목표 경로 '+(b?'B':'A'),color:b?'#b05b19':'#315dc9',values:r.map(f=>(f[0]-r[0][0])*1000)})),null,range);
  if(step===1)return;
  plot($('lessonChart'),[{name:'목표 reference(t)',color:'#172033',dash:true,values:s.reference.map(f=>(f[0]-s.reference[0][0])*1000)},...s.runs.map((r,i)=>({name:NAMES[i],color:i?'#b05b19':'#315dc9',values:r.frames.map(f=>(f.state[0]-s.reference[0][0])*1000)}))],tick,range);
  $('lessonSeek').value=tick;$('lessonSeek').setAttribute('aria-valuetext',`${(tick*.02).toFixed(2)}초, 표본 ${tick}/28`);$('lessonTime').textContent=`${(tick*.02).toFixed(2)} s · 표본 ${tick}/28`;$('lessonPrev').disabled=tick===0;$('lessonNext').disabled=tick===28;
  $('lessonValues').innerHTML=s.frames.map((f,i)=>`<tr data-condition="${CONDITIONS[i]}"><th>${NAMES[i]}</th>${f.state.map(v=>`<td>${v.toFixed(4)}</td>`).join('')}<td>${f.decision?f.decision.appliedForceN.toFixed(3):'—'}</td><td>${f.decision?f.decision.token.join(', '):'—'}</td></tr>`).join('');
  $('lessonDecision').textContent=tick===28?'기록 종료: 새 행동 없음':`t=${(tick*.02).toFixed(2)}s 결정 → 다음 구간 [${(tick*.02).toFixed(2)}, ${((tick+1)*.02).toFixed(2)}]s 힘`;
  $('lessonCue').textContent=s.frames.map((f,i)=>NAMES[i]+': '+JSON.stringify({caseId,state:f.state,referenceNow:f.referenceNow,decision:f.decision,endpointMaxError:s.case.endpointMaxError})).join('\n');
  const maes=s.runs.map(r=>r.metrics.cartMae*1000);const nom=CONDITIONS.map(condition=>s.case.runs.find(r=>r.condition===condition&&r.branch===branch&&r.offset==='nominal').metrics.cartMae*1000);const pert=CONDITIONS.map(condition=>s.case.runs.find(r=>r.condition===condition&&r.branch===branch&&r.offset==='perturbed').metrics.cartMae*1000);const gap=maes[1]-maes[0];
  $('lessonStats').textContent=`사례 ${data.cases.indexOf(s.case)+1} · 경로 ${branch?'B':'A'} · ${offset}: 미래 구간 MAE ${maes[0].toFixed(4)} mm, 현재·도착점 MAE ${maes[1].toFixed(4)} mm (28개 다음 상태의 |x−reference| 평균). 완주: ${s.runs.map((r,i)=>NAMES[i]+' '+(r.metrics.completed?'예':'아니오')).join(' / ')}.`;
  $('lessonDiagnosis').textContent=`선택 실행에서는 ${Math.abs(gap)<1e-9?'두 MAE가 같습니다':gap>0?'미래 구간 학습의 MAE가 더 작습니다':'현재·도착점 학습의 MAE가 더 작습니다'}. 현재·도착점 MAE − 미래 구간 MAE: 기준 ${(nom[1]-nom[0]).toFixed(4)} mm → 초기 편차 ${(pert[1]-pert[0]).toFixed(4)} mm. ${pert[1]-pert[0]<nom[1]-nom[0]?'초기 편차에서 미래 구간 조건의 상대 이점이 줄어듭니다.':'이 사례에서는 초기 편차가 상대 이점을 줄이지 않습니다.'}`;
 }
 function rewind(){stop();tick=0;draw();}
 function animate(now){if(!playing||!active||document.hidden)return stop();if(!last)last=now;elapsed+=now-last;last=now;const next=Math.min(28,Math.floor(elapsed/80));if(tick!==next){tick=next;draw();}if(tick===28)return stop();timer=requestAnimationFrame(animate);}
 async function load(){stop();data=null;$('lessonBody').hidden=true;$('lessonRetry').hidden=true;$('lessonStatus').textContent='native 실행 기록을 불러오는 중…';try{const response=await fetch('./evidence/guided_lesson/traces.json',{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);data=validateLessonData(await response.json());caseId=data.cases[0].id;range=Math.max(.1,...data.cases.flatMap(c=>c.runs.flatMap(r=>r.frames.map(f=>Math.abs((f.state[0]-c.reference[r.branch][0][0])*1000)))),...data.cases.flatMap(c=>c.reference.flatMap(r=>r.map(f=>Math.abs((f[0]-r[0][0])*1000)))))*1.12;$('lessonCase').replaceChildren(...data.cases.map(c=>new Option("사례 "+(data.cases.indexOf(c)+1),c.id)));$('lessonStatus').textContent='검증된 native 기록 · 세 표시 사례 (전체 벤치마크 아님)';$('lessonBody').hidden=false;tick=0;stage(1);}catch(e){$('lessonStatus').textContent='기록을 표시할 수 없습니다. '+e.message+' · 수치 기록을 대체하지 않습니다.';$('lessonRetry').hidden=false;}}
 root.querySelectorAll('[data-stage]').forEach(b=>b.onclick=()=>{if(data)stage(Number(b.dataset.stage));});root.querySelectorAll('[data-predict]').forEach(b=>b.onclick=()=>{$('lessonFeedback').textContent=b.dataset.predict==='no'?'같은 현재 상태와 도착점만으로는 중간 경로를 구별할 수 없습니다. 이제 실제 힘과 오차를 확인해 보세요.':'현재 상태와 도착점이 같으면 입력 단서도 같습니다. 중간 경로의 차이를 구별하려면 추가 정보가 필요합니다.';});$('lessonSkip').onclick=()=>stage(2);$('lessonReveal').onclick=()=>stage(2);$('lessonInterpret').onclick=()=>stage(3);$('lessonRetry').onclick=load;
 $('lessonCase').onchange=e=>{caseId=e.target.value;rewind();};$('lessonBranch').onchange=e=>{branch=Number(e.target.value);rewind();};$('lessonOffset').onchange=e=>{offset=e.target.value;rewind();};$('lessonSeek').oninput=e=>{stop();tick=Number(e.target.value);draw();};$('lessonPrev').onclick=()=>{stop();tick=Math.max(0,tick-1);draw();};$('lessonNext').onclick=()=>{stop();tick=Math.min(28,tick+1);draw();};$('lessonPlay').onclick=()=>{if(playing)return stop();if(!data||!active||document.hidden)return;if(tick===28)tick=0;playing=true;last=0;elapsed=tick*80;$('lessonPlay').textContent='일시정지';timer=requestAnimationFrame(animate);};
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);window.addEventListener('resize',()=>{if(active)draw();});
 return {get active(){return active;},snapshot:()=>({activeSurface:active?'native-lesson':'exploration',lesson:{caseId,branch,offset,tick,playing,stage:step,loaded:Boolean(data),recorded:active&&data?structuredClone({source:'native-recorded-reexecution',time:tick*.02,conditions:CONDITIONS,frames:selectLessonFrame(data,{caseId,branch,offset,tick}).frames}):null}}),enter(){onEnter();active=true;root.hidden=false;document.body.classList.add('native-lesson-active');if(data)draw();else load();},leave(){stop();active=false;root.hidden=true;document.body.classList.remove('native-lesson-active');onLeave();}};
}
