// Evidence-only view: the dots are measured PPO runs, not simulated browser training.
let budget=50,view='budget',variantId='two-default';
export const optimizerViewState=()=>({budget,view,variantId});
const $=id=>document.getElementById(id);
const number=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
const scientific=v=>Number.isFinite(v)?v.toExponential(2):'—';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const meanSd=s=>`${number(s.mean)} ± ${number(s.sampleStd)}`;
const armLabel=id=>`${id.startsWith('one')?'1 token':'2 tokens'} · actor ×${id.endsWith('005')?'0.05':'1'}`;

function drawRuns(evidence,beginCanvas){
  const canvas=$('optimizerRuns');if(!canvas)return;
  const {ctx,w,h}=beginCanvas(canvas),study=evidence.pairedSeedStudy;
  const rows=study.summary.filter(r=>r.ppoIterations===budget);
  const values=study.runs.flatMap(r=>r.arms.map(a=>a.checkpoints.find(c=>c.ppoIterations===budget).cleanMae)).filter(Number.isFinite);
  const max=Math.max(.1,...values,...rows.map(r=>(r.cleanMae.mean??0)+(r.cleanMae.sampleStd??0)))*1.12;
  const pad={l:150,r:22,t:27,b:26},X=v=>pad.l+v/max*(w-pad.l-pad.r),rowH=(h-pad.t-pad.b)/4;
  ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
  ctx.font='11px system-ui';ctx.fillStyle='#475467';ctx.textAlign='left';ctx.fillText('Clean tracking MAE [m] ↓ · dots = individual PPO runs',8,14);
  for(let i=0;i<=4;i++){
    const v=max*i/4,x=X(v);ctx.strokeStyle='#eaecf0';ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,h-pad.b);ctx.stroke();
    ctx.fillStyle='#667085';ctx.font='10px ui-monospace';ctx.textAlign='center';ctx.fillText(v.toFixed(2),x,h-8);
  }
  rows.forEach((row,i)=>{
    const y=pad.t+(i+.5)*rowH,color=row.id.startsWith('one')?'#ad6819':'#315dc9';
    ctx.fillStyle='#344054';ctx.font='11px system-ui';ctx.textAlign='right';ctx.fillText(armLabel(row.id),pad.l-9,y+4);
    const s=row.cleanMae;
    if(Number.isFinite(s.mean)){
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(X(Math.max(0,s.mean-(s.sampleStd??0))),y);ctx.lineTo(X(s.mean+(s.sampleStd??0)),y);ctx.stroke();
      ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(X(s.mean),y-7);ctx.lineTo(X(s.mean),y+7);ctx.stroke();
    }
    study.runs.forEach((run,j)=>{
      const value=run.arms.find(a=>a.id===row.id).checkpoints.find(c=>c.ppoIterations===budget).cleanMae;
      if(!Number.isFinite(value))return;
      ctx.strokeStyle=color;ctx.fillStyle='#fff';ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(X(value),y+(j-1)*5,3,0,Math.PI*2);ctx.fill();ctx.stroke();
    });
  });
}

export function renderOptimizerEvidencePanel({evidence,showHtml,beginCanvas,rerender}){
  if(!evidence?.pairedSeedStudy){showHtml('<div class="optimizer-audit">Updated optimizer evidence is loading…</div>','Evidence is fetched from the repository; no synthetic results are substituted.');return;}
  const ev=evidence,study=ev.pairedSeedStudy;
  const nav=`<div class="opt-nav"><button id="optBudget" class="${view==='budget'?'primary':''}">같은 예산으로 비교</button><button id="optDiagnostics" class="${view==='diagnostics'?'primary':''}">원인 계측 보기</button><span>stored evidence · v${ev.revision}</span></div>`;
  let content;
  if(view==='budget'){
    const rows=study.summary.filter(r=>r.ppoIterations===budget);
    const table=rows.map(r=>`<tr><th scope="row">${armLabel(r.id)}</th><td>${meanSd(r.cleanMae)}</td><td>${meanSd(r.pushMae)}</td><td>${r.cleanSuccesses}/36 · ${r.pushSuccesses}/36</td></tr>`).join('');
    const raw=study.runs.flatMap(run=>run.arms.map(a=>{const c=a.checkpoints.find(p=>p.ppoIterations===budget);return `<tr><td>${run.seed}</td><th scope="row">${armLabel(a.id)}</th><td>${number(c.cleanMae)}</td><td>${number(c.pushMae)}</td><td>${c.cleanSuccesses}/12 · ${c.pushSuccesses}/12</td></tr>`;})).join('');
    content=`<div class="opt-budget-row"><b>학습 횟수를 맞추면 결론이 유지되는가?</b><div><button id="opt10" class="${budget===10?'primary':''}">+10 PPO</button><button id="opt50" class="${budget===50?'primary':''}">+50 PPO</button></div></div>
      <div class="opt-run-chart"><canvas id="optimizerRuns" role="img" aria-label="Three individual PPO runs per condition; mean and one sample standard deviation. Exact values appear in the following table."></canvas></div>
      <table class="opt-data"><caption>${budget} iterations · ${budget*768} environment steps per run · mean ± sample SD [m]</caption><thead><tr><th>조건</th><th>Clean MAE ↓</th><th>Push MAE ↓</th><th>생존 clean · push</th></tr></thead><tbody>${table}</tbody></table>
      <p class="opt-explain"><b>점 3개 = PPO 난수 seed 3개.</b> 선은 평균 ± 표본 표준편차이며 신뢰구간이 아닙니다. 각 구조의 초기 bootstrap은 고정했습니다. 새로운 초기 모델 3개를 학습한 실험이 아닙니다.</p>
      <details class="opt-details"><summary>개별 seed 결과와 해석 한계</summary><table class="opt-data"><thead><tr><th>Seed</th><th>조건</th><th>Clean</th><th>Push</th><th>생존</th></tr></thead><tbody>${raw}</tbody></table><p>Clean MAE는 완주한 episode에 대한 값입니다. 반드시 생존율과 함께 읽으세요. 각 조건은 같은 evaluation episodes를 쓰지만, 행동이 달라지면 이후 물리 상태도 달라집니다. token 수만의 인과 효과나 SONIC의 최적 설정은 입증하지 않습니다.</p></details>`;
  }else{
    const v=ev.variants.find(x=>x.id===variantId)||ev.variants[0],d=v.diagnostics,p=v.parameters;
    const one=ev.variants.find(x=>x.id==='one-default').parameters;
    const opts=ev.variants.map(x=>`<option value="${esc(x.id)}" ${x.id===v.id?'selected':''}>${esc(x.label)}</option>`).join('');
    const modules=['encoder','dynamic','kinematic','critic'].map(k=>{const u=d.updates[k];return `<tr><th scope="row">${esc(k)}</th><td>${scientific(u.preClipNorm)}</td><td>${scientific(u.postClipNorm)}</td><td>${scientific(u.updateNorm)}</td><td>${scientific(u.relativeUpdateNorm)}</td></tr>`;}).join('');
    content=`<label class="opt-select">+10 PPO 조건 <select id="optVariant">${opts}</select></label>
      <div class="opt-cards"><div><span>전체 / actor 파라미터</span><b>${p.total} / ${p.actor}</b><small>1-token 기준: ${one.total} / ${one.actor}</small></div><div><span>실제 Dynamic Decoder 변화량</span><b>${scientific(d.updates.dynamic.updateNorm)}</b><small>Adam 전후 가중치 차이의 L2 크기</small></div><div><span>정책 분포 변화 · KL</span><b>${scientific(d.postUpdateKl)}</b><small>동일한 상태·reference에서 전후 비교</small></div><div><span>Token이 바뀐 reference</span><b>${number(100*d.tokenChangeFraction,1)}%</b><small>한 scalar 이상 바뀐 표본의 비율</small></div></div>
      <div class="opt-process">gradient → norm clipping → Adam moments → 실제 Δweights → 정책 변화</div>
      <table class="opt-data"><caption>마지막 PPO iteration · minibatch별 측정값의 평균</caption><thead><tr><th>Module</th><th>Raw ∥g∥</th><th>Clipped ∥g∥</th><th>∥Δweights∥</th><th>상대 변화량</th></tr></thead><tbody>${modules}</tbody></table>
      <p class="opt-explain"><b>∥gradient∥ × learning rate ≠ Adam의 실제 이동량.</b> clipping과 누적 moment가 중간에 작용합니다. KL은 motor 입력을 자르기 전 Gaussian 분포의 변화이며, token 변화율은 같은 입력으로 측정합니다.</p>
      <details class="opt-details"><summary>확률 계산·평가·파라미터 비교의 주의점</summary><p>Gaussian 원본 sample은 PPO likelihood에 보존하고, [-1,1] 제한은 물리 입력에만 적용합니다. 실제 sample clipping은 전체 ${v.steps}회 PPO에서 평균 ${number(100*d.meanSampleClipAcrossIterations,3)}%였습니다. 평균 action의 saturation과는 다른 값입니다.</p><p>이 조건의 ratio 범위 이탈률은 ${number(100*d.clipFraction,1)}%, advantage 방향까지 고려한 objective clipping은 ${number(100*d.objectiveClipFraction,1)}%입니다. 두 정의를 분리했습니다.</p><p>총 파라미터만 맞춘 2-token은 total 1220 / actor 643입니다. actor를 맞춘 조건은 total 1268 / actor 607입니다. 둘 모두 layer width가 달라지므로 기능적 표현력을 완전히 맞췄다는 뜻은 아닙니다. 외란 이전에 실패한 episode의 post-push 오차는 0이 아니라 미관측(null)입니다.</p></details>`;
  }
  showHtml(`<div class="optimizer-audit">${nav}${content}</div>`,
    '<b>EVIDENCE · matched-budget PPO audit.</b> 1·2-token 모두 +10/+50을 비교합니다. 실제 Adam 변화량·KL·token 변화를 계측하며, 관측된 결과보다 강한 원인 결론을 내리지 않습니다.',
    ['3 PPO seeds · fixed bootstrap','10 / 50 matched iterations','raw Gaussian likelihood','measured Adam displacement']);
  if(!$('optBudget'))return;
  requestAnimationFrame(()=>{if(view==='budget'&&$('optimizerRuns'))drawRuns(ev,beginCanvas);});
    $('optBudget').onclick=()=>{view='budget';rerender();};
    $('optDiagnostics').onclick=()=>{view='diagnostics';rerender();};
    for(const n of [10,50])if($('opt'+n))$('opt'+n).onclick=()=>{budget=n;rerender();};
    if($('optVariant'))$('optVariant').onchange=e=>{variantId=e.target.value;rerender();};
}
