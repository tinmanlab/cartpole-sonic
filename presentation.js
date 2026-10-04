// Presentation only: these values never feed the controller or physics.
export function controlSample(t,state,referenceNow,reference,p,lastAppliedForce,referenceScale=1){
  return {t,targetTime:t,referenceNow,previewTargetTime:t+.08,referencePreview:reference[0]*referenceScale,state:Array.from(state),reference:Array.from(reference),z:Array.from(p.z),q:Array.from(p.q),force:p.force,plannedForce:p.force,lastAppliedForce};
}
export function controllerIdentity(p,mode,temporalPolicy=null){
  const activeMode=temporalPolicy?'fsq':mode,dim=p?.q?.length??0;
  return {source:'LIVE browser calculation',mode:activeMode,tokens:activeMode==='ae'?0:temporalPolicy?.tokens??1,scalarDimsPerToken:activeMode==='ae'?0:temporalPolicy?.tokenDim??dim,flattenedDim:dim};
}
export function controllerSummary(identity){
  if(!identity.flattenedDim)return '제어기 준비 중';
  if(identity.mode==='ae')return 'AE · 연속 latent '+identity.flattenedDim+'D · 양자화 없음';
  return identity.mode.toUpperCase()+' · '+identity.tokens+' token × '+identity.scalarDimsPerToken+' scalars ('+identity.flattenedDim+'D)';
}
export function blockShape(id,identity,fallback){
  if(!identity.flattenedDim)return fallback;
  const dim=identity.flattenedDim;
  if(id==='encoder')return '16D reference → '+dim+'D latent';
  if(id==='quantizer')return identity.mode==='ae'?'현재 경로: 양자화 없음':identity.mode.toUpperCase()+' · '+dim+' scalar values';
  if(id==='token')return identity.mode==='ae'?'현재 경로: 연속 latent '+dim+'D':identity.tokens+' token × '+identity.scalarDimsPerToken+' scalars';
  if(id==='control-decoder')return dim+'D '+(identity.mode==='ae'?'latent':'token')+' + state4 → force';
  return fallback;
}
export function decoderLayout(w){
  const margin=16,gap=24,width=(w-2*margin-gap)/2;
  return Array.from({length:4},(_,i)=>({x:margin+(i%2)*(width+gap),y:24+Math.floor(i/2)*94,w:width,h:72}));
}
export function robotGeometry(w,h,length,x,theta){
  const scale=Math.min(118,(w-32)/(2*(1.8+Math.sin(.75)*length+.32)),(h-42)/(length+.58));
  const railY=h-14,pivotY=railY-.46*scale,cx=w/2+x*scale;
  return {scale,railY,pivotY,cx,tx:cx+Math.sin(theta)*length*scale,ty:pivotY-Math.cos(theta)*length*scale,cartW:.64*scale,cartH:.24*scale,wheelY:railY-.09*scale};
}

export function decoderConnections(w){
  const [token,state,decoder,force]=decoderLayout(w);
  const tx=token.x+token.w/2,sx=state.x+state.w/2,dy=decoder.y-11;
  return [
    {points:[[tx,token.y+token.h],[tx,dy],[tx,decoder.y]],arrow:'down'},
    {points:[[sx,state.y+state.h],[sx,dy],[tx,dy]],arrow:null},
    {points:[[decoder.x+decoder.w,decoder.y+decoder.h/2],[force.x,force.y+force.h/2]],arrow:'right'}
  ];
}

// Keep captions inside their own region without shrinking the font.
export function canvasLines(ctx,text,x,y,width,lineHeight=17){
  let line='',row=0;
  for(const char of String(text)){
    if(line&&ctx.measureText(line+char).width>width){ctx.fillText(line,x,y+row++*lineHeight);line='';}
    line+=char;
  }
  if(line)ctx.fillText(line,x,y+row*lineHeight);
  return row+1;
}
export function canvasTicks(ctx,labels,positions,y,gap=8){
  let edge=-Infinity;
  labels.forEach((label,i)=>{const width=ctx.measureText(label).width,left=positions[i]-width/2;if(left>=edge+gap){ctx.fillText(label,positions[i],y);edge=left+width;}});
}
