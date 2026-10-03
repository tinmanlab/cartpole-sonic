import { MuJoCoCartPole } from "./mujoco_sim.js";

const REF_FRAMES = 8;
const REF_DIM = REF_FRAMES * 2;
const STATE_SCALE = [1.8, 3.0, 0.55, 4.0];
const ACTION_FORCE = 10.0;
const FIXED_STD = 0.12;
const LOG_SQRT_2PI = 0.9189385332046727;

function clamp(x,a,b){ return Math.max(a,Math.min(b,x)); }
function mean(a){ return a.length ? a.reduce((s,v)=>s+v,0)/a.length : 0; }

export class RNG {
  constructor(seed=123){this.s=seed>>>0;this.spare=null;}
  uniform(){let t=this.s=(this.s+0x6D2B79F5)>>>0;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;}
  normal(){if(this.spare!==null){const v=this.spare;this.spare=null;return v;}let x,y,q;do{x=2*this.uniform()-1;y=2*this.uniform()-1;q=x*x+y*y;}while(q===0||q>=1);const k=Math.sqrt(-2*Math.log(q)/q);this.spare=y*k;return x*k;}
}

export class MLP {
  constructor(rng,n,h,o,{outTanh=false,outScale=0.1}={}){
    this.n=n;this.h=h;this.o=o;this.outTanh=outTanh;
    this.w1=0;this.b1=h*n;this.w2=this.b1+h;this.b2=this.w2+o*h;
    this.p=new Float64Array(this.b2+o);this.m=new Float64Array(this.p.length);this.v=new Float64Array(this.p.length);this.t=0;
    for(let i=0;i<h*n;i++)this.p[i]=rng.normal()*Math.sqrt(2/(n+h));
    for(let i=this.w2;i<this.b2;i++)this.p[i]=rng.normal()*outScale;
  }
  forward(x){
    const h=new Float64Array(this.h),preOut=new Float64Array(this.o),y=new Float64Array(this.o);
    for(let i=0;i<this.h;i++){let z=this.p[this.b1+i];for(let j=0;j<this.n;j++)z+=this.p[i*this.n+j]*x[j];h[i]=Math.tanh(z);}
    for(let i=0;i<this.o;i++){let z=this.p[this.b2+i];for(let j=0;j<this.h;j++)z+=this.p[this.w2+i*this.h+j]*h[j];preOut[i]=z;y[i]=this.outTanh?Math.tanh(z):z;}
    return{x:Array.from(x),h,y,preOut};
  }
  backward(f,dy,G,scale=1){
    const dOut=new Float64Array(this.o),dh=new Float64Array(this.h),dx=new Float64Array(this.n);
    for(let i=0;i<this.o;i++)dOut[i]=dy[i]*(this.outTanh?(1-f.y[i]*f.y[i]):1);
    for(let i=0;i<this.o;i++){
      G[this.b2+i]+=dOut[i]*scale;
      for(let j=0;j<this.h;j++)G[this.w2+i*this.h+j]+=dOut[i]*f.h[j]*scale;
    }
    for(let j=0;j<this.h;j++){
      let d=0;for(let i=0;i<this.o;i++)d+=this.p[this.w2+i*this.h+j]*dOut[i];
      const dz=d*(1-f.h[j]*f.h[j]);dh[j]=dz;G[this.b1+j]+=dz*scale;
      for(let k=0;k<this.n;k++){G[j*this.n+k]+=dz*f.x[k]*scale;dx[k]+=this.p[j*this.n+k]*dz;}
    }
    return dx;
  }
  adam(G,lr,maxNorm=1.0){
    const norm=Math.sqrt(G.reduce((s,v)=>s+v*v,0)),scale=Math.min(1,maxNorm/(norm+1e-12));this.t++;
    const bc1=1-.9**this.t,bc2=1-.999**this.t;
    for(let i=0;i<this.p.length;i++){const g=G[i]*scale;this.m[i]=.9*this.m[i]+.1*g;this.v[i]=.999*this.v[i]+.001*g*g;const mh=this.m[i]/bc1,vh=this.v[i]/bc2;this.p[i]-=lr*mh/(Math.sqrt(vh)+1e-8);}
    return norm;
  }
  snapshot(){return{n:this.n,h:this.h,o:this.o,outTanh:this.outTanh,p:Array.from(this.p),m:Array.from(this.m),v:Array.from(this.v),t:this.t};}
}

function fsqScalar(z){
  const bound=Math.tanh(z)*1.998;
  return Math.round(bound)/2;
}
function fsqDeriv(z){const t=Math.tanh(z);return .999*(1-t*t);}
function logProbGaussian(a,mu,std=FIXED_STD){
  const z=(a-mu)/std;return -.5*z*z-Math.log(std)-LOG_SQRT_2PI;
}

export function planReference(context,goal,{frames=REF_FRAMES,frameDt=.08,k=2.5}={}){
  const [x,xd]=context;
  const out=new Float64Array(frames*2);
  const e=x-goal,B=xd+k*e;
  for(let i=0;i<frames;i++){
    const t=(i+1)*frameDt,exp=Math.exp(-k*t);
    const xr=goal+(e+B*t)*exp;
    const vr=(xd-k*B*t)*exp;
    out[i*2]=clamp(xr/STATE_SCALE[0],-1,1);
    out[i*2+1]=clamp(vr/STATE_SCALE[1],-1,1);
  }
  return out;
}

export function advancePlannerContext(context,goal,dt=.02,k=2.5){
  const [x,xd]=context,e=x-goal,B=xd+k*e,exp=Math.exp(-k*dt);
  return [goal+(e+B*dt)*exp,(xd-k*B*dt)*exp];
}

function plannerPhysical(ref){
  const frames=[];for(let i=0;i<REF_FRAMES;i++)frames.push({x:ref[i*2]*STATE_SCALE[0],xd:ref[i*2+1]*STATE_SCALE[1],t:(i+1)*.08});
  return frames;
}

function normalizeState(s){return s.map((v,i)=>clamp(v/STATE_SCALE[i],-1.5,1.5));}

export class SonicCartPolePolicy {
  constructor(seed=123,mode="fsq"){
    this.rng=new RNG(seed);this.mode=mode;
    this.encoder=new MLP(this.rng,REF_DIM,16,2,{outScale:.22});
    this.dynamicDecoder=new MLP(this.rng,6,24,1,{outTanh:true,outScale:.08});
    this.kinematicDecoder=new MLP(this.rng,2,20,REF_DIM,{outTanh:true,outScale:.08});
    this.critic=new MLP(this.rng,5,20,1,{outScale:.15});
    this.codebook=new Float64Array([-0.75,-0.75,-0.25,-0.75,0.25,-0.75,0.75,-0.75,-0.75,0.25,-0.25,0.25,0.25,0.25,0.75,0.25]);
  }
  quantize(z,mode=this.mode){
    if(mode==="ae")return{q:[z[0],z[1]],index:null,deriv:[1,1]};
    if(mode==="fsq")return{q:[fsqScalar(z[0]),fsqScalar(z[1])],index:null,deriv:[fsqDeriv(z[0]),fsqDeriv(z[1])]};
    let best=0,bd=Infinity;for(let k=0;k<8;k++){const dx=z[0]-this.codebook[k*2],dy=z[1]-this.codebook[k*2+1],d=dx*dx+dy*dy;if(d<bd){bd=d;best=k;}}
    return{q:[this.codebook[best*2],this.codebook[best*2+1]],index:best,deriv:[1,1]};
  }
  forward(state,ref,goal,{sample=false,rng=this.rng}={}){
    const ef=this.encoder.forward(ref),z=Array.from(ef.y),quant=this.quantize(z),sNorm=normalizeState(state);
    const dynIn=[quant.q[0],quant.q[1],...sNorm],df=this.dynamicDecoder.forward(dynIn),mu=clamp(df.y[0],-.999,.999);
    let action=mu;if(sample)action=clamp(mu+rng.normal()*FIXED_STD,-1,1);
    const kf=this.kinematicDecoder.forward(quant.q),cf=this.critic.forward([...sNorm,goal/STATE_SCALE[0]]);
    return{ref,z,q:quant.q,qIndex:quant.index,qDeriv:quant.deriv,mu,action,force:action*ACTION_FORCE,value:cf.y[0],kinRecon:Array.from(kf.y),cache:{ef,df,kf,cf,sNorm}};
  }
  codeUsage(refs){
    if(this.mode!=="vq")return null;const c=new Array(8).fill(0);for(const ref of refs){const z=Array.from(this.encoder.forward(ref).y),q=this.quantize(z);c[q.index]++;}return c;
  }
}

function rewardFor(nextState,goal,action){
  const [x,xd,th,thd]=nextState;
  return 1
    - 0.65*Math.min((x-goal)*(x-goal),4)
    - 8.0*th*th
    - .020*xd*xd
    - .020*thd*thd
    - .002*action*action;
}

function doneFor(s){return Math.abs(s[0])>1.78||Math.abs(s[2])>.65||!s.every(Number.isFinite);}

class TrainEnv {
  constructor(sim,rng){
    this.sim=sim;this.rng=rng;this.data=sim.makeData();this.goal=0;this.steps=0;this.reset();
  }
  reset(){
    this.data.qpos[0]=(this.rng.uniform()*2-1)*.12;this.data.qpos[1]=(this.rng.uniform()*2-1)*.08;
    this.data.qvel[0]=(this.rng.uniform()*2-1)*.08;this.data.qvel[1]=(this.rng.uniform()*2-1)*.08;this.data.ctrl[0]=0;this.data.time=0;
    this.sim.mujoco.mj_forward(this.sim.model,this.data);this.goal=(this.rng.uniform()*2-1)*0.8;this.refContext=[0,0];this.steps=0;return this.sim.getState(this.data);
  }
  state(){return this.sim.getState(this.data);}
  reference(){return planReference(this.refContext,this.goal);}
  step(action){
    this.data.ctrl[0]=clamp(action,-1,1);
    // 50 Hz policy/control rate over native MuJoCo 10 ms physics.
    this.sim.mujoco.mj_step(this.sim.model,this.data);
    this.sim.mujoco.mj_step(this.sim.model,this.data);
    this.steps++;
    this.refContext=advancePlannerContext(this.refContext,this.goal,.02);
    const s=this.state(),targetX=this.refContext[0],done=doneFor(s)||this.steps>=500,r=rewardFor(s,targetX,action);
    return{s,r,targetX,done,terminated:done&&this.steps<500};
  }
  delete(){this.data.delete();}
}

function gae(data,n,gamma=.99,lambda=.95){
  const carry=new Float64Array(n);
  for(let i=data.length-1;i>=0;i--){const q=data[i];q.delta=q.r+gamma*(q.terminated?0:q.nextV)-q.oldV;q.rawAdv=q.delta+gamma*lambda*(q.done?0:carry[q.env]);q.ret=q.rawAdv+q.oldV;carry[q.env]=q.rawAdv;}
  const m=mean(data.map(q=>q.rawAdv)),sd=Math.sqrt(mean(data.map(q=>(q.rawAdv-m)**2))+1e-8);for(const q of data)q.adv=(q.rawAdv-m)/(sd+1e-8);
}

export class SonicToyTrainer {
  constructor(sim,{seed=20261003,mode="fsq",n=8,horizon=96,epochs=4,batch=128}={}){
    this.sim=sim;this.rng=new RNG(seed);this.policy=new SonicCartPolePolicy(seed,mode);this.mode=mode;this.n=n;this.horizon=horizon;this.epochs=epochs;this.batch=batch;
    this.envs=Array.from({length:n},()=>new TrainEnv(sim,this.rng));this.iter=0;this.envSteps=0;this.episodes=0;this.history=[];this.last={reward:NaN,tracking:NaN,piLoss:NaN,valueLoss:NaN,auxLoss:NaN,entropy:Math.log(FIXED_STD*Math.sqrt(2*Math.PI*Math.E)),clipFraction:NaN};
  }
  setMode(mode){this.mode=mode;this.policy.mode=mode;}
  restorePolicy(snapshot,{teacherBootstrap=null,rngState=null}={}){
    if(!snapshot)throw new Error("policy snapshot required");
    const load=(net,src,name)=>{
      if(!src?.p||src.p.length!==net.p.length)throw new Error("invalid "+name+" checkpoint");
      net.p.set(src.p);
      if(src.m?.length===net.m.length)net.m.set(src.m);else net.m.fill(0);
      if(src.v?.length===net.v.length)net.v.set(src.v);else net.v.fill(0);
      net.t=Number.isFinite(src.t)?src.t:0;
    };
    load(this.policy.encoder,snapshot.encoder,"encoder");
    load(this.policy.dynamicDecoder,snapshot.dynamicDecoder,"dynamicDecoder");
    load(this.policy.kinematicDecoder,snapshot.kinematicDecoder,"kinematicDecoder");
    load(this.policy.critic,snapshot.critic,"critic");
    if(snapshot.codebook){
      if(snapshot.codebook.length!==this.policy.codebook.length)throw new Error("invalid codebook checkpoint");
      this.policy.codebook.set(snapshot.codebook);
    }
    if(rngState){
      this.rng.s=(rngState.s>>>0);
      this.rng.spare=Number.isFinite(rngState.spare)?rngState.spare:null;
    }
    this.teacherBootstrap=teacherBootstrap;
    return this;
  }
  collect(){
    const data=[];let rewardSum=0,trackSum=0;
    for(let t=0;t<this.horizon;t++){
      for(let i=0;i<this.n;i++){
        const e=this.envs[i],s=e.state(),goal=e.goal,ref=e.reference(),out=this.policy.forward(s,ref,goal,{sample:true,rng:this.rng}),oldLogp=logProbGaussian(out.action,out.mu),oldV=out.value;
        const step=e.step(out.action),nextRef=e.reference(),next=this.policy.forward(step.s,nextRef,goal,{sample:false}),nextV=next.value;
        const q={env:i,state:s.slice(),goal,ref:Array.from(out.ref),action:out.action,oldMu:out.mu,oldLogp,oldV,r:step.r,nextV,done:step.done,terminated:step.terminated};
        data.push(q);rewardSum+=step.r;trackSum+=Math.abs(step.s[0]-e.refContext[0]);this.envSteps++;
        if(step.done){this.episodes++;e.reset();}
      }
    }
    gae(data,this.n);return{data,rewardMean:rewardSum/data.length,trackingMae:trackSum/data.length};
  }
  optimize(data){
    const ids=Array.from({length:data.length},(_,i)=>i);let piSum=0,vSum=0,auxSum=0,count=0,clipped=0;
    for(let ep=0;ep<this.epochs;ep++){
      for(let i=ids.length-1;i>0;i--){const j=Math.floor(this.rng.uniform()*(i+1));[ids[i],ids[j]]=[ids[j],ids[i]];}
      for(let st=0;st<ids.length;st+=this.batch){
        const end=Math.min(st+this.batch,ids.length),bs=end-st;
        const GE=new Float64Array(this.policy.encoder.p.length),GD=new Float64Array(this.policy.dynamicDecoder.p.length),GK=new Float64Array(this.policy.kinematicDecoder.p.length),GC=new Float64Array(this.policy.critic.p.length);
        const cbSums=Array.from({length:8},()=>[0,0]),cbCount=new Uint32Array(8);
        for(let ii=st;ii<end;ii++){
          const qd=data[ids[ii]],out=this.policy.forward(qd.state,qd.ref,qd.goal,{sample:false}),ratio=Math.exp(logProbGaussian(qd.action,out.mu)-qd.oldLogp);
          const active=qd.adv>=0?ratio<=1.2:ratio>=.8;if(!active)clipped++;
          const dlogp=active?-qd.adv*ratio:0, dmu=dlogp*(qd.action-out.mu)/(FIXED_STD*FIXED_STD);
          const dqPolicy=this.policy.dynamicDecoder.backward(out.cache.df,[dmu],GD,1/bs).slice(0,2);
          const auxDiff=out.kinRecon.map((v,j)=>v-qd.ref[j]),auxLoss=auxDiff.reduce((s,v)=>s+v*v,0)/REF_DIM;
          const dqAux=this.policy.kinematicDecoder.backward(out.cache.kf,auxDiff.map(v=>.20*2*v/REF_DIM),GK,1/bs);
          const dz=[(dqPolicy[0]+dqAux[0])*out.qDeriv[0],(dqPolicy[1]+dqAux[1])*out.qDeriv[1]];
          if(this.mode==="vq"){dz[0]+=.12*(out.z[0]-out.q[0]);dz[1]+=.12*(out.z[1]-out.q[1]);if(out.qIndex!=null){cbSums[out.qIndex][0]+=out.z[0];cbSums[out.qIndex][1]+=out.z[1];cbCount[out.qIndex]++;}}
          this.policy.encoder.backward(out.cache.ef,dz,GE,1/bs);
          const criticIn=[...normalizeState(qd.state),qd.goal/STATE_SCALE[0]],cf=this.policy.critic.forward(criticIn),dv=cf.y[0]-qd.ret;this.policy.critic.backward(cf,[dv],GC,1/bs);
          piSum+=-Math.min(ratio*qd.adv,clamp(ratio,.8,1.2)*qd.adv);vSum+=.5*dv*dv;auxSum+=auxLoss;count++;
        }
        this.policy.encoder.adam(GE,5e-4,.7);this.policy.dynamicDecoder.adam(GD,8e-4,.7);this.policy.kinematicDecoder.adam(GK,8e-4,.7);this.policy.critic.adam(GC,1.5e-3,1.0);
        if(this.mode==="vq"){for(let k=0;k<8;k++)if(cbCount[k]){for(let d=0;d<2;d++){const m=cbSums[k][d]/cbCount[k];this.policy.codebook[k*2+d]+=.08*(m-this.policy.codebook[k*2+d]);}}}
      }
    }
    return{piLoss:piSum/count,valueLoss:vSum/count,auxLoss:auxSum/count,clipFraction:clipped/count};
  }
  bootstrapFromTeacher(teacher,{steps=160,batch=128,lr=8e-4,auxCoef=.12}={}){
    if(!teacher)throw new Error("teacher required");
    const t0=performance.now();
    let actionLossSum=0,auxLossSum=0,count=0;
    for(let step=0;step<steps;step++){
      const GE=new Float64Array(this.policy.encoder.p.length),GD=new Float64Array(this.policy.dynamicDecoder.p.length),GK=new Float64Array(this.policy.kinematicDecoder.p.length);
      const cbSums=Array.from({length:8},()=>[0,0]),cbCount=new Uint32Array(8);
      for(let b=0;b<batch;b++){
        const goal=(this.rng.uniform()*2-1)*.8;
        const refContext=[(this.rng.uniform()*2-1)*.6,(this.rng.uniform()*2-1)*.5];
        const ref=planReference(refContext,goal);
        const state=[
          (this.rng.uniform()*2-1)*.8,
          (this.rng.uniform()*2-1)*1.0,
          (this.rng.uniform()*2-1)*.22,
          (this.rng.uniform()*2-1)*1.8
        ];
        const out=this.policy.forward(state,ref,goal,{sample:false});
        const target=teacher.forward(state,goal).normalizedForce;
        const err=out.mu-target;
        actionLossSum+=err*err;
        const dmu=2*err;
        const dqPolicy=this.policy.dynamicDecoder.backward(out.cache.df,[dmu],GD,1/batch).slice(0,2);

        const auxDiff=out.kinRecon.map((v,j)=>v-ref[j]);
        const auxLoss=auxDiff.reduce((ss,v)=>ss+v*v,0)/REF_DIM;
        auxLossSum+=auxLoss;
        const dqAux=this.policy.kinematicDecoder.backward(out.cache.kf,auxDiff.map(v=>auxCoef*2*v/REF_DIM),GK,1/batch);

        const dz=[(dqPolicy[0]+dqAux[0])*out.qDeriv[0],(dqPolicy[1]+dqAux[1])*out.qDeriv[1]];
        if(this.mode==="vq"){
          dz[0]+=.12*(out.z[0]-out.q[0]);dz[1]+=.12*(out.z[1]-out.q[1]);
          if(out.qIndex!=null){cbSums[out.qIndex][0]+=out.z[0];cbSums[out.qIndex][1]+=out.z[1];cbCount[out.qIndex]++;}
        }
        this.policy.encoder.backward(out.cache.ef,dz,GE,1/batch);
        count++;
      }
      this.policy.encoder.adam(GE,lr,.8);
      this.policy.dynamicDecoder.adam(GD,lr,.8);
      this.policy.kinematicDecoder.adam(GK,lr,.8);
      if(this.mode==="vq"){
        for(let k=0;k<8;k++)if(cbCount[k])for(let d=0;d<2;d++){const m=cbSums[k][d]/cbCount[k];this.policy.codebook[k*2+d]+=.08*(m-this.policy.codebook[k*2+d]);}
      }
    }
    this.teacherBootstrap={steps,batch,actionMse:actionLossSum/count,auxMse:auxLossSum/count,ms:performance.now()-t0,teacher:teacher.metadata?.()||null};
    return this.teacherBootstrap;
  }

  iteration(){
    const t0=performance.now(),col=this.collect(),opt=this.optimize(col.data);this.iter++;
    this.last={reward:col.rewardMean,tracking:col.trackingMae,...opt,entropy:Math.log(FIXED_STD*Math.sqrt(2*Math.PI*Math.E)),ms:performance.now()-t0,envSteps:this.envSteps,episodes:this.episodes};
    this.history.push({iter:this.iter,...this.last});if(this.history.length>160)this.history.shift();return this.last;
  }
  preview(state,ref,goal){return this.policy.forward(state,ref,goal,{sample:false});}
  evaluate(episodes=6,seed=8080){
    const rng=new RNG(seed);let successes=0,steps=[],allErrs=[],completedErrs=[];
    for(let ep=0;ep<episodes;ep++){
      const e=new TrainEnv(this.sim,rng);let err=0,n=0;
      while(e.steps<500){const s=e.state(),ref=e.reference(),o=this.policy.forward(s,ref,e.goal,{sample:false}),r=e.step(o.mu);err+=Math.abs(r.s[0]-r.targetX);n++;if(r.done)break;}
      const epMae=err/Math.max(1,n),completed=e.steps>=500&&!doneFor(e.state());
      if(completed){successes++;completedErrs.push(epMae);}
      steps.push(e.steps);allErrs.push(epMae);e.delete();
    }
    return{episodes,successes,meanSteps:mean(steps),trackingMae:completedErrs.length?mean(completedErrs):null,allStepTrackingMae:mean(allErrs)};
  }
  snapshot(){
    return{mode:this.mode,iter:this.iter,envSteps:this.envSteps,episodes:this.episodes,rng:{s:this.rng.s,spare:this.rng.spare},teacherBootstrap:this.teacherBootstrap||null,last:this.last,history:this.history.slice(-120),policy:{encoder:this.policy.encoder.snapshot(),dynamicDecoder:this.policy.dynamicDecoder.snapshot(),kinematicDecoder:this.policy.kinematicDecoder.snapshot(),critic:this.policy.critic.snapshot(),codebook:Array.from(this.policy.codebook)}};
  }
  delete(){for(const e of this.envs)e.delete();}
}

export function referenceFramesPhysical(context,goal){return plannerPhysical(planReference(context,goal));}
export const SONIC_TOY_CONSTANTS={REF_FRAMES,REF_DIM,STATE_SCALE,ACTION_FORCE,FIXED_STD};
