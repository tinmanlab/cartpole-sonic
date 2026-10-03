import {MLP,RNG,planReference,advancePlannerContext,SONIC_TOY_CONSTANTS} from "./sonic_toy.js";

const REF_DIM=SONIC_TOY_CONSTANTS.REF_DIM;
const STATE_SCALE=SONIC_TOY_CONSTANTS.STATE_SCALE;
const ACTION_FORCE=SONIC_TOY_CONSTANTS.ACTION_FORCE;
const FIXED_STD=SONIC_TOY_CONSTANTS.FIXED_STD;
const LOG_SQRT_2PI=0.9189385332046727;

const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
const normalizeState=s=>s.map((v,i)=>clamp(v/STATE_SCALE[i],-1.5,1.5));
const fsqScalar=z=>Math.round(Math.tanh(z)*1.998)/2;
const fsqDeriv=z=>{const t=Math.tanh(z);return .999*(1-t*t);};
const logProbGaussian=(a,mu,std=FIXED_STD)=>{const z=(a-mu)/std;return -.5*z*z-Math.log(std)-LOG_SQRT_2PI;};

function rewardFor(nextState,goal,action){
  const [x,xd,th,thd]=nextState;
  return 1
    -.65*Math.min((x-goal)*(x-goal),4)
    -8*th*th
    -.020*xd*xd
    -.020*thd*thd
    -.002*action*action;
}
function doneFor(s){return Math.abs(s[0])>1.78||Math.abs(s[2])>.65||!s.every(Number.isFinite);}
function gae(data,n,gamma=.99,lambda=.95){
  const carry=new Float64Array(n);
  for(let i=data.length-1;i>=0;i--){
    const q=data[i];
    q.delta=q.r+gamma*(q.terminated?0:q.nextV)-q.oldV;
    q.rawAdv=q.delta+gamma*lambda*(q.done?0:carry[q.env]);
    q.ret=q.rawAdv+q.oldV;carry[q.env]=q.rawAdv;
  }
  const m=mean(data.map(q=>q.rawAdv)),sd=Math.sqrt(mean(data.map(q=>(q.rawAdv-m)**2))+1e-8);
  for(const q of data)q.adv=(q.rawAdv-m)/(sd+1e-8);
}

class TemporalControlPolicy {
  constructor({tokens=1,seed=9001}={}){
    this.tokens=tokens;
    this.tokenDim=2;
    this.latentDim=tokens*this.tokenDim;
    const rng=new RNG(seed);
    this.encoder=new MLP(rng,REF_DIM,20,this.latentDim,{outScale:.20});
    this.dynamicDecoder=new MLP(rng,this.latentDim+4,28,1,{outTanh:true,outScale:.08});
    this.kinematicDecoder=new MLP(rng,this.latentDim,24,REF_DIM,{outTanh:true,outScale:.08});
    this.critic=new MLP(rng,5,20,1,{outScale:.15});
  }
  quantize(z){
    return{
      q:z.map(fsqScalar),
      deriv:z.map(fsqDeriv),
    };
  }
  forward(state,ref,goal,{sample=false,rng=null}={}){
    const ef=this.encoder.forward(ref),z=Array.from(ef.y),quant=this.quantize(z),sNorm=normalizeState(state);
    const df=this.dynamicDecoder.forward([...quant.q,...sNorm]),mu=clamp(df.y[0],-.999,.999);
    let action=mu;
    if(sample&&rng)action=clamp(mu+rng.normal()*FIXED_STD,-1,1);
    const kf=this.kinematicDecoder.forward(quant.q);
    const cf=this.critic.forward([...sNorm,goal/STATE_SCALE[0]]);
    return{
      z,q:quant.q,qDeriv:quant.deriv,mu,action,force:action*ACTION_FORCE,
      value:cf.y[0],kinRecon:Array.from(kf.y),cache:{ef,df,kf,cf,sNorm}
    };
  }
  snapshot(){
    return{
      tokens:this.tokens,tokenDim:this.tokenDim,
      encoder:this.encoder.snapshot(),
      dynamicDecoder:this.dynamicDecoder.snapshot(),
      kinematicDecoder:this.kinematicDecoder.snapshot(),
      critic:this.critic.snapshot(),
    };
  }
  restore(snapshot){
    if(snapshot?.tokens!==this.tokens)throw new Error("controller token-count mismatch");
    const load=(net,src,name)=>{
      if(!src?.p||src.p.length!==net.p.length)throw new Error("invalid "+name+" checkpoint");
      net.p.set(src.p);
      if(src.m?.length===net.m.length)net.m.set(src.m);else net.m.fill(0);
      if(src.v?.length===net.v.length)net.v.set(src.v);else net.v.fill(0);
      net.t=Number.isFinite(src.t)?src.t:0;
    };
    load(this.encoder,snapshot.encoder,"encoder");
    load(this.dynamicDecoder,snapshot.dynamicDecoder,"dynamicDecoder");
    load(this.kinematicDecoder,snapshot.kinematicDecoder,"kinematicDecoder");
    load(this.critic,snapshot.critic,"critic");
    return this;
  }
}

class ControlEnv {
  constructor(sim,rng){
    this.sim=sim;this.rng=rng;this.data=sim.makeData();this.goal=0;this.steps=0;this.reset();
  }
  reset(){
    this.data.qpos[0]=(this.rng.uniform()*2-1)*.12;
    this.data.qpos[1]=(this.rng.uniform()*2-1)*.08;
    this.data.qvel[0]=(this.rng.uniform()*2-1)*.08;
    this.data.qvel[1]=(this.rng.uniform()*2-1)*.08;
    this.data.ctrl[0]=0;this.data.time=0;
    this.sim.mujoco.mj_forward(this.sim.model,this.data);
    this.goal=(this.rng.uniform()*2-1)*.8;
    this.refContext=[0,0];this.steps=0;
    return this.state();
  }
  state(){return this.sim.getState(this.data);}
  reference(){return planReference(this.refContext,this.goal);}
  impulse({xDotDelta=.65,thetaDotDelta=-1.0}={}){
    this.data.qvel[0]+=xDotDelta;this.data.qvel[1]+=thetaDotDelta;
    this.sim.mujoco.mj_forward(this.sim.model,this.data);
  }
  step(action){
    this.data.ctrl[0]=clamp(action,-1,1);
    this.sim.mujoco.mj_step(this.sim.model,this.data);
    this.sim.mujoco.mj_step(this.sim.model,this.data);
    this.steps++;
    this.refContext=advancePlannerContext(this.refContext,this.goal,.02);
    const s=this.state(),targetX=this.refContext[0],done=doneFor(s)||this.steps>=500;
    return{s,targetX,r:rewardFor(s,targetX,action),done,terminated:done&&this.steps<500};
  }
  delete(){this.data.delete();}
}

export class TemporalControlTrainer {
  constructor(sim,{tokens=1,seed=9101,n=8,horizon=96,epochs=4,batch=128}={}){
    this.sim=sim;this.tokens=tokens;this.seed=seed;this.rng=new RNG(seed+5000);
    this.policy=new TemporalControlPolicy({tokens,seed});
    this.n=n;this.horizon=horizon;this.epochs=epochs;this.batch=batch;
    this.envs=Array.from({length:n},()=>new ControlEnv(sim,this.rng));
    this.iter=0;this.envSteps=0;this.episodes=0;this.history=[];
    this.last={reward:NaN,tracking:NaN,piLoss:NaN,valueLoss:NaN,auxLoss:NaN};
  }
  preview(state,ref,goal){return this.policy.forward(state,ref,goal);}
  bootstrapFromTeacher(teacher,{steps=500,batch=256,lr=.0015,auxCoef=.08,dataSeed=0xC011AB}={}){
    if(!teacher)throw new Error("teacher required");
    const dataRng=new RNG(dataSeed);
    let actionLoss=0,auxLoss=0,count=0;
    for(let step=0;step<steps;step++){
      const GE=new Float64Array(this.policy.encoder.p.length);
      const GD=new Float64Array(this.policy.dynamicDecoder.p.length);
      const GK=new Float64Array(this.policy.kinematicDecoder.p.length);
      for(let b=0;b<batch;b++){
        const goal=(dataRng.uniform()*2-1)*.8;
        const refContext=[(dataRng.uniform()*2-1)*.6,(dataRng.uniform()*2-1)*.5];
        const ref=planReference(refContext,goal);
        const state=[
          (dataRng.uniform()*2-1)*.8,
          (dataRng.uniform()*2-1)*1.0,
          (dataRng.uniform()*2-1)*.22,
          (dataRng.uniform()*2-1)*1.8,
        ];
        const out=this.policy.forward(state,ref,goal);
        const target=teacher.forward(state,goal).normalizedForce;
        const err=out.mu-target;actionLoss+=err*err;
        const dqPolicy=this.policy.dynamicDecoder.backward(out.cache.df,[2*err],GD,1/batch).slice(0,this.policy.latentDim);
        const auxDiff=out.kinRecon.map((v,j)=>v-ref[j]);
        auxLoss+=auxDiff.reduce((s,v)=>s+v*v,0)/REF_DIM;
        const dqAux=this.policy.kinematicDecoder.backward(out.cache.kf,auxDiff.map(v=>auxCoef*2*v/REF_DIM),GK,1/batch);
        const dz=new Float64Array(this.policy.latentDim);
        for(let i=0;i<dz.length;i++)dz[i]=(dqPolicy[i]+dqAux[i])*out.qDeriv[i];
        this.policy.encoder.backward(out.cache.ef,dz,GE,1/batch);
        count++;
      }
      this.policy.encoder.adam(GE,lr,.8);
      this.policy.dynamicDecoder.adam(GD,lr,.8);
      this.policy.kinematicDecoder.adam(GK,lr,.8);
    }
    this.bootstrap={steps,batch,actionMse:actionLoss/count,auxMse:auxLoss/count,dataSeed};
    return this.bootstrap;
  }
  collect(){
    const data=[];let rewardSum=0,trackSum=0;
    for(let t=0;t<this.horizon;t++){
      for(let i=0;i<this.n;i++){
        const e=this.envs[i],s=e.state(),goal=e.goal,ref=e.reference();
        const out=this.policy.forward(s,ref,goal,{sample:true,rng:this.rng});
        const oldLogp=logProbGaussian(out.action,out.mu),oldV=out.value;
        const step=e.step(out.action),next=this.policy.forward(step.s,e.reference(),goal),nextV=next.value;
        data.push({env:i,state:s.slice(),goal,ref:Array.from(ref),action:out.action,oldLogp,oldV,r:step.r,nextV,done:step.done,terminated:step.terminated});
        rewardSum+=step.r;trackSum+=Math.abs(step.s[0]-e.refContext[0]);this.envSteps++;
        if(step.done){this.episodes++;e.reset();}
      }
    }
    gae(data,this.n);
    return{data,rewardMean:rewardSum/data.length,trackingMae:trackSum/data.length};
  }
  optimize(data){
    const ids=Array.from({length:data.length},(_,i)=>i);
    let piSum=0,vSum=0,auxSum=0,count=0,clipped=0;
    for(let ep=0;ep<this.epochs;ep++){
      for(let i=ids.length-1;i>0;i--){
        const j=Math.floor(this.rng.uniform()*(i+1));[ids[i],ids[j]]=[ids[j],ids[i]];
      }
      for(let st=0;st<ids.length;st+=this.batch){
        const end=Math.min(st+this.batch,ids.length),bs=end-st;
        const GE=new Float64Array(this.policy.encoder.p.length);
        const GD=new Float64Array(this.policy.dynamicDecoder.p.length);
        const GK=new Float64Array(this.policy.kinematicDecoder.p.length);
        const GC=new Float64Array(this.policy.critic.p.length);
        for(let ii=st;ii<end;ii++){
          const qd=data[ids[ii]],out=this.policy.forward(qd.state,qd.ref,qd.goal);
          const ratio=Math.exp(logProbGaussian(qd.action,out.mu)-qd.oldLogp);
          const active=qd.adv>=0?ratio<=1.2:ratio>=.8;if(!active)clipped++;
          const dlogp=active?-qd.adv*ratio:0;
          const dmu=dlogp*(qd.action-out.mu)/(FIXED_STD*FIXED_STD);
          const dqPolicy=this.policy.dynamicDecoder.backward(out.cache.df,[dmu],GD,1/bs).slice(0,this.policy.latentDim);
          const auxDiff=out.kinRecon.map((v,j)=>v-qd.ref[j]);
          const aux=auxDiff.reduce((s,v)=>s+v*v,0)/REF_DIM;
          const dqAux=this.policy.kinematicDecoder.backward(out.cache.kf,auxDiff.map(v=>.20*2*v/REF_DIM),GK,1/bs);
          const dz=new Float64Array(this.policy.latentDim);
          for(let i=0;i<dz.length;i++)dz[i]=(dqPolicy[i]+dqAux[i])*out.qDeriv[i];
          this.policy.encoder.backward(out.cache.ef,dz,GE,1/bs);
          const cf=this.policy.critic.forward([...normalizeState(qd.state),qd.goal/STATE_SCALE[0]]);
          const dv=cf.y[0]-qd.ret;this.policy.critic.backward(cf,[dv],GC,1/bs);
          piSum+=-Math.min(ratio*qd.adv,clamp(ratio,.8,1.2)*qd.adv);vSum+=.5*dv*dv;auxSum+=aux;count++;
        }
        this.policy.encoder.adam(GE,5e-4,.7);
        this.policy.dynamicDecoder.adam(GD,8e-4,.7);
        this.policy.kinematicDecoder.adam(GK,8e-4,.7);
        this.policy.critic.adam(GC,1.5e-3,1.0);
      }
    }
    return{piLoss:piSum/count,valueLoss:vSum/count,auxLoss:auxSum/count,clipFraction:clipped/count};
  }
  iteration(){
    const col=this.collect(),opt=this.optimize(col.data);this.iter++;
    this.last={reward:col.rewardMean,tracking:col.trackingMae,...opt,envSteps:this.envSteps,episodes:this.episodes};
    this.history.push({iter:this.iter,...this.last});if(this.history.length>160)this.history.shift();
    return this.last;
  }
  evaluate({episodes=12,seed=8080,disturbance=false}={}){
    const rng=new RNG(seed);let successes=0;const allErr=[],completedErr=[],postPushErr=[],recoverySteps=[];
    for(let ep=0;ep<episodes;ep++){
      const e=new ControlEnv(this.sim,rng);let err=0,n=0,postErr=0,postN=0,recovered=null;
      while(e.steps<500){
        if(disturbance&&e.steps===120)e.impulse();
        const s=e.state(),ref=e.reference(),o=this.policy.forward(s,ref,e.goal),step=e.step(o.mu);
        const absErr=Math.abs(step.s[0]-step.targetX);err+=absErr;n++;
        if(disturbance&&e.steps>120){
          postErr+=absErr;postN++;
          if(recovered===null&&e.steps>=125&&absErr<.18&&Math.abs(step.s[2])<.18)recovered=e.steps-120;
        }
        if(step.done)break;
      }
      const epMae=err/Math.max(1,n),completed=e.steps>=500&&!doneFor(e.state());
      if(completed){successes++;completedErr.push(epMae);}
      allErr.push(epMae);
      if(disturbance){postPushErr.push(postErr/Math.max(1,postN));recoverySteps.push(recovered??380);}
      e.delete();
    }
    return{
      episodes,successes,
      trackingMae:completedErr.length?mean(completedErr):null,
      allStepTrackingMae:mean(allErr),
      disturbance:disturbance?{postPushTrackingMae:mean(postPushErr),meanRecoverySteps:mean(recoverySteps)}:null,
    };
  }
  snapshot(){
    return{
      schema:"cartpole-sonic-temporal-control-trainer/v1",
      tokens:this.tokens,iter:this.iter,envSteps:this.envSteps,episodes:this.episodes,
      rng:{s:this.rng.s,spare:this.rng.spare},
      bootstrap:this.bootstrap||null,last:this.last,history:this.history.slice(-120),
      policy:this.policy.snapshot(),
    };
  }
  restore(snapshot){
    if(snapshot?.schema!=="cartpole-sonic-temporal-control-trainer/v1"||snapshot.tokens!==this.tokens)throw new Error("invalid temporal control checkpoint");
    this.policy.restore(snapshot.policy);
    for(const e of this.envs)e.delete();
    this.rng=new RNG(this.seed+5000);
    this.envs=Array.from({length:this.n},()=>new ControlEnv(this.sim,this.rng));
    if(snapshot.rng){this.rng.s=snapshot.rng.s>>>0;this.rng.spare=Number.isFinite(snapshot.rng.spare)?snapshot.rng.spare:null;}
    this.bootstrap=snapshot.bootstrap||null;
    this.iter=snapshot.iter||0;this.envSteps=snapshot.envSteps||0;this.episodes=snapshot.episodes||0;
    this.last=snapshot.last||{reward:NaN,tracking:NaN,piLoss:NaN,valueLoss:NaN,auxLoss:NaN};
    this.history=Array.isArray(snapshot.history)?snapshot.history.slice(-120):[];
    return this;
  }
  delete(){for(const e of this.envs)e.delete();}
}

export class TemporalControlLab {
  constructor(sim,{seed=9101}={}){
    this.sim=sim;this.seed=seed;
    this.one=new TemporalControlTrainer(sim,{tokens:1,seed});
    this.two=new TemporalControlTrainer(sim,{tokens:2,seed:seed+17});
    this.evalHistory=[];
  }
  bootstrap(teacher,opts={}){
    const one=this.one.bootstrapFromTeacher(teacher,opts);
    const two=this.two.bootstrapFromTeacher(teacher,opts);
    return{one,two};
  }
  ppo(steps=10){
    for(let i=0;i<steps;i++){this.one.iteration();this.two.iteration();}
    return this.evaluate();
  }
  evaluate(){
    const cleanOne=this.one.evaluate({episodes:12,seed:8181});
    const cleanTwo=this.two.evaluate({episodes:12,seed:8181});
    const pushOne=this.one.evaluate({episodes:12,seed:9191,disturbance:true});
    const pushTwo=this.two.evaluate({episodes:12,seed:9191,disturbance:true});
    const row={ppoIterations:this.one.iter,one:{clean:cleanOne,push:pushOne},two:{clean:cleanTwo,push:pushTwo}};
    this.evalHistory.push(row);if(this.evalHistory.length>80)this.evalHistory.shift();
    return row;
  }
  snapshot(){
    return{
      schema:"cartpole-sonic-temporal-control-lab/v1",
      one:this.one.snapshot(),two:this.two.snapshot(),
      evalHistory:this.evalHistory.slice(-40),
      claimBoundary:"Compares two small controllers with different token-slot capacity under matched teacher/bootstrap/PPO budgets. It does not isolate token count from parameter-count differences and does not prove the SONIC release choice is globally optimal.",
    };
  }
  restore(snapshot){
    if(snapshot?.schema!=="cartpole-sonic-temporal-control-lab/v1")throw new Error("invalid temporal control lab checkpoint");
    this.one.restore(snapshot.one);
    this.two.restore(snapshot.two);
    this.evalHistory=Array.isArray(snapshot.evalHistory)?snapshot.evalHistory.slice(-40):[];
    return this;
  }
  delete(){this.one.delete();this.two.delete();}
}
