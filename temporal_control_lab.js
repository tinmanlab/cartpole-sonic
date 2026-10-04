import {normalizeState,doneFor,gae,CartPoleTrainEnv,validateRng,integerCount} from "./control_contract.js";
import {MLP,RNG,planReference,advancePlannerContext,SONIC_TOY_CONSTANTS} from "./sonic_toy.js";

const REF_DIM=SONIC_TOY_CONSTANTS.REF_DIM;
const STATE_SCALE=SONIC_TOY_CONSTANTS.STATE_SCALE;
const ACTION_FORCE=SONIC_TOY_CONSTANTS.ACTION_FORCE;
const FIXED_STD=SONIC_TOY_CONSTANTS.FIXED_STD;
const LOG_SQRT_2PI=0.9189385332046727;

const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
const fsqScalar=z=>Math.round(Math.tanh(z)*1.998)/2;
const fsqDeriv=z=>{const t=Math.tanh(z);return .999*(1-t*t);};
const logProbGaussian=(a,mu,std=FIXED_STD)=>{const z=(a-mu)/std;return -.5*z*z-Math.log(std)-LOG_SQRT_2PI;};

class TemporalControlPolicy {
  constructor({tokens=1,seed=9001,widths={}}={}){
    this.tokens=tokens;
    this.tokenDim=2;
    this.latentDim=tokens*this.tokenDim;
    this.widths={
      encoder:widths.encoder??20,
      dynamic:widths.dynamic??28,
      kinematic:widths.kinematic??24,
      critic:widths.critic??20,
    };
    const rng=new RNG(seed);
    this.encoder=new MLP(rng,REF_DIM,this.widths.encoder,this.latentDim,{outScale:.20});
    this.dynamicDecoder=new MLP(rng,this.latentDim+4,this.widths.dynamic,1,{outTanh:true,outScale:.08});
    this.kinematicDecoder=new MLP(rng,this.latentDim,this.widths.kinematic,REF_DIM,{outTanh:true,outScale:.08});
    this.critic=new MLP(rng,5,this.widths.critic,1,{outScale:.15});
  }
  parameterBreakdown(){
    const encoder=this.encoder.p.length,dynamic=this.dynamicDecoder.p.length,kinematic=this.kinematicDecoder.p.length,critic=this.critic.p.length;
    return{encoder,dynamic,kinematic,critic,actor:encoder+dynamic,total:encoder+dynamic+kinematic+critic};
  }
  parameterCount(){
    return this.encoder.p.length+this.dynamicDecoder.p.length+this.kinematicDecoder.p.length+this.critic.p.length;
  }
  quantize(z){
    return{q:z.map(fsqScalar),deriv:z.map(fsqDeriv)};
  }
  forward(state,ref,goal,{sample=false,rng=null}={}){
    const ef=this.encoder.forward(ref),z=Array.from(ef.y),quant=this.quantize(z),sNorm=normalizeState(state);
    const df=this.dynamicDecoder.forward([...quant.q,...sNorm]),mu=clamp(df.y[0],-.999,.999);
    // Preserve the original Gaussian sample for PPO; only physical input is clipped.
    const rawAction=sample&&rng?mu+rng.normal()*FIXED_STD:mu,action=clamp(rawAction,-1,1);
    const kf=this.kinematicDecoder.forward(quant.q);
    const cf=this.critic.forward([...sNorm,goal/STATE_SCALE[0]]);
    return{
      z,q:quant.q,qDeriv:quant.deriv,mu,rawAction,action,force:action*ACTION_FORCE,
      value:cf.y[0],kinRecon:Array.from(kf.y),cache:{ef,df,kf,cf,sNorm}
    };
  }
  snapshot(){
    return{
      tokens:this.tokens,tokenDim:this.tokenDim,widths:{...this.widths},parameterCount:this.parameterCount(),parameters:this.parameterBreakdown(),
      encoder:this.encoder.snapshot(),dynamicDecoder:this.dynamicDecoder.snapshot(),
      kinematicDecoder:this.kinematicDecoder.snapshot(),critic:this.critic.snapshot(),
    };
  }
  validateSnapshot(snapshot){
    if(snapshot?.tokens!==this.tokens)throw new Error("controller token-count mismatch");
    for(const name of ["encoder","dynamicDecoder","kinematicDecoder","critic"])this[name].validateSnapshot(snapshot[name],name);
  }
  restore(snapshot){
    this.validateSnapshot(snapshot);
    for(const name of ["encoder","dynamicDecoder","kinematicDecoder","critic"])this[name].applySnapshot(snapshot[name]);
    return this;
  }
}

export class TemporalControlTrainer {
  constructor(sim,{tokens=1,seed=9101,n=8,horizon=96,epochs=4,batch=128,widths={},ppoLrScale=1,ppoModuleScales={}}={}){
    integerCount(n,"environment count",1,1024);integerCount(horizon,"rollout horizon",1,100000);
    integerCount(epochs,"epochs",1,1000);integerCount(batch,"batch",1,1000000);
    this.sim=sim;this.tokens=tokens;this.seed=seed;this.rng=new RNG(seed+5000);
    this.widths={...widths};this.ppoLrScale=ppoLrScale;
    this.ppoModuleScales={
      encoder:ppoModuleScales.encoder??1,dynamic:ppoModuleScales.dynamic??1,
      kinematic:ppoModuleScales.kinematic??1,critic:ppoModuleScales.critic??1,
    };
    this.policy=new TemporalControlPolicy({tokens,seed,widths});
    this.n=n;this.horizon=horizon;this.epochs=epochs;this.batch=batch;
    this.envs=Array.from({length:n},()=>new CartPoleTrainEnv(sim,this.rng));
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
        const state=[(dataRng.uniform()*2-1)*.8,(dataRng.uniform()*2-1)*1.0,(dataRng.uniform()*2-1)*.22,(dataRng.uniform()*2-1)*1.8];
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
    const data=[];let rewardSum=0,trackSum=0,saturation=0,sampleClips=0;
    for(let t=0;t<this.horizon;t++){
      for(let i=0;i<this.n;i++){
        const e=this.envs[i],s=e.state(),goal=e.goal,ref=e.reference();
        const out=this.policy.forward(s,ref,goal,{sample:true,rng:this.rng});
        const oldLogp=logProbGaussian(out.rawAction,out.mu),oldV=out.value;
        if(Math.abs(out.mu)>.95)saturation++;
        if(out.rawAction!==out.action)sampleClips++;
        const step=e.step(out.action),next=this.policy.forward(step.s,e.reference(),goal),nextV=next.value;
        data.push({env:i,state:s.slice(),goal,ref:Array.from(ref),rawAction:out.rawAction,action:out.action,oldMu:out.mu,oldQ:out.q.slice(),oldLogp,oldV,r:step.r,nextV,done:step.done,terminated:step.terminated});
        rewardSum+=step.r;trackSum+=Math.abs(step.s[0]-e.refContext[0]);this.envSteps++;
        if(step.done){this.episodes++;e.reset();}
      }
    }
    gae(data,this.n);
    return{data,rewardMean:rewardSum/data.length,trackingMae:trackSum/data.length,actionSaturation:saturation/data.length,sampleClipFraction:sampleClips/data.length};
  }
  optimize(data){
    const ids=Array.from({length:data.length},(_,i)=>i);
    let piSum=0,vSum=0,auxSum=0,count=0,clipped=0,objectiveClipped=0,gradBatches=0;
    let gradEncoder=0,gradDynamic=0,gradKinematic=0,gradCritic=0;
    const updates=Object.fromEntries(["encoder","dynamic","kinematic","critic"].map(k=>[k,{preClipNorm:0,postClipNorm:0,updateNorm:0,relativeUpdateNorm:0,gradClipFraction:0}]));
    for(let ep=0;ep<this.epochs;ep++){
      for(let i=ids.length-1;i>0;i--){const j=Math.floor(this.rng.uniform()*(i+1));[ids[i],ids[j]]=[ids[j],ids[i]];}
      for(let st=0;st<ids.length;st+=this.batch){
        const end=Math.min(st+this.batch,ids.length),bs=end-st;
        const GE=new Float64Array(this.policy.encoder.p.length),GD=new Float64Array(this.policy.dynamicDecoder.p.length);
        const GK=new Float64Array(this.policy.kinematicDecoder.p.length),GC=new Float64Array(this.policy.critic.p.length);
        for(let ii=st;ii<end;ii++){
          const qd=data[ids[ii]],out=this.policy.forward(qd.state,qd.ref,qd.goal);
          const ratio=Math.exp(logProbGaussian(qd.rawAction,out.mu)-qd.oldLogp);
          const active=qd.adv>=0?ratio<=1.2:ratio>=.8;if(!active)objectiveClipped++;if(ratio<.8||ratio>1.2)clipped++;
          const dlogp=active?-qd.adv*ratio:0;
          const dmu=dlogp*(qd.rawAction-out.mu)/(FIXED_STD*FIXED_STD);
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
        gradEncoder+=this.policy.encoder.adam(GE,5e-4*this.ppoLrScale*this.ppoModuleScales.encoder,.7);
        gradDynamic+=this.policy.dynamicDecoder.adam(GD,8e-4*this.ppoLrScale*this.ppoModuleScales.dynamic,.7);
        gradKinematic+=this.policy.kinematicDecoder.adam(GK,8e-4*this.ppoLrScale*this.ppoModuleScales.kinematic,.7);
        gradCritic+=this.policy.critic.adam(GC,1.5e-3*this.ppoLrScale*this.ppoModuleScales.critic,1.0);
        for(const [key,net] of [["encoder",this.policy.encoder],["dynamic",this.policy.dynamicDecoder],["kinematic",this.policy.kinematicDecoder],["critic",this.policy.critic]]){
          for(const metric of ["preClipNorm","postClipNorm","updateNorm","relativeUpdateNorm"])updates[key][metric]+=net.lastUpdate[metric];
          updates[key].gradClipFraction+=net.lastUpdate.clipScale<1?1:0;
        }
        gradBatches++;
      }
    }
    for(const values of Object.values(updates))for(const k of Object.keys(values))values[k]/=Math.max(1,gradBatches);
    // Fixed-std Gaussian KL before actuator clipping, on identical collected states/references.
    let kl=0,approxKl=0,shiftSq=0,changed=0,postClipped=0;
    for(const row of data){
      const out=this.policy.forward(row.state,row.ref,row.goal),shift=out.mu-row.oldMu;
      const logRatio=logProbGaussian(row.rawAction,out.mu)-row.oldLogp,ratio=Math.exp(logRatio);
      shiftSq+=shift*shift;kl+=shift*shift/(2*FIXED_STD*FIXED_STD);approxKl+=Math.expm1(logRatio)-logRatio;
      if(out.q.some((q,i)=>q!==row.oldQ[i]))changed++;
      if(ratio<.8||ratio>1.2)postClipped++;
    }
    return{
      piLoss:piSum/count,valueLoss:vSum/count,auxLoss:auxSum/count,clipFraction:clipped/count,objectiveClipFraction:objectiveClipped/count,
      gradEncoder:gradEncoder/Math.max(1,gradBatches),gradDynamic:gradDynamic/Math.max(1,gradBatches),
      gradKinematic:gradKinematic/Math.max(1,gradBatches),gradCritic:gradCritic/Math.max(1,gradBatches),
      updates,optimizerBatches:gradBatches,postUpdateKl:kl/data.length,postUpdateApproxKl:approxKl/data.length,
      postUpdateClipFraction:postClipped/data.length,tokenChangeFraction:changed/data.length,actionMeanShiftRms:Math.sqrt(shiftSq/data.length),
    };
  }
  iteration(){
    const col=this.collect(),opt=this.optimize(col.data);this.iter++;
    this.last={reward:col.rewardMean,tracking:col.trackingMae,actionSaturation:col.actionSaturation,sampleClipFraction:col.sampleClipFraction,...opt,envSteps:this.envSteps,episodes:this.episodes};
    this.history.push({iter:this.iter,...this.last});if(this.history.length>160)this.history.shift();
    return this.last;
  }
  evaluate({episodes=12,seed=8080,disturbance=false}={}){
    const rng=new RNG(seed);let successes=0;const allErr=[],completedErr=[],postPushErr=[],recoverySteps=[],recoveredOnly=[];
    for(let ep=0;ep<episodes;ep++){
      const e=new CartPoleTrainEnv(this.sim,rng);let err=0,n=0,postErr=0,postN=0,recovered=null;
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
      if(disturbance&&postN>0){postPushErr.push(postErr/postN);recoverySteps.push(recovered??380);if(recovered!==null)recoveredOnly.push(recovered);}
      e.delete();
    }
    return{
      episodes,successes,trackingMae:completedErr.length?mean(completedErr):null,allStepTrackingMae:mean(allErr),
      disturbance:disturbance?{postPushTrackingMae:postPushErr.length?mean(postPushErr):null,meanRecoverySteps:recoverySteps.length?mean(recoverySteps):null,pushReachedEpisodes:postPushErr.length,recoveredEpisodes:recoveredOnly.length,meanRecoveredSteps:recoveredOnly.length?mean(recoveredOnly):null,recoveryDefinition:"First tolerance entry, not sustained recovery; unrecovered reached episodes capped at 380 steps."}:null,
    };
  }
  snapshot(){
    return{
      schema:"cartpole-sonic-temporal-control-trainer/v1",ppoContract:"raw-gaussian-sample/v2",
      tokens:this.tokens,widths:{...this.policy.widths},parameterCount:this.policy.parameterCount(),parameters:this.policy.parameterBreakdown(),ppoLrScale:this.ppoLrScale,ppoModuleScales:{...this.ppoModuleScales},
      iter:this.iter,envSteps:this.envSteps,episodes:this.episodes,rng:{s:this.rng.s,spare:this.rng.spare},
      bootstrap:this.bootstrap||null,last:this.last,history:this.history.slice(-120),policy:this.policy.snapshot(),
    };
  }
  validateSnapshot(snapshot){
    if(snapshot?.schema!=="cartpole-sonic-temporal-control-trainer/v1"||snapshot.tokens!==this.tokens)throw new Error("invalid temporal control checkpoint");
    this.policy.validateSnapshot(snapshot.policy);
    if(Object.hasOwn(snapshot,"rng"))validateRng(snapshot.rng);
    for(const key of ["iter","envSteps","episodes"])if(Object.hasOwn(snapshot,key))integerCount(snapshot[key],key,0,Number.MAX_SAFE_INTEGER);
  }
  restore(snapshot){
    if(snapshot?.schema!=="cartpole-sonic-temporal-control-trainer/v1"||snapshot.tokens!==this.tokens)throw new Error("invalid temporal control checkpoint");
    this.validateSnapshot(snapshot);
    this.policy.restore(snapshot.policy);
    for(const e of this.envs)e.delete();
    this.rng=new RNG(this.seed+5000);
    this.envs=Array.from({length:this.n},()=>new CartPoleTrainEnv(this.sim,this.rng));
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
    const one=this.one.bootstrapFromTeacher(teacher,opts),two=this.two.bootstrapFromTeacher(teacher,opts);
    return{one,two};
  }
  ppo(steps=10){
    for(let i=0;i<steps;i++){this.one.iteration();this.two.iteration();}
    return this.evaluate();
  }
  evaluate(){
    const cleanOne=this.one.evaluate({episodes:12,seed:8181}),cleanTwo=this.two.evaluate({episodes:12,seed:8181});
    const pushOne=this.one.evaluate({episodes:12,seed:9191,disturbance:true}),pushTwo=this.two.evaluate({episodes:12,seed:9191,disturbance:true});
    const row={ppoIterations:this.one.iter,one:{clean:cleanOne,push:pushOne},two:{clean:cleanTwo,push:pushTwo}};
    this.evalHistory.push(row);if(this.evalHistory.length>80)this.evalHistory.shift();
    return row;
  }
  snapshot(){
    return{
      schema:"cartpole-sonic-temporal-control-lab/v1",ppoContract:"raw-gaussian-sample/v2",
      one:this.one.snapshot(),two:this.two.snapshot(),evalHistory:this.evalHistory.slice(-40),
      claimBoundary:"Matched teacher/bootstrap/PPO budgets and evaluation seeds; initialization and training RNG differ in this legacy pair. Total or actor parameter counts do not establish equal functional capacity. No universal token-count superiority is established.",
    };
  }
  restore(snapshot){
    if(snapshot?.schema!=="cartpole-sonic-temporal-control-lab/v1")throw new Error("invalid temporal control lab checkpoint");
    this.one.validateSnapshot(snapshot.one);this.two.validateSnapshot(snapshot.two);
    this.one.restore(snapshot.one);this.two.restore(snapshot.two);
    this.evalHistory=Array.isArray(snapshot.evalHistory)?snapshot.evalHistory.slice(-40):[];
    return this;
  }
  delete(){this.one.delete();this.two.delete();}
}
