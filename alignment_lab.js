import {MLP,RNG,planReference} from "./sonic_toy.js";

const STATE_SCALE=[1.8,3.0,0.55,4.0];
const FORCE_SCALE=10;
const KEY_FRAMES=[0,2,5,7];

const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const normalizeState=s=>s.map((v,i)=>clamp(v/STATE_SCALE[i],-1.5,1.5));

export function sparseKeypointReference(ref){
  const out=[];
  for(const i of KEY_FRAMES){
    out.push(ref[i*2],ref[i*2+1]);
  }
  return out;
}

export class AlignmentLab {
  constructor(policy,{seed=9402}={}){
    this.policy=policy;
    this.seed=seed;
    this.reset();
  }

  reset(){
    this.encoder=new MLP(new RNG(this.seed),8,16,2,{outScale:.22});
    this.step=0;
    this.history=[];
    this.baseline=this.evaluate(192);
    this.current=this.baseline;
    return this.snapshot();
  }

  secondary(ref){
    const sparse=sparseKeypointReference(ref);
    const f=this.encoder.forward(sparse);
    const z=Array.from(f.y);
    const quant=this.policy.quantize(z,"fsq");
    return{sparse,f,z,q:quant.q};
  }

  forceFromToken(q,state){
    const sNorm=normalizeState(state);
    const out=this.policy.dynamicDecoder.forward([q[0],q[1],...sNorm]);
    const mu=clamp(out.y[0],-.999,.999);
    return mu*FORCE_SCALE;
  }

  compare(ref,state,goal){
    const primary=this.policy.forward(state,ref,goal,{sample:false});
    const secondary=this.secondary(ref);
    const secondaryForce=this.forceFromToken(secondary.q,state);
    const dz0=primary.z[0]-secondary.z[0],dz1=primary.z[1]-secondary.z[1];
    return{
      primary:{z:Array.from(primary.z),q:Array.from(primary.q),force:primary.force},
      secondary:{z:Array.from(secondary.z),q:Array.from(secondary.q),force:secondaryForce,sparse:Array.from(secondary.sparse)},
      latentDistance:Math.hypot(dz0,dz1),
      tokenSame:primary.q[0]===secondary.q[0]&&primary.q[1]===secondary.q[1],
      actionDifference:Math.abs(primary.force-secondaryForce),
    };
  }

  sample(rng){
    const context=[(rng.uniform()*2-1)*.85,(rng.uniform()*2-1)*1.1];
    const goal=(rng.uniform()*2-1)*.9;
    const ref=planReference(context,goal);
    const state=[
      (rng.uniform()*2-1)*.9,
      (rng.uniform()*2-1)*1.2,
      (rng.uniform()*2-1)*.20,
      (rng.uniform()*2-1)*1.4,
    ];
    return{ref,state,goal};
  }

  trainStep({batch=96,lr=.0025}={}){
    const rng=new RNG((this.seed+this.step*7919+17)>>>0);
    const G=new Float64Array(this.encoder.p.length);
    let loss=0;
    for(let i=0;i<batch;i++){
      const {ref}=this.sample(rng);
      const target=Array.from(this.policy.encoder.forward(ref).y);
      const sparse=sparseKeypointReference(ref);
      const f=this.encoder.forward(sparse);
      const d0=f.y[0]-target[0],d1=f.y[1]-target[1];
      loss+=(d0*d0+d1*d1)/2;
      this.encoder.backward(f,[d0,d1],G,1/batch);
    }
    const gradNorm=this.encoder.adam(G,lr,1.0);
    this.step++;
    const metrics=this.evaluate(192);
    this.current=metrics;
    this.history.push({step:this.step,trainMse:loss/batch,gradNorm,...metrics});
    if(this.history.length>240)this.history.shift();
    return this.history.at(-1);
  }

  train(steps=25,opts={}){
    let out=null;
    for(let i=0;i<steps;i++)out=this.trainStep(opts);
    return out;
  }

  evaluate(n=192){
    const rng=new RNG(0xA11CE);
    let latentMse=0,tokenSame=0,actionMae=0,qMse=0;
    for(let i=0;i<n;i++){
      const {ref,state,goal}=this.sample(rng);
      const c=this.compare(ref,state,goal);
      const dz0=c.primary.z[0]-c.secondary.z[0],dz1=c.primary.z[1]-c.secondary.z[1];
      const dq0=c.primary.q[0]-c.secondary.q[0],dq1=c.primary.q[1]-c.secondary.q[1];
      latentMse+=(dz0*dz0+dz1*dz1)/2;
      qMse+=(dq0*dq0+dq1*dq1)/2;
      if(c.tokenSame)tokenSame++;
      actionMae+=c.actionDifference;
    }
    return{
      latentMse:latentMse/n,
      tokenAgreement:tokenSame/n,
      tokenMse:qMse/n,
      actionMae:actionMae/n,
      samples:n,
    };
  }

  snapshot(){
    return{
      schema:"cartpole-sonic-alignment-lab/v1",
      representationA:{name:"full trajectory",shape:"8 frames × [x,xDot] = 16D",trainable:false,role:"primary anchor encoder input"},
      representationB:{name:"sparse keypoints",shape:"frames [1,3,6,8] × [x,xDot] = 8D",trainable:true,role:"secondary encoder input"},
      step:this.step,
      baseline:this.baseline,
      current:this.current,
      history:this.history.slice(-120),
      simplification:"CartPole freezes the primary Encoder as an anchor and trains only the sparse secondary Encoder. SONIC trains multiple modality Encoders jointly with alignment auxiliaries.",
    };
  }
}
