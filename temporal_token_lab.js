import {MLP,RNG,planReference} from "./sonic_toy.js";

const REF_DIM=16;
const TOKEN_DIM=2;

const fsqScalar=z=>Math.round(Math.tanh(z)*1.998)/2;
const fsqDeriv=z=>{const t=Math.tanh(z);return .999*(1-t*t);};

class TemporalAE {
  constructor({tokens=1,seed=1}={}){
    this.tokens=tokens;
    this.dim=tokens*TOKEN_DIM;
    this.rng=new RNG(seed);
    this.encoder=new MLP(this.rng,REF_DIM,24,this.dim,{outScale:.20});
    this.decoder=new MLP(this.rng,this.dim,36,REF_DIM,{outTanh:true,outScale:.08});
  }
  forward(ref){
    const ef=this.encoder.forward(ref);
    const z=Array.from(ef.y);
    const q=z.map(fsqScalar);
    const df=this.decoder.forward(q);
    return{ef,z,q,df,recon:Array.from(df.y)};
  }
  trainBatch(refs,{lr=.0025}={}){
    const Ge=new Float64Array(this.encoder.p.length);
    const Gd=new Float64Array(this.decoder.p.length);
    let loss=0;
    for(const ref of refs){
      const f=this.forward(ref);
      const dy=new Float64Array(REF_DIM);
      for(let i=0;i<REF_DIM;i++){
        const d=f.recon[i]-ref[i];
        loss+=d*d/REF_DIM;
        dy[i]=2*d/REF_DIM;
      }
      const dq=this.decoder.backward(f.df,dy,Gd,1/refs.length);
      const dz=new Float64Array(this.dim);
      for(let i=0;i<this.dim;i++)dz[i]=dq[i]*fsqDeriv(f.z[i]);
      this.encoder.backward(f.ef,dz,Ge,1/refs.length);
    }
    const gradDecoder=this.decoder.adam(Gd,lr,1.0);
    const gradEncoder=this.encoder.adam(Ge,lr,1.0);
    return{loss:loss/refs.length,gradEncoder,gradDecoder};
  }
}

function sampleReference(rng){
  const context=[(rng.uniform()*2-1)*1.15,(rng.uniform()*2-1)*1.6];
  const goal=(rng.uniform()*2-1)*1.15;
  return planReference(context,goal);
}
function tokenGroups(arr,tokens){
  const out=[];
  for(let i=0;i<tokens;i++)out.push(arr.slice(i*TOKEN_DIM,(i+1)*TOKEN_DIM));
  return out;
}
function vectorDistance(a,b){
  let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d;}return Math.sqrt(s);
}

export class TemporalTokenLab {
  constructor({seed=73021}={}){
    this.seed=seed;
    this.reset();
  }
  reset(){
    this.one=new TemporalAE({tokens:1,seed:this.seed});
    this.two=new TemporalAE({tokens:2,seed:this.seed+11});
    this.step=0;
    this.history=[];
    this.baseline=this.evaluate(256);
    this.current=this.baseline;
    return this.snapshot();
  }
  batch(step=this.step,batch=128){
    const rng=new RNG((this.seed+0x51ED+step*104729)>>>0);
    return Array.from({length:batch},()=>sampleReference(rng));
  }
  trainStep({batch=128,lr=.0025}={}){
    const refs=this.batch(this.step,batch);
    const a=this.one.trainBatch(refs,{lr});
    const b=this.two.trainBatch(refs,{lr});
    this.step++;
    const metrics=this.evaluate(256);
    this.current=metrics;
    const row={step:this.step,trainOne:a.loss,trainTwo:b.loss,...metrics};
    this.history.push(row);
    if(this.history.length>240)this.history.shift();
    return row;
  }
  train(steps=50,opts={}){
    let r=null;for(let i=0;i<steps;i++)r=this.trainStep(opts);return r;
  }
  evaluate(n=256){
    const rng=new RNG(0x71A3);
    let mseOne=0,mseTwo=0;
    const codesOne=new Set(),codesTwo=new Set();
    for(let k=0;k<n;k++){
      const ref=sampleReference(rng);
      const a=this.one.forward(ref),b=this.two.forward(ref);
      for(let i=0;i<REF_DIM;i++){
        mseOne+=(a.recon[i]-ref[i])**2/(REF_DIM*n);
        mseTwo+=(b.recon[i]-ref[i])**2/(REF_DIM*n);
      }
      codesOne.add(a.q.map(v=>v.toFixed(2)).join(","));
      codesTwo.add(b.q.map(v=>v.toFixed(2)).join(","));
    }
    return{
      mseOne,mseTwo,
      mseRatio:mseTwo/Math.max(mseOne,1e-12),
      uniqueCodesOne:codesOne.size,
      uniqueCodesTwo:codesTwo.size,
      samples:n,
    };
  }
  compare(ref){
    const one=this.one.forward(ref),two=this.two.forward(ref);
    return{
      one:{z:tokenGroups(one.z,1),q:tokenGroups(one.q,1),recon:one.recon},
      two:{z:tokenGroups(two.z,2),q:tokenGroups(two.q,2),recon:two.recon},
    };
  }
  sensitivity(ref,{delta=.12}={}){
    const base=this.two.forward(ref);
    const perturb=(start,end)=>{
      const r=Float64Array.from(ref);
      for(let i=start;i<end;i++){
        // perturb both x and xdot coherently but keep normalized range bounded
        const sign=(i%2===0)?1:.55;
        r[i]=Math.max(-1,Math.min(1,r[i]+delta*sign));
      }
      return this.two.forward(r);
    };
    const early=perturb(0,8),late=perturb(8,16);
    const perToken=(a,b)=>{
      const ga=tokenGroups(a,this.two.tokens),gb=tokenGroups(b,this.two.tokens);
      return ga.map((x,i)=>vectorDistance(x,gb[i]));
    };
    return{
      description:"Diagnostic only: token roles are not pre-assigned. Values show how much each continuous token latent changes when early or late half of the same input window is perturbed.",
      earlyLatentDelta:perToken(early.z,base.z),
      lateLatentDelta:perToken(late.z,base.z),
      earlyTokenChanged:tokenGroups(early.q,2).map((q,i)=>vectorDistance(q,tokenGroups(base.q,2)[i])>1e-12),
      lateTokenChanged:tokenGroups(late.q,2).map((q,i)=>vectorDistance(q,tokenGroups(base.q,2)[i])>1e-12),
    };
  }
  snapshot(){
    return{
      schema:"cartpole-sonic-temporal-token-lab/v1",
      step:this.step,
      baseline:this.baseline,
      current:this.current,
      history:this.history.slice(-120),
      oneToken:{tokens:1,tokenDim:TOKEN_DIM,flattenedDim:TOKEN_DIM,capacity:"5^2 implicit FSQ combinations"},
      twoToken:{tokens:2,tokenDim:TOKEN_DIM,flattenedDim:2*TOKEN_DIM,capacity:"5^4 implicit FSQ combinations"},
      officialMapping:"SONIC MLP encoders read the whole future window, then reshape encoder output to (max_num_tokens, token_dim). Tokens are jointly learned slots; near/far semantics are not hard-coded by token index.",
      simplification:"The toy compares 1 vs 2 tokens with token_dim=2 and reconstruction-only training. SONIC uses token_dim=32, max_num_tokens=2 and joint PPO/auxiliary training.",
    };
  }
}
