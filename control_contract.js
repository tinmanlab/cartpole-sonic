// Shared browser teaching task; the native continuing task remains separate.
export const REF_FRAMES=8, REF_DIM=16, ACTION_FORCE=10, FIXED_STD=.12;
export const STATE_ORDER=Object.freeze(["x","xDot","theta","thetaDot"]);
export const STATE_SCALE=Object.freeze([1.8,3,.55,4]);
export const PHYSICS_DT=.01, CONTROL_DT=.02, CONTROL_SUBSTEPS=2, JS_TIMEOUT_STEPS=500;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
export function finiteNumber(value,name,min=-Infinity,max=Infinity){
  if(typeof value!=="number"||!Number.isFinite(value)||value<min||value>max)throw new Error(name+" must be a finite number in ["+min+", "+max+"]");
  return value;
}
export function integerCount(value,name,min=1,max=100){
  finiteNumber(value,name,min,max);if(!Number.isSafeInteger(value))throw new Error(name+" must be an integer");return value;
}
export function numericVector(value,length,name,{nonnegative=false}={}){
  if(!(Array.isArray(value)||ArrayBuffer.isView(value))||value.length!==length)throw new Error("invalid "+name+" length");
  for(const x of value)finiteNumber(x,name,nonnegative?0:-Infinity);return value;
}
export function validateRng(value){
  if(!value||typeof value!=="object")throw new Error("invalid RNG checkpoint");
  integerCount(value.s,"RNG state",0,4294967295);
  if(value.spare!==null)finiteNumber(value.spare,"RNG spare");return value;
}
export function physicalFailureReason(s){
  if(s.length!==4||!s.every(x=>typeof x==="number"&&Number.isFinite(x)))return "non-finite state";
  if(Math.abs(s[0])>1.78)return "track limit (1.78 m)";
  if(Math.abs(s[2])>.65)return "pole angle (0.65 rad)";
  return null;
}
export const doneFor=s=>physicalFailureReason(s)!==null;
export const normalizeState=s=>s.map((v,i)=>clamp(v/STATE_SCALE[i],-1.5,1.5));
export function planReference(context,goal,{frames=REF_FRAMES,frameDt=.08,k=2.5}={}){
  numericVector(context,2,"planner context");finiteNumber(goal,"goal");
  integerCount(frames,"reference frames",1,1000);finiteNumber(frameDt,"frameDt",Number.MIN_VALUE);finiteNumber(k,"k",Number.MIN_VALUE);
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
  numericVector(context,2,"planner context");finiteNumber(goal,"goal");finiteNumber(dt,"dt",0);finiteNumber(k,"k",Number.MIN_VALUE);
  const [x,xd]=context,e=x-goal,B=xd+k*e,exp=Math.exp(-k*dt);
  return [goal+(e+B*dt)*exp,(xd-k*B*dt)*exp];
}

export function rewardFor(nextState,goal,action){
  const [x,xd,th,thd]=nextState;
  return 1
    - 0.65*Math.min((x-goal)*(x-goal),4)
    - 8.0*th*th
    - .020*xd*xd
    - .020*thd*thd
    - .002*action*action;
}

export class CartPoleTrainEnv {
  constructor(sim,rng){
    this.sim=sim;this.rng=rng;this.data=sim.makeData();this.goal=0;this.steps=0;this.reset();
  }
  reset(){
    this.sim.mujoco.mj_resetData(this.sim.model,this.data);
    this.data.qpos[0]=(this.rng.uniform()*2-1)*.12;this.data.qpos[1]=(this.rng.uniform()*2-1)*.08;
    this.data.qvel[0]=(this.rng.uniform()*2-1)*.08;this.data.qvel[1]=(this.rng.uniform()*2-1)*.08;this.data.ctrl[0]=0;this.data.time=0;
    this.sim.mujoco.mj_forward(this.sim.model,this.data);this.goal=(this.rng.uniform()*2-1)*0.8;this.refContext=[0,0];this.steps=0;return this.sim.getState(this.data);
  }
  state(){return this.sim.getState(this.data);}
  reference(){return planReference(this.refContext,this.goal);}
  impulse({xDotDelta=.65,thetaDotDelta=-1.0}={}){
    finiteNumber(xDotDelta,"xDotDelta");finiteNumber(thetaDotDelta,"thetaDotDelta");
    this.data.qvel[0]+=xDotDelta;this.data.qvel[1]+=thetaDotDelta;
    this.sim.mujoco.mj_forward(this.sim.model,this.data);
  }
  step(action){
    finiteNumber(action,"action");
    this.data.ctrl[0]=clamp(action,-1,1);
    // 50 Hz policy/control rate over native MuJoCo 10 ms physics.
    for(let i=0;i<CONTROL_SUBSTEPS;i++)this.sim.mujoco.mj_step(this.sim.model,this.data);
    this.steps++;
    this.refContext=advancePlannerContext(this.refContext,this.goal,CONTROL_DT);
    const s=this.state(),targetX=this.refContext[0],done=doneFor(s)||this.steps>=JS_TIMEOUT_STEPS,r=rewardFor(s,targetX,action);
    return{s,r,targetX,done,terminated:doneFor(s)};
  }
  delete(){this.data.delete();}
}

export function gae(data,n,gamma=.99,lambda=.95){
  const carry=new Float64Array(n);
  for(let i=data.length-1;i>=0;i--){const q=data[i];q.delta=q.r+gamma*(q.terminated?0:q.nextV)-q.oldV;q.rawAdv=q.delta+gamma*lambda*(q.done?0:carry[q.env]);q.ret=q.rawAdv+q.oldV;carry[q.env]=q.rawAdv;}
  const m=mean(data.map(q=>q.rawAdv)),sd=Math.sqrt(mean(data.map(q=>(q.rawAdv-m)**2))+1e-8);for(const q of data)q.adv=(q.rawAdv-m)/(sd+1e-8);
}

