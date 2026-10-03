export class FrozenCartPoleTeacher {
  constructor(spec){
    if(spec?.schema!=="tinmanlab-cartpole-teacher/v1") throw new Error("unsupported teacher schema");
    this.spec=spec;this.actor=spec.actor;
  }
  forward(state,goal){
    const [x,v,t,w]=state;
    const obs=[(goal-x)/2.4,v/2.5,t/(Math.PI/15),w/2.5,goal/2.4];
    const m=this.actor,p=m.p,h=new Float64Array(m.h),y=new Float64Array(m.o);
    const b1=m.h*m.n,w2=b1+m.h,b2=w2+m.o*m.h;
    for(let i=0;i<m.h;i++){let z=p[b1+i];for(let j=0;j<m.n;j++)z+=p[i*m.n+j]*obs[j];h[i]=Math.tanh(z);}
    for(let i=0;i<m.o;i++){let z=p[b2+i];for(let j=0;j<m.h;j++)z+=p[w2+i*m.h+j]*h[j];y[i]=z;}
    const actionIndex=y[1]>=y[0]?1:0;
    const forceN=(actionIndex?1:-1)*(this.spec.source_force_n||8);
    return{obs,logits:Array.from(y),actionIndex,forceN,normalizedForce:forceN/10};
  }
  metadata(){
    return{
      source_repo:this.spec.source_repo,
      source_file:this.spec.source_file,
      source_iteration:this.spec.source_iteration,
      trained_profile:this.spec.trained_profile,
      source_force_n:this.spec.source_force_n,
      note:this.spec.note
    };
  }
}

export async function loadTeacherPolicy(url="./assets/teacher_cartpole_ppo.json"){
  const r=await fetch(url,{cache:"no-store"});
  if(!r.ok)throw new Error("teacher fetch failed: "+r.status);
  return new FrozenCartPoleTeacher(await r.json());
}
