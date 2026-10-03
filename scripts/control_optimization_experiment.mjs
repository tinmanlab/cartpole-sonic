import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {MuJoCoCartPole} from '../mujoco_sim.js';
import {FrozenCartPoleTeacher} from '../teacher_policy.js';
import {TemporalControlTrainer} from '../temporal_control_lab.js';

const root=new URL('../',import.meta.url);
const read=p=>JSON.parse(fs.readFileSync(new URL(p,root),'utf8'));
const hash=p=>createHash('sha256').update(fs.readFileSync(new URL(p,root))).digest('hex');
export const STUDY_SEEDS=[17011,27011,37011];
export const STUDY_ARMS=[
  {id:'one-default',label:'1 token · actor ×1',tokens:1,scale:1},
  {id:'two-default',label:'2 tokens · actor ×1',tokens:2,scale:1},
  {id:'one-actor-005',label:'1 token · actor ×0.05',tokens:1,scale:.05},
  {id:'two-actor-005',label:'2 tokens · actor ×0.05',tokens:2,scale:.05},
];
export function describe(values){
  const finite=values.filter(Number.isFinite),n=finite.length;
  const mean=n?finite.reduce((a,b)=>a+b,0)/n:null;
  return {n,mean,sampleStd:n>1?Math.sqrt(finite.reduce((s,x)=>s+(x-mean)**2,0)/(n-1)):null,min:n?Math.min(...finite):null,max:n?Math.max(...finite):null};
}
function evaluate(t,{cleanSeed=8181,pushSeed=9191}={}){
  const clean=t.evaluate({episodes:12,seed:cleanSeed});
  const push=t.evaluate({episodes:12,seed:pushSeed,disturbance:true});
  return {
    cleanMae:clean.trackingMae,cleanAllStepMae:clean.allStepTrackingMae,cleanSuccesses:clean.successes,
    pushMae:push.disturbance.postPushTrackingMae,pushSuccesses:push.successes,
    pushReachedEpisodes:push.disturbance.pushReachedEpisodes,recoveredEpisodes:push.disturbance.recoveredEpisodes,
    recoverySteps:push.disturbance.meanRecoverySteps,
  };
}
function diagnostics(t){
  if(!t.iter)return null;
  return {
    rolloutMae:t.last.tracking,clipFraction:t.last.clipFraction,objectiveClipFraction:t.last.objectiveClipFraction,
    actionSaturation:t.last.actionSaturation,sampleClipFraction:t.last.sampleClipFraction,
    postUpdateKl:t.last.postUpdateKl,postUpdateApproxKl:t.last.postUpdateApproxKl,
    tokenChangeFraction:t.last.tokenChangeFraction,actionMeanShiftRms:t.last.actionMeanShiftRms,
    gradEncoder:t.last.gradEncoder,gradDynamic:t.last.gradDynamic,updates:t.last.updates,
    maxKlAcrossIterations:Math.max(...t.history.map(h=>h.postUpdateKl)),
    meanSampleClipAcrossIterations:describe(t.history.map(h=>h.sampleClipFraction)).mean,
  };
}
function checkpoint(t,evalOptions){
  return {ppoIterations:t.iter,envSteps:t.envSteps,optimizerBatches:t.history.reduce((s,h)=>s+h.optimizerBatches,0),...evaluate(t,evalOptions),diagnostics:diagnostics(t)};
}
function trajectory(t,targets,evalOptions){
  const points=[];
  for(const target of targets){while(t.iter<target)t.iteration();points.push(checkpoint(t,evalOptions));}
  return points;
}

/** Run actual native-MuJoCo experiments. Never mutate stored evidence during verification. */
export async function runOptimizationExperiment({onProgress=()=>{}}={}){
  const bootstrap=read('assets/temporal_control_bootstrap.json');
  const teacher=new FrozenCartPoleTeacher(read('assets/teacher_cartpole_ppo.json'));
  const sim=new MuJoCoCartPole();await sim.init('playground');
  const sources={one:bootstrap.one,two:bootstrap.two};
  const variants=[],budgetTrajectories=[];
  const make=({tokens=2,seed=9118,widths={},scales={},source=null,paired=false})=>{
    const t=new TemporalControlTrainer(sim,{tokens,seed,widths,ppoModuleScales:scales});
    if(source){
      // Paired trials retain fresh environments/RNG. Only identical per-architecture weights and Adam moments are copied.
      if(paired){t.policy.restore(source.policy);t.bootstrap=source.bootstrap;}
      else t.restore(source);
    }
    return t;
  };
  const variant=({id,label,tokens=2,seed=9118,widths={},scales={},source=null,long=false})=>{
    const t=make({tokens,seed,widths,scales,source});
    try{
      if(!source)t.bootstrapFromTeacher(teacher,{steps:300,batch:256,lr:.0015,auxCoef:.08,dataSeed:0xC011AB});
      const points=trajectory(t,long?[0,10,20,50]:[0,10]);
      const after=points.find(p=>p.ppoIterations===10);
      variants.push({id,label,tokens,steps:10,parameterCount:t.policy.parameterCount(),parameters:t.policy.parameterBreakdown(),widths:{...t.policy.widths},ppoLrScale:1,ppoModuleScales:{...t.ppoModuleScales},before:points[0],after,diagnostics:after.diagnostics});
      if(long)budgetTrajectories.push({id,label,checkpoints:points});
      onProgress(id);
    }finally{t.delete();}
  };
  try{
    variant({id:'one-default',label:'1-token default',tokens:1,seed:9101,source:sources.one,long:true});
    variant({id:'two-default',label:'2-token default',source:sources.two,long:true});
    variant({id:'two-matched-capacity',label:'2-token matched TOTAL parameters',widths:{encoder:18,dynamic:26,kinematic:20,critic:20}});
    variant({id:'two-matched-actor',label:'2-token matched ACTOR parameters',widths:{encoder:22,dynamic:14,kinematic:24,critic:20}});
    variant({id:'two-freeze-encoder',label:'2-token freeze Encoder',source:sources.two,scales:{encoder:0}});
    for(const scale of [.25,.10,.05])variant({id:'two-actor-'+String(scale).replace('.',''),label:'2-token actor LR ×'+scale,source:sources.two,scales:{encoder:scale,dynamic:scale},long:scale===.05});
    variant({id:'one-actor-005',label:'1-token actor LR ×0.05',tokens:1,seed:9101,source:sources.one,scales:{encoder:.05,dynamic:.05},long:true});

    const runs=[];
    for(const seed of STUDY_SEEDS){
      const arms=[];
      for(const arm of STUDY_ARMS){
        const t=make({tokens:arm.tokens,seed,source:arm.tokens===1?sources.one:sources.two,scales:{encoder:arm.scale,dynamic:arm.scale},paired:true});
        try{
          arms.push({id:arm.id,label:arm.label,parameters:t.policy.parameterBreakdown(),checkpoints:trajectory(t,[0,10,50],{cleanSeed:18181,pushSeed:19191})});
        }finally{t.delete();}
      }
      runs.push({seed,arms});onProgress('paired PPO seed '+seed);
    }
    const summary=[];
    for(const arm of STUDY_ARMS)for(const iterations of [0,10,50]){
      const rows=runs.map(r=>r.arms.find(a=>a.id===arm.id).checkpoints.find(c=>c.ppoIterations===iterations));
      summary.push({id:arm.id,label:arm.label,ppoIterations:iterations,cleanMae:describe(rows.map(c=>c.cleanMae)),pushMae:describe(rows.map(c=>c.pushMae)),cleanSuccesses:rows.reduce((s,c)=>s+c.cleanSuccesses,0),pushSuccesses:rows.reduce((s,c)=>s+c.pushSuccesses,0),episodes:36});
    }
    const pairedDifferences=[];
    for(const suffix of ['default','actor-005'])for(const iterations of [10,50]){
      const differences=runs.map(r=>{
        const row=id=>r.arms.find(a=>a.id===id).checkpoints.find(c=>c.ppoIterations===iterations);
        const one=row('one-'+suffix).cleanMae,two=row('two-'+suffix).cleanMae;
        return {seed:r.seed,deltaTwoMinusOne:Number.isFinite(one)&&Number.isFinite(two)?two-one:null};
      });
      pairedDifferences.push({actorScale:suffix==='default'?1:.05,ppoIterations:iterations,differences,summary:describe(differences.map(d=>d.deltaTwoMinusOne))});
    }
    return {
      schema:'cartpole-sonic-control-optimization-ablation/v1',revision:2,ppoContract:'raw-gaussian-sample/v2',
      provenance:{physics:sim.backend,nodeMajor:Number(process.versions.node.split('.')[0]),sha256:Object.fromEntries(['sonic_toy.js','temporal_control_lab.js','scripts/control_optimization_experiment.mjs','assets/temporal_control_bootstrap.json','assets/teacher_cartpole_ppo.json'].map(p=>[p,hash(p)]))},
      protocol:{bootstrap:'300 steps × 256 teacher-imitation examples; fixed weights and Adam moments for each architecture',ppo:'8 env × 96 steps; 4 epochs, batch 128; action std 0.12; auxiliary coefficient 0.20',evaluation:'12 clean + 12 push episodes, 500 control steps each; failed-before-push has null post-push error',metricDefinitions:{updateNorm:'mean per-minibatch ||weights_after_Adam - weights_before_Adam||2, after clipping and moment preconditioning',postUpdateKl:'exact raw-Gaussian KL with fixed std, measured on the same collected states/references after all PPO epochs; not clipped-action KL',tokenChangeFraction:'fraction of fixed rollout references with at least one quantized scalar changed',clipFraction:'ratio outside [0.8,1.2], separate from advantage-dependent objective clipping',sampleClipFraction:'fraction of Gaussian samples clipped for physical actuation; not policy-mean saturation',recovery:'first tolerance entry, not sustained recovery; non-recovery capped at 380 after push',cleanMae:'episode-mean tracking MAE conditional on completed episodes; always interpret with survival counts'}},
      variants,budgetTrajectories,tunedTrajectory:budgetTrajectories.find(t=>t.id==='two-actor-005').checkpoints,
      pairedSeedStudy:{seeds:STUDY_SEEDS,modelInitializationReplicates:1,randomness:'paired PPO rollout/reset/shuffle seeds, not independent model initialization; trajectories can diverge after different actions or terminations',evaluationSeeds:{clean:18181,push:19191},budgets:[10,50],selection:'actor scales 1 and 0.05 declared before these trials, from the earlier exploratory sweep; no selection on these evaluation episodes',uncertainty:'sample standard deviation over three PPO runs conditional on fixed bootstrap; not a confidence interval and not a general token-ranking claim',runs,summary,pairedDifferences},
      conclusionsSupported:[
        'Total-parameter matching does not match actor parameter count or functional capacity. Actor-parameter matching also changes layer widths.',
        'A raw gradient norm times learning rate is not the actual Adam parameter displacement.',
        'Matched 10/50 budgets and paired PPO seeds permit conditional comparisons; inspect per-run outcomes and survival rather than require a winner.',
        'These are small fixed-bootstrap CartPole experiments. They neither isolate token count causally nor prescribe SONIC production hyperparameters.'
      ]
    };
  }finally{sim.dispose();}
}
