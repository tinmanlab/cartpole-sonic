#!/usr/bin/env python3
"""Bounded native learning experiment, not a new trainer or a performance guarantee."""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from core_smoke import import_native, UPSTREAM_COMMIT
from training import build_models, verify_sources
from cartpole_env import CartPoleEnv


def build_experiment_models(root,env,fsq_levels=5):
    """Reuse the established native builder; change only a parameter-free FSQ configuration."""
    if type(fsq_levels) is not int or fsq_levels not in (5,32):
        raise ValueError('Only the declared FSQ 5/32 comparison is supported')
    policy,critic,algo=build_models(root,env)
    if fsq_levels!=5:
        from vector_quantize_pytorch import FSQ
        module=policy.actor_module
        if type(module.quantizer) is not FSQ:
            raise RuntimeError('Expected original FSQ implementation')
        module.quantizer=FSQ(levels=[fsq_levels]*module.token_dim,return_indices=False)
        module.fsq_level_list=[fsq_levels]*module.token_dim
    return policy,critic,algo


def token_probe(policy):
    env=CartPoleEnv(65,seed=31)
    env.goals=np.linspace(-.4,.4,65);env.reference_time[:]=5.
    obs=env.observations();obs['actor_obs'].zero_()
    with torch.no_grad():
        out=policy.actor_module({k:v[:,None] for k,v in obs.items()},return_dict=True)
        q=out['encoded_tokens']['cartpole'].flatten(1)
        z=out['encoded_latents']['cartpole'];action=out['action_mean']
    count=len(torch.unique(q,dim=0));span=float(action.max()-action.min())
    return {'goals':65,'goal_range':[-.4,.4],'unique_token_vectors':count,
            'latent_min':float(z.min()),'latent_max':float(z.max()),
            'action_span_same_state':span,'reference_conditioned':bool(count>1 and span>1e-5),
            'interpretation':'Nonzero response is necessary but not sufficient for correct tracking.'}


def evaluate_policy(policy,*,seed,episodes=16,horizon=500,push=False,reference_mode='normal'):
    if reference_mode not in ['normal','zero','negate']:
        raise ValueError('Unknown reference intervention')
    if type(episodes) is not int or episodes<1 or type(horizon) is not int or horizon<1:
        raise ValueError('Positive episode count/horizon required')
    env=CartPoleEnv(episodes,seed=seed)
    goals=env.goals.copy();initial=env.states().copy();obs=env.observations()
    alive=np.ones(episodes,dtype=bool);errors=np.zeros(episodes);steps=np.zeros(episodes,dtype=int)
    late_errors=np.zeros(episodes);late_steps=np.zeros(episodes,dtype=int)
    post_errors=np.zeros(episodes);post_steps=np.zeros(episodes,dtype=int)
    pushed=np.zeros(episodes,dtype=bool);end_x=initial[:,0].copy()
    for step in range(horizon):
        if push and step==120:
            for i in np.flatnonzero(alive):
                env.data[i].qvel[0]+=.65;env.data[i].qvel[1]-=1.;pushed[i]=True
            obs=env.observations()
        inputs={k:v.unsqueeze(1) for k,v in obs.items()}
        if reference_mode=='zero':inputs['tokenizer']=torch.zeros_like(inputs['tokenizer'])
        if reference_mode=='negate':inputs['tokenizer']=-inputs['tokenizer']
        with torch.no_grad():action=policy(inputs)[:,0]
        obs,_,done,_=env.step({'actions':action})
        err=np.abs(env.last_step_states[:,0]-env.last_targets)
        errors[alive]+=err[alive];steps[alive]+=1;end_x[alive]=env.last_step_states[alive,0]
        if step>=max(0,horizon-100):
            late_errors[alive]+=err[alive];late_steps[alive]+=1
        post=alive&pushed;post_errors[post]+=err[post];post_steps[post]+=1
        alive&=~done.numpy()
        if not alive.any():break
    rows=[{'goal':float(goals[i]),'initial_state':initial[i].tolist(),'steps':int(steps[i]),
        'survived':bool(alive[i]),'executed_mae':float(errors[i]/max(1,steps[i])),
        'final_x':float(end_x[i]),'pushed':bool(pushed[i])} for i in range(episodes)]
    return {'seed':seed,'episodes':episodes,'horizon':horizon,'reference_mode':reference_mode,
        'survived':int(alive.sum()),'mean_steps':float(steps.mean()),
        'executed_mae':float(np.mean(errors/np.maximum(steps,1))),
        'completed_mae':float(np.mean(errors[alive]/steps[alive])) if alive.any() else None,
        'completed_tail_mae':float(np.mean(late_errors[alive]/late_steps[alive])) if alive.any() else None,
        'push_reached':int(pushed.sum()),'post_push_survived':int((pushed&alive).sum()),
        'post_push_mae':float(np.mean(post_errors[pushed]/np.maximum(post_steps[pushed],1))) if pushed.any() else None,
        'per_episode':rows,
        'metric_boundary':'Completed metrics are survivor-conditioned. Read with survival; missing push data are null, not zero.'}


def audit_policy(policy,seed,episodes=16,interventions=True):
    result={'token_probe':token_probe(policy),
            'clean':evaluate_policy(policy,seed=seed,episodes=episodes),
            'push':evaluate_policy(policy,seed=seed,episodes=episodes,push=True)}
    if interventions:
        result['zero_reference']=evaluate_policy(policy,seed=seed,episodes=episodes,reference_mode='zero')
        result['negated_reference']=evaluate_policy(policy,seed=seed,episodes=episodes,reference_mode='negate')
    return result


def load_policy(upstream,checkpoint):
    root=Path(upstream).resolve(strict=True);verify_sources(root);import_native(root)
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=data['experiment']
    policy,_,_=build_experiment_models(root,CartPoleEnv(1,seed=31),fsq_levels=config['fsq_levels'])
    policy.load_state_dict(data['policy']);policy.eval()
    return policy


def write_result(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as out:
        json.dump(result,out,indent=2,allow_nan=False);out.write('\n');temporary=out.name
    os.replace(temporary,path)


def run_experiment(upstream:Path,output_dir:Path,iterations=128,seed=9101,checkpoints=None,fsq_levels=5,learning_rate=3e-4):
    if type(iterations) is not int or not 1<=iterations<=512:
        raise ValueError('This bounded experiment permits 1..512 iterations')
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Seed must be a nonnegative uint32 integer')
    if type(learning_rate) not in (int,float) or not 0<float(learning_rate)<=.003:
        raise ValueError('Learning rate must be finite and in (0,0.003]')
    points=sorted(set(checkpoints if checkpoints is not None else [n for n in [16,64,iterations] if n<=iterations]))
    if any(type(n) is not int or n<1 or n>iterations for n in points):
        raise ValueError('Checkpoints must fall inside this run')
    root=upstream.resolve(strict=True);sources=verify_sources(root);import_native(root)
    local_sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['learning.py','training.py','cartpole_env.py','core_smoke.py','cartpole.xml']}
    from accelerate import Accelerator
    from transformers import TrainerCallback
    from trl.experimental.ppo import PPOConfig
    from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
    from loguru import logger
    logger.disable('gear_sonic');torch.set_num_threads(1);torch.manual_seed(seed)
    output_dir.mkdir(parents=True,exist_ok=True)
    env=CartPoleEnv(8,seed=seed);policy,critic,algo=build_experiment_models(root,env,fsq_levels)
    policy.eval()
    baseline={'iteration':0,**audit_policy(policy,8080,episodes=8,interventions=False)}
    history=[baseline];progress=output_dir/'progress.jsonl'
    progress.write_text(json.dumps(baseline,allow_nan=False)+'\n')
    metadata={'fsq_levels':fsq_levels,'learning_rate':learning_rate,'seed':seed}
    class Measurements(TrainerCallback):
        def on_step_end(self,args,state,control,model=None,**kwargs):
            if state.global_step not in points:return control
            previous=model.policy.training;model.policy.eval()
            row={'iteration':int(state.global_step),**audit_policy(model.policy,8080,episodes=8,interventions=False)}
            model.policy.train(previous)
            row['last_training_metrics']={k:v for k,v in state.log_history[-1].items()
                if isinstance(v,(float,int)) and np.isfinite(v)}
            history.append(row)
            torch.save({'policy':model.policy.state_dict(),'critic':model.value_model.state_dict(),'experiment':metadata},output_dir/f'weights-{state.global_step}.pt')
            with progress.open('a') as out:out.write(json.dumps(row,allow_nan=False)+'\n')
            return control
    args=PPOConfig(output_dir=str(output_dir/'trainer'),use_cpu=True,bf16=False,fp16=False,
        num_total_batches=iterations,num_ppo_epochs=2,num_mini_batches=2,
        per_device_train_batch_size=4,gradient_accumulation_steps=1,
        learning_rate=learning_rate,gamma=.99,lam=.95,vf_coef=.5,max_grad_norm=1.,
        num_sample_generations=0,report_to='none',save_strategy='no',lr_scheduler_type='constant',
        gradient_checkpointing=False,disable_tqdm=True,seed=seed,push_to_hub=False)
    args.exp_name='cartpole-native-learning'
    trainer=TRLAuxLossPPOTrainer(args,algo,env,policy,value_model=critic,
        accelerator=Accelerator(cpu=True),local_seed=seed,log_dir=str(output_dir),callbacks=[Measurements()])
    calls=Counter();method_codes={getattr(type(trainer),name).__code__:name for name in
        ['train','_rollout_step','_compute_returns','_compute_ppo_loss','_compute_aux_loss']}
    def profiler(frame,event,arg):
        if event=='call' and frame.f_code in method_codes:calls[method_codes[frame.f_code]]+=1
    optimizer=trainer.optimizer.optimizer if hasattr(trainer.optimizer,'optimizer') else trainer.optimizer
    updates=[];hook=optimizer.register_step_post_hook(lambda *args:updates.append(1));old=sys.getprofile()
    try:
        sys.setprofile(profiler)
        with (output_dir/'train.log').open('w') as log,contextlib.redirect_stdout(log):trainer.train()
    finally:
        sys.setprofile(old);hook.remove()
    if len(updates)!=iterations*4 or env.control_transitions!=iterations*1024:
        raise RuntimeError('Official trainer did not complete the requested budget')
    policy.eval();validation=audit_policy(policy,8080,episodes=8)
    checkpoint=output_dir/'weights-final.pt'
    torch.save({'policy':policy.state_dict(),'critic':critic.state_dict(),'experiment':metadata},checkpoint)
    restored=load_policy(root,checkpoint);obs={k:v[:,None] for k,v in env.observations().items()}
    with torch.no_grad():parity=float((policy(obs)-restored(obs)).abs().max())
    if parity!=0:raise RuntimeError('Checkpoint configuration/weight roundtrip failed')
    if verify_sources(root)!=sources:raise RuntimeError('Upstream source changed')
    report={'execution':'COMPLETE','performance_claim':'see survival and reference interventions; not implied by execution',
        'upstream_commit':UPSTREAM_COMMIT,'source_sha256':sources,'local_source_sha256':local_sources,'training_seed':seed,
        'iterations':iterations,'control_transitions':env.control_transitions,'optimizer_steps':len(updates),
        'official_method_calls':dict(calls),'physics_steps':env.physics_steps,
        'initialization':'random; no teacher or stabilizing controller','history':history,'validation':validation,
        'configuration':{**metadata,'tokens':2,'token_dim':2,'hidden_widths':[32,32],
            'num_envs':8,'horizon':128,'epochs':2,'minibatches':2},
        'checkpoint_action_parity_max_abs':parity,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    write_result(output_dir/'learning.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=128)
    parser.add_argument('--seed',type=int,default=9101)
    parser.add_argument('--fsq-levels',type=int,choices=[5,32],default=5)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    try:
        result=run_experiment(args.upstream,args.output_dir,iterations=args.iterations,seed=args.seed,
            fsq_levels=args.fsq_levels,learning_rate=args.learning_rate)
        print(json.dumps({'execution':result['execution'],'configuration':result['configuration'],
            'survival':result['validation']['clean']['survived'],'mean_steps':result['validation']['clean']['mean_steps'],
            'completed_mae':result['validation']['clean']['completed_mae'],
            'token_probe':result['validation']['token_probe']},allow_nan=False))
        return 0
    except Exception as exc:
        result={'execution':'FAILED','performance_claim':'none','error':str(exc)}
        write_result(args.output_dir/'learning.json',result);print(json.dumps(result));return 1

if __name__=='__main__':raise SystemExit(main())
