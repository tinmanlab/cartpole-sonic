#!/usr/bin/env python3
"""Original SONIC PPO refinement of the offline concept model on physical reference replay.

Environment adaptation and reconstruction binding are local; trainer/Actor/Critic
and UniversalTokenModule implementations remain the unmodified pinned originals.
"""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import mujoco
from omegaconf import OmegaConf

from cartpole_env import CartPoleEnv,STATE_SCALE
from core_smoke import import_native,digest_parameters
from training import verify_sources
from learning import write_result
from concepts import pack,make_model,windows,align_marker,replay,alignment_metrics,dataset_metrics


class ReplayTask:
    def __init__(self,bank,ids,num_envs=8,seed=6107):
        self.bank=bank;self.ids=np.asarray(ids,dtype=int)
        if not len(self.ids) or any(bank['lengths'][i]<=29 for i in self.ids):raise ValueError('complete reference needed')
        self.num_envs=num_envs;self.base=CartPoleEnv(num_envs,seed);self.rng=np.random.default_rng(seed)
        self.selected=np.zeros(num_envs,dtype=int);self.cursor=np.zeros(num_envs,dtype=int)
        self.transitions=0;self.completed_motions=0;self.failed_motions=0;self.use_symmetry=False
        self.config=OmegaConf.create({'num_envs':num_envs,'use_symmetry':False,
            'robot':{'actions_dim':1,'algo_obs_dim_dict':{'actor_obs':4,'tokenizer':82,'critic_obs':37}},
            'obs':{'obs_dict':{'actor_obs':['state'],'tokenizer':['paired_reference'],'critic_obs':['state_reference_clock']}},
            'rewards':{'num_critics':1}})
        self.reset_all()

    def _reset(self,j):
        i=int(self.rng.choice(self.ids));self.selected[j]=i;self.cursor[j]=0
        s=self.bank['states'][i,0].copy()
        # Reference is unchanged. Tiny actual-state perturbations make the task off-reference.
        s+=self.rng.uniform(-1,1,4)*[.008,.01,.003,.015]
        d=self.base.data[j];mujoco.mj_resetData(self.base.model,d);d.qpos[:]=s[[0,2]];d.qvel[:]=s[[1,3]]
        mujoco.mj_forward(self.base.model,d)

    def reset_all(self):
        for j in range(self.num_envs):self._reset(j)
        return self.observations()

    def observations(self):
        ref=np.stack([self.bank['states'][i,t+np.arange(8)*4] for i,t in zip(self.selected,self.cursor)])
        inputs=pack(ref,self.base.states(),0)
        clock=np.array([(self.bank['lengths'][i]-28-t)/(self.bank['lengths'][i]-28) for i,t in zip(self.selected,self.cursor)])
        actor=inputs['actor_obs'][:,0];tokenizer=inputs['tokenizer'][:,0]
        critic=torch.cat((actor,torch.tensor((ref/STATE_SCALE).reshape(self.num_envs,32),dtype=torch.float32),torch.tensor(clock[:,None],dtype=torch.float32)),dim=-1)
        return {'actor_obs':actor,'tokenizer':tokenizer,'critic_obs':critic}

    def set_is_evaluating(self,is_evaluating,log_info=False):
        self.is_evaluating=bool(is_evaluating)

    def step(self,policy_state_dict):
        _,_,physical_failure,_=self.base.step(policy_state_dict)
        self.cursor+=1;self.transitions+=self.num_envs
        target=np.array([self.bank['states'][i,t] for i,t in zip(self.selected,self.cursor)])
        error=self.base.last_step_states-target
        u=np.clip(policy_state_dict['actions'].detach().cpu().numpy()[:,0],-1,1)
        reward=1-3*error[:,0]**2-8*error[:,2]**2-.1*error[:,1]**2-.03*error[:,3]**2-.002*u**2
        end=self.cursor>=np.array([self.bank['lengths'][i]-28 for i in self.selected])
        failed=physical_failure.numpy();completed=end&~failed;done=end|failed
        self.completed_motions+=int(completed.sum());self.failed_motions+=int(failed.sum())
        for j in np.flatnonzero(done):self._reset(int(j))
        return self.observations(),torch.tensor(reward,dtype=torch.float32),torch.tensor(done),{
            'episode':{},'time_outs':torch.zeros(self.num_envs,dtype=torch.bool),
            'to_log':{'tracking_mae':torch.tensor(np.abs(error[:,0]),dtype=torch.float32)}}


class Reconstruction(torch.nn.Module):
    def forward(self,inputs):
        return torch.nn.functional.mse_loss(inputs['decoded_outputs']['motion_kin']['joint_reference'],inputs['tokenizer_obs']['joint_reference'])


def native_backbone(env_config,algo_config,obs_dim_dict,module_dim_dict,upstream):
    # Configuration factory returns the ORIGINAL UniversalTokenModule, not a wrapper model.
    if env_config.robot.actions_dim!=1 or obs_dim_dict['actor_obs']!=4:raise ValueError('CartPole adapter dimensions mismatch')
    model=make_model(Path(upstream))
    model.aux_loss_func=torch.nn.ModuleDict({'motion_reconstruction':Reconstruction()})
    model.aux_loss_coef={'motion_reconstruction':.1}
    return model


def run(upstream,concept_dir,output_dir,iterations=192):
    if type(iterations) is not int or not 1<=iterations<=256:raise ValueError('budget must be 1..256')
    upstream=Path(upstream).resolve(strict=True);source=verify_sources(upstream);import_native(upstream)
    concept_dir=Path(concept_dir);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    from accelerate import Accelerator
    from trl.experimental.ppo import PPOConfig
    from gear_sonic.trl.modules.actor_critic_modules import Actor
    from gear_sonic.trl.utils.common import custom_instantiate
    from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
    from loguru import logger
    logger.disable('gear_sonic');torch.set_num_threads(1);torch.manual_seed(91071)
    with np.load(concept_dir/'motions.npz',allow_pickle=False) as data:bank={k:data[k].copy() for k in data.files}
    env=ReplayTask(bank,list(range(24)))
    algo=OmegaConf.create({'init_noise_std':.12,'use_log_std':True,'freeze_noise_std':True,
        'num_steps_per_env':96,'entropy_coef':0.,'desired_kl':None,'compute_aux_loss':True,'aux_loss_scale':1.,'use_padding_mask':False})
    backbone=OmegaConf.create({'_target_':'concept_control.native_backbone','upstream':str(upstream)})
    policy=Actor(env.config,algo,backbone,input_obs_dict=True,has_aux_loss=True,running_mean_std=False)
    saved=torch.load(concept_dir/'concept-weights.pt',weights_only=True,map_location='cpu')
    policy.actor_module.load_state_dict(saved['model'])
    # On-policy refinement updates the joint path and shared decoders, not the marker encoder.
    for p in policy.actor_module.parameters():p.requires_grad_(True)
    for p in policy.actor_module.encoders['markers'].parameters():p.requires_grad_(False)
    cfg=OmegaConf.load(upstream/'gear_sonic/config/actor_critic/critics/mlp.yaml');cfg.running_mean_std=False
    cfg.backbone.module_config_dict.layer_config.hidden_dims=[64,64]
    critic=custom_instantiate(cfg,env_config=env.config,algo_config=algo,_resolve=False)
    args=PPOConfig(output_dir=str(output_dir/'trainer'),use_cpu=True,bf16=False,fp16=False,
        num_total_batches=iterations,num_ppo_epochs=2,num_mini_batches=2,per_device_train_batch_size=4,
        learning_rate=.001,gamma=.99,lam=.95,vf_coef=.5,max_grad_norm=1.,num_sample_generations=0,
        report_to='none',save_strategy='no',lr_scheduler_type='constant',gradient_checkpointing=False,disable_tqdm=True,seed=91071,push_to_hub=False)
    args.exp_name='sonic-physical-motion-replay'
    trainer=TRLAuxLossPPOTrainer(args,algo,env,policy,value_model=critic,accelerator=Accelerator(cpu=True),local_seed=91071,log_dir=str(output_dir))
    calls=Counter();codes={getattr(type(trainer),n).__code__:n for n in ['train','_rollout_step','_compute_returns','_compute_ppo_loss','_compute_aux_loss']}
    def profiler(frame,event,arg):
        if event=='call' and frame.f_code in codes:calls[codes[frame.f_code]]+=1
    updates=[];optimizer=trainer.optimizer.optimizer if hasattr(trainer.optimizer,'optimizer') else trainer.optimizer
    hook=optimizer.register_step_post_hook(lambda *args:updates.append(1));old=sys.getprofile()
    before=digest_parameters(policy.actor_module)
    try:
        sys.setprofile(profiler)
        with (output_dir/'train.log').open('w') as log,contextlib.redirect_stdout(log):trainer.train()
    finally:sys.setprofile(old);hook.remove()
    if len(updates)!=iterations*4 or env.transitions!=iterations*8*96:raise RuntimeError('native budget mismatch')
    if before==digest_parameters(policy.actor_module):raise RuntimeError('no policy update')
    model=policy.actor_module;model.eval()
    heldout=windows(bank,range(24,32));training=windows(bank,range(24))
    alignment_before=alignment_metrics(model,heldout)
    marker_fit=align_marker(model,training,steps=900)
    alignment_after=alignment_metrics(model,heldout)
    outcomes={}
    for route,name in [(0,'joints'),(1,'markers')]:
        outcomes[name]={'clean':replay(model,bank,list(range(24,32)),route),'push':replay(model,bank,list(range(24,32)),route,push=True)}
    # Interventions preserve immutable true target, including endpoint for the interior reversal.
    outcomes['current_only']=replay(model,bank,list(range(24,32)),0,frame_mode='current_only')
    outcomes['reverse_interior']=replay(model,bank,list(range(24,32)),0,frame_mode='reverse_future')
    checkpoint=output_dir/'refined-concept-weights.pt';torch.save({'model':model.state_dict(),'config':saved['config']},checkpoint)
    report={'schema':'sonic-replay-refinement/v1','execution':'COMPLETE','original_trainer':type(trainer).__module__+'.'+type(trainer).__name__,
        'iterations':iterations,'control_transitions':env.transitions,'optimizer_steps':len(updates),'official_method_calls':dict(calls),
        'failed_training_motions':env.failed_motions,'completed_training_motions':env.completed_motions,
        'source_sha256':source,'local_source_sha256':{f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['concept_control.py','concepts.py','cartpole_env.py']},
        'initialization':'offline concept imitation model; not the original teacher-free PPO experiment',
        'task':'finite recorded-motion episodes; recorded completion is true terminal, not a continuing-task time-limit; critic receives time-to-go',
        'training_episode_ids':list(range(24)),'evaluation_episode_ids':list(range(24,32)),
        'evaluation_boundary':'eight episodes excluded from gradients, previously used to diagnose offline failure; not a pristine final holdout',
        'configuration':{'envs':8,'horizon':96,'learning_rate':.001,'epochs':2,'minibatches':2,'seed':91071,'reconstruction_coef':.1},
        'marker_refit':marker_fit,'marker_before_refit':alignment_before,'marker_after_refit':alignment_after,
        'outcomes':outcomes,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'boundaries':['One training seed and short replay tasks; no performance guarantee.',
                      'No camera, human retargeting, multi-actuator contact or cross-robot weight-transfer validation.',
                      'Both feature branches call the same original dynamic decoder; marker refit leaves it frozen.']}
    if source!=verify_sources(upstream):raise RuntimeError('upstream changed')
    write_result(output_dir/'refinement.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    p.add_argument('--concept-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--iterations',type=int,default=192)
    a=p.parse_args()
    try:
        r=run(a.upstream,a.concept_dir,a.output_dir,a.iterations);print(json.dumps(r,allow_nan=False));return 0
    except Exception as e:
        import traceback;traceback.print_exc();write_result(a.output_dir/'refinement.json',{'execution':'FAILED','error':str(e)});return 1

if __name__=='__main__':raise SystemExit(main())
