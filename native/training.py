#!/usr/bin/env python3
"""Run the original SONIC PPO trainer on a small, explicit MuJoCo CartPole task.

No copied/overridden PPO, GAE, trainer loop, policy, critic or optimizer algorithms.
This bounded integration check is not a learned tracking-performance benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

from core_smoke import UPSTREAM_COMMIT, configuration, git, import_native, validate_upstream

TRAINING_SOURCES = (
    'gear_sonic/trl/trainer/ppo_trainer.py',
    'gear_sonic/trl/trainer/ppo_trainer_aux_loss.py',
    'gear_sonic/trl/modules/actor_critic_modules.py',
    'gear_sonic/trl/modules/data_utils.py',
    'gear_sonic/trl/utils/rl.py',
    'gear_sonic/trl/utils/scheduler.py',
    'gear_sonic/trl/callbacks/hv_callback_handler.py',
    'gear_sonic/utils/average_meters.py',
    'gear_sonic/utils/batch_normalizer.py',
    'gear_sonic/utils/running_mean_std.py',
    'gear_sonic/config/actor_critic/critics/mlp.yaml',
)


def verify_sources(root):
    hashes=validate_upstream(root)
    for name in TRAINING_SOURCES:
        data=(root/name).read_bytes()
        if data!=git(root,'show',f'{UPSTREAM_COMMIT}:{name}'):
            raise RuntimeError('Modified upstream source: '+name)
        hashes[name]=hashlib.sha256(data).hexdigest()
    return hashes


def build_models(root,env):
    from omegaconf import OmegaConf
    from gear_sonic.trl.modules.actor_critic_modules import Actor, Critic
    from gear_sonic.trl.utils.common import custom_instantiate
    algo=OmegaConf.create({'init_noise_std':.12,'use_log_std':True,'freeze_noise_std':True,
        'num_steps_per_env':128,'entropy_coef':.0,'desired_kl':None,
        'compute_aux_loss':True,'aux_loss_scale':1.0,'use_padding_mask':False})
    backbone=configuration(root,2,2,5)
    backbone['_target_']='gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule'
    for key in ['env_config','algo_config','obs_dim_dict']:del backbone[key]
    backbone.aux_loss_func={'cartpole_reconstruction':{'_target_':'cartpole_env.CartPoleReconstructionLoss'}}
    backbone.aux_loss_coef={'cartpole_reconstruction':.1}
    policy=Actor(env.config,algo,backbone,input_obs_dict=True,has_aux_loss=True,running_mean_std=False)
    critic_cfg=OmegaConf.load(root/'gear_sonic/config/actor_critic/critics/mlp.yaml')
    critic_cfg.running_mean_std=False
    critic_cfg.backbone.module_config_dict.layer_config.hidden_dims=[32,32]
    critic=custom_instantiate(critic_cfg,env_config=env.config,algo_config=algo,_resolve=False)
    if type(policy) is not Actor or type(critic) is not Critic:
        raise RuntimeError('Expected original official Actor and Critic')
    return policy,critic,algo


def evaluate(policy,push=False):
    import numpy as np
    import torch
    from cartpole_env import CartPoleEnv
    env=CartPoleEnv(num_envs=8,seed=8080)
    obs=env.observations()
    alive=np.ones(8,dtype=bool)
    errors=np.zeros(8)
    steps=np.zeros(8,dtype=int)
    push_count=0
    # Record only each fixed initial episode; auto-resets of failed lanes are ignored.
    for step in range(500):
        if push and step==120:
            for i in np.flatnonzero(alive):
                env.data[i].qvel[0]+=.65
                env.data[i].qvel[1]-=1.
                push_count+=1
            obs=env.observations()
        with torch.no_grad():
            means=policy({k:v.unsqueeze(1) for k,v in obs.items()})[:,0]
        obs,_,done,_=env.step({'actions':means})
        errors[alive]+=np.abs(env.last_step_states[alive,0]-env.last_targets[alive])
        steps[alive]+=1
        alive&=~done.numpy()
        if not alive.any():break
    return {'episodes':8,'seed':8080,'horizon_steps':500,'survived':int(alive.sum()),
            'mean_steps':float(steps.mean()),
            'all_executed_step_mae':float((errors/np.maximum(steps,1)).mean()),
            'completed_episode_mae':float((errors[alive]/steps[alive]).mean()) if alive.any() else None,
            'push_reached_episodes':push_count if push else None,
            'interpretation':'MAE before failure can look small; survival is reported separately. Evaluation horizon is not a training timeout.'}


def run_training(upstream:Path,output_dir:Path,iterations=2):
    if not isinstance(iterations,int) or not 1<=iterations<=10:
        raise ValueError('Bounded integration run requires 1..10 iterations')
    root=upstream.resolve(strict=True)
    provenance=verify_sources(root)
    import_native(root)
    import numpy as np
    import torch
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from loguru import logger
    from trl.experimental.ppo import PPOConfig
    from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer
    from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
    from cartpole_env import CartPoleEnv
    for cls,name in [(TRLPPOTrainer,TRAINING_SOURCES[0]),(TRLAuxLossPPOTrainer,TRAINING_SOURCES[1])]:
        if Path(inspect.getfile(cls)).resolve()!=(root/name).resolve():
            raise RuntimeError('Trainer imported from wrong source')
    logger.disable('gear_sonic')
    torch.set_num_threads(1)
    torch.manual_seed(9101)
    np.random.seed(9101)
    output_dir.mkdir(parents=True,exist_ok=True)
    env=CartPoleEnv(num_envs=8,seed=9101)
    policy,critic,algo=build_models(root,env)
    before={'clean':evaluate(policy),'push':evaluate(policy,True)}
    modules={'encoder':policy.actor_module.encoders['cartpole'],
             'dynamic':policy.actor_module.decoders['g1_dyn'],
             'kinematic':policy.actor_module.decoders['cartpole_kin'],'critic':critic}
    original={k:torch.cat([p.detach().flatten().clone() for p in m.parameters()]) for k,m in modules.items()}
    args=PPOConfig(output_dir=str(output_dir/'trainer'),use_cpu=True,bf16=False,fp16=False,
        num_total_batches=iterations,num_ppo_epochs=2,num_mini_batches=2,
        per_device_train_batch_size=4,gradient_accumulation_steps=1,
        learning_rate=3e-4,gamma=.99,lam=.95,vf_coef=.5,max_grad_norm=1.,
        num_sample_generations=0,report_to='none',save_strategy='no',
        lr_scheduler_type='constant',gradient_checkpointing=False,disable_tqdm=True,
        seed=9101,push_to_hub=False)
    args.exp_name="cartpole-native-integration" # SONIC adds this field beyond HF PPOConfig
    accelerator=Accelerator(cpu=True,gradient_accumulation_steps=1)
    trainer=TRLAuxLossPPOTrainer(args,algo,env,policy,value_model=critic,
        accelerator=accelerator,local_seed=9101,log_dir=str(output_dir))
    if type(trainer) is not TRLAuxLossPPOTrainer:
        raise RuntimeError('Trainer must be the original class, not a subclass/replacement')
    # Passive profiling and the optimizer's supported hook measure execution;
    # no trainer method or source is changed.
    calls=Counter()
    loss_records=[]
    wanted={'train','_rollout_step','_compute_returns','_compute_ppo_loss','_compute_aux_loss','_gradient_clipping'}
    source_paths={str(root/TRAINING_SOURCES[i]) for i in (0,1)}
    def profile(frame,event,arg):
        if frame.f_code.co_filename not in source_paths:return
        name=frame.f_code.co_name
        if event=='call' and name in wanted:calls[name]+=1
        if event=='return' and name=='_compute_loss' and frame.f_code.co_filename==str(root/TRAINING_SOURCES[1]):
            if arg is not None:
                ppo=arg['ppo_loss_dict'];aux=arg['aux_loss_dict']
                loss_records.append({'total':float(arg['loss'].detach()),'policy':float(ppo['pg_loss'].detach()),
                    'value':float(ppo['vf_loss'].detach()),'auxiliary':float(aux['total_aux_loss'].detach())})
    steps=[]
    optimizer=trainer.optimizer.optimizer if hasattr(trainer.optimizer,'optimizer') else trainer.optimizer
    hook=optimizer.register_step_post_hook(lambda *unused:steps.append(1))
    previous_profile=sys.getprofile()
    try:
        sys.setprofile(profile)
        trainer.train() # Original complete rollout -> GAE -> PPO/aux -> optimizer loop
    finally:
        sys.setprofile(previous_profile)
        hook.remove()
    deltas={k:float(torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in m.parameters()])-original[k])) for k,m in modules.items()}
    after={'clean':evaluate(policy),'push':evaluate(policy,True)}
    checkpoint=output_dir/'native-weights.pt'
    torch.save({'policy':policy.state_dict(),'critic':critic.state_dict()},checkpoint)
    restored,restored_critic,_=build_models(root,env)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    restored.load_state_dict(saved['policy']);restored_critic.load_state_dict(saved['critic'])
    obs={k:v.unsqueeze(1) for k,v in env.observations().items()}
    with torch.no_grad():parity=float((policy(obs)-restored(obs)).abs().max())
    valid=(len(steps)==iterations*4 and env.control_transitions==iterations*8*128
           and len(loss_records)==len(steps) and all(np.isfinite(v) for row in loss_records for v in row.values())
           and all(np.isfinite(v) and v>0 for v in deltas.values()) and parity==0
           and all(calls[k]>0 for k in wanted))
    if not valid:raise RuntimeError('Native trainer integration measurement failed')
    if provenance!=verify_sources(root):raise RuntimeError('Upstream changed during training')
    report={'status':'PASS','scope':'native_trainer_physics_integration','upstream_commit':UPSTREAM_COMMIT,
        'upstream_clean':True,'source_sha256':provenance,
        'local_source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['training.py','cartpole_env.py','cartpole.xml']},
        'versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','mujoco','trl','transformers','accelerate','tensordict','vector-quantize-pytorch']},
        'trainer_class':TRLAuxLossPPOTrainer.__module__+'.'+TRLAuxLossPPOTrainer.__name__,
        'official_ppo_executed':True,'physical_rollout_executed':True,'official_method_calls':dict(calls),
        'iterations':iterations,'optimizer_steps':len(steps),'control_transitions':env.control_transitions,'physics_steps':env.physics_steps,
        'terminated_training_episodes':env.terminated_episodes,'parameter_delta_l2':deltas,
        'losses':loss_records,'losses_finite':True,'reconstruction_loss_observed':any(row['auxiliary']>0 for row in loss_records),
        'evaluation':{'before':before,'after':after},
        'checkpoint':{'kind':'weights-only; not exact-resume trainer state','action_parity_max_abs':parity,'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        'initialization':'random; no teacher, no stabilizing controller',
        'performance_claim':'not established by this integration run',
        'adaptations':{'environment':'custom MuJoCo CartPole environment; not Isaac Lab or a humanoid task',
            'auxiliary_loss':'local CartPole MSE binding through original SONIC auxiliary-loss extension; not G1 kinematic or multimodal alignment loss',
            'network':'official Actor/Critic/UniversalTokenModule/BaseModule/FSQ; two 2D tokens, five levels, reduced [32,32] MLP widths',
            'action':'original Gaussian sample retained for PPO; clip only physical input to [-1,1], force=10*input N',
            'timing':'MuJoCo dt=0.01 s, two physics steps per action, 50 Hz control',
            'termination':'physical |x|>1.78 or |theta|>0.65 only; no artificial training timeouts; rollout boundary bootstrapped by original trainer',
            'reference':'independent critically damped x/xd trajectory, goal sampled in [-0.4,0.4] m; no independent commanded pole angle'},
        'configuration':{'algo':OmegaConf.to_container(algo,resolve=True),'num_envs':8,'horizon':128,'epochs':2,'minibatches':2,'learning_rate':3e-4,'seed':9101}}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=2)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    try:
        report=run_training(Path(args.upstream),args.output_dir,args.iterations)
        code=0
    except Exception as exc:
        report={'status':'FAIL','scope':'native_trainer_physics_integration','error':str(exc)}
        code=1
        import traceback
        traceback.print_exc()
    with tempfile.NamedTemporaryFile(mode='w',dir=args.output_dir,delete=False) as out:
        json.dump(report,out,indent=2,allow_nan=False);out.write('\n');name=out.name
    os.replace(name,args.output_dir/'native_training.json')
    print(json.dumps({k:report[k] for k in ['status','scope','optimizer_steps','control_transitions','error'] if k in report}))
    return code

if __name__=='__main__':raise SystemExit(main())
