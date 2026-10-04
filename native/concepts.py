#!/usr/bin/env python3
"""Small SONIC concept experiments, separate from the established PPO controller.

Uses unmodified UniversalTokenModule/BaseModule/FSQ. Local contributions are
CartPole data representations, offline imitation/alignment losses and audits.
Offline training is NOT called official SONIC PPO or humanoid multimodal training.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from cartpole_env import CartPoleEnv, STATE_SCALE
from core_smoke import configuration, import_native, digest_parameters, UPSTREAM_COMMIT
from learning import load_policy, write_result
from training import verify_sources

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
OFFSETS=np.arange(8)*4  # t+0, .08, ..., .56 seconds. First frame is CURRENT reference.
MARKER_SCALE=np.array([1.8,2.8,1.,3.,7.,4.])


def to_markers(states):
    """Cart x, pole-tip x/z (z relative to pivot height), and their velocities.

    This is exact coordinate data, not images/estimated keypoints. Pole length=1 m.
    """
    s=np.asarray(states,dtype=np.float64)
    if s.shape[-1]!=4 or not np.isfinite(s).all():raise ValueError('finite state4 required')
    x,v,theta,w=np.moveaxis(s,-1,0)
    return np.stack((x,x+np.sin(theta),np.cos(theta),v,v+np.cos(theta)*w,-np.sin(theta)*w),axis=-1)


def from_markers(markers):
    m=np.asarray(markers,dtype=np.float64)
    if m.shape[-1]!=6 or not np.isfinite(m).all():raise ValueError('finite marker6 required')
    x,tx,tz,v,tvx,tvz=np.moveaxis(m,-1,0)
    if not np.allclose((tx-x)**2+tz**2,1.,atol=1e-7):raise ValueError('marker violates pole length')
    theta=np.arctan2(tx-x,tz)
    w=np.cos(theta)*(tvx-v)-np.sin(theta)*tvz
    return np.stack((x,v,theta,w),axis=-1)


def make_model(upstream,seed=7101):
    verify_sources(upstream);Universal,_,_=import_native(upstream)
    torch.manual_seed(seed)
    cfg=configuration(upstream,tokens=2,dim=4,levels=32)
    encoder=copy.deepcopy(cfg.encoders.cartpole)
    encoder.inputs=['joint_reference']
    marker=copy.deepcopy(encoder);marker.inputs=['marker_reference']
    cfg.encoders={'joints':encoder,'markers':marker}
    cfg.encoder_sample_probs={'joints':.5,'markers':.5}
    cfg.env_config.obs.group_obs_dims.tokenizer={'joint_reference':[8,4],'marker_reference':[8,6],'encoder_index':[2]}
    cfg.env_config.obs.group_obs_names.tokenizer=['joint_reference','marker_reference','encoder_index']
    kin=copy.deepcopy(cfg.decoders.cartpole_kin);kin.outputs=['joint_reference']
    cfg.decoders={'g1_dyn':cfg.decoders.g1_dyn,'motion_kin':kin}
    for net in [*cfg.encoders.values(),*cfg.decoders.values()]:
        net.params.module_config_dict.layer_config.hidden_dims=[64,64]
    return Universal(**cfg).cpu()


def pack(reference,proprioception,route):
    ref=np.asarray(reference,dtype=np.float64);state=np.asarray(proprioception,dtype=np.float64)
    if ref.ndim!=3 or ref.shape[1:]!=(8,4) or state.shape!=(len(ref),4):raise ValueError('expected [B,8,4] reference and [B,4] state')
    if not np.isfinite(ref).all() or not np.isfinite(state).all():raise ValueError('non-finite inputs')
    if isinstance(route,(int,np.integer)):
        route=np.full(len(ref),route)
    route=np.asarray(route)
    if route.shape!=(len(ref),) or not np.isin(route,[0,1]).all():raise ValueError('one valid encoder per sample required')
    selector=np.eye(2)[route.astype(int)]
    data=np.concatenate(((ref/STATE_SCALE).reshape(len(ref),32),(to_markers(ref)/MARKER_SCALE).reshape(len(ref),48),selector),axis=-1)
    return {'actor_obs':torch.tensor(np.clip(state/STATE_SCALE,-1.5,1.5)[:,None],dtype=torch.float32),
            'tokenizer':torch.tensor(data[:,None],dtype=torch.float32)}


def intervene(reference,mode):
    ref=np.array(reference,copy=True)
    if mode=='normal':return ref
    if mode in ('current_only','repeat_first'):return np.repeat(ref[:,:1],8,axis=1)
    if mode=='reverse_future':
        ref[:,1:-1]=ref[:,1:-1][:,::-1]
        return ref  # current and endpoint retained; corrupted input, not a feasible new motion
    if mode=='endpoints_only':
        alpha=np.linspace(0,1,8)[None,:,None]
        return ref[:,:1]*(1-alpha)+ref[:,-1:] * alpha
    raise ValueError('unknown frame intervention')


def physics_forks():
    """Two actually executed futures from the same state, under different bounded inputs."""
    env=CartPoleEnv(2,11)
    for d in env.data:
        d.qpos[:]=[0,.02];d.qvel[:]=0;mujoco.mj_forward(env.model,d)
    initial=env.states().copy();states=[initial.copy()]
    for step in range(28):
        u=np.array([.025,-.025]) if step<8 else np.zeros(2)
        _,_,done,_=env.step({'actions':torch.tensor(u[:,None],dtype=torch.float32)})
        if done.any():raise RuntimeError('fork fixture failed physically')
        states.append(env.last_step_states.copy())
    traces=np.stack(states,axis=1)
    return {'source':'native MuJoCo stepped with recorded bounded inputs','initial_states':initial,
            'reference':traces[:,OFFSETS],'first_actions':np.array([.025,-.025]),'physics_steps':env.physics_steps}


def desired(times,amplitude,kind):
    """Diverse commands for the *frozen data-generating controller*. Only measured
    resulting state/action trajectories, not these requested curves, enter the bank.
    """
    t=np.asarray(times);a=np.asarray(amplitude)
    k=2.5;d=np.exp(-k*t)
    rise=1-(1+k*t)*d;rise_v=k*k*t*d
    if kind==0:return a*rise,a*rise_v
    if kind==1:
        w=.9
        return a*rise*np.sin(w*t),a*(rise_v*np.sin(w*t)+rise*w*np.cos(w*t))
    if kind==2:
        delay=np.maximum(t-2.,0);d2=np.exp(-k*delay)
        return a*(rise-2*(1-(1+k*delay)*d2)),a*(rise_v-2*k*k*delay*d2)
    decay=np.exp(-.45*t)
    return a*rise*decay,a*(rise_v-.45*rise)*decay


def motion_bank(upstream,episodes=32,steps=300):
    if type(episodes) is not int or episodes<2 or type(steps) is not int or steps<40:raise ValueError('invalid bank size')
    policy=load_policy(upstream,REPO/'evidence/native_learning/weights-final.pt')
    env=CartPoleEnv(episodes,50191)
    rng=np.random.default_rng(8701);amp=rng.uniform(.12,.3,episodes)*rng.choice([-1,1],episodes)
    states=np.zeros((episodes,steps+1,4));actions=np.zeros((episodes,steps));alive=np.ones(episodes,dtype=bool)
    lengths=np.zeros(episodes,dtype=int);states[:,0]=env.states()
    families=np.arange(episodes)%4
    with torch.no_grad():
        for t in range(steps):
            obs=env.observations();reference=np.zeros((episodes,8,2))
            times=t*.02+np.arange(1,9)*.08
            for i in range(episodes):
                x,v=desired(times,amp[i],int(families[i]));reference[i,:,0]=x/1.8;reference[i,:,1]=v/3.
            obs['tokenizer']=torch.tensor(reference.reshape(episodes,16),dtype=torch.float32)
            u=policy({k:v[:,None] for k,v in obs.items()})[:,0]
            _,_,done,_=env.step({'actions':u})
            actions[alive,t]=u.numpy()[alive,0];states[alive,t+1]=env.last_step_states[alive]
            lengths[alive]+=1;alive&=~done.numpy()
    return {'states':states,'actions':actions,'lengths':lengths,'family':families,'amplitude':amp,
            'source':'recorded actual MuJoCo transitions under frozen native learned policy; changed commands are not certified as perfectly tracked',
            'checkpoint_sha256':hashlib.sha256((REPO/'evidence/native_learning/weights-final.pt').read_bytes()).hexdigest(),
            'plant_sha256':hashlib.sha256((HERE/'cartpole.xml').read_bytes()).hexdigest()}


def windows(bank,ids):
    refs=[];state=[];actions=[];episode=[];ticks=[]
    for i in ids:
        # Never pad missing future frames or cross terminal/reset boundaries.
        for t in range(0,int(bank['lengths'][i])-int(OFFSETS[-1]),3):
            refs.append(bank['states'][i,t+OFFSETS]);state.append(bank['states'][i,t]);actions.append(bank['actions'][i,t]);episode.append(i);ticks.append(t)
    if not refs:raise ValueError('no complete physical windows')
    return {'reference':np.array(refs),'state':np.array(state),'action':np.array(actions),
            'episode_id':np.array(episode),'tick':np.array(ticks)}


def train_anchor(model,data,steps=900,seed=701,reference_mode='normal',aux_coef=.5):
    for name,net in model.encoders.items():
        for p in net.parameters():p.requires_grad_(name=='joints')
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    rng=np.random.default_rng(seed);losses=[]
    for _ in range(steps):
        idx=rng.integers(0,len(data['state']),64)
        original=data['reference'][idx];ref=intervene(original,reference_mode)
        output=model(pack(ref,data['state'][idx],0),return_dict=True)
        action=output['action_mean'][:,0,0];target=torch.tensor(data['action'][idx],dtype=torch.float32)
        reconstruction=output['decoded_outputs']['motion_kin']['joint_reference'][:,0]
        target_ref=torch.tensor(original/STATE_SCALE,dtype=torch.float32)
        action_loss=torch.nn.functional.mse_loss(action,target)
        recon_loss=torch.nn.functional.mse_loss(reconstruction,target_ref)
        loss=action_loss+aux_coef*recon_loss
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        losses.append(float(loss.detach()))
    return {'steps':steps,'batch':64,'learning_rate':.001,'aux_coef':aux_coef,'first_loss':losses[0],'last_loss':losses[-1],
            'reference_mode':reference_mode,'method':'local offline imitation and reconstruction with official neural modules, not PPO'}


def freeze_anchor(model):
    for p in model.parameters():p.requires_grad_(False)
    for p in model.encoders['markers'].parameters():p.requires_grad_(True)
    return ['encoders.joints','decoders','quantizer']


def alignment_step(model,optimizer,ref,state):
    with torch.no_grad():anchor=model(pack(ref,state,0),return_dict=True)
    student=model(pack(ref,state,1),return_dict=True)
    z=student['encoded_latents']['markers'];target=anchor['encoded_latents']['joints']
    latent=torch.nn.functional.mse_loss(z,target)
    action=torch.nn.functional.mse_loss(student['action_mean'],anchor['action_mean'])
    loss=latent+action
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.encoders['markers'].parameters(),1.);optimizer.step()
    return float(loss.detach())


def align_marker(model,data,steps=900):
    freeze_anchor(model);before=digest_parameters(model.decoders);encoder_before=digest_parameters(model.encoders['joints'])
    optimizer=torch.optim.Adam(model.encoders['markers'].parameters(),lr=.001)
    rng=np.random.default_rng(8071);losses=[]
    for _ in range(steps):
        idx=rng.integers(0,len(data['state']),64)
        losses.append(alignment_step(model,optimizer,data['reference'][idx],data['state'][idx]))
    if digest_parameters(model.decoders)!=before or digest_parameters(model.encoders['joints'])!=encoder_before:raise RuntimeError('frozen shared anchor changed')
    return {'steps':steps,'decoder_unchanged':True,'joint_encoder_unchanged':True,'first_loss':losses[0],'last_loss':losses[-1],
            'method':'local paired latent/action MSE; not G1/SMPL loss and not the official PPO trainer'}


def dataset_metrics(model,data,route=0,mode='normal'):
    ref=intervene(data['reference'],mode)
    with torch.no_grad():out=model(pack(ref,data['state'],route),return_dict=True)
    action=out['action_mean'][:,0,0].numpy();recon=out['decoded_outputs']['motion_kin']['joint_reference'][:,0].numpy()*STATE_SCALE
    return {'samples':len(action),'route':int(route),'frame_mode':mode,
            'action_rmse_normalized':float(np.sqrt(np.mean((action-data['action'])**2))),
            'action_rmse_newtons':float(10*np.sqrt(np.mean((action-data['action'])**2))),
            'reconstruction_mae_state4':np.mean(np.abs(recon-data['reference']),axis=(0,1)).tolist()}


def alignment_metrics(model,data):
    with torch.no_grad():
        a=model(pack(data['reference'],data['state'],0),return_dict=True)
        b=model(pack(data['reference'],data['state'],1),return_dict=True)
    za=a['encoded_latents']['joints'];zb=b['encoded_latents']['markers']
    qa=a['encoded_tokens']['joints'];qb=b['encoded_tokens']['markers']
    return {'latent_mse':float((za-zb).square().mean()),'exact_token_vector_agreement':float((qa==qb).flatten(1).all(1).float().mean()),
            'scalar_agreement':float((qa==qb).float().mean()),'action_disagreement_rmse_newtons':float(10*(a['action_mean']-b['action_mean']).square().mean().sqrt()),
            'warning':'exact token equality is neither the only criterion nor a guarantee of control quality'}


def replay(model,bank,ids,route=0,push=False,frame_mode='normal'):
    env=CartPoleEnv(len(ids),9081)
    for j,i in enumerate(ids):
        s=bank['states'][i,0];d=env.data[j];d.qpos[:]=s[[0,2]];d.qvel[:]=s[[1,3]];mujoco.mj_forward(env.model,d)
    alive=np.ones(len(ids),dtype=bool);counts=np.zeros(len(ids),dtype=int);errors=np.zeros(len(ids));pushed=np.zeros(len(ids),dtype=bool)
    horizons=np.array([min(250,int(bank['lengths'][i])-28) for i in ids])
    for t in range(int(max(horizons))):
        active=alive&(t<horizons)
        if not active.any():break
        if push and t==70:
            for j in np.flatnonzero(active):env.data[j].qvel[0]+=.2;env.data[j].qvel[1]-=.3;pushed[j]=True
        ref=np.stack([bank['states'][i,min(t,max(0,int(bank['lengths'][i])-29))+OFFSETS] for i in ids])
        actual=env.states().copy()
        with torch.no_grad():action=model(pack(intervene(ref,frame_mode),actual,route))[:,0]
        _,_,done,_=env.step({'actions':action})
        # Immutable external prerecorded target, never the current simulator's future state.
        target=np.array([bank['states'][i,min(t+1,int(bank['lengths'][i])),0] for i in ids])
        errors[active]+=np.abs(env.last_step_states[active,0]-target[active]);counts[active]+=1
        alive[active]&=~done.numpy()[active]
    completed=counts==horizons
    completed&=alive
    return {'route':route,'frame_mode':frame_mode,'episodes':len(ids),'completed':int(completed.sum()),
            'horizons':horizons.tolist(),'executed_mae':float(np.mean(errors/np.maximum(counts,1))),
            'completed_mae':float(np.mean(errors[completed]/counts[completed])) if completed.any() else None,
            'mean_steps':float(counts.mean()),'push_reached':int(pushed.sum()),
            'reference_source':'immutable prerecorded physical trajectory; current proprioception is measured live',
            'claim':'replay-conditioned evaluation only; no online planner or formal stability claim'}


def run(upstream,output_dir,steps=900):
    if type(steps) is not int or not 1<=steps<=1500:raise ValueError('offline budget must be 1..1500')
    upstream=Path(upstream).resolve(strict=True);sources=verify_sources(upstream)
    torch.set_num_threads(1)
    from loguru import logger
    logger.disable('gear_sonic')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    bank=motion_bank(upstream);train=windows(bank,range(24));heldout=windows(bank,range(24,32))
    np.savez_compressed(output_dir/'motions.npz',**{k:bank[k] for k in ['states','actions','lengths','family','amplitude']})
    anchor=make_model(upstream);anchor_before=dataset_metrics(anchor,heldout)
    fit=train_anchor(anchor,train,steps=steps)
    trained=dataset_metrics(anchor,heldout)
    marker_before=alignment_metrics(anchor,heldout)
    align=align_marker(anchor,train,steps=steps)
    marker_after=alignment_metrics(anchor,heldout)
    model_current=make_model(upstream);current_fit=train_anchor(model_current,train,steps=steps,reference_mode='current_only')
    ablations={mode:dataset_metrics(anchor,heldout,mode=mode) for mode in ['normal','current_only','reverse_future','endpoints_only']}
    current_only=dataset_metrics(model_current,heldout,mode='current_only')
    fork=physics_forks()
    with torch.no_grad():branch_out=anchor(pack(fork['reference'],fork['initial_states'],0),return_dict=True)
    replay_results={}
    for route,name in [(0,'joints'),(1,'markers')]:
        replay_results[name]={'clean':replay(anchor,bank,list(range(24,32)),route),
                              'push':replay(anchor,bank,list(range(24,32)),route,push=True)}
    replay_results['current_only_matched_training']=replay(model_current,bank,list(range(24,32)),0,frame_mode='current_only')
    report={'schema':'cartpole-sonic-concepts/v1','execution':'COMPLETE','upstream_commit':UPSTREAM_COMMIT,'source_sha256':sources,
        'local_source_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['concepts.py','cartpole_env.py','cartpole.xml']},
        'versions':{'torch':torch.__version__,'mujoco':mujoco.__version__},
        'method':'official neural modules with local offline imitation/reconstruction/alignment; existing PPO path unchanged',
        'configuration':{'tokens':2,'token_dim':4,'fsq_levels':32,'hidden_widths':[64,64],'future_offsets_steps':OFFSETS.tolist(),
            'joint_reference_dim':4,'marker_reference_dim':6,'one_shared_dynamic_decoder':True,'first_reference_frame':'current reference, not first future frame'},
        'bank':{k:v for k,v in bank.items() if k not in ['states','actions','lengths','family','amplitude']},
        'data_split':{'train_episode_ids':list(range(24)),'heldout_episode_ids':list(range(24,32)),
            'train_windows':len(train['state']),'heldout_windows':len(heldout['state']),'recorded_lengths':bank['lengths'].tolist(),
            'no_cross_terminal_windows':True,'no_padding':True,'failed_episodes_are_not_hidden':True},
        'anchor_fit':fit,'anchor_before':anchor_before,'anchor_after':trained,
        'marker_fit':align,'marker_before':marker_before,'marker_after':marker_after,
        'marker_task_metrics':dataset_metrics(anchor,heldout,1),
        'temporal_interventions':ablations,'matched_current_only_fit':current_fit,'matched_current_only_metrics':current_only,
        'fork_probe':{'current_reference_max_delta':float(np.max(np.abs(fork['reference'][0,0]-fork['reference'][1,0]))),
            'future_max_delta':float(np.max(np.abs(fork['reference'][0,1:]-fork['reference'][1,1:]))),
            'action_delta':float((branch_out['action_mean'][0]-branch_out['action_mean'][1]).abs().max()),
            'boundary':'physical futures, not equal-endpoint paths; action difference alone does not demonstrate useful anticipation'},
        'replay':replay_results,
        'boundaries':['Clean invertible coordinates, not camera/VR noise, occlusion or ambiguous human retargeting.',
            'One offline training seed; shared-decoder transfer is not cross-robot weight transfer.',
            'Frame corruption is an input intervention, not a new physically feasible motion command.',
            'Replay performance and offline prediction do not prove online planning or optimal anticipatory control.',
            'No new PPO sweep, browser policy replacement, new robot or hardware actuation was performed.']}
    torch.save({'model':anchor.state_dict(),'config':report['configuration']},output_dir/'concept-weights.pt')
    report['artifacts_sha256']={f:hashlib.sha256((output_dir/f).read_bytes()).hexdigest() for f in ['motions.npz','concept-weights.pt']}
    if sources!=verify_sources(upstream):raise RuntimeError('upstream changed during experiment')
    write_result(output_dir/'concepts.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=900)
    args=p.parse_args()
    try:
        r=run(args.upstream,args.output_dir,args.steps)
        print(json.dumps({k:r[k] for k in ['execution','data_split','anchor_after','marker_before','marker_after','matched_current_only_metrics','replay']},allow_nan=False));return 0
    except Exception as exc:
        import traceback;traceback.print_exc()
        write_result(args.output_dir/'concepts.json',{'execution':'FAILED','error':str(exc),'performance_claim':'none'});return 1

if __name__=='__main__':raise SystemExit(main())
