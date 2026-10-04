#!/usr/bin/env python3
"""Add coordinate encoders to the known working native token/decoder space.

The original learned Encoder, Dynamic Decoder, Kinematic Decoder and FSQ are
unchanged. Only new joint-motion and marker-motion Encoders learn. This is local
paired latent distillation, not a new PPO benchmark or a humanoid modality claim.
"""
import argparse,copy,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from core_smoke import configuration,import_native,digest_parameters
from training import verify_sources
from cartpole_env import STATE_SCALE
from learning import load_policy,write_result
from concepts import MARKER_SCALE,to_markers,windows,desired,replay
HERE=Path(__file__).resolve().parent


def make_transfer(upstream):
    sources=verify_sources(upstream);Universal,_,_=import_native(upstream)
    old=load_policy(upstream,HERE.parent/'evidence/native_learning/weights-final.pt')
    cfg=configuration(upstream,tokens=2,dim=2,levels=32)
    native=copy.deepcopy(cfg.encoders.cartpole);joint=copy.deepcopy(native);marker=copy.deepcopy(native)
    joint.inputs=['joint_reference'];marker.inputs=['marker_reference']
    for c in [joint,marker]:c.params.module_config_dict.layer_config.hidden_dims=[64,64]
    cfg.encoders={'cartpole':native,'joints':joint,'markers':marker}
    cfg.encoder_sample_probs={'cartpole':1/3,'joints':1/3,'markers':1/3}
    cfg.env_config.obs.group_obs_dims.tokenizer={'cartpole_reference':[8,2],'joint_reference':[8,4],'marker_reference':[8,6],'encoder_index':[3]}
    cfg.env_config.obs.group_obs_names.tokenizer=['cartpole_reference','joint_reference','marker_reference','encoder_index']
    torch.manual_seed(7091);model=Universal(**cfg).cpu()
    model.encoders['cartpole'].load_state_dict(old.actor_module.encoders['cartpole'].state_dict())
    for name in model.decoders:model.decoders[name].load_state_dict(old.actor_module.decoders[name].state_dict())
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('encoders.joints.') or name.startswith('encoders.markers.'))
    model.eval();return model


def pack_transfer(ref,state,route,command=None):
    ref=np.asarray(ref);state=np.asarray(state);n=len(ref)
    if ref.shape!=(n,8,4) or state.shape!=(n,4) or route not in (0,1,2):raise ValueError('invalid transfer input')
    if not np.isfinite(ref).all() or not np.isfinite(state).all():raise ValueError('nonfinite transfer input')
    if command is None:
        if route==0:raise ValueError('native encoder needs its real command input')
        command=np.zeros((n,8,2))  # inactive native branch placeholder, never an active fake modality
    if np.asarray(command).shape!=(n,8,2):raise ValueError('command shape mismatch')
    selector=np.repeat(np.eye(3)[route:route+1],n,axis=0)
    flat=np.concatenate((np.asarray(command).reshape(n,16),(ref/STATE_SCALE).reshape(n,32),(to_markers(ref)/MARKER_SCALE).reshape(n,48),selector),axis=-1)
    return {'actor_obs':torch.tensor(np.clip(state/STATE_SCALE,-1.5,1.5)[:,None],dtype=torch.float32),
            'tokenizer':torch.tensor(flat[:,None],dtype=torch.float32)}


def commands_for(bank,data):
    refs=[]
    for i,t in zip(data['episode_id'],data['tick']):
        x,v=desired(t*.02+np.arange(1,9)*.08,bank['amplitude'][i],int(bank['family'][i]))
        refs.append(np.stack((x/1.8,v/3.),axis=-1))
    return np.array(refs)


def metrics(model,data,command,route):
    with torch.no_grad():
        target=model(pack_transfer(data['reference'],data['state'],0,command),return_dict=True)
        out=model(pack_transfer(data['reference'],data['state'],route),return_dict=True)
    name=['cartpole','joints','markers'][route];z=out['encoded_latents'][name];q=out['encoded_tokens'][name]
    tz=target['encoded_latents']['cartpole'];tq=target['encoded_tokens']['cartpole']
    return {'latent_mse':float((z-tz).square().mean()),'token_vector_agreement':float((q==tq).flatten(1).all(1).float().mean()),
            'action_disagreement_rmse_newtons':float(10*(out['action_mean']-target['action_mean']).square().mean().sqrt()),
            'action_imitation_rmse_newtons':float(10*np.sqrt(np.mean((out['action_mean'][:,0,0].numpy()-data['action'])**2)))}


class Bridge:
    """Adapt the existing replay evaluator's input layout to the expanded native model.
    No original command is available here: it is zero only in the inactive branch.
    """
    def __init__(self,model,route):self.model=model;self.route=route
    def __call__(self,inputs):
        ref=inputs['tokenizer'][:,0,:32].numpy().reshape(-1,8,4)*STATE_SCALE
        state=inputs['actor_obs'][:,0].numpy()*STATE_SCALE
        return self.model(pack_transfer(ref,state,self.route))


def run(upstream,concept_dir,output_dir,steps=1200):
    if type(steps) is not int or not 1<=steps<=1500:raise ValueError('budget must be 1..1500')
    upstream=Path(upstream);concept_dir=Path(concept_dir);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    source=verify_sources(upstream);torch.set_num_threads(1)
    from loguru import logger
    logger.disable('gear_sonic')
    with np.load(concept_dir/'motions.npz',allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    training=windows(bank,range(24));heldout=windows(bank,range(24,32))
    command=commands_for(bank,training);held_command=commands_for(bank,heldout)
    model=make_transfer(upstream)
    fingerprint={'original_encoder':digest_parameters(model.encoders['cartpole']),'shared_decoders':digest_parameters(model.decoders)}
    before={name:metrics(model,heldout,held_command,r) for r,name in [(1,'joints'),(2,'markers')]}
    rng=np.random.default_rng(8101)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    losses=[]
    for _ in range(steps):
        idx=rng.integers(0,len(training['state']),64);ref=training['reference'][idx];state=training['state'][idx]
        with torch.no_grad():target=model(pack_transfer(ref,state,0,command[idx]),return_dict=True)
        loss=0.
        for r,name in [(1,'joints'),(2,'markers')]:
            out=model(pack_transfer(ref,state,r),return_dict=True)
            loss=loss+torch.nn.functional.mse_loss(out['encoded_latents'][name],target['encoded_latents']['cartpole'])
            loss=loss+.1*torch.nn.functional.mse_loss(out['action_mean'],target['action_mean'])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();losses.append(float(loss.detach()))
    after={name:metrics(model,heldout,held_command,r) for r,name in [(1,'joints'),(2,'markers')]}
    if fingerprint['original_encoder']!=digest_parameters(model.encoders['cartpole']) or fingerprint['shared_decoders']!=digest_parameters(model.decoders):raise RuntimeError('known working anchor changed')
    outcomes={}
    for r,name in [(1,'joints'),(2,'markers')]:
        agent=Bridge(model,r)
        outcomes[name]={'clean':replay(agent,bank,list(range(24,32))),
            'push':replay(agent,bank,list(range(24,32)),push=True),
            'current_only':replay(agent,bank,list(range(24,32)),frame_mode='current_only'),
            'reverse_interior':replay(agent,bank,list(range(24,32)),frame_mode='reverse_future')}
        for result in outcomes[name].values():result['active_native_encoder']=name;result['route_note']='route=0 is the old evaluator layout; bridge selects the named native encoder'
    ck=output_dir/'encoder-transfer.pt';torch.save({'model':model.state_dict(),'source_native_checkpoint_sha256':hashlib.sha256((HERE.parent/'evidence/native_learning/weights-final.pt').read_bytes()).hexdigest()},ck)
    report={'schema':'sonic-frozen-decoder-transfer/v1','execution':'COMPLETE','source_sha256':source,
        'local_source_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['encoder_transfer.py','concepts.py']},
        'training':{'steps':steps,'batch':64,'learning_rate':.001,'seed':8101,'model_seed':7091,'loss_first':losses[0],'loss_last':losses[-1],
          'method':'local paired latent/action distillation; only new encoders updated; no PPO update'},
        'frozen_anchor_hashes':fingerprint,'frozen_anchor_unchanged':True,
        'dataset':{'training_episodes':list(range(24)),'diagnostic_episodes':list(range(24,32)),'bank_sha256':hashlib.sha256((concept_dir/'motions.npz').read_bytes()).hexdigest()},
        'before':before,'after':after,'outcomes':outcomes,'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
        'boundaries':['Full invertible kinematic measurements; no camera/VR or missing-sensor inference.',
         'New input encoders learn the previously trained native controller token space; no cross-robot weights are claimed.',
         'Recorded future motion is a fixed reference during replay, not the actual simulator future.',
         'These eight gradient-excluded episodes were previously inspected; this is a diagnostic evaluation, not a pristine final holdout.',
         'Input-corruption effects alone do not establish optimal anticipation or future horizon necessity.']}
    if source!=verify_sources(upstream):raise RuntimeError('upstream changed')
    write_result(output_dir/'transfer.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    p.add_argument('--concept-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=1200)
    a=p.parse_args()
    try:r=run(a.upstream,a.concept_dir,a.output_dir,a.steps);print(json.dumps(r,allow_nan=False))
    except Exception as e:
        import traceback;traceback.print_exc();write_result(a.output_dir/'transfer.json',{'execution':'FAILED','error':str(e)});raise SystemExit(1)
