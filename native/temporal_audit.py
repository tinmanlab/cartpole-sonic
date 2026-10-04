#!/usr/bin/env python3
"""Frozen-policy same-present/same-endpoint temporal-information audit.

No policy training or data selection by policy performance. Reduced-cue inputs
are interventions on the same pretrained model, not retrained best baselines.
The recorded-control prediction bound is not an optimal-control lower bound.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import mujoco
import torch
from core_smoke import digest_parameters
from encoder_transfer import make_transfer, pack_transfer
from learning import write_result
from training import verify_sources

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
PROTOCOL=HERE/'temporal_audit_protocol.json'
OFFSETS=np.arange(8)*4
MODES=('full_future','current_only','history_only','endpoints_only')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(upstream,checkpoint=None):
    protocol=json.loads(PROTOCOL.read_text())
    checkpoint=Path(checkpoint) if checkpoint is not None else REPO/protocol['frozen_policy']
    if sha256(checkpoint)!=protocol['frozen_policy_sha256']:
        raise ValueError('Frozen policy hash differs from the declared checkpoint')
    verify_sources(upstream)
    model=make_transfer(upstream)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    return model


def simulate(model,initial,controls):
    initial=np.asarray(initial,dtype=np.float64);controls=np.asarray(controls,dtype=np.float64)
    if initial.shape!=(4,) or controls.ndim!=1 or not np.isfinite(initial).all() or not np.isfinite(controls).all():
        raise ValueError('finite initial state4 and scalar control sequence required')
    if (np.abs(controls)>1).any():raise ValueError('simulation controls exceed physical normalized limit')
    data=mujoco.MjData(model)
    mujoco.mj_resetData(model,data)
    data.qpos[:]=initial[[0,2]];data.qvel[:]=initial[[1,3]]
    mujoco.mj_forward(model,data)
    states=[initial.copy()]
    for u in controls:
        data.ctrl[0]=u
        mujoco.mj_step(model,data);mujoco.mj_step(model,data)
        states.append(np.array([data.qpos[0],data.qvel[0],data.qpos[1],data.qvel[1]]))
    return np.asarray(states)


def reference_inputs(states,history,tick,mode,common_endpoint=None):
    states=np.asarray(states,dtype=np.float64);history=np.asarray(history,dtype=np.float64)
    if mode not in MODES or type(tick) is not int or tick<0:
        raise ValueError('invalid tick or reference mode')
    if states.ndim!=3 or states.shape[2]!=4 or history.shape!=(len(states),29,4):
        raise ValueError('expected [batch,time,4] states and [batch,29,4] history')
    if tick+28>=states.shape[1]:raise ValueError('missing physically generated future; padding prohibited')
    if not np.isfinite(states).all() or not np.isfinite(history).all():raise ValueError('nonfinite trajectory')
    future=states[:,tick+OFFSETS].copy()
    if common_endpoint is not None:
        if tick!=0:raise ValueError('endpoint canonicalization is only for the branch-point probe')
        endpoint=np.asarray(common_endpoint,dtype=np.float64)
        if endpoint.shape!=(4,) or not np.isfinite(endpoint).all():raise ValueError('invalid canonical endpoint')
        if np.max(np.abs(future[:,-1]-endpoint))>1e-7:raise ValueError('canonicalization exceeds protocol tolerance')
        future[:,-1]=endpoint
    if mode=='full_future':return future
    if mode=='current_only':return np.repeat(future[:,:1],8,axis=1)
    if mode=='endpoints_only':
        alpha=np.linspace(0,1,8)[None,:,None]
        return future[:,:1]*(1-alpha)+future[:,-1:]*alpha
    # Explicit backward-time history intervention: current, -.08,...,-.56 s.
    # It is not a network independently trained to consume causal history.
    timeline=np.concatenate((history[:,:-1],states),axis=1)
    return timeline[:,28+tick-OFFSETS].copy()


def predict(frozen,reference,actual_state,route):
    if route not in (1,2):raise ValueError('audit only the admitted joints/markers encoders')
    with torch.no_grad():
        out=frozen(pack_transfer(reference,actual_state,route),return_dict=True)
    name={1:'joints',2:'markers'}[route]
    actions=out['action_mean'][:,0,0].cpu().numpy().copy()
    tokens=out['encoded_tokens'][name].detach().cpu().numpy().copy()
    if not np.isfinite(actions).all() or not np.isfinite(tokens).all():raise RuntimeError('nonfinite policy output')
    return actions,tokens


def blind_action_mse_bound(recorded_force):
    y=np.asarray(recorded_force,dtype=np.float64)
    if y.shape!=(2,) or not np.isfinite(y).all():raise ValueError('finite paired recorded force required')
    return float(np.mean((y-y.mean())**2))


def initial_probe(frozen,states,history,recorded_actions,route):
    actual=np.repeat(states[:1,0],2,axis=0)
    common=states[0,0].copy() # equal equilibrium endpoint specified before optimization
    recorded=np.asarray(recorded_actions)[:,0]*10.
    result={}
    for mode in MODES:
        ref=reference_inputs(states,history,0,mode,common_endpoint=common)
        action,tokens=predict(frozen,ref,actual,route)
        force=action*10.
        result[mode]={'reference_pair_max_abs_delta':float(np.max(np.abs(ref[0]-ref[1]))),
            'predicted_force_N':force.tolist(),'force_pair_abs_delta_N':float(abs(force[0]-force[1])),
            'tokens_identical':bool(np.array_equal(tokens[0],tokens[1])),
            'recorded_first_force_mse_N2':float(np.mean((force-recorded)**2))}
    raw_action,raw_tokens=predict(frozen,reference_inputs(states,history,0,'full_future'),actual,route)
    return {'route':{1:'joints',2:'markers'}[route],'recorded_first_force_N':recorded.tolist(),
        'best_common_prediction_mse_N2':blind_action_mse_bound(recorded),
        'conditions':result,'raw_vs_canonical_force_max_abs_delta_N':float(np.max(np.abs(raw_action*10-np.asarray(result['full_future']['predicted_force_N'])))),
        'canonical_endpoint_max_change_state4':np.max(np.abs(states[:,28]-common),axis=0).tolist(),
        'interpretation':'Paired mean bound concerns imitation of recorded first forces. It is not a bound on every successful controller or an optimal-control proof.'}


def control_rollout(frozen,model,states,history,route,mode,actual_offset):
    states=np.asarray(states,dtype=np.float64);offset=np.asarray(actual_offset,dtype=np.float64)
    if states.shape!=(2,57,4) or offset.shape!=(4,) or not np.isfinite(offset).all():raise ValueError('invalid rollout inputs')
    data=[mujoco.MjData(model) for _ in range(2)]
    for j,d in enumerate(data):
        initial=states[j,0]+offset
        mujoco.mj_resetData(model,d);d.qpos[:]=initial[[0,2]];d.qvel[:]=initial[[1,3]];mujoco.mj_forward(model,d)
    alive=np.ones(2,dtype=bool);steps=np.zeros(2,dtype=int);absolute=np.zeros((2,4));maxforce=np.zeros(2)
    first_force=np.zeros(2);final=np.stack([states[j,0]+offset for j in range(2)])
    for tick in range(28):
        ref=reference_inputs(states,history,tick,mode)
        actual=np.array([[d.qpos[0],d.qvel[0],d.qpos[1],d.qvel[1]] for d in data])
        action,_=predict(frozen,ref,actual,route)
        applied=np.clip(action,-1,1)
        if tick==0:first_force=applied*10.
        for j,d in enumerate(data):
            if not alive[j]:continue
            d.ctrl[0]=applied[j];mujoco.mj_step(model,d);mujoco.mj_step(model,d)
            s=np.array([d.qpos[0],d.qvel[0],d.qpos[1],d.qvel[1]])
            if not np.isfinite(s).all():raise RuntimeError('nonfinite controlled trajectory; do not turn into success')
            final[j]=s;steps[j]+=1;absolute[j]+=np.abs(s-states[j,tick+1]);maxforce[j]=max(maxforce[j],abs(applied[j]*10))
            if abs(s[0])>1.78 or abs(s[2])>.65:alive[j]=False
    episodes=[]
    for j in range(2):
        episodes.append({'branch':j,'steps':int(steps[j]),'completed':bool(steps[j]==28 and alive[j]),
            'executed_state_mae':(absolute[j]/max(steps[j],1)).tolist(),
            'cart_mae_m':float(absolute[j,0]/max(steps[j],1)),
            'final_state':final[j].tolist(),'target_terminal_state':states[j,28].tolist(),
            'terminal_abs_error_state4':np.abs(final[j]-states[j,28]).tolist(),
            'first_applied_force_N':float(first_force[j]),'max_abs_applied_force_N':float(maxforce[j])})
    return {'encoder':{1:'joints',2:'markers'}[route],'condition':mode,'actual_offset_state4':offset.tolist(),
        'horizon_steps':28,'horizon_s':.56,'episodes':episodes,
        'completed':sum(e['completed'] for e in episodes),
        'mean_executed_cart_mae_m':float(np.mean([e['cart_mae_m'] for e in episodes])),
        'target_source':'immutable recorded nonlinear reference; zero-control tail generated by actual mj_step',
        'interpretation':'Short-horizon local replay, not sustained stability, unseen-robot transfer or an optimally retrained baseline.'}


def extend_reference(model,states):
    states=np.asarray(states,dtype=np.float64)
    if states.shape!=(2,29,4):raise ValueError('raw pair states must be [2,29,4]')
    extended=[];histories=[]
    for branch in range(2):
        tail=simulate(model,states[branch,-1],np.zeros(28))
        history=simulate(model,states[branch,0],np.zeros(28))
        if not np.allclose(history,states[branch,0],rtol=0,atol=1e-12):
            raise ValueError('declared stationary prehistory is not physically stationary')
        extended.append(np.concatenate((states[branch],tail[1:]),axis=0));histories.append(history)
    return np.asarray(extended),np.asarray(histories)


def audit_arrays(frozen,model,states,actions,protocol):
    extended,history=extend_reference(model,states)
    before=digest_parameters(frozen);rng=torch.get_rng_state().clone()
    probes=[initial_probe(frozen,extended,history,actions,r) for r in (1,2)]
    rollouts=[control_rollout(frozen,model,extended,history,r,mode,np.asarray(offset))
        for r in (1,2) for mode in MODES for offset in protocol['initial_actual_state_offsets']]
    if digest_parameters(frozen)!=before or not torch.equal(torch.get_rng_state(),rng):
        raise RuntimeError('audit mutated frozen weights or Torch RNG')
    return {'initial_probes':probes,'rollouts':rollouts,
            'simulated_tail_endpoint_pair_abs_delta_state4':np.abs(extended[0,-1]-extended[1,-1]).tolist()}


def summarize(pairs,declared_count):
    result={'declared_pairs':declared_count,'accepted_pairs':len(pairs),'rejected_pairs':declared_count-len(pairs),
            'case_scope':'paired trajectories from one declared generator; one fixed pretrained policy; not independent training seeds',
            'initial_action_probe':{},'rollout_comparisons':[]}
    for encoder in ['joints','markers']:
        probes=[next(x for x in p['audit']['initial_probes'] if x['route']==encoder) for p in pairs]
        def avg(key):return float(np.mean(key)) if key else None
        result['initial_action_probe'][encoder]={
            'pairs_with_different_full_future_tokens':sum(not p['conditions']['full_future']['tokens_identical'] for p in probes),
            'pairs_with_full_future_force_delta_above_1e_6_N':sum(p['conditions']['full_future']['force_pair_abs_delta_N']>1e-6 for p in probes),
            'mean_full_future_recorded_force_mse_N2':avg([p['conditions']['full_future']['recorded_first_force_mse_N2'] for p in probes]),
            'mean_blind_best_recorded_force_mse_N2':avg([p['best_common_prediction_mse_N2'] for p in probes]),
            'full_future_below_paired_mean_bound_count':sum(p['conditions']['full_future']['recorded_first_force_mse_N2']<p['best_common_prediction_mse_N2'] for p in probes)}
        for mode in MODES:
            for perturb in range(2):
                rows=[next(r for r in p['audit']['rollouts'] if r['encoder']==encoder and r['condition']==mode and bool(np.any(r['actual_offset_state4']))==bool(perturb)) for p in pairs]
                episodes=[e for r in rows for e in r['episodes']]
                completed=[e for e in episodes if e['completed']]
                result['rollout_comparisons'].append({'encoder':encoder,'condition':mode,'perturbed':bool(perturb),
                    'episodes':len(episodes),'completed':len(completed),
                    'executed_cart_mae_m':avg([e['cart_mae_m'] for e in episodes]),
                    'completed_cart_mae_m':avg([e['cart_mae_m'] for e in completed])})
    return result


def run(upstream,output_dir):
    """Solve every declared case before loading/scoring the frozen policy."""
    from temporal_pairs import generate_bank,save_bank,declared_cases
    protocol=json.loads(PROTOCOL.read_text())
    cases=declared_cases()
    expected=[(tick,amp) for tick in protocol['interior_ticks'] for amp in protocol['interior_displacements_m']]
    if [(c.tick,c.amplitude) for c in cases]!=expected or len(cases)!=protocol['declared_case_count']:
        raise ValueError('generator case list differs from the preregistered protocol')
    upstream=Path(upstream).resolve(strict=True);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    sources=verify_sources(upstream)
    model=mujoco.MjModel.from_xml_path(str(HERE/'cartpole.xml'))
    bank=generate_bank(model=model)
    save_bank(bank,output_dir/'physical_pairs.npz')
    # Policy cannot affect generation or accepted/rejected membership.
    torch.set_num_threads(1)
    frozen=load_frozen(upstream)
    before=digest_parameters(frozen)
    evaluated=[];rejected=[];all_case_metadata=[]
    for index,record in enumerate(bank['records']):
        case=record['case']
        meta={'case_id':index,'initial_x_m':float(case.x),'interior_tick':int(case.tick),'amplitude_m':float(case.amplitude),
              'accepted':bool(record['accepted']),'reason':record['reason'],
              'first_action_informative':bool(record['first_action_informative']),
              'candidate_count':int(record['candidate_count']),'solver_verification_status':int(record['solver_status']),
              'endpoint_errors_state4':np.asarray(record['endpoint_errors']).tolist(),
              'interior_pair_delta_m':float(record['interior_reference_delta']),
              'first_action_difference':float(record['first_action_delta'])}
        all_case_metadata.append(meta)
        if not record['accepted']:
            rejected.append(meta);continue
        # Independent reset and native replay from the archived controls, not generator status alone.
        for branch in range(2):
            replayed=simulate(model,record['states'][branch,0],record['actions'][branch])
            if not np.allclose(replayed,record['states'][branch],rtol=0,atol=1e-12):
                raise RuntimeError('reference controls do not independently reproduce recorded states')
        evaluated.append({'case':meta,'audit':audit_arrays(frozen,model,record['states'],record['actions'],protocol)})
    after=digest_parameters(frozen)
    if before!=after:raise RuntimeError('frozen policy changed')
    report={'schema':'cartpole-sonic-temporal-information/v1','execution':'COMPLETE',
        'protocol':protocol,'protocol_sha256':sha256(PROTOCOL),'source_sha256':sources,
        'local_source_sha256':{name:sha256(HERE/name) for name in ['temporal_pairs.py','temporal_audit.py','encoder_transfer.py','concepts.py','cartpole.xml']},
        'versions':bank['provenance']['versions'],'generator_provenance':bank['provenance'],
        'generation_summary':bank['summary'],'all_cases':all_case_metadata,'rejected_cases':rejected,
        'frozen_model_parameter_sha256':before,'frozen_model_unchanged':True,'optimizer_updates':0,
        'pairs':evaluated,'summary':summarize(evaluated,len(cases)),
        'physical_pair_artifact_sha256':sha256(output_dir/'physical_pairs.npz'),
        'interpretation_boundaries':protocol['comparison_rules'],
        'task_limits':{'new_motion_family':True,'single_fixed_checkpoint':True,'local_horizon_seconds':.56,
            'limited_force_and_amplitude_cases':True,'no_new_robot':True,'browser_unchanged':True}}
    if verify_sources(upstream)!=sources:raise RuntimeError('upstream source changed')
    write_result(output_dir/'audit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=run(args.upstream,args.output_dir)
        print(json.dumps({'execution':result['execution'],'generation':result['generation_summary'],'summary':result['summary']},allow_nan=False))
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        write_result(args.output_dir/'audit.json',{'execution':'FAILED','error':str(error),'performance_claim':'none'})
        return 1

if __name__=='__main__':raise SystemExit(main())
