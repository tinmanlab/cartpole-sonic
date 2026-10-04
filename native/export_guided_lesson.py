#!/usr/bin/env python3
"""Export synchronized *recorded native executions* for one public guided lesson.

No optimizer, new model, solver, policy selection by score, or browser inference.
Existing separately trained cue Encoders and their frozen SONIC Decoder are reused.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from matched_cues import reload_checkpoint, fingerprints
from encoder_transfer import pack_transfer
from temporal_audit import extend_reference, reference_inputs, control_rollout
from training import verify_sources
from learning import write_result

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CONDITIONS=('full_future','endpoints_only')
OFFSETS={'nominal':np.zeros(4),'perturbed':np.array([0.,.02,.001,-.02])}
ENCODER_SEED=52101


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_trace(policy,physics,raw_states,condition,offset):
    """Record decisions at t and resulting states at t+dt; never pad missing actions."""
    extended,history=extend_reference(physics,raw_states)
    data=[mujoco.MjData(physics) for _ in range(2)]
    traces=[[],[]]
    stopped=np.zeros(2,dtype=bool)
    for branch,d in enumerate(data):
        start=raw_states[branch,0]+offset
        mujoco.mj_resetData(physics,d);d.qpos[:]=start[[0,2]];d.qvel[:]=start[[1,3]]
        mujoco.mj_forward(physics,d)
    for tick in range(29):
        states=np.array([[d.qpos[0],d.qvel[0],d.qpos[1],d.qvel[1]] for d in data])
        decision=None
        if tick<28 and not stopped.all():
            cue=reference_inputs(extended,history,tick,condition)
            with torch.no_grad():
                out=policy(pack_transfer(cue,states,1),return_dict=True)
            actions=out['action_mean'][:,0,0].numpy().copy()
            latent=out['encoded_latents']['joints'].flatten(1).numpy().copy()
            token=out['encoded_tokens']['joints'].flatten(1).numpy().copy()
            if not all(np.isfinite(x).all() for x in [actions,latent,token]):raise RuntimeError('Nonfinite native inference')
            # Roundtrip force through the physical unit is recorded honestly.
            force=np.clip(actions.astype(np.float64),-1.,1.)*10.
            decision=[{'referenceWindow':cue[j].tolist(),'latent':latent[j].tolist(),'token':token[j].tolist(),
                'requestedForceN':float(actions[j])*10.,'appliedForceN':float(force[j])} for j in range(2)]
        for branch,d in enumerate(data):
            if stopped[branch]:continue
            state=states[branch];target=raw_states[branch,tick]
            frame={'tick':tick,'time':tick*.02,'state':state.tolist(),'referenceNow':target.tolist(),
                   'errorX':float(state[0]-target[0]),'decision':decision[branch] if tick<28 else None}
            traces[branch].append(frame)
            if tick>=28:continue
            d.ctrl[0]=decision[branch]['appliedForceN']/10.
            mujoco.mj_step(physics,d);mujoco.mj_step(physics,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('Nonfinite physics')
            if abs(d.qpos[0])>1.78 or abs(d.qpos[1])>.65:
                stopped[branch]=True
                state=np.array([d.qpos[0],d.qvel[0],d.qpos[1],d.qvel[1]])
                target=raw_states[branch,tick+1]
                traces[branch].append({'tick':tick+1,'time':(tick+1)*.02,'state':state.tolist(),
                    'referenceNow':target.tolist(),'errorX':float(state[0]-target[0]),'decision':None})
    runs=[]
    for branch,frames in enumerate(traces):
        errors=[abs(f['errorX']) for f in frames[1:]]
        runs.append({'branch':branch,'condition':condition,'frames':frames,
            'metrics':{'steps':len(frames)-1,'completed':bool(not stopped[branch] and len(frames)==29),
                       'cartMae':float(np.mean(errors)) if errors else None}})
    return runs


def export_lesson(upstream,output):
    upstream=Path(upstream).resolve(strict=True);sources=verify_sources(upstream)
    source_dir=ROOT/'evidence/matched_cues'
    summary=json.loads((source_dir/'summary.json').read_text())
    compressed=(source_dir/'matched.json.gz').read_bytes()
    if hashlib.sha256(compressed).hexdigest()!=summary['compressed_report_sha256']:
        raise RuntimeError('Stored comparison report hash mismatch')
    raw=gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest()!=summary['raw_report_sha256']:
        raise RuntimeError('Stored raw comparison hash mismatch')
    evidence=json.loads(raw)
    if sha(source_dir/'pairs.npz')!=evidence['pairs_sha256']:
        raise RuntimeError('Stored physical pair hash mismatch')
    records={}
    for condition in CONDITIONS:
        matches=[r for r in evidence['runs'] if r['config']['modality']=='joints' and r['config']['seed']==ENCODER_SEED and r['config']['mode']==condition]
        if len(matches)!=1:raise RuntimeError('Missing or ambiguous declared policy')
        records[condition]=matches[0]
    with np.load(source_dir/'pairs.npz',allow_pickle=False) as bank:
        candidates=[]
        for i in range(72):
            split=bank[f'case_{i}_split'].tobytes().decode()
            if split=='test' and bool(bank[f'case_{i}_accepted']):
                candidates.append((bank[f'case_{i}_pair_id'].tobytes().decode(),bank[f'case_{i}_states'].copy()))
    candidates.sort(key=lambda x:x[0]);selected=candidates[:3]
    if len(selected)!=3:raise RuntimeError('Three display cases required; never select by performance')
    torch.set_num_threads(1)
    from loguru import logger
    logger.disable('gear_sonic')
    physics=mujoco.MjModel.from_xml_path(str(HERE/'cartpole.xml'))
    files={'native/export_guided_lesson.py':sha(__file__),
           'native/cartpole.xml':sha(HERE/'cartpole.xml'),
           'evidence/matched_cues/summary.json':sha(source_dir/'summary.json'),
           'evidence/matched_cues/matched.json.gz':sha(source_dir/'matched.json.gz'),
           'evidence/matched_cues/pairs.npz':sha(source_dir/'pairs.npz')}
    for name in ['matched_cues.py','encoder_transfer.py','temporal_audit.py']:
        files['native/'+name]=sha(HERE/name)
    cases=[{'id':pair_id,'reference':states.tolist(),'endpointMaxError':float(np.max(np.abs(states[:,-1]-states[:,0]))),'runs':[]} for pair_id,states in selected]
    frozen=True
    comparisons=0
    for condition in CONDITIONS:
        row=records[condition];checkpoint=source_dir/row['checkpoint']
        if sha(checkpoint)!=row['checkpoint_sha256']:raise RuntimeError('Policy hash mismatch')
        files['evidence/matched_cues/'+row['checkpoint']]=sha(checkpoint)
        policy,_=reload_checkpoint(upstream,checkpoint)
        for p in policy.parameters():p.requires_grad_(False)
        before=fingerprints(policy)
        for case,(pair_id,states) in zip(cases,selected):
            for offset_name,offset in OFFSETS.items():
                traces=pair_trace(policy,physics,states,condition,offset)
                # Validate against both a fresh existing evaluator and stored study metrics.
                ext,hist=extend_reference(physics,states)
                oracle=control_rollout(policy,physics,ext,hist,1,condition,offset)
                prior=next(r for r in row['rollouts'] if r['pair_id']==pair_id and r['actual_offset_state4']==offset.tolist())
                for trace,actual,stored in zip(traces,oracle['episodes'],prior['episodes']):
                    if trace['metrics']['completed']!=actual['completed'] or actual['completed']!=stored['completed']:
                        raise RuntimeError('Completion result changed')
                    for value in [actual['cart_mae_m'],stored['cart_mae_m']]:
                        if not np.isclose(trace['metrics']['cartMae'],value,atol=1e-8,rtol=1e-5):
                            raise RuntimeError('Trace no longer matches measured comparison')
                    trace['offset']=offset_name
                    case['runs'].append(trace);comparisons+=1
        frozen=frozen and fingerprints(policy)==before
    if not frozen or verify_sources(upstream)!=sources:raise RuntimeError('Frozen model/source changed')
    result={'schema':'cartpole-sonic-guided-lesson/v1',
        'source':{'kind':'native-recorded-reexecution',
            'training':'matched selected-Encoder supervised imitation; shared original SONIC Decoder frozen',
            'upstreamCommit':evidence['protocol']['upstream_commit'],'encoderSeed':ENCODER_SEED,'modality':'joints',
            'files':files,'scope':'three display examples, not complete benchmark',
            'optimizerUpdates':0,'frozenModelsUnchanged':True,'matchedStoredOutcomes':True,'outcomesCompared':comparisons,
            'runtime':{'mujoco':mujoco.__version__,'torch':torch.__version__}},
        'protocol':{'controlDt':.02,'horizonSteps':28,'referenceOffsets':[0,4,8,12,16,20,24,28],
            'stateOrder':['x','xdot','theta','thetadot'],'stateUnits':['m','m/s','rad','rad/s'],
            'conditions':list(CONDITIONS),'offsets':{k:v.tolist() for k,v in OFFSETS.items()},
            'selection':'first three accepted test pairs by stored pair ID; first declared Encoder seed; no score selection'},
        'selection':{'acceptedTestPairIds':[x[0] for x in candidates],'displayedPairIds':[x[0] for x in selected]},
        'cases':cases,
        'limits':['one fixed Decoder and one displayed Encoder seed; aggregate three-seed study separate',
            '0.56 seconds, fixed near-upright CartPole; no hardware/general robustness claim',
            'comparison policies trained separately; not changing one policy input at playback',
            'nominal/perturbed are distinct prerecorded runs; no live force injection',
            'terminal sample has no new action; state and error use the same sample time'],
        'benchmark':{'summaryPath':'evidence/matched_cues/summary.json','documentationPath':'native/MATCHED_CUES.md'}}
    write_result(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        report=export_lesson(args.upstream,args.output)
        print(json.dumps({'schema':report['schema'],'cases':len(report['cases']),'traces':report['source']['outcomesCompared'],'matchedStoredOutcomes':True}));return 0
    except Exception as exc:
        write_result(args.output,{'schema':'cartpole-sonic-guided-lesson/v1','error':str(exc),'status':'FAILED'})
        import traceback;traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
