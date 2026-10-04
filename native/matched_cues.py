#!/usr/bin/env python3
"""Matched Encoder imitation, conditional on one fixed native learned Decoder.

The protocol file is immutable. Short runs are smoke evidence, never the final
protocol. Pair/sign/tick/split are audit metadata and never network features.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from concepts import MARKER_SCALE, from_markers, to_markers
from core_smoke import import_native
from encoder_transfer import make_transfer, pack_transfer
from learning import write_result
from temporal_pairs import Case, generate_pair, load_model, save_bank
from temporal_audit import extend_reference, reference_inputs, control_rollout, blind_action_mse_bound
from training import verify_sources

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'matched_cue_protocol.json'
ROUTES = {'joints': 1, 'markers': 2}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return json.loads(PROTOCOL.read_text())


def validate_steps(steps):
    if type(steps) is not int or not 1 <= steps <= 1200:
        raise ValueError('steps must be an integer in 1..1200')


def declared_cases(config=None):
    config = protocol() if config is None else config
    result = {}
    for split, spec in config['data_splits'].items():
        rng = np.random.default_rng(spec['seed'])
        result[split] = [Case(float(rng.uniform(-.25,.25)), tick, amplitude)
                         for _ in range(spec['replicates']) for tick in spec['ticks']
                         for amplitude in spec['amplitudes_m']]
    return result


def generate_data(plant, config=None):
    records = []
    for split, cases in declared_cases(config).items():
        for i, case in enumerate(cases):
            record = generate_pair(case, model=plant)
            record.update(split=split, pair_id=f'{split}-{i:03d}')
            records.append(record)
    return records


def save_pairs(records, path, summary, provenance):
    numeric_records = [{k:np.frombuffer(v.encode(),dtype=np.uint8) if k in ('split','pair_id') else v
                        for k,v in r.items()} for r in records]
    save_bank(dict(records=numeric_records,summary=summary,provenance=provenance),path)


def build_windows(records, plant, mode):
    """Keep raw true targets/actions separately from intervened reference cues."""
    rows = {k: [] for k in ('reference','state','target','action','split','pair_id','sign','tick')}
    for record in records:
        if not record['accepted']:
            continue
        extended, history = extend_reference(plant, record['states'])
        for tick in range(28):
            ref = reference_inputs(extended, history, tick, mode,
                common_endpoint=record['canonical_endpoint'] if tick == 0 else None)
            if mode == 'full_future':
                markers = to_markers(ref)
                validate_markers(markers, np.ones(markers.shape,dtype=bool))
            for branch, sign in enumerate((1,-1)):
                for key, value in dict(reference=ref[branch], state=record['states'][branch,tick],
                    target=record['states'][branch,tick+1], action=record['actions'][branch,tick],
                    split=record['split'], pair_id=record['pair_id'], sign=sign, tick=tick).items():
                    rows[key].append(value)
    if not rows['action']:
        raise ValueError('no accepted physical pairs')
    return {k: np.asarray(v) for k,v in rows.items()}


def validate_markers(markers, mask):
    markers = np.asarray(markers)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != markers.shape or not mask.all():
        raise ValueError('all cart/tip positions and velocities must be explicitly present')
    return from_markers(markers)


def noisy_marker_inputs(inputs, position_sigma, velocity_sigma, seed=94104):
    """Offline sensitivity only; alter the active marker block in physical units."""
    if position_sigma not in (.001,.005) or velocity_sigma not in (.01,.05):
        raise ValueError('noise outside declared physical sigmas')
    result = {k:v.clone() for k,v in inputs.items()}
    block = result['tokenizer'][...,48:96]
    if not torch.all(result['tokenizer'][...,96:99] == torch.tensor([0.,0.,1.])):
        raise ValueError('marker route required')
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(*block.shape[:-1],8,6))
    noise *= np.array([position_sigma]*3+[velocity_sigma]*3) / MARKER_SCALE
    block.add_(torch.tensor(noise.reshape(block.shape), dtype=block.dtype))
    return result


def state_hash(module):
    """Hash parameters and all buffers, including nonpersistent FSQ levels/basis."""
    h = hashlib.sha256()
    tensors = dict(module.state_dict())
    for name, tensor in module.named_buffers():
        tensors.setdefault(name, tensor)
    for name, tensor in sorted(tensors.items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fingerprints(model):
    return dict(encoders={k:state_hash(v) for k,v in model.encoders.items()},
                decoders={k:state_hash(v) for k,v in model.decoders.items()},
                fsq=state_hash(model.quantizer), model=state_hash(model))


def initialize(upstream, modality, seed):
    if modality not in ROUTES or seed not in protocol()['encoder_initialization_seeds']:
        raise ValueError('undeclared Encoder or seed')
    model = make_transfer(Path(upstream))
    Universal, Base, FSQ = import_native(Path(upstream))
    if type(model) is not Universal or type(model.quantizer) is not FSQ or any(
        type(x) is not Base for x in [*model.encoders.values(), *model.decoders.values()]):
        raise RuntimeError('native module identity mismatch')
    for p in model.parameters(): p.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in model.encoders[modality].modules():
            if isinstance(module, torch.nn.Linear): module.reset_parameters()
    for p in model.encoders[modality].parameters(): p.requires_grad_(True)
    return model.eval()


def sample_stream(pair_count, steps, seed):
    validate_steps(steps)
    if pair_count < 1: raise ValueError('training requires accepted pairs')
    rng = np.random.default_rng(seed)
    stream = []
    for _ in range(steps):
        pairs = rng.integers(pair_count, size=32)
        ticks = rng.integers(1,28,size=32)
        ticks[rng.random(32)<.5] = 0
        stream.append((pairs[:,None]*56 + ticks[:,None]*2 + np.arange(2)).reshape(-1))
    return np.asarray(stream)


def forward_actions(model, data, route, indices=None):
    ix = slice(None) if indices is None else indices
    return model(pack_transfer(data['reference'][ix],data['state'][ix],route),
                 return_dict=True)['action_mean'][:,0,0]


def errors(model, data, route):
    with torch.no_grad():
        predictions = np.concatenate([forward_actions(model, data, route, np.arange(i,min(i+256,len(data['action'])))).numpy()
                                      for i in range(0,len(data['action']),256)])
    force_error = (predictions-data['action'])*10
    first = data['tick']==0
    forces = data['action'][first].reshape(-1,2)*10
    return dict(action_rmse_N=float(np.sqrt(np.mean(force_error**2))),
                first_force_mse_N2=float(np.mean(force_error[first]**2)),
                paired_common_first_force_bound_N2=float(np.mean([blind_action_mse_bound(y) for y in forces])),
                samples=len(predictions), predictions_normalized=predictions.tolist())


def train_encoder(model, data, modality, seed, steps):
    validate_steps(steps)
    before = fingerprints(model)
    stream = sample_stream(len(data['action'])//56,steps,seed)
    params = list(model.encoders[modality].parameters())
    opt = torch.optim.Adam(params,lr=.001)
    losses = []
    for indices in stream:
        prediction = forward_actions(model,data,ROUTES[modality],indices)
        labels = torch.tensor(data['action'][indices],dtype=prediction.dtype)
        loss = torch.nn.functional.mse_loss(prediction,labels)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        opt.step(); losses.append(float(loss.detach()))
    after = fingerprints(model)
    if before['decoders'] != after['decoders'] or before['fsq'] != after['fsq'] or any(
        before['encoders'][k] != after['encoders'][k] for k in before['encoders'] if k != modality):
        raise RuntimeError('frozen native anchor changed')
    return dict(steps=steps, init=before, final=after, losses=losses,
                sampling_stream_sha256=hashlib.sha256(stream.tobytes()).hexdigest())


def save_checkpoint(model, path, config, evidence):
    payload = dict(selected_encoder=model.encoders[config['modality']].state_dict(),
                   config=config, evidence=evidence)
    torch.save(payload,path)
    return sha256(path)


def reload_checkpoint(upstream, path):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    cfg = payload['config']
    provenance = cfg.get('provenance', {})
    if provenance.get('protocol_sha256') != sha256(PROTOCOL):
        raise RuntimeError('checkpoint protocol provenance mismatch')
    if 'checkpoint_sha256' in provenance and provenance['checkpoint_sha256'] != sha256(HERE.parent / protocol()['native_checkpoint']):
        raise RuntimeError('checkpoint learned anchor provenance mismatch')
    if 'source_sha256' in provenance and provenance['source_sha256'] != verify_sources(Path(upstream)):
        raise RuntimeError('checkpoint source provenance mismatch')
    model = initialize(upstream,cfg['modality'],cfg['seed'])
    if fingerprints(model) != payload['evidence']['init']:
        raise RuntimeError('checkpoint initial anchor/provenance mismatch')
    model.encoders[cfg['modality']].load_state_dict(payload['selected_encoder'])
    if fingerprints(model) != payload['evidence']['final']:
        raise RuntimeError('checkpoint final hash mismatch')
    return model, payload


def aggregate(runs):
    summaries = []
    for modality in ROUTES:
        for mode in protocol()['conditions']:
            rows = [r for r in runs if r['config']['modality']==modality and r['config']['mode']==mode]
            metrics = {}
            for key in ('action_rmse_N','first_force_mse_N2','paired_common_first_force_bound_N2'):
                values = [r['final_errors']['test'][key] for r in rows]
                metrics[key] = dict(mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None)
            survival = []
            for offset in protocol()['evaluations']['closed_loop']['actual_state_offsets']:
                counts = [(sum(p['completed'] for p in r['rollouts'] if p['actual_offset_state4']==offset),
                           sum(len(p['episodes']) for p in r['rollouts'] if p['actual_offset_state4']==offset)) for r in rows]
                rates = [n/d for n,d in counts if d]
                cart_metrics = {}
                for completed_only in (False,True):
                    per_seed = []
                    for r in rows:
                        episodes = [e for p in r['rollouts'] if p['actual_offset_state4']==offset
                                    for e in p['episodes'] if not completed_only or e['completed']]
                        per_seed.append(float(np.mean([e['cart_mae_m'] for e in episodes])) if episodes else None)
                    values = [v for v in per_seed if v is not None]
                    cart_metrics['completed_cart_mae_m' if completed_only else 'executed_cart_mae_m'] = dict(
                        per_seed=per_seed,mean=float(np.mean(values)) if values else None,
                        sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None)
                survival.append(dict(offset=offset,per_seed_counts=counts,mean=float(np.mean(rates)) if rates else None,
                    sample_std=float(np.std(rates,ddof=1)) if len(rates)>1 else None,cart_metrics=cart_metrics))
            summaries.append(dict(modality=modality,mode=mode,encoder_initializations=len(rows),metrics=metrics,survival=survival))
    return summaries


def run(upstream, output_dir, steps=1200):
    validate_steps(steps)
    cfg = protocol(); upstream = Path(upstream); output = Path(output_dir)
    sources = verify_sources(upstream)
    checkpoint = HERE.parent / cfg['native_checkpoint']
    provenance = dict(protocol_sha256=sha256(PROTOCOL), checkpoint_sha256=sha256(checkpoint),
        plant_sha256=sha256(HERE/'cartpole.xml'),
        source_sha256=sources, local_source_sha256={f:sha256(HERE/f) for f in
            ('matched_cues.py','temporal_pairs.py','temporal_audit.py','encoder_transfer.py','concepts.py','cartpole.xml')})
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    plant = load_model(); records = generate_data(plant,cfg)
    accepted = [r for r in records if r['accepted']]
    summary = {s:dict(declared=sum(r['split']==s for r in records),accepted=sum(r['split']==s for r in accepted)) for s in cfg['data_splits']}
    save_pairs(records,output/'pairs.npz',summary,provenance)
    windows = {mode:{s:build_windows([r for r in records if r['split']==s],plant,mode)
                     for s in cfg['data_splits']} for mode in cfg['conditions']}
    runs = []; initial_weights = {}; shared = None
    # Complete all training before any final-model evaluation/rollout.
    for modality in cfg['encoders']:
        for seed in cfg['encoder_initialization_seeds']:
            for mode in cfg['conditions']:
                model = initialize(upstream,modality,seed)
                start = fingerprints(model)
                anchor = dict(decoders=start['decoders'],fsq=start['fsq'],original=start['encoders']['cartpole'])
                if shared is not None and shared != anchor: raise RuntimeError('shared learned anchor differs')
                shared = anchor
                key = (modality,seed)
                if key in initial_weights and initial_weights[key] != start['encoders'][modality]:
                    raise RuntimeError('cue conditions have different initialization')
                initial_weights[key] = start['encoders'][modality]
                evidence = train_encoder(model,windows[mode]['train'],modality,seed,steps)
                config = dict(modality=modality,mode=mode,seed=seed,steps=steps,
                    final_protocol=steps==1200,training={**cfg['training'],'steps':steps},provenance=provenance)
                name = f'{modality}-{mode}-{seed}-{steps}steps.pt'
                digest = save_checkpoint(model,output/name,config,evidence)
                runs.append(dict(config=config,training=evidence,
                                 checkpoint=name,checkpoint_sha256=digest))
                write_result(output/'progress.json',dict(phase='training',finished_models=len(runs),total_models=24))
    for number,row in enumerate(runs,1):
        model,_ = reload_checkpoint(upstream,output/row['checkpoint'])
        modality,mode = row['config']['modality'],row['config']['mode']; route=ROUTES[modality]
        initial_model = initialize(upstream,modality,row['config']['seed'])
        row['init_errors'] = {s:errors(initial_model,d,route) for s,d in windows[mode].items()}
        row['final_errors'] = {s:errors(model,d,route) for s,d in windows[mode].items()}
        row['pair_errors'] = []
        for split,d in windows[mode].items():
            predictions = np.asarray(row['final_errors'][split]['predictions_normalized'])
            for pair_id in dict.fromkeys(d['pair_id']):
                ix = d['pair_id']==pair_id; first=ix & (d['tick']==0)
                row['pair_errors'].append(dict(split=split,pair_id=str(pair_id),
                    action_rmse_N=float(10*np.sqrt(np.mean((predictions[ix]-d['action'][ix])**2))),
                    first_force_mse_N2=float(100*np.mean((predictions[first]-d['action'][first])**2)),
                    paired_common_first_force_bound_N2=blind_action_mse_bound(d['action'][first]*10)))
        row['rollouts'] = []
        for record in accepted:
            if record['split'] != 'test': continue
            extended,history = extend_reference(plant,record['states'])
            for offset in cfg['evaluations']['closed_loop']['actual_state_offsets']:
                result = control_rollout(model,plant,extended,history,route,mode,np.asarray(offset))
                result['pair_id'] = record['pair_id']; row['rollouts'].append(result)
        if modality=='markers' and mode=='full_future':
            d=windows[mode]['test']; inputs=pack_transfer(d['reference'],d['state'],2)
            row['marker_noise_probe'] = []
            for ps in (.001,.005):
                for vs in (.01,.05):
                    noisy = noisy_marker_inputs(inputs,ps,vs)
                    with torch.no_grad(): pred=model(noisy,return_dict=True)['action_mean'][:,0,0].numpy()
                    row['marker_noise_probe'].append(dict(position_sigma_m=ps,velocity_sigma_m_s=vs,
                        seed=94104,action_rmse_N=float(10*np.sqrt(np.mean((pred-d['action'])**2))),
                        predictions_normalized=pred.tolist(),scope='offline input sensitivity only'))
        if fingerprints(model) != row['training']['final']: raise RuntimeError('evaluation mutated model')
        write_result(output/'progress.json',dict(phase='evaluation',finished_models=number,total_models=24))
    if (verify_sources(upstream)!=sources or sha256(checkpoint)!=provenance['checkpoint_sha256']
        or sha256(PROTOCOL)!=provenance['protocol_sha256']
        or any(sha256(HERE/f)!=h for f,h in provenance['local_source_sha256'].items())):
        raise RuntimeError('source/checkpoint changed during execution')
    report = dict(schema='cartpole-sonic-matched-cues/v1',execution='COMPLETE',
        final_protocol=steps==1200,actual_steps=steps,protocol=cfg,provenance=provenance,
        generation=summary,pairs_sha256=sha256(output/'pairs.npz'),shared_anchor=shared,
        runs=runs,summary=aggregate(runs),
        interpretation='Three Encoder initializations conditional on one fixed learned Decoder; not three independent end-to-end policy runs.')
    write_result(output/'matched.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1200)
    args=parser.parse_args(argv)
    try:
        result=run(args.upstream,args.output_dir,args.steps)
        print(json.dumps(dict(execution=result['execution'],actual_steps=args.steps,final_protocol=result['final_protocol'])))
        return 0
    except Exception as exc:
        write_result(args.output_dir/'matched.json',dict(execution='FAILED',error_type=type(exc).__name__,
            actual_steps=args.steps,performance_claim='none'))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
