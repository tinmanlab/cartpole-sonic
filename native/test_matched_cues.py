"""Bounded independent checks; no full experiment is executed here."""
import gzip
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
import matched_cues as mc


def test_cases_and_budget():
    cases = mc.declared_cases()
    assert {k: len(v) for k, v in cases.items()} == {'train': 36, 'validation': 12, 'test': 24}
    assert len({c.x for rows in cases.values() for c in rows}) == 72
    assert [(c.tick, c.amplitude) for c in cases['train'][:12]] == [(t, a) for t in (8,12,16) for a in (.001,.002,.004,.006)]
    for budget in (0, 1201, True, 1.5):
        with pytest.raises(ValueError): mc.validate_steps(budget)


def test_marker_contract():
    from concepts import to_markers
    a = np.array([0.,0.,.2,0.])
    b = np.array([2*np.sin(.2),0.,-.2,0.])
    assert np.allclose(to_markers(a)[[1,2,4,5]], to_markers(b)[[1,2,4,5]])
    assert not np.array_equal(a,b)
    assert np.allclose(mc.validate_markers(to_markers(a), np.ones(6,bool)), a)
    with pytest.raises(ValueError): mc.validate_markers(to_markers(a), [False,True,True,False,True,True])


@pytest.fixture(scope='module')
def physical():
    from temporal_pairs import Case, generate_pair, load_model
    plant = load_model()
    records = []
    for i,(split,x) in enumerate((('train',.03),('test',-.04))):
        r=generate_pair(Case(x,8,.001),model=plant)
        assert r['accepted'], r['reason']
        r.update(split=split,pair_id=f'{split}-{i}')
        records.append(r)
    return plant,records


def test_windows_grouping_and_information(physical):
    plant,records=physical
    for mode in mc.protocol()['conditions']:
        d=mc.build_windows(records,plant,mode)
        assert len(d['action'])==112
        assert set(d)=={'reference','state','target','action','split','pair_id','sign','tick'}
        for i,r in enumerate(records):
            rows=slice(i*56,(i+1)*56)
            assert set(d['split'][rows])=={r['split']}
            assert set(d['pair_id'][rows])=={r['pair_id']}
            assert np.array_equal(d['action'][rows].reshape(28,2).T,r['actions'])
            assert np.array_equal(d['state'][rows].reshape(28,2,4).transpose(1,0,2),r['states'][:,:28])
            assert np.array_equal(d['target'][rows].reshape(28,2,4).transpose(1,0,2),r['states'][:,1:])
            first=d['reference'][i*56:i*56+2]
            if mode=='full_future': assert not np.array_equal(first[0],first[1])
            else: assert first[0].tobytes()==first[1].tobytes()
        packed=mc.pack_transfer(d['reference'],d['state'],1)
        assert set(packed)=={'actor_obs','tokenizer'}
        assert packed['tokenizer'].shape==(112,1,99)


def test_noise_isolation(physical):
    plant,records=physical
    d=mc.build_windows(records,plant,'full_future')
    snapshot={k:v.copy() for k,v in d.items()}
    inputs=mc.pack_transfer(d['reference'],d['state'],2)
    out=mc.noisy_marker_inputs(inputs,.001,.01)
    assert torch.equal(inputs['actor_obs'],out['actor_obs'])
    assert torch.equal(inputs['tokenizer'][...,:48],out['tokenizer'][...,:48])
    assert torch.equal(inputs['tokenizer'][...,96:],out['tokenizer'][...,96:])
    assert not torch.equal(inputs['tokenizer'][...,48:96],out['tokenizer'][...,48:96])
    assert all(np.array_equal(d[k],v) for k,v in snapshot.items())
    assert torch.equal(out['tokenizer'],mc.noisy_marker_inputs(inputs,.001,.01)['tokenizer'])
    with pytest.raises(ValueError): mc.noisy_marker_inputs(mc.pack_transfer(d['reference'],d['state'],1),.001,.01)


def test_native_training_checkpoint_and_rollout(physical,tmp_path):
    import os
    from core_smoke import import_native
    from temporal_audit import extend_reference, control_rollout
    root=Path(os.environ['SONIC_UPSTREAM'])
    torch.set_num_threads(1)
    plant,records=physical
    Universal,Base,FSQ=import_native(root)
    for modality in ('joints','markers'):
        model=mc.initialize(root,modality,52101)
        twin=mc.initialize(root,modality,52101)
        assert mc.fingerprints(model)==mc.fingerprints(twin)
        assert type(model) is Universal and type(model.quantizer) is FSQ
        assert all(type(v) is Base for v in [*model.encoders.values(),*model.decoders.values()])
        assert all(p.requires_grad == name.startswith(f'encoders.{modality}.') for name,p in model.named_parameters())
        before={k:v.clone() for k,v in model.state_dict().items()}
        data=mc.build_windows(records[:1],plant,'full_future')
        stream=mc.sample_stream(1,2,52101)
        assert np.array_equal(stream[:,1::2],stream[:,::2]+1)
        evidence=mc.train_encoder(model,data,modality,52101,2)
        after=model.state_dict()
        assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith(f'encoders.{modality}.'))
        assert all(torch.equal(v,after[k]) for k,v in before.items() if not k.startswith(f'encoders.{modality}.'))
        cfg=dict(modality=modality,seed=52101,mode='full_future',steps=2,provenance={'protocol_sha256':mc.sha256(mc.PROTOCOL)})
        path=tmp_path/f'{modality}.pt'
        digest=mc.save_checkpoint(model,path,cfg,evidence)
        restored,payload=mc.reload_checkpoint(root,path)
        assert digest==mc.sha256(path) and payload['config']==cfg
        assert mc.fingerprints(restored)==mc.fingerprints(model)
        assert mc.errors(restored,data,mc.ROUTES[modality])==mc.errors(model,data,mc.ROUTES[modality])
        extended,history=extend_reference(plant,records[1]['states'])
        result=control_rollout(restored,plant,extended,history,mc.ROUTES[modality],'full_future',np.zeros(4))
        assert len(result['episodes'])==2 and result['horizon_steps']==28
        payload['evidence']['final']['fsq']='invalid'
        torch.save(payload,path)
        with pytest.raises(RuntimeError): mc.reload_checkpoint(root,path)


def test_invalid_cli_replaces_stale_complete(tmp_path,monkeypatch):
    path=tmp_path/'matched.json'
    path.write_text('{"execution":"COMPLETE"}')
    monkeypatch.setattr(mc,'generate_data',lambda *a: pytest.fail('heavy work before budget validation'))
    assert mc.main(['--output-dir',str(tmp_path),'--steps','1201'])==1
    import json
    assert json.loads(path.read_text())['execution']=='FAILED'


def test_numeric_archive_retains_all_solver_records(physical,tmp_path):
    plant,records=physical
    rejected={**records[1],'accepted':False,'reason':'test rejection','pair_id':'test-rejected'}
    path=tmp_path/'pairs.npz'
    mc.save_pairs([records[0],rejected],path,{'declared':2},{'protocol_sha256':mc.sha256(mc.PROTOCOL)})
    with np.load(path,allow_pickle=False) as archive:
        assert all(archive[k].dtype.kind in 'biuf' for k in archive.files)
        for i,r in enumerate((records[0],rejected)):
            for key in ('actions','states','candidate_controls','candidate_residuals','candidate_failures','solver_controls','solver_log'):
                assert np.array_equal(archive[f'case_{i}_{key}'],r[key])
        assert archive['case_1_reason'].tobytes().decode()=='test rejection'
        assert archive['case_1_split'].tobytes().decode()=='test'


def test_fingerprint_includes_nonpersistent_quantization_buffers():
    from vector_quantize_pytorch import FSQ
    q=FSQ(levels=[32,32],return_indices=False)
    # Upstream deliberately excludes these buffers from state_dict().
    assert not q.state_dict()
    original=mc.state_hash(q)
    q._levels[0]=31
    assert mc.state_hash(q)!=original, 'quantization configuration must not hash as empty state'


def test_committed_protocol_artifacts_and_matched_budgets():
    import json
    root=Path(__file__).resolve().parent.parent/'evidence/matched_cues'
    report=json.loads(gzip.decompress((root/'matched.json.gz').read_bytes()))
    assert report['execution']=='COMPLETE' and report['final_protocol']
    assert report['actual_steps']==1200 and len(report['runs'])==24
    assert report['pairs_sha256']==mc.sha256(root/'pairs.npz')
    assert report['provenance']['protocol_sha256']==mc.sha256(mc.PROTOCOL)
    for file,digest in report['provenance']['local_source_sha256'].items():
        assert mc.sha256(mc.HERE/file)==digest
    initial={};streams={};checkpoints=set()
    for r in report['runs']:
        c=r['config'];e=r['training'];key=(c['modality'],c['seed'])
        assert c['steps']==e['steps']==len(e['losses'])==1200
        assert e['init']['decoders']==e['final']['decoders']==report['shared_anchor']['decoders']
        assert e['init']['fsq']==e['final']['fsq']==report['shared_anchor']['fsq']
        assert e['init']['encoders']['cartpole']==e['final']['encoders']['cartpole']
        if key in initial:assert initial[key]==e['init']['encoders'][c['modality']]
        initial[key]=e['init']['encoders'][c['modality']]
        if c['seed'] in streams:assert streams[c['seed']]==e['sampling_stream_sha256']
        streams[c['seed']]=e['sampling_stream_sha256']
        assert mc.sha256(root/r['checkpoint'])==r['checkpoint_sha256']
        checkpoints.add(r['checkpoint'])
    assert len(checkpoints)==24 and len(initial)==6 and len(streams)==3
    with np.load(root/'pairs.npz',allow_pickle=False) as bank:
        seen=set()
        counts={split:[0,0] for split in mc.protocol()['data_splits']}
        for i in range(72):
            split=bank[f'case_{i}_split'].tobytes().decode()
            pair_id=bank[f'case_{i}_pair_id'].tobytes().decode()
            assert pair_id not in seen;seen.add(pair_id)
            counts[split][0]+=1;counts[split][1]+=int(bank[f'case_{i}_accepted'])
        for split,(declared,accepted) in counts.items():
            assert report['generation'][split]=={'declared':declared,'accepted':accepted}
        assert len(seen)==72
    # Summaries are recomputed from individual runs, not asserted to favor future input.
    assert report['summary']==json.loads(json.dumps(mc.aggregate(report['runs'])))


def test_committed_retrained_encoder_checkpoints_reproduce_test_predictions():
    import json,os
    root=Path(__file__).resolve().parent.parent/'evidence/matched_cues'
    report=json.loads(gzip.decompress((root/'matched.json.gz').read_bytes()))
    plant=mc.load_model();records=[]
    with np.load(root/'pairs.npz',allow_pickle=False) as bank:
        for i in range(72):
            split=bank[f'case_{i}_split'].tobytes().decode()
            if split!='test' or not bank[f'case_{i}_accepted']:continue
            records.append(dict(accepted=True,split=split,pair_id=bank[f'case_{i}_pair_id'].tobytes().decode(),
                states=bank[f'case_{i}_states'],actions=bank[f'case_{i}_actions'],
                canonical_endpoint=bank[f'case_{i}_canonical_endpoint']))
    assert records
    # All 24 configurations are reloaded; no re-training in CI.
    torch.set_num_threads(1)
    for row in report['runs']:
        model,_=mc.reload_checkpoint(Path(os.environ['SONIC_UPSTREAM']),root/row['checkpoint'])
        data=mc.build_windows(records,plant,row['config']['mode'])
        actual=mc.errors(model,data,mc.ROUTES[row['config']['modality']])
        expected=row['final_errors']['test']
        for metric in ('action_rmse_N','first_force_mse_N2','paired_common_first_force_bound_N2'):
            assert actual[metric]==pytest.approx(expected[metric],rel=1e-5,abs=1e-7)


def test_compact_summary_is_a_projection_of_the_compressed_raw_report():
    import json,hashlib
    root=mc.HERE.parent/'evidence/matched_cues'
    packed=(root/'matched.json.gz').read_bytes();raw=gzip.decompress(packed)
    report=json.loads(raw);summary=json.loads((root/'summary.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==summary['raw_report_sha256']
    assert hashlib.sha256(packed).hexdigest()==summary['compressed_report_sha256']
    for key in ('execution','final_protocol','actual_steps','provenance','protocol','generation','shared_anchor','summary'):
        assert summary[key]==report[key]
    assert summary['models']==len(report['runs'])==24
    for row in summary['checkpoints']:
        assert row['sha256']==mc.sha256(root/row['file'])


def test_committed_models_replay_their_own_trained_cue_conditions():
    import json,os
    from temporal_audit import extend_reference,control_rollout
    root=mc.HERE.parent/'evidence/matched_cues'
    report=json.loads(gzip.decompress((root/'matched.json.gz').read_bytes()))
    with np.load(root/'pairs.npz',allow_pickle=False) as bank:
        indices=[i for i in range(72) if bank[f'case_{i}_split'].tobytes().decode()=='test' and bank[f'case_{i}_accepted']]
        i=indices[0];pair_id=bank[f'case_{i}_pair_id'].tobytes().decode();states=bank[f'case_{i}_states'].copy()
    plant=mc.load_model();extended,history=extend_reference(plant,states);torch.set_num_threads(1)
    for row in report['runs']:
        if row['config']['seed']!=52101:continue
        model,_=mc.reload_checkpoint(Path(os.environ['SONIC_UPSTREAM']),root/row['checkpoint'])
        before=mc.fingerprints(model)
        for offset in mc.protocol()['evaluations']['closed_loop']['actual_state_offsets']:
            actual=control_rollout(model,plant,extended,history,mc.ROUTES[row['config']['modality']],row['config']['mode'],np.asarray(offset))
            expected=next(p for p in row['rollouts'] if p['pair_id']==pair_id and p['actual_offset_state4']==offset)
            assert actual['completed']==expected['completed']
            assert actual['mean_executed_cart_mae_m']==pytest.approx(expected['mean_executed_cart_mae_m'],rel=1e-4,abs=1e-7)
        assert mc.fingerprints(model)==before
