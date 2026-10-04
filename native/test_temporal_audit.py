"""Frozen-policy temporal-information tests; correctness does not require a winning policy."""
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def test_audit_module_exists():
    assert (HERE/'temporal_audit.py').is_file(), 'frozen temporal-information evaluator is missing'


@pytest.fixture(scope='module')
def audit():return importlib.import_module('temporal_audit')


@pytest.fixture(scope='module')
def frozen(audit):
    torch.set_num_threads(1)
    return audit.load_frozen(Path(os.environ['SONIC_UPSTREAM']))


def test_no_frozen_parameter_remains_trainable(frozen):
    assert not any(p.requires_grad for p in frozen.parameters())


def test_equal_cues_have_equal_first_reference_and_do_not_leak_interior(audit):
    initial=np.array([.1,0,0,0]);states=np.tile(initial,(2,57,1))
    states[0,4:24,0]+=.003;states[1,4:24,0]-=.003
    history=np.tile(initial,(2,29,1));raw=states.copy()
    for mode in ['current_only','history_only','endpoints_only']:
        out=audit.reference_inputs(states,history,0,mode,common_endpoint=initial)
        np.testing.assert_array_equal(out[0],out[1])
    out=audit.reference_inputs(states,history,0,'full_future',common_endpoint=initial)
    assert not np.array_equal(out[0],out[1])
    np.testing.assert_array_equal(states,raw)


def test_missing_future_is_not_padded(audit):
    states=np.zeros((2,29,4));history=np.zeros((2,29,4))
    with pytest.raises(ValueError):audit.reference_inputs(states,history,1,'full_future')


def test_paired_mean_bound_is_only_for_recorded_action_prediction(audit):
    assert audit.blind_action_mse_bound(np.array([1.,-1.]))==1.
    assert audit.blind_action_mse_bound(np.array([2.,4.]))==1.
    assert audit.blind_action_mse_bound(np.array([3.,3.]))==0.
    with pytest.raises(ValueError):audit.blind_action_mse_bound(np.array([1.,float('nan')]))


def test_zero_control_tail_is_actually_simulated(audit):
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(HERE/'cartpole.xml'))
    initial=np.array([.1,.03,.001,.02])
    rollout=audit.simulate(model,initial,np.zeros(28))
    assert rollout.shape==(29,4)
    assert np.isfinite(rollout).all()
    assert not np.array_equal(rollout[-1],rollout[0]), 'nonstationary state cannot be copied as padding'
    np.testing.assert_array_equal(rollout,audit.simulate(model,initial,np.zeros(28)))


def test_same_initial_reference_never_changes_actual_state(frozen,audit):
    initial=np.array([.12,0,0,0]);ref=np.tile(initial,(2,8,1))
    ref[0,3,0]+=.01;ref[1,3,0]-=.01
    actual=np.tile(initial,(2,1));saved=actual.copy()
    action,tokens=audit.predict(frozen,ref,actual,1)
    assert action.shape==(2,) and tokens.shape[0]==2
    np.testing.assert_array_equal(actual,saved)
    assert np.isfinite(action).all()


def test_protocol_rejects_changed_checkpoint(audit,tmp_path):
    file=tmp_path/'weights.pt';file.write_bytes(b'not the accepted model')
    with pytest.raises(ValueError,match='hash'):
        audit.load_frozen(Path(os.environ['SONIC_UPSTREAM']),checkpoint=file)


def test_stationary_reference_rollout_is_replayable_and_model_unchanged(audit,frozen):
    from core_smoke import digest_parameters
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(HERE/'cartpole.xml'))
    initial=np.array([0.,0.,0.,0.]);states=np.tile(initial,(2,57,1));history=np.tile(initial,(2,29,1))
    before=digest_parameters(frozen);rng=torch.get_rng_state().clone()
    a=audit.control_rollout(frozen,model,states,history,1,'full_future',np.zeros(4))
    b=audit.control_rollout(frozen,model,states,history,1,'full_future',np.zeros(4))
    assert a==b
    assert before==digest_parameters(frozen) and torch.equal(rng,torch.get_rng_state())
    assert len(a['episodes'])==2
    assert all(0<=e['steps']<=28 for e in a['episodes'])
    json.dumps(a,allow_nan=False)


def test_committed_pairs_reproduce_physics_and_all_declared_cases(audit):
    import mujoco
    from temporal_pairs import declared_cases
    folder=HERE.parent/'evidence/temporal_audit'
    report=json.loads((folder/'audit.json').read_text())
    assert report['optimizer_updates']==0 and report['frozen_model_unchanged']
    assert len(report['all_cases'])==12
    assert report['generation_summary']['accepted']+report['generation_summary']['rejected']==12
    assert audit.sha256(folder/'physical_pairs.npz')==report['physical_pair_artifact_sha256']
    assert audit.sha256(HERE/'temporal_audit_protocol.json')==report['protocol_sha256']
    for name,digest in report['local_source_sha256'].items():
        assert audit.sha256(HERE/name)==digest
    model=mujoco.MjModel.from_xml_path(str(HERE/'cartpole.xml'))
    with np.load(folder/'physical_pairs.npz',allow_pickle=False) as bank:
        for i,case in enumerate(declared_cases()):
            states=bank[f'case_{i}_states'];actions=bank[f'case_{i}_actions']
            for branch in range(2):
                replayed=audit.simulate(model,states[branch,0],actions[branch])
                np.testing.assert_allclose(replayed,states[branch],atol=1e-12,rtol=0)
            if bank[f'case_{i}_accepted']:
                np.testing.assert_array_equal(states[0,0],states[1,0])
                assert np.max(np.abs(states[:,-1]-states[0,0]))<=1e-7
                assert np.max(np.abs(states[0,4:25:4,0]-states[1,4:25:4,0]))>1e-4


def test_committed_information_audit_replays_without_winner_assertions(audit,frozen):
    import mujoco
    folder=HERE.parent/'evidence/temporal_audit';report=json.loads((folder/'audit.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(HERE/'cartpole.xml'))
    with np.load(folder/'physical_pairs.npz',allow_pickle=False) as bank:
        for pair in report['pairs']:
            i=pair['case']['case_id'];states=bank[f'case_{i}_states'];actions=bank[f'case_{i}_actions']
            extended,history=audit.extend_reference(model,states)
            for route,name in [(1,'joints'),(2,'markers')]:
                probe=audit.initial_probe(frozen,extended,history,actions,route)
                stored=next(p for p in pair['audit']['initial_probes'] if p['route']==name)
                for mode in audit.MODES:
                    a=probe['conditions'][mode];b=stored['conditions'][mode]
                    np.testing.assert_allclose(a['predicted_force_N'],b['predicted_force_N'],atol=1e-7,rtol=1e-5)
                    if mode!='full_future':
                        assert a['reference_pair_max_abs_delta']==0
                        assert a['tokens_identical']
                        # Float32 CPU kernels can differ by a few ulps for equal rows.
                        # Keep this below the audit's 1e-6 N distinguishability criterion.
                        assert a['force_pair_abs_delta_N']<=1e-7
                for mode in ['full_future','endpoints_only']:
                    now=audit.control_rollout(frozen,model,extended,history,route,mode,np.zeros(4))
                    then=next(r for r in pair['audit']['rollouts'] if r['encoder']==name and r['condition']==mode and not np.any(r['actual_offset_state4']))
                    assert now['completed']==then['completed']
                    assert now['mean_executed_cart_mae_m']==pytest.approx(then['mean_executed_cart_mae_m'],abs=1e-7,rel=1e-5)


def test_cli_ordinary_failure_replaces_stale_success(tmp_path):
    import subprocess
    (tmp_path/'audit.json').write_text('{"execution":"COMPLETE"}')
    p=subprocess.run([sys.executable,str(HERE/'temporal_audit.py'),'--upstream',str(tmp_path/'missing'),
                      '--output-dir',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode!=0
    assert json.loads((tmp_path/'audit.json').read_text())['execution']=='FAILED'
