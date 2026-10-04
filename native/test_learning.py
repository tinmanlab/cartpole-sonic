"""Measurement contracts; no test requires a preferred performance winner."""
import importlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def test_measurement_driver_exists():
    assert (HERE/'learning.py').is_file(), 'bounded native learning/evaluation driver is missing'


@pytest.fixture(scope='module')
def policy():
    from core_smoke import import_native
    from training import build_models
    from cartpole_env import CartPoleEnv
    root=Path(os.environ['SONIC_UPSTREAM']);import_native(root)
    torch.set_num_threads(1);torch.manual_seed(9101)
    p,_,_=build_models(root,CartPoleEnv(8,9101))
    return p.eval()


def test_detects_initial_token_collapse_instead_of_calling_it_tracking(policy):
    learning=importlib.import_module('learning')
    result=learning.token_probe(policy)
    assert result['goals']==65 and result['unique_token_vectors']==1
    assert result['action_span_same_state']<1e-6
    assert result['reference_conditioned'] is False


def test_evaluation_is_replayable_and_does_not_change_policy_or_torch_rng(policy):
    learning=importlib.import_module('learning')
    from core_smoke import digest_parameters
    before=digest_parameters(policy);rng=torch.get_rng_state().clone()
    a=learning.evaluate_policy(policy,seed=18081,episodes=8,horizon=150)
    b=learning.evaluate_policy(policy,seed=18081,episodes=8,horizon=150)
    assert a==b and digest_parameters(policy)==before and torch.equal(rng,torch.get_rng_state())
    assert a['episodes']==8 and len(a['per_episode'])==8
    if a['survived']==0:assert a['completed_mae'] is None
    else:assert a['completed_mae']>=0
    json.dumps(a,allow_nan=False)


def test_no_reached_push_means_no_push_performance(policy):
    learning=importlib.import_module('learning')
    r=learning.evaluate_policy(policy,seed=8080,episodes=8,horizon=500,push=True)
    assert r['survived']==0 and r['push_reached']==0
    assert r['post_push_mae'] is None and r['post_push_survived']==0


def test_counterfactual_input_does_not_change_evaluation_targets(policy):
    learning=importlib.import_module('learning')
    a=learning.evaluate_policy(policy,seed=18081,episodes=8,horizon=150)
    b=learning.evaluate_policy(policy,seed=18081,episodes=8,horizon=150,reference_mode='zero')
    assert [e['goal'] for e in a['per_episode']]==[e['goal'] for e in b['per_episode']]
    assert a['reference_mode']=='normal' and b['reference_mode']=='zero'


def test_invalid_budget_is_rejected_before_model_or_environment_work(tmp_path):
    learning=importlib.import_module('learning')
    for value in [0,-1,513,True,2.5]:
        with pytest.raises(ValueError):learning.run_experiment(Path('/missing'),tmp_path,iterations=value)


def test_cli_failure_replaces_stale_result(tmp_path):
    import subprocess
    result=tmp_path/'learning.json';result.write_text('{"execution":"COMPLETE"}')
    child=subprocess.run([sys.executable,str(HERE/'learning.py'),'--iterations','0',
        '--output-dir',str(tmp_path)],capture_output=True,text=True)
    assert child.returncode!=0 and json.loads(result.read_text())['execution']=='FAILED'


def test_fsq_resolution_is_a_parameter_free_controlled_change():
    from core_smoke import import_native
    from cartpole_env import CartPoleEnv
    learning=importlib.import_module('learning')
    root=Path(os.environ['SONIC_UPSTREAM']);import_native(root)
    torch.manual_seed(9101);a,_,_=learning.build_experiment_models(root,CartPoleEnv(8,31),fsq_levels=5)
    torch.manual_seed(9101);b,_,_=learning.build_experiment_models(root,CartPoleEnv(8,31),fsq_levels=32)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert a.actor_module.quantizer._levels.tolist()==[5,5]
    assert b.actor_module.quantizer._levels.tolist()==[32,32]
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())


def test_short_experiment_uses_native_updates_and_preserves_checkpoint_configuration(tmp_path):
    learning=importlib.import_module('learning')
    result=learning.run_experiment(Path(os.environ['SONIC_UPSTREAM']),tmp_path,iterations=2,fsq_levels=32)
    assert result['execution']=='COMPLETE'
    assert result['control_transitions']==2048 and result['optimizer_steps']==8
    assert result['official_method_calls']['_compute_ppo_loss']==8
    assert result['checkpoint_action_parity_max_abs']==0
    restored=learning.load_policy(Path(os.environ['SONIC_UPSTREAM']),tmp_path/'weights-final.pt')
    assert restored.actor_module.quantizer._levels.tolist()==[32,32]


def test_stored_learning_artifacts_are_consistent_not_a_winner_assertion():
    import hashlib
    root=HERE.parent/'evidence/native_learning'
    summary=json.loads((root/'summary.json').read_text())
    for name,digest in summary['artifacts_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    assert summary['protocol']['distinct_training_seeds']==1
    assert len(summary['matched_budget_128'])==4
    for row in summary['matched_budget_128']:
        assert row['control_transitions']==128*1024 and row['optimizer_steps']==128*4
        assert 0<=row['survived']<=row['episodes']
    holdout=json.loads((root/'holdout.json').read_text())
    for run in holdout['runs']:
        conditions=run['results'];normal=conditions['clean']['per_episode']
        for key in ['zero_reference','negated_reference','push']:
            altered=conditions[key]['per_episode']
            assert [x['goal'] for x in normal]==[x['goal'] for x in altered]
            assert [x['initial_state'] for x in normal]==[x['initial_state'] for x in altered]
    for key,aggregate in summary['holdout_aggregate'].items():
        rows=[r['results'][key] for r in holdout['runs']]
        survived=sum(r['survived'] for r in rows)
        assert aggregate['survived']==survived
        if survived:
            expected=sum(r['completed_mae']*r['survived'] for r in rows if r['completed_mae'] is not None)/survived
            assert aggregate['completed_mae']==pytest.approx(expected)
        else:assert aggregate['completed_mae'] is None
