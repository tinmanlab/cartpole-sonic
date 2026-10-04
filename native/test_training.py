"""Tests exercise real MuJoCo and the original SONIC trainer, never a substitute PPO."""
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def test_training_adapter_exists():
    assert (HERE/'training.py').is_file(), 'native trainer integration is missing'
    assert (HERE/'cartpole_env.py').is_file(), 'physical environment adapter is missing'


@pytest.fixture
def env():
    cls = importlib.import_module('cartpole_env').CartPoleEnv
    return cls(num_envs=4, seed=41)


def test_real_physics_action_clipping_and_clock(env):
    before = env.states().copy()
    actions = torch.tensor([[2.],[-2.],[.5],[-.5]])
    obs,reward,done,info = env.step({'actions': actions})
    assert torch.equal(actions, torch.tensor([[2.],[-2.],[.5],[-.5]])), 'retain raw PPO samples'
    assert np.allclose(env.last_applied_force, [10.,-10.,5.,-5.])
    assert not np.array_equal(before, env.states())
    assert all(abs(d.time-.02)<1e-12 for d in env.data)
    assert env.physics_steps == 8 and env.control_transitions == 4
    assert obs['actor_obs'].shape==(4,4) and obs['tokenizer'].shape==(4,16)
    assert obs['critic_obs'].shape==(4,20)
    assert torch.isfinite(reward).all() and done.dtype==torch.bool
    assert not info['time_outs'].any()


def test_state_perturbation_does_not_rewrite_reference(env):
    before=env.observations()
    env.data[0].qvel[0] += .6
    env.data[0].qvel[1] -= .7
    after=env.observations()
    assert torch.equal(before['tokenizer'],after['tokenizer'])
    assert not torch.equal(before['actor_obs'],after['actor_obs'])


def test_failure_resets_only_failed_environment_and_reports_terminal(env):
    other_before=env.data[1].time
    env.data[0].qpos[1]=.9
    _,_,done,info=env.step({'actions': torch.zeros(4,1)})
    assert done.tolist()==[True,False,False,False]
    assert not info['time_outs'].any()
    assert abs(env.data[0].qpos[1])<.1
    assert env.data[1].time>other_before
    assert abs(env.last_terminal_states[0,2])>.65
    assert env.terminated_episodes==1


def test_reward_penalizes_tracking_error_without_using_actual_state_to_move_reference(env):
    targets=np.zeros(4)
    ideal=np.zeros((4,4))
    off=ideal.copy(); off[:,0]=.6
    assert np.all(env.reward(ideal,targets,np.zeros(4)) > env.reward(off,targets,np.zeros(4)))


def test_bad_actions_fail_before_stepping(env):
    with pytest.raises(ValueError): env.step({'actions': torch.full((4,1),float('nan'))})
    with pytest.raises(ValueError): env.step({'actions': torch.zeros(3,1)})
    assert env.control_transitions==0


@pytest.fixture(scope='module')
def report(tmp_path_factory):
    mod=importlib.import_module('training')
    return mod.run_training(Path(os.environ['SONIC_UPSTREAM']), tmp_path_factory.mktemp('actual-ppo'), iterations=2)


def test_official_trainer_methods_executed_unmodified(report):
    assert report['status']=='PASS'
    assert report['official_ppo_executed'] and report['physical_rollout_executed']
    assert report['trainer_class']=='gear_sonic.trl.trainer.ppo_trainer_aux_loss.TRLAuxLossPPOTrainer'
    assert report['upstream_clean']
    calls=report['official_method_calls']
    assert calls['train']==1 and calls['_rollout_step']==2 and calls['_compute_returns']==2
    assert calls['_compute_ppo_loss']>0 and calls['_compute_aux_loss']>0
    assert report['optimizer_steps']==8
    assert report['control_transitions']==2048 and report['physics_steps']==4096


def test_native_weights_are_updated_and_checkpoint_roundtrip_is_exact(report):
    assert all(v>0 for v in report['parameter_delta_l2'].values())
    assert report['checkpoint']['action_parity_max_abs']==0
    assert report['checkpoint']['kind']=='weights-only; not exact-resume trainer state'
    assert all(np.isfinite(v) for v in report['parameter_delta_l2'].values())


def test_real_training_loss_and_evaluation_are_reported_without_success_requirement(report):
    assert report['losses_finite'] and report['reconstruction_loss_observed']
    assert report['initialization']=='random; no teacher, no stabilizing controller'
    assert report['performance_claim']=='not established by this integration run'
    for phase in ['before','after']:
        for scenario in ['clean','push']:
            metrics=report['evaluation'][phase][scenario]
            assert metrics['episodes']==8 and metrics['horizon_steps']==500
            assert 0<=metrics['survived']<=8
            assert metrics['all_executed_step_mae']>=0
            assert 0<metrics['mean_steps']<=500
    json.dumps(report,allow_nan=False)


def test_rollout_or_evaluation_horizon_does_not_fake_training_termination(env):
    for d in env.data:
        d.qpos[:]=0;d.qvel[:]=0
    for _ in range(501):
        _,_,done,info=env.step({'actions':torch.zeros(4,1)})
        assert not done.any() and not info['time_outs'].any()
    assert (env.episode_steps==501).all()


def test_actual_official_actor_std_is_not_the_upstream_printer_fallback():
    # Upstream's console reports zero when log_std is used. Verify the real distribution.
    from core_smoke import import_native
    from training import build_models
    from cartpole_env import CartPoleEnv
    root=Path(os.environ['SONIC_UPSTREAM']);import_native(root)
    policy,_,_=build_models(root,CartPoleEnv(2))
    assert float(policy.get_std.detach()[0])==pytest.approx(.12,rel=1e-6)


def test_failed_training_does_not_leave_a_stale_pass(tmp_path):
    import subprocess
    target=tmp_path/'native_training.json'
    target.write_text('{"status":"PASS"}')
    result=subprocess.run([sys.executable,'-S',str(HERE/'training.py'),
        '--upstream',os.environ['SONIC_UPSTREAM'],'--output-dir',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode!=0
    assert json.loads(target.read_text())['status']=='FAIL'


def test_committed_training_evidence_matches_fresh_run(report):
    expected=json.loads((HERE.parent/'evidence/native_training.json').read_text())
    for key in ['source_sha256','local_source_sha256','versions','official_method_calls','configuration']:
        assert report[key]==expected[key]
    assert report['parameter_delta_l2']==pytest.approx(expected['parameter_delta_l2'],rel=1e-5,abs=1e-7)
    for actual,stored in zip(report['losses'],expected['losses']):
        assert actual==pytest.approx(stored,rel=1e-5,abs=1e-7)
    for phase in ['before','after']:
        for scenario in ['clean','push']:
            assert report['evaluation'][phase][scenario]['survived']==expected['evaluation'][phase][scenario]['survived']
