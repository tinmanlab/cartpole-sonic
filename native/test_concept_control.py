"""Physical replay-task contracts before connecting original SONIC PPO."""
import importlib,os,sys
from pathlib import Path
import numpy as np
import pytest
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))


def test_replay_task_exists():
    assert (HERE/'concept_control.py').is_file()


@pytest.fixture(scope='module')
def bank():
    from concepts import motion_bank
    return motion_bank(Path(os.environ['SONIC_UPSTREAM']),episodes=6,steps=160)


def test_reference_does_not_follow_current_robot_or_cross_recorded_end(bank):
    mod=importlib.import_module('concept_control');env=mod.ReplayTask(bank,[0,1,2,3],num_envs=4)
    old=env.observations();env.base.data[0].qvel[0]+=.25
    changed=env.observations()
    assert torch.equal(old['tokenizer'],changed['tokenizer'])
    assert not torch.equal(old['actor_obs'],changed['actor_obs'])
    assert changed['critic_obs'].shape==(4,37)
    for _ in range(140):
        _,_,_,info=env.step({'actions':torch.zeros(4,1)})
        assert not info['time_outs'].any()
        assert all(env.cursor[i]+28<=bank['lengths'][env.selected[i]] for i in range(4))
    assert env.transitions==560


def test_finite_motion_completion_is_distinct_from_physical_failure(bank):
    mod=importlib.import_module('concept_control');env=mod.ReplayTask(bank,[0],num_envs=1)
    # Place at end of prerecorded task with its actual state, not a fake ideal state.
    env.cursor[0]=int(bank['lengths'][0])-29
    s=bank['states'][0,env.cursor[0]];d=env.base.data[0];d.qpos[:]=s[[0,2]];d.qvel[:]=s[[1,3]]
    u=bank['actions'][0,env.cursor[0]]
    _,_,done,info=env.step({'actions':torch.tensor([[u]],dtype=torch.float32)})
    assert done[0] and not info['time_outs'][0]
    assert env.completed_motions==1 and env.failed_motions==0


def test_bounded_training_arguments_fail_early(tmp_path):
    mod=importlib.import_module('concept_control')
    with pytest.raises(ValueError):mod.run(Path('/missing'),Path('/missing'),tmp_path,iterations=0)
