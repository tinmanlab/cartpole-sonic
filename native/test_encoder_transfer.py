"""Reuse the known working native decoder; verify no hidden label path or weight change."""
import importlib,os,sys
from pathlib import Path
import numpy as np
import torch
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))


def test_transfer_exists():assert (HERE/'encoder_transfer.py').is_file()


@pytest.fixture(scope='module')
def transfer():return importlib.import_module('encoder_transfer')


@pytest.fixture(scope='module')
def model(transfer):
    torch.set_num_threads(1)
    return transfer.make_transfer(Path(os.environ['SONIC_UPSTREAM']))


def test_native_branch_reproduces_saved_policy_exactly(transfer,model):
    from learning import load_policy
    from cartpole_env import CartPoleEnv
    old=load_policy(Path(os.environ['SONIC_UPSTREAM']),HERE.parent/'evidence/native_learning/weights-final.pt')
    env=CartPoleEnv(4,21);obs=env.observations();state=env.states()
    motion=np.repeat(state[:,None,:],8,axis=1);command=obs['tokenizer'].numpy().reshape(4,8,2)
    with torch.no_grad():
        expected=old({k:v[:,None] for k,v in obs.items()})
        actual=model(transfer.pack_transfer(motion,state,0,command))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_inactive_original_reference_cannot_leak_to_new_encoder(transfer,model):
    rng=np.random.default_rng(4);ref=rng.normal(0,.05,(4,8,4));state=ref[:,0]
    a=transfer.pack_transfer(ref,state,1)
    b={k:v.clone() for k,v in a.items()};b['tokenizer'][...,:16]=100
    with torch.no_grad():
        x=model(a,return_dict=True);y=model(b,return_dict=True)
    assert torch.equal(x['action_mean'],y['action_mean'])
    assert torch.equal(x['encoded_tokens']['joints'],y['encoded_tokens']['joints'])


def test_only_new_encoders_are_trainable(model):
    assert set(model.encoders)=={'cartpole','joints','markers'}
    for name,p in model.named_parameters():
        assert p.requires_grad==(name.startswith('encoders.joints.') or name.startswith('encoders.markers.'))


def test_transfer_budget_is_bounded(transfer,tmp_path):
    with pytest.raises(ValueError):transfer.run(Path('/missing'),Path('/missing'),tmp_path,steps=0)


def test_committed_transfer_preserves_original_native_weights(transfer):
    import hashlib,json
    from learning import load_policy
    root=Path(os.environ['SONIC_UPSTREAM']);folder=HERE.parent/'evidence/native_concepts'
    report=json.loads((folder/'transfer.json').read_text());checkpoint=folder/'encoder-transfer.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256']
    assert hashlib.sha256((folder/'motions.npz').read_bytes()).hexdigest()==report['dataset']['bank_sha256']
    adapted=transfer.make_transfer(root);adapted.load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
    original=load_policy(root,HERE.parent/'evidence/native_learning/weights-final.pt').actor_module
    for name,p in original.encoders['cartpole'].state_dict().items():assert torch.equal(p,adapted.encoders['cartpole'].state_dict()[name])
    for name,p in original.decoders.state_dict().items():assert torch.equal(p,adapted.decoders.state_dict()[name])


def test_committed_transfer_replays_measured_outcomes_not_a_winner_threshold(transfer):
    import json
    from concepts import replay
    folder=HERE.parent/'evidence/native_concepts';report=json.loads((folder/'transfer.json').read_text())
    model=transfer.make_transfer(Path(os.environ['SONIC_UPSTREAM']))
    model.load_state_dict(torch.load(folder/'encoder-transfer.pt',weights_only=True)['model'])
    with np.load(folder/'motions.npz',allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    for route,name in [(1,'joints'),(2,'markers')]:
        for case,kwargs in [('clean',{}),('push',{'push':True}),('current_only',{'frame_mode':'current_only'}),('reverse_interior',{'frame_mode':'reverse_future'})]:
            actual=replay(transfer.Bridge(model,route),bank,list(range(24,32)),**kwargs)
            expected=report['outcomes'][name][case]
            assert actual['completed']==expected['completed']
            assert actual['mean_steps']==pytest.approx(expected['mean_steps'],abs=1e-7)
            assert actual['executed_mae']==pytest.approx(expected['executed_mae'],rel=1e-5,abs=1e-7)
            assert actual['push_reached']==expected['push_reached']
