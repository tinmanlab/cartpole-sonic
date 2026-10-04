"""Concept contracts, not an assertion that SONIC must beat a baseline."""
import importlib
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def test_concept_runner_exists():
    assert (HERE/'concepts.py').is_file(), 'native multi-representation / future tests are missing'


@pytest.fixture(scope='module')
def concept():
    return importlib.import_module('concepts')


@pytest.fixture(scope='module')
def model(concept):
    root=Path(os.environ['SONIC_UPSTREAM'])
    torch.set_num_threads(1)
    return concept.make_model(root,seed=7101)


def test_marker_mapping_is_invertible_and_matches_mujoco(concept):
    import mujoco
    from cartpole_env import CartPoleEnv
    env=CartPoleEnv(1,71)
    states=np.random.default_rng(12).uniform([-1,-.6,-.5,-1],[1,.6,.5,1],(20,4))
    markers=concept.to_markers(states)
    np.testing.assert_allclose(concept.from_markers(markers),states,atol=1e-12)
    for s,m in zip(states,markers):
        d=env.data[0];d.qpos[:]=s[[0,2]];d.qvel[:]=s[[1,3]]
        mujoco.mj_forward(env.model,d)
        body=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'pole')
        p=d.xpos[body]+d.xmat[body].reshape(3,3)@np.array([0.,0.,1.])
        jac=np.zeros((3,2));mujoco.mj_jac(env.model,d,jac,None,p,body)
        velocity=jac@d.qvel
        np.testing.assert_allclose(m, [s[0],p[0],p[2]-1,s[1],velocity[0],velocity[2]],atol=1e-12)


def test_bad_route_and_shape_fail_before_official_scatter(concept):
    states=np.zeros((4,8,4));proprio=np.zeros((4,4))
    for route in [-1,2,'both']:
        with pytest.raises(ValueError):concept.pack(states,proprio,route)
    with pytest.raises(ValueError):concept.pack(states[:,:7],proprio,0)
    with pytest.raises(ValueError):concept.pack(states,np.zeros((3,4)),0)


def test_real_dual_encoder_and_single_shared_decoder(model):
    from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule
    from gear_sonic.trl.modules.base_module import BaseModule
    from vector_quantize_pytorch import FSQ
    assert type(model) is UniversalTokenModule
    assert set(model.encoders)=={'joints','markers'}
    assert set(model.decoders)=={'g1_dyn','motion_kin'}
    assert all(type(x) is BaseModule for x in [*model.encoders.values(),*model.decoders.values()])
    assert type(model.quantizer) is FSQ
    assert model.decoder_input_features['g1_dyn']==['token_flattened','proprioception']
    assert model.decoder_input_features['motion_kin']==['token']


def test_mixed_selector_rows_match_separate_routes(concept,model):
    states=np.random.default_rng(4).normal(0,.1,(4,8,4));proprio=states[:,0].copy()
    with torch.no_grad():
        a=model(concept.pack(states,proprio,0),return_dict=True)
        b=model(concept.pack(states,proprio,1),return_dict=True)
        mixed=model(concept.pack(states,proprio,np.array([0,1,0,1])),return_dict=True)
    expected=torch.cat([a['action_mean'][0:1],b['action_mean'][1:2],a['action_mean'][2:3],b['action_mean'][3:4]])
    torch.testing.assert_close(mixed['action_mean'],expected)


def test_unselected_modality_and_actual_state_cannot_change_reference_token(concept,model):
    states=np.random.default_rng(13).normal(0,.1,(4,8,4));proprio=np.zeros((4,4))
    x=concept.pack(states,proprio,0);other={k:v.clone() for k,v in x.items()}
    # Joint reference occupies the first 32 scalars. Corrupt only unused markers.
    other['tokenizer'][...,32:80]=900
    other['actor_obs']+=.15
    with torch.no_grad():
        a=model(x,return_dict=True);b=model(other,return_dict=True)
    assert torch.equal(a['encoded_tokens']['joints'],b['encoded_tokens']['joints'])
    assert not torch.equal(a['action_mean'],b['action_mean'])
    torch.testing.assert_close(a['decoded_outputs']['motion_kin']['joint_reference'],b['decoded_outputs']['motion_kin']['joint_reference'])


def test_feasible_forks_share_current_state_but_not_future(concept):
    pair=concept.physics_forks()
    assert pair['reference'].shape==(2,8,4)
    np.testing.assert_array_equal(pair['reference'][0,0],pair['reference'][1,0])
    np.testing.assert_array_equal(pair['initial_states'][0],pair['initial_states'][1])
    assert not np.allclose(pair['reference'][0,1:],pair['reference'][1,1:])
    assert pair['first_actions'][0]!=pair['first_actions'][1]
    assert pair['source']=='native MuJoCo stepped with recorded bounded inputs'


def test_short_bank_splits_by_episode_not_correlated_windows(concept):
    root=Path(os.environ['SONIC_UPSTREAM'])
    bank=concept.motion_bank(root,episodes=6,steps=160)
    a=concept.windows(bank,ids=[0,1,2,3]);b=concept.windows(bank,ids=[4,5])
    assert len(a['reference'])>0 and len(b['reference'])>0
    assert set(a['episode_id']).isdisjoint(set(b['episode_id']))
    assert a['reference'].shape[1:]==(8,4)
    assert np.isfinite(a['reference']).all()
    for ref,s in zip(a['reference'],a['state']):np.testing.assert_array_equal(ref[0],s)


def test_frame_order_interventions_do_not_edit_reference_bank(concept):
    ref=np.random.default_rng(3).normal(size=(3,8,4));saved=ref.copy()
    for mode in ['current_only','repeat_first','reverse_future','normal']:
        altered=concept.intervene(ref,mode)
        np.testing.assert_array_equal(ref,saved)
        np.testing.assert_array_equal(altered[:,0],ref[:,0])
    assert concept.intervene(ref,'current_only').shape==ref.shape


def test_fixed_decoders_really_remain_fixed_during_alignment(concept,model):
    from core_smoke import digest_parameters
    frozen=concept.freeze_anchor(model)
    assert 'encoders.joints' in frozen and 'decoders' in frozen
    expected=digest_parameters(model.decoders)
    ref=np.random.default_rng(17).normal(0,.1,(8,8,4))
    opt=torch.optim.Adam(model.encoders['markers'].parameters(),lr=1e-3)
    concept.alignment_step(model,opt,ref,ref[:,0])
    assert digest_parameters(model.decoders)==expected
    assert all(not p.requires_grad for p in model.encoders['joints'].parameters())


def test_invalid_budget_rejected_before_loading_data(concept,tmp_path):
    with pytest.raises(ValueError):concept.run(Path('/missing'),tmp_path,steps=0)


def test_tip_only_observation_can_be_ambiguous_even_in_cartpole(concept):
    theta=.2
    states=np.array([[0.,0.,theta,0.],[2*np.sin(theta),0.,-theta,0.]])
    full=concept.to_markers(states)
    # Same tip position/velocity, different cart position/pole angle.
    np.testing.assert_allclose(full[0,[1,2,4,5]],full[1,[1,2,4,5]],atol=1e-12)
    assert not np.allclose(states[0],states[1])
    assert full[0,0]!=full[1,0]  # The cart marker resolves this ambiguity.


def test_recorded_concept_evidence_has_matching_inputs_and_scope():
    import hashlib,json
    folder=HERE.parent/'evidence/native_concepts'
    offline=json.loads((folder/'concepts.json').read_text())
    for name,digest in offline['artifacts_sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    no_aux=json.loads((folder/'no_aux.json').read_text())
    assert no_aux['shared_bank_sha256']==offline['artifacts_sha256']['motions.npz']
    assert no_aux['training']['steps']==offline['anchor_fit']['steps']
    assert no_aux['training']['aux_coef']==0 and offline['anchor_fit']['aux_coef']==.5
    refined=json.loads((folder/'refinement.json').read_text())
    assert refined['original_trainer']=='gear_sonic.trl.trainer.ppo_trainer_aux_loss.TRLAuxLossPPOTrainer'
    assert refined['control_transitions']==refined['iterations']*8*96
    assert refined['optimizer_steps']==refined['iterations']*4
    assert hashlib.sha256((folder/'refined-concept-weights.pt').read_bytes()).hexdigest()==refined['checkpoint_sha256']
    for r in refined['outcomes'].values():
        if 'clean' in r:
            for item in r.values():
                if item['completed']==0:assert item['completed_mae'] is None
