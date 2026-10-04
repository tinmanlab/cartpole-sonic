"""Recorded lesson values must come from actual frozen native execution, not UI mockups."""
import importlib
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))


def test_exporter_exists():
    assert (HERE/'export_guided_lesson.py').is_file(), 'native trace exporter is not implemented'


@pytest.fixture(scope='module')
def exporter():return importlib.import_module('export_guided_lesson')


@pytest.fixture(scope='module')
def fresh(exporter,tmp_path_factory):
    torch.set_num_threads(1)
    return exporter.export_lesson(Path(os.environ['SONIC_UPSTREAM']),tmp_path_factory.mktemp('lesson')/'traces.json')


def test_selection_is_fixed_by_stored_pair_id_not_performance(fresh):
    assert fresh['schema']=='cartpole-sonic-guided-lesson/v1'
    assert fresh['source']['encoderSeed']==52101 and fresh['source']['modality']=='joints'
    assert fresh['source']['optimizerUpdates']==0 and fresh['source']['frozenModelsUnchanged']
    assert len(fresh['cases'])==3
    assert [c['id'] for c in fresh['cases']]==sorted(fresh['selection']['acceptedTestPairIds'])[:3]
    assert fresh['source']['matchedStoredOutcomes']


def test_every_value_is_aligned_to_the_actual_sample_time(fresh):
    for case in fresh['cases']:
        assert len(case['runs'])==8
        refs=np.asarray(case['reference'])
        np.testing.assert_allclose(refs[:,0],refs[:,-1],atol=1e-7,rtol=0)
        for run in case['runs']:
            assert len(run['frames'])==run['metrics']['steps']+1
            assert run['frames'][-1]['decision'] is None
            for tick,frame in enumerate(run['frames']):
                assert frame['tick']==tick and frame['time']==pytest.approx(tick*.02)
                np.testing.assert_array_equal(frame['referenceNow'],refs[run['branch'],tick])
                assert frame['errorX']==pytest.approx(frame['state'][0]-frame['referenceNow'][0])
                if frame['decision'] is not None:
                    d=frame['decision']
                    assert len(d['referenceWindow'])==8 and len(d['token'])==len(d['latent'])==4
                    assert np.isfinite([*d['token'],*d['latent'],d['requestedForceN'],d['appliedForceN']]).all()
                    assert d['appliedForceN']==pytest.approx(np.clip(d['requestedForceN'],-10,10),abs=1e-6)
            errors=[abs(f['errorX']) for f in run['frames'][1:]]
            assert run['metrics']['cartMae']==pytest.approx(np.mean(errors),abs=1e-12)


def test_recorded_applied_actions_reproduce_every_native_state(fresh):
    model=mujoco.MjModel.from_xml_path(str(HERE/'cartpole.xml'))
    for case in fresh['cases']:
        for run in case['runs']:
            d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
            s=np.array(run['frames'][0]['state']);d.qpos[:]=s[[0,2]];d.qvel[:]=s[[1,3]]
            mujoco.mj_forward(model,d)
            for before,after in zip(run['frames'],run['frames'][1:]):
                d.ctrl[0]=before['decision']['appliedForceN']/10
                mujoco.mj_step(model,d);mujoco.mj_step(model,d)
                expected=[d.qpos[0],d.qvel[0],d.qpos[1],d.qvel[1]]
                np.testing.assert_allclose(after['state'],expected,atol=1e-10,rtol=1e-10)


def test_condition_and_offset_selection_never_rewrite_true_targets(fresh):
    for case in fresh['cases']:
        for branch in (0,1):
            runs=[r for r in case['runs'] if r['branch']==branch]
            for run in runs:
                np.testing.assert_array_equal([f['referenceNow'] for f in run['frames']],case['reference'][branch])
            nominal=[r for r in runs if r['offset']=='nominal']
            perturbed=[r for r in runs if r['offset']=='perturbed']
            for a,b in zip(nominal,perturbed):
                np.testing.assert_allclose(np.array(b['frames'][0]['state'])-a['frames'][0]['state'],[0,.02,.001,-.02],atol=1e-12)


def test_committed_lesson_is_exactly_the_fresh_export(fresh):
    stored=json.loads((ROOT/'evidence/guided_lesson/traces.json').read_text())
    assert stored['source']==fresh['source']
    assert stored['protocol']==fresh['protocol']
    assert len(stored['cases'])==len(fresh['cases'])
    for actual,expected in zip(fresh['cases'],stored['cases']):
        assert actual['id']==expected['id']
        for a,b in zip(actual['runs'],expected['runs']):
            assert a['condition']==b['condition'] and a['offset']==b['offset'] and a['branch']==b['branch']
            for x,y in zip(a['frames'],b['frames']):
                np.testing.assert_allclose(x['state'],y['state'],rtol=1e-6,atol=1e-8)
                if x['decision']:
                    np.testing.assert_allclose(x['decision']['token'],y['decision']['token'],rtol=0,atol=0)
                    assert x['decision']['requestedForceN']==pytest.approx(y['decision']['requestedForceN'],rel=1e-5,abs=1e-7)


def test_invalid_source_fails_without_reusing_stored_success(exporter,tmp_path):
    output=tmp_path/'traces.json'
    output.write_text('{"schema":"stale-success"}')
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):
        exporter.export_lesson(tmp_path/'missing',output)


def test_initial_state_offset_changes_feedback_state_not_reference_token(fresh):
    for case in fresh['cases']:
        for condition in fresh['protocol']['conditions']:
            for branch in (0,1):
                runs={r['offset']:r for r in case['runs'] if r['condition']==condition and r['branch']==branch}
                a=runs['nominal']['frames'][0];b=runs['perturbed']['frames'][0]
                np.testing.assert_array_equal(a['decision']['referenceWindow'],b['decision']['referenceWindow'])
                np.testing.assert_array_equal(a['decision']['token'],b['decision']['token'])
                assert not np.array_equal(a['state'],b['state'])
