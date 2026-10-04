"""Cross-runtime task-contract check: same base plant, inputs, units and clocks.

Different controllers/native task timeouts are NOT equated. No stored success
report, network request or new simulation wrapper is used as an oracle.
"""
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent.parent


def test_wasm_and_native_base_physics_use_same_units_and_control_clock():
    node=shutil.which('node')
    assert node, 'Node is required for actual WASM/native differential verification'
    rng=np.random.default_rng(78133)
    cases=[{'state':rng.uniform([-0.3,-0.1,-0.04,-0.1],[0.3,0.1,0.04,0.1]).tolist(),
            'forces':([12.,-12.,0.,1.,-1.,.2,-.2,0.]*2)} for _ in range(4)]
    run=subprocess.run([node,str(ROOT/'scripts/export_runtime_fixture.mjs')],input=json.dumps({'cases':cases}),
                       text=True,capture_output=True,cwd=ROOT,timeout=30)
    assert run.returncode==0, run.stderr
    # Engine diagnostic stdout, if any, cannot replace the final JSON measurement.
    result=json.loads(run.stdout.strip().splitlines()[-1])
    model=mujoco.MjModel.from_xml_path(str(ROOT/'native/cartpole.xml'))
    assert result['timestep']==model.opt.timestep==.01 and result['controlSteps']==2
    errors=[]
    for item,measured in zip(cases,result['cases']):
        data=mujoco.MjData(model);mujoco.mj_resetData(model,data)
        initial=np.asarray(item['state']);data.qpos[:]=initial[[0,2]];data.qvel[:]=initial[[1,3]]
        mujoco.mj_forward(model,data)
        for request,row in zip(item['forces'],measured['rows']):
            applied=float(np.clip(request,-10,10));data.ctrl[0]=applied/10
            mujoco.mj_step(model,data);mujoco.mj_step(model,data)
            expected=np.array([data.qpos[0],data.qvel[0],data.qpos[1],data.qvel[1]])
            assert row['requestedForceN']==request and row['appliedForceN']==applied
            assert row['time']==pytest.approx(data.time,abs=1e-14)
            np.testing.assert_allclose(row['state'],expected,rtol=1e-10,atol=1e-11)
            errors.append(float(np.max(np.abs(np.asarray(row['state'])-expected))))
    assert len(errors)==64
    assert all(row['terminated'] and row['done'] for row in result['trainerContracts'])
