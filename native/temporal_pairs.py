"""Policy-free physical reference pairs for the original native CartPole.

State order/units: x (m), xd (m/s), theta (rad), thetad (rad/s).
Seven normalized controls (10 N/unit), each held for four .02 s steps.
Endpoints are measured, never snapped. Status: 0 verified, 1 unverified,
2 solver budget exhausted, 3 nonfinite physics. Solver trace does not
expose MuJoCo's internal termination enum; status here is verification status.
"""
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

import mujoco
from mujoco import minimize
import numpy as np

HERE = Path(__file__).resolve().parent
OFFSETS = np.arange(0, 29, 4)
SEED = 73421
ENDPOINT_TOL = 1e-7
TARGET_TOL = 1e-7
FORCE_BOUND = .25
# Residual denominators in m, m/s, rad, rad/s, m respectively.
RESIDUAL_SCALE = np.array([.001, .01, .001, .01, .001])
MAX_ITER = 80
MAX_CANDIDATES = 1200


@dataclass(frozen=True)
class Case:
    x: float
    tick: int
    amplitude: float


def declared_cases():
    rng = np.random.default_rng(SEED)
    return tuple(Case(float(rng.uniform(-.25, .25)), tick, amplitude)
                 for tick in (8, 12, 16) for amplitude in (.001, .002, .004, .006))


def _plant(root=None):
    if root is None:
        return HERE / 'cartpole.xml'
    root = Path(root)
    return root / ('cartpole.xml' if root.name == 'native' else 'native/cartpole.xml')


def load_model(root=None):
    return mujoco.MjModel.from_xml_path(str(_plant(root)))


def _model_bytes(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer.tobytes()


def _state(data):
    return np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])


def simulate(model, x, actions):
    """Reset for every rollout; retain raw samples and check every substep."""
    actions = np.asarray(actions, dtype=np.float64)
    if (actions.shape != (28,) or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > FORCE_BOUND)
            or not np.isfinite(x) or abs(x) > .25):
        raise ValueError('finite bounded x and 28 bounded normalized controls required')
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = [x, 0.]
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    states = np.empty((29, 4))
    states[0] = _state(data)
    failed = False
    for tick, action in enumerate(actions):
        data.ctrl[0] = action
        for _ in range(2):
            mujoco.mj_step(model, data)
            s = _state(data)
            failed |= not np.isfinite(s).all() or abs(s[0]) > 1.78 or abs(s[2]) > .65
        states[tick + 1] = s
    return states, bool(failed)


class _BudgetExceeded(Exception):
    pass


def generate_pair(case, model=None, root=None, *, max_iter=MAX_ITER,
                  max_candidates=MAX_CANDIDATES):
    """Return both actual branches, stationary prehistories and all diagnostics.

    Fixed acceptance thresholds cannot be overridden. Exhausted budgets reject
    even a physically close last candidate. Rejected trajectories remain present.
    """
    if (not isinstance(case, Case) or type(case.tick) is not int
            or case.tick not in (8, 12, 16)
            or not np.isfinite([case.x, case.amplitude]).all()
            or abs(case.x) > .25 or not .001 <= abs(case.amplitude) <= .006):
        raise ValueError('case outside declared experiment bounds')
    if (type(max_iter) is not int or not 1 <= max_iter <= MAX_ITER
            or type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES):
        raise ValueError('solver budget outside finite experiment limits')
    original = load_model()
    if model is None:
        model = load_model(root) if root is not None else original
    if _model_bytes(model) != _model_bytes(original):
        raise ValueError('model must be the unmodified original native plant')
    initial = np.array([case.x, 0., 0., 0.])
    candidates, residuals, failures = [], [], []
    profile = np.zeros(7)
    logs = []
    status = 1
    budget_reason = ''

    def residual(controls):
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim == 1:
            controls = controls[:, None]
        if controls.ndim != 2 or controls.shape[0] != 7:
            raise ValueError('expected [7, ncandidates] controls')
        out = np.empty((5, controls.shape[1]))
        for column, u in enumerate(controls.T):
            if len(candidates) >= max_candidates:
                raise _BudgetExceeded
            states, failed = simulate(model, case.x, np.repeat(u, 4))
            r = np.r_[states[-1] - initial,
                      states[case.tick, 0] - case.x - case.amplitude] / RESIDUAL_SCALE
            candidates.append(u.copy())
            residuals.append(r.copy())
            failures.append(failed)
            if not np.isfinite(r).all():
                raise FloatingPointError
            out[:, column] = r
        return out

    # Keep logs via callback too, so budget interruptions retain earlier iterates.
    def log_callback(trace):
        logs[:] = [(entry.candidate.ravel().copy(), float(entry.objective),
                    float(entry.reduction), float(entry.regularizer)) for entry in trace]

    try:
        profile, _ = minimize.least_squares(
            profile, residual, bounds=(-np.full(7, FORCE_BOUND), np.full(7, FORCE_BOUND)),
            max_iter=max_iter, gtol=1e-10, xtol=1e-10,
            verbose=minimize.Verbosity.SILENT, iter_callback=log_callback)
        # The official API returns no termination enum. Conservatively reject
        # runs using the entire iteration allowance, even if physically close.
        if len(logs) - 1 >= max_iter:
            status = 2
            budget_reason = 'iteration_budget_exhausted'
    except (_BudgetExceeded, FloatingPointError) as exc:
        status = 2 if isinstance(exc, _BudgetExceeded) else 3
        budget_reason = 'candidate_budget_exhausted' if status == 2 else ''
        if candidates:
            # Retain an evaluated candidate, but never promote interrupted solves.
            profile = candidates[-1].copy()
    actions = np.stack((np.repeat(profile, 4), np.repeat(-profile, 4)))
    rollouts = [simulate(model, case.x, action) for action in actions]
    states = np.stack([r[0] for r in rollouts])
    history, history_failed = simulate(model, case.x, np.zeros(28))
    histories = np.stack((history, history.copy()))
    references = states[:, OFFSETS].copy()
    endpoint_errors = states[:, -1] - initial
    target_errors = states[:, case.tick, 0] - case.x - np.array([case.amplitude, -case.amplitude])
    interior_delta = float(np.max(np.abs(references[0, 1:-1, 0] - references[1, 1:-1, 0])))
    first_delta = float(abs(actions[0, 0] - actions[1, 0]))
    reasons = []
    if status == 2:
        reasons.append(budget_reason)
    if status == 3 or not np.isfinite(states).all():
        reasons.append('nonfinite_physics')
    if any(r[1] for r in rollouts) or history_failed:
        reasons.append('physical_failure_bound')
    if not np.array_equal(history, np.broadcast_to(initial, history.shape)):
        reasons.append('nonstationary_history')
    if not np.all(np.abs(endpoint_errors) <= ENDPOINT_TOL):
        reasons.append('endpoint_tolerance')
    if not np.all(np.abs(target_errors) <= TARGET_TOL):
        reasons.append('interior_target_tolerance')
    if not interior_delta > 1e-4:
        reasons.append('interior_not_distinct')
    informative = first_delta > 1e-5
    if not reasons:
        status = 0
    return dict(case=case, actions=actions, states=states, references=references,
                histories=histories, canonical_endpoint=initial,
                endpoint_errors=endpoint_errors, target_errors=target_errors,
                interior_reference_delta=interior_delta, first_action_delta=first_delta,
                first_action_informative=informative, accepted=not reasons,
                branch_failures=np.asarray([r[1] for r in rollouts], dtype=bool),
                history_failed=history_failed,
                reason=';'.join(reasons) or 'accepted', solver_status=status,
                candidate_count=len(candidates), max_iter=max_iter, max_candidates=max_candidates,
                candidate_controls=np.asarray(candidates).reshape(-1, 7),
                candidate_residuals=np.asarray(residuals).reshape(-1, 5),
                candidate_failures=np.asarray(failures, dtype=bool),
                solver_controls=np.asarray([l[0] for l in logs]).reshape(-1, 7),
                solver_log=np.asarray([l[1:] for l in logs]).reshape(-1, 3))


def generate_bank(model=None, root=None):
    """Execute exactly the frozen declared cases, without retries or policy use."""
    records = [generate_pair(case, model, root) for case in declared_cases()]
    accepted = sum(r['accepted'] for r in records)
    return dict(records=records, summary=dict(candidates=len(records), accepted=accepted,
                rejected=len(records) - accepted,
                accepted_first_action_informative=sum(r['accepted'] and r['first_action_informative'] for r in records),
                accepted_not_first_action_informative=sum(r['accepted'] and not r['first_action_informative'] for r in records)),
                provenance=dict(seed=SEED, plant_sha256=hashlib.sha256((HERE / 'cartpole.xml').read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  mujoco=mujoco.__version__, torch=version('torch')),
                    endpoint_tolerance=ENDPOINT_TOL, target_tolerance=TARGET_TOL,
                    interior_difference_min=1e-4, first_action_difference_min=1e-5,
                    residual_scale=RESIDUAL_SCALE.tolist(), control_dt=.02, substeps=2,
                    physics_dt=.01, horizon_steps=28, offsets=OFFSETS.tolist(),
                    force_bound_normalized=FORCE_BOUND, force_newtons_per_unit=10.,
                    x_failure_bound=1.78, theta_failure_bound=.65,
                    solver_gtol=1e-10, solver_xtol=1e-10,
                    max_iter=MAX_ITER, max_candidates=MAX_CANDIDATES))


def save_bank(bank, path):
    """Numeric NPZ payload, including rejected records and every solver diagnostic.

    Metadata is UTF-8 JSON bytes, never a Python object/pickle or host path.
    """
    import json
    arrays = {'metadata_json_utf8': np.frombuffer(json.dumps(
        dict(summary=bank['summary'], provenance=bank['provenance']), allow_nan=False).encode(), dtype=np.uint8)}
    for index, record in enumerate(bank['records']):
        for key, value in record.items():
            if key == 'case':
                value = [value.x, value.tick, value.amplitude]
            elif key == 'reason':
                value = np.frombuffer(value.encode(), dtype=np.uint8)
            arrays[f'case_{index}_{key}'] = np.asarray(value)
    np.savez_compressed(path, **arrays)


if __name__ == '__main__':
    import json
    bank = generate_bank()
    print(json.dumps(dict(summary=bank['summary'], provenance=bank['provenance'],
                         cases=[dict(x=r['case'].x, tick=r['case'].tick, amplitude=r['case'].amplitude,
                                     accepted=r['accepted'], reason=r['reason'], status=r['solver_status'],
                                     evaluations=r['candidate_count'], endpoint_max=float(np.abs(r['endpoint_errors']).max()),
                                     target_max=float(np.abs(r['target_errors']).max()),
                                     first_action_informative=r['first_action_informative']) for r in bank['records']]), allow_nan=False))
