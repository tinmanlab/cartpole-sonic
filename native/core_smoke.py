#!/usr/bin/env python3
"""Issue #2: execute unmodified SONIC modules, not a reimplementation or controller.

No physical environment, PPO, teacher checkpoint or optimizer update is run here.
The only local adaptation is configuration and deterministic tensor fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

UPSTREAM_COMMIT = "b042411fae38ee4d1af9aac82a37a1f8d14d6dd0"
UPSTREAM_URL = "https://github.com/NVlabs/GR00T-WholeBodyControl"
SOURCE_FILES = (
    "gear_sonic/trl/modules/universal_token_modules.py",
    "gear_sonic/trl/modules/base_module.py",
    "gear_sonic/trl/utils/common.py",
    "gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml",
    "gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml",
    "gear_sonic/config/actor_critic/decoders/g1_kin_mf_mlp.yaml",
    "gear_sonic/config/actor_critic/quantizers/fsq.yaml",
)
CASES = (("reduced-one", 1, 2, 5), ("reduced-two", 2, 2, 5),
         ("release-token-shape", 2, 32, 32))


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Cannot verify upstream checkout: " + result.stderr.decode().strip())
    return result.stdout


def validate_upstream(root: Path) -> dict:
    """Fail closed on wrong revision or modified source; never download or repair it."""
    root = root.resolve(strict=True)
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(f"Upstream revision must be {UPSTREAM_COMMIT}; got {revision}")
    if git(root, "status", "--porcelain", "--untracked-files=all").strip():
        raise RuntimeError("Upstream checkout is not clean; use an unmodified pinned checkout")
    sources = {}
    for name in SOURCE_FILES:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise RuntimeError("Upstream source escapes its checkout: " + name)
        data = path.read_bytes()
        if data != git(root, "show", f"{UPSTREAM_COMMIT}:{name}"):
            raise RuntimeError("Upstream source differs from the pinned blob: " + name)
        sources[name] = hashlib.sha256(data).hexdigest()
    return sources


def import_native(root: Path):
    # A normal import from the pinned checkout: no source copying, monkey-patching or mocks.
    sys.dont_write_bytecode = True
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule
    from gear_sonic.trl.modules.base_module import BaseModule
    from vector_quantize_pytorch import FSQ
    from gear_sonic.trl.utils import common
    for obj, relative in ((UniversalTokenModule, SOURCE_FILES[0]),
                          (BaseModule, SOURCE_FILES[1]), (common, SOURCE_FILES[2])):
        if Path(inspect.getfile(obj)).resolve() != (root / relative).resolve():
            raise RuntimeError("Imported SONIC code is not from the pinned checkout")
    if importlib.metadata.version("vector-quantize-pytorch") != "1.31.6":
        raise RuntimeError("This smoke pins vector-quantize-pytorch==1.31.6")
    return UniversalTokenModule, BaseModule, FSQ


def configuration(root: Path, tokens: int, dim: int, levels: int):
    """Reuse upstream configs and its BaseModule builders; change declared dimensions only."""
    from omegaconf import OmegaConf
    actor = root / "gear_sonic/config/actor_critic"
    encoder = OmegaConf.load(actor / "encoders/g1_mf_mlp.yaml").g1
    dynamic = OmegaConf.load(actor / "decoders/g1_dyn_mlp.yaml").g1_dyn
    kinematic = OmegaConf.load(actor / "decoders/g1_kin_mf_mlp.yaml").g1_kin
    quantizer = OmegaConf.load(actor / "quantizers/fsq.yaml")
    encoder.inputs = ["cartpole_reference"]
    encoder.params.num_input_temporal_dims = 8
    encoder.params.num_output_temporal_dims = tokens
    kinematic.outputs = ["cartpole_reference"]
    kinematic.params.num_input_temporal_dims = tokens
    kinematic.params.num_output_temporal_dims = 8
    for network in (encoder, dynamic, kinematic):
        network.params.module_config_dict.layer_config.hidden_dims = [32, 32]
    # The controller consumes numeric codes, not a giant integer codebook index.
    # This disables optional index/table materialization, NOT scalar quantization.
    quantizer.return_indices = False
    env = OmegaConf.create({
        "obs": {"group_obs_dims": {"tokenizer": {"cartpole_reference": [8, 2]}},
                "group_obs_names": {"tokenizer": ["cartpole_reference"]}},
        "robot": {"actions_dim": 1, "algo_obs_dim_dict": {"actor_obs": 4}},
    })
    config = OmegaConf.create({
        "env_config": env, "algo_config": {}, "obs_dim_dict": {"actor_obs": 4},
        "proprioception_features": ["actor_obs"], "num_future_frames": 8,
        "max_num_tokens": tokens, "num_fsq_levels": dim, "fsq_level_list": levels,
        "quantizer": quantizer, "encoders": {"cartpole": encoder},
        # forward() selects the action from this legacy dispatch name.
        "decoders": {"g1_dyn": dynamic, "cartpole_kin": kinematic},
        "encoder_sample_probs": {"cartpole": 1.0},
    })
    # Pass DictConfigs through: upstream expects config attribute access during instantiation.
    return config


def fixtures(torch):
    """Physical meanings are specified; these are synthetic fixtures, not robot measurements."""
    # Four smooth position/velocity reference trajectories at two sequence times.
    goal = torch.tensor([-.7, -.2, .35, .8]).reshape(4, 1, 1)
    time = torch.arange(1, 9).reshape(1, 1, 8)*.08 + torch.tensor([0., .02]).reshape(1, 2, 1)
    decay = torch.exp(-2.5*time)
    x = goal*(1-(1+2.5*time)*decay)
    xd = goal*6.25*time*decay
    reference = torch.stack((x/1.8, xd/3.0), dim=-1)
    state = torch.linspace(-.35, .35, 4*2*4).reshape(4, 2, 4)
    scales = torch.tensor([1.8, 3.0, .55, 4.0])
    proprioception = state/scales
    return reference.contiguous(), proprioception.contiguous()


def digest_parameters(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_norm(module) -> float:
    import torch
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    if not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Non-finite native-module gradient")
    return float(torch.sqrt(sum(g.square().sum() for g in grads)))


def run_case(root, classes, case):
    import torch
    from omegaconf import OmegaConf
    UniversalTokenModule, BaseModule, FSQ = classes
    case_id, tokens, dim, levels = case
    torch.manual_seed(9101)
    config = configuration(root, tokens, dim, levels)
    model = UniversalTokenModule(**config).cpu().eval()
    if type(model.quantizer) is not FSQ or any(type(n) is not BaseModule for n in
            [*model.encoders.values(), *model.decoders.values()]):
        raise RuntimeError("Native module identity mismatch")
    reference, proprio = fixtures(torch)
    inputs = {"actor_obs": proprio, "tokenizer": reference.flatten(-2)}
    weight_hash = digest_parameters(model)
    calls = []
    handle = model.quantizer.register_forward_hook(lambda module, args, out: calls.append(1))
    try:
        out = model(inputs, return_dict=True)
        q = out["encoded_tokens"]["cartpole"]
        z = out["encoded_latents"]["cartpole"]
        action = out["action_mean"]
        recon = out["decoded_outputs"]["cartpole_kin"]["cartpole_reference"]
        if action is None:
            raise RuntimeError("No action: upstream g1_dyn dispatch was not configured")
        norms = lambda: {"encoder": gradient_norm(model.encoders["cartpole"]),
                         "dynamic": gradient_norm(model.decoders["g1_dyn"]),
                         "kinematic": gradient_norm(model.decoders["cartpole_kin"])}
        # Deliberately separate gradient paths; these losses are diagnostic, NOT SONIC PPO.
        action.square().mean().backward()
        action_grad = norms()
        model.zero_grad(set_to_none=True)
        recon_out = model(inputs, return_dict=True)
        recon2 = recon_out["decoded_outputs"]["cartpole_kin"]["cartpole_reference"]
        (recon2-reference).square().mean().backward()
        recon_grad = norms()
        with torch.no_grad():
            changed = model({**inputs, "actor_obs": proprio + .2}, return_dict=True)
            new_ref = model({**inputs, "tokenizer": inputs["tokenizer"] + .25}, return_dict=True)
        quantized = q.detach()
        half = levels//2
        all_finite = all(torch.isfinite(v).all().item() for v in [q,z,action,recon,recon2])
        data = {
            "id": case_id, "reference_shape": list(reference.shape),
            "proprioception_shape": list(proprio.shape), "token_shape": list(q.shape),
            "action_shape": list(action.shape), "reconstruction_shape": list(recon.shape),
            "hidden_widths": [32,32], "encoder_names": list(model.encoders),
            "dynamic_dispatch_name": "g1_dyn", "return_indices": model.quantizer.return_indices,
            "quantizer_levels": model.quantizer._levels.tolist(), "quantizer_calls": len(calls),
            "quantizer_parameters": sum(p.numel() for p in model.quantizer.parameters()),
            "on_quantization_grid": bool(torch.allclose(quantized*half, (quantized*half).round(), atol=1e-6, rtol=0)),
            "quantized_equals_latent": bool(torch.equal(q,z)), "finite": bool(all_finite),
            "action_gradient_norms": action_grad, "reconstruction_gradient_norms": recon_grad,
            "same_reference_changed_state": {
                "tokens_equal": bool(torch.equal(q,changed["encoded_tokens"]["cartpole"])),
                "action_max_abs_delta": float((action-changed["action_mean"]).abs().max().detach())},
            "reference_latent_response": float((z-new_ref["encoded_latents"]["cartpole"]).abs().max().detach()),
            "weights_unchanged": digest_parameters(model) == weight_hash,
            "parameters": sum(p.numel() for p in model.parameters()),
            "resolved_configuration": OmegaConf.to_container(config,resolve=True),
        }
        # A successful report is itself gated, even when the runner is used without pytest.
        required = (data["finite"] and data["on_quantization_grid"] and not data["quantized_equals_latent"]
                    and data["same_reference_changed_state"]["tokens_equal"]
                    and data["same_reference_changed_state"]["action_max_abs_delta"] > 1e-7
                    and data["reference_latent_response"] > 1e-7 and data["weights_unchanged"]
                    and action_grad["encoder"] > 0 and action_grad["dynamic"] > 0
                    and action_grad["kinematic"] == 0 and recon_grad["encoder"] > 0
                    and recon_grad["kinematic"] > 0 and recon_grad["dynamic"] == 0)
        if not required:
            raise RuntimeError("Native computation contract failed: " + case_id)
        return data
    finally:
        handle.remove()


def run_smoke(upstream: Path) -> dict:
    root = upstream.resolve(strict=True)
    sources = validate_upstream(root)
    classes = import_native(root)
    import torch
    from loguru import logger
    logger.disable("gear_sonic")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            cases = [run_case(root, classes, case) for case in CASES]
    finally:
        torch.set_num_threads(old_threads)
    modules = {}
    for name, cls in zip(("universal_token", "base", "fsq"), classes):
        path = Path(inspect.getfile(cls)).resolve()
        modules[name] = {"class": cls.__module__ + "." + cls.__name__,
                         "source": str(path.relative_to(root)) if path.is_relative_to(root) else cls.__module__.replace(".","/")+".py",
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # Recheck after computation: imports must not have patched tracked upstream files.
    if sources != validate_upstream(root):
        raise RuntimeError("Upstream changed during execution")
    return {
        "status": "PASS", "scope": "native_module_smoke_only", "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT, "upstream_clean": True,
        "source_verification": "working files byte-equal to pinned git blobs",
        "source_sha256": sources, "modules": modules,
        "versions": {n:importlib.metadata.version(n) for n in
                     ["torch","torchvision","numpy","omegaconf","loguru","wandb","vector-quantize-pytorch","einops","einx"]},
        "device": "cpu", "dtype": "float32", "seed": 9101,
        "fixture_source": "deterministic synthetic CartPole references and states",
        "normalization": {"reference": [1.8,3.0], "state": [1.8,3.0,.55,4.0]},
        "official_ppo_executed": False, "physical_rollout_executed": False, "optimizer_steps": 0,
        "adaptations": {
            "g1_dyn": "legacy dispatch name retained solely because upstream forward selects action_mean from this key; no G1 observations or weights are used",
            "release-token-shape": "2 tokens x 32 scalars x 32 levels; not production network widths, modalities, weights, training or humanoid capability",
            "return_indices": "False suppresses optional integer indices/implicit codebook table; real upstream FSQ scalar quantization remains enabled",
            "widths": "All three BaseModules use [32,32] instead of released large MLP widths; upstream SiLU activation is retained",
            "losses": "Independent action-squared and reference-reconstruction MSE gradient probes only; not official PPO or its configured auxiliary-loss training",
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=os.environ.get("SONIC_UPSTREAM"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.upstream is None:
            raise RuntimeError("Supply --upstream or SONIC_UPSTREAM")
        report = run_smoke(Path(args.upstream))
        exit_code = 0
    except Exception as error:
        report = {"status":"FAIL", "scope":"native_module_smoke_only", "error":str(error)}
        exit_code = 1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=args.output.parent,delete=False) as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary,args.output)
    print(json.dumps({"status":report["status"],"scope":report["scope"],
                      "cases":[c["id"] for c in report.get("cases",[])],
                      "error":report.get("error")},ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
