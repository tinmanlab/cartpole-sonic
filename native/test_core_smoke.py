"""Acceptance tests for issue #2; no mocked neural modules or skipped dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
PIN = "b042411fae38ee4d1af9aac82a37a1f8d14d6dd0"

@pytest.fixture(scope="module")
def core():
    path = HERE / "core_smoke.py"
    assert path.is_file(), "native SONIC module adapter has not been implemented"
    spec = importlib.util.spec_from_file_location("core_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="module")
def report(core):
    upstream = os.environ.get("SONIC_UPSTREAM")
    assert upstream, "SONIC_UPSTREAM must identify the unmodified pinned checkout"
    return core.run_smoke(Path(upstream))

def test_actual_pinned_modules_are_used(report):
    assert report["upstream_commit"] == PIN
    assert report["upstream_clean"] is True
    modules = report["modules"]
    assert modules["universal_token"]["class"] == "gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule"
    assert modules["base"]["class"] == "gear_sonic.trl.modules.base_module.BaseModule"
    assert modules["fsq"]["class"] == "vector_quantize_pytorch.finite_scalar_quantization.FSQ"
    assert report["versions"]["vector-quantize-pytorch"] == "1.31.6"
    for entry in modules.values():
        assert len(entry["sha256"]) == 64
    assert report["source_verification"] == "working files byte-equal to pinned git blobs"

def test_three_shapes_use_actual_quantization(report):
    assert [c["id"] for c in report["cases"]] == ["reduced-one", "reduced-two", "release-token-shape"]
    for case, tokens, dim, levels in zip(report["cases"], [1,2,2], [2,2,32], [5,5,32]):
        assert case["reference_shape"] == [4,2,8,2]
        assert case["proprioception_shape"] == [4,2,4]
        assert case["token_shape"] == [8,tokens,dim]
        assert case["action_shape"] == [4,2,1]
        assert case["reconstruction_shape"] == [4,2,8,2]
        assert case["quantizer_levels"] == [levels]*dim
        assert case["quantizer_calls"] >= 3
        assert case["quantizer_parameters"] == 0
        assert case["on_quantization_grid"] is True
        assert case["quantized_equals_latent"] is False
        assert case["finite"] is True
        assert case["return_indices"] is False

def test_both_gradient_paths_reach_the_official_encoder(report):
    for case in report["cases"]:
        action = case["action_gradient_norms"]
        recon = case["reconstruction_gradient_norms"]
        assert action["encoder"] > 0 and action["dynamic"] > 0
        assert action["kinematic"] == 0
        assert recon["encoder"] > 0 and recon["kinematic"] > 0
        assert recon["dynamic"] == 0
        assert case["weights_unchanged"] is True

def test_proprioception_is_decoder_condition_not_motion_token(report):
    for case in report["cases"]:
        assert case["same_reference_changed_state"]["tokens_equal"] is True
        assert case["same_reference_changed_state"]["action_max_abs_delta"] > 1e-7
        assert case["reference_latent_response"] > 1e-7

def test_release_shape_is_not_mislabeled_as_production_controller(report):
    assert report["scope"] == "native_module_smoke_only"
    assert report["official_ppo_executed"] is False
    assert report["physical_rollout_executed"] is False
    assert report["optimizer_steps"] == 0
    assert report["fixture_source"] == "deterministic synthetic CartPole references and states"
    for case in report["cases"]:
        assert case["encoder_names"] == ["cartpole"]
        assert case["dynamic_dispatch_name"] == "g1_dyn"
        assert case["hidden_widths"] == [32,32]
    assert "legacy dispatch" in report["adaptations"]["g1_dyn"]
    assert "not" in report["adaptations"]["release-token-shape"]

def test_missing_checkout_fails_instead_of_faking_a_pass(core,tmp_path):
    with pytest.raises((RuntimeError,FileNotFoundError)):
        core.run_smoke(tmp_path / "missing")

def test_nonmatching_git_revision_is_rejected(core,tmp_path):
    subprocess.run(["git","init","-q",str(tmp_path)],check=True)
    (tmp_path/"x").write_text("not upstream")
    subprocess.run(["git","-C",str(tmp_path),"add","x"],check=True)
    subprocess.run(["git","-C",str(tmp_path),"-c","user.name=Test","-c","user.email=test@example.invalid","commit","-qm","fixture"],check=True)
    with pytest.raises(RuntimeError, match="revision"):
        core.validate_upstream(tmp_path)

def test_report_is_json_serializable_and_replayable(core,report):
    json.dumps(report,allow_nan=False)
    replay=core.run_smoke(Path(os.environ["SONIC_UPSTREAM"]))
    assert report == replay

def test_adapter_file_exists():
    assert (HERE / "core_smoke.py").is_file(), "missing native SONIC adapter"

def test_missing_dependencies_replace_stale_pass_with_failure(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"status":"PASS"}')
    # -S deliberately removes site-packages. Missing real dependencies cannot be a skip/PASS.
    result = subprocess.run([sys.executable,"-S",str(HERE/"core_smoke.py"),
                             "--upstream",os.environ["SONIC_UPSTREAM"],
                             "--output",str(output)],capture_output=True,text=True)
    assert result.returncode != 0
    assert json.loads(output.read_text())["status"] == "FAIL"

def test_committed_evidence_matches_fresh_computation(report):
    expected = json.loads((HERE.parent/"evidence/native_core_smoke.json").read_text())
    assert report["upstream_commit"] == expected["upstream_commit"]
    assert report["source_sha256"] == expected["source_sha256"]
    assert report["modules"] == expected["modules"]
    assert report["versions"] == expected["versions"]
    for actual, stored in zip(report["cases"],expected["cases"]):
        for field in ["id","token_shape","action_shape","reconstruction_shape","resolved_configuration"]:
            assert actual[field] == stored[field]
        for field in ["action_gradient_norms","reconstruction_gradient_norms"]:
            assert actual[field] == pytest.approx(stored[field],rel=1e-5,abs=1e-8)

def test_modified_upstream_source_is_rejected(core,tmp_path):
    checkout = tmp_path / "upstream"
    # Copy the already materialized sparse checkout locally. A git clone of a
    # partial/promisor repository may fetch unrelated missing objects from the network.
    import shutil
    shutil.copytree(Path(os.environ["SONIC_UPSTREAM"]), checkout, symlinks=True)
    core.validate_upstream(checkout)
    source = checkout / "gear_sonic/trl/modules/universal_token_modules.py"
    source.write_bytes(source.read_bytes()+b'\n# intentional dirty-source fixture\n')
    with pytest.raises(RuntimeError,match="not clean"):
        core.validate_upstream(checkout)

def test_disabling_integer_indices_does_not_disable_quantization():
    import torch
    from vector_quantize_pytorch import FSQ
    z = torch.linspace(-2.,2.,10).reshape(1,5,2)
    codes_with_ids, ids = FSQ(levels=[5,5],return_indices=True)(z)
    codes_only, no_ids = FSQ(levels=[5,5],return_indices=False)(z)
    assert ids is not None and no_ids is None
    assert torch.equal(codes_with_ids,codes_only)
    assert not torch.equal(codes_only,z)
