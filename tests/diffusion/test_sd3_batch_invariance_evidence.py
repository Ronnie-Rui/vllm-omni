# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.batch_invariance_support import (
    CASES_BY_BATCH_SIZE,
    REQUESTS,
    REQUIRED_TENSOR_LAYERS,
    SD3BatchInvarianceProbeExtension,
    assert_tensor_evidence_equal,
    compare_matrix_results,
    tensor_metadata,
    validate_case_evidence,
)

# ``examples/`` has no ``__init__.py``, so it is only importable as a namespace
# package. Once ``tests/examples/offline_inference/__init__.py`` puts a regular
# ``examples.offline_inference`` on the path (any full-suite collection does),
# it shadows the namespace portion and ``examples.offline_inference.text_to_image``
# disappears. Load the evidence script by path instead, like the other tests that
# cover example scripts do.
_GPU_EVIDENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "offline_inference"
    / "text_to_image"
    / "sd3_batch_invariance_gpu.py"
)
_GPU_EVIDENCE_MODULE_NAME = "sd3_batch_invariance_gpu_evidence_test"


def _load_gpu_evidence():
    spec = importlib.util.spec_from_file_location(_GPU_EVIDENCE_MODULE_NAME, _GPU_EVIDENCE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[_GPU_EVIDENCE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


gpu_evidence = _load_gpu_evidence()
_assign_repetitions_to_gpus = gpu_evidence._assign_repetitions_to_gpus

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _layer(value: int, request_ids: list[str]) -> dict:
    by_request = {
        request_id: {**tensor_metadata(torch.tensor([value + index], dtype=torch.int64)), "batch_index": index}
        for index, request_id in enumerate(request_ids)
    }
    return {"batch_dim": len(request_ids), "by_request": by_request}


def _case(case_id: str, request_ids: list[str]) -> dict:
    tensors = {layer: _layer(10 * layer_index, request_ids) for layer_index, layer in enumerate(REQUIRED_TENSOR_LAYERS)}
    return {
        "case_id": case_id,
        "expected_ids": request_ids,
        "request_ids": request_ids,
        "seeds": [REQUESTS[request_id]["seed"] for request_id in request_ids],
        "seed_was_explicit": [True] * len(request_ids),
        "output_count": len(request_ids),
        "error": None,
        "tensors": tensors,
        "request_outputs": {
            request_id: tensors["decoded_pixels"]["by_request"][request_id] for request_id in request_ids
        },
    }


def _matrix_results() -> list[dict]:
    results = []
    for repetition in range(3):
        for batch_size, cases in CASES_BY_BATCH_SIZE.items():
            result_cases = {}
            for case_id, request_ids in cases:
                case = _case(case_id, request_ids)
                for layer_index, layer in enumerate(REQUIRED_TENSOR_LAYERS):
                    case["tensors"][layer]["by_request"] = {
                        request_id: {
                            **tensor_metadata(torch.tensor([10 * layer_index], dtype=torch.int64)),
                            "batch_index": index,
                        }
                        for index, request_id in enumerate(request_ids)
                    }
                case["request_outputs"] = {
                    request_id: case["tensors"]["decoded_pixels"]["by_request"][request_id]
                    for request_id in request_ids
                }
                case["worker_pid"] = 2000 + repetition * 10 + batch_size
                result_cases[case_id] = case
            results.append(
                {
                    "repetition": repetition,
                    "batch_size": batch_size,
                    "engine_max_num_seqs": 4 if batch_size == 3 else batch_size,
                    "outer_pid": 1000 + repetition * 10 + batch_size,
                    "cases": result_cases,
                }
            )
    return results


def test_tensor_metadata_hashes_raw_typed_bytes() -> None:
    first = tensor_metadata(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    same = tensor_metadata(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    different_dtype = tensor_metadata(torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    assert first == same
    assert first["sha256"] != different_dtype["sha256"]
    assert first["dtype"] == "torch.float32"
    assert first["shape"] == [1, 2]


def test_each_repetition_keeps_all_batch_sizes_on_one_gpu_queue() -> None:
    queues = _assign_repetitions_to_gpus(["0", "1", "2", "3"])
    assert queues[3] == []
    for repetition, queue in enumerate(queues[:3]):
        assert queue == [(repetition, 1), (repetition, 2), (repetition, 3), (repetition, 4)]


def test_child_job_argv_round_trips_through_parser(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    child_dir = run_dir / "processes" / "repetition_2" / "bs3"
    model_snapshot = tmp_path / "model"
    model_snapshot.mkdir()
    captured: dict = {}

    def fake_run(command, **kwargs):
        monkeypatch.setattr(gpu_evidence.sys, "argv", command[1:])
        captured["args"] = gpu_evidence._parse_args()
        captured["env"] = kwargs["env"]
        gpu_evidence._json_dump(child_dir / "result.json", {})
        return SimpleNamespace(returncode=0, stdout="pass\n")

    monkeypatch.setattr(gpu_evidence.subprocess, "run", fake_run)
    result_path = gpu_evidence._run_child_job(
        args=SimpleNamespace(model_revision="b" * 40, request_batch_max_wait_ms=250.0),
        run_dir=run_dir,
        model_snapshot=model_snapshot,
        model_source={"source_kind": "local_directory"},
        expected_git_head="a" * 40,
        repetition=2,
        batch_size=3,
        gpu_id="7",
    )

    parsed = captured["args"]
    assert result_path == child_dir / "result.json"
    assert parsed.child_run is True
    assert parsed.evidence_dir == str(run_dir)
    assert parsed.child_evidence_dir == str(child_dir)
    assert parsed.repetition == 2
    assert parsed.batch_size == 3
    assert parsed.master_port == 24230
    assert captured["env"]["VLLM_BATCH_INVARIANT"] == "1"
    assert captured["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "7"


@pytest.mark.parametrize(
    ("batch_invariant", "attention_backend"),
    [(None, "TORCH_SDPA"), ("0", "TORCH_SDPA"), ("1", None), ("1", "FLASH_ATTN")],
)
def test_evidence_process_requires_supported_environment(
    monkeypatch: pytest.MonkeyPatch,
    batch_invariant: str | None,
    attention_backend: str | None,
) -> None:
    for name, value in (
        ("VLLM_BATCH_INVARIANT", batch_invariant),
        ("DIFFUSION_ATTENTION_BACKEND", attention_backend),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError):
        gpu_evidence._require_process_environment()


@pytest.mark.parametrize("max_wait_ms", [0.0, -1.0])
def test_evidence_main_requires_positive_batch_admission_wait(
    monkeypatch: pytest.MonkeyPatch,
    max_wait_ms: float,
) -> None:
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setattr(
        gpu_evidence,
        "_parse_args",
        lambda: SimpleNamespace(request_batch_max_wait_ms=max_wait_ms, child_run=False),
    )

    with pytest.raises(ValueError, match="must be greater than zero"):
        gpu_evidence.main()


def test_evidence_child_binds_exact_probe_extension(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from vllm_omni.diffusion import data as diffusion_data
    from vllm_omni.diffusion import diffusion_engine

    captured: dict = {}
    start_methods = []

    class EngineConstructionObservedError(Exception):
        pass

    class CapturingConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def enrich_config(self):
            captured["enriched"] = True

    class CapturingEngine:
        @staticmethod
        def make_engine(config):
            assert start_methods == [("spawn", True)]
            assert captured["enriched"] is True
            raise EngineConstructionObservedError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(major=8, minor=0))
    monkeypatch.setattr(diffusion_data, "OmniDiffusionConfig", CapturingConfig)
    monkeypatch.setattr(diffusion_engine, "DiffusionEngine", CapturingEngine)
    monkeypatch.setattr(
        gpu_evidence.mp,
        "set_start_method",
        lambda method, *, force: start_methods.append((method, force)),
    )
    monkeypatch.setattr(gpu_evidence, "_require_clean_git_state", lambda **_: "a" * 40)

    with pytest.raises(EngineConstructionObservedError):
        gpu_evidence._run_child(
            SimpleNamespace(
                batch_size=2,
                expected_git_head="a" * 40,
                resolved_model_path=str(tmp_path),
                model_source_json='{"source_kind": "local_directory"}',
                child_evidence_dir=str(tmp_path / "evidence"),
                request_batch_max_wait_ms=250.0,
                master_port=24120,
            )
        )

    assert captured["worker_extension_cls"] == gpu_evidence.PROBE_EXTENSION
    assert captured["request_batch_max_wait_ms"] == 250.0


def test_resolve_snapshot_fingerprints_every_local_model_file(tmp_path) -> None:
    model_dir = tmp_path / "model"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "model_index.json").write_bytes(b'{"class": "sd3"}')
    (model_dir / "transformer" / "weights.safetensors").write_bytes(b"weight-bytes")
    (model_dir / ".cache").mkdir()
    (model_dir / ".cache" / "transient").write_bytes(b"ignored")

    resolved, source = gpu_evidence._resolve_snapshot(str(model_dir), "A" * 40, False)

    assert resolved == model_dir.resolve()
    assert source["source_kind"] == "local_directory"
    assert source["requested_source"] == str(model_dir)
    assert source["requested_revision_label"] == "a" * 40
    files = source["content_fingerprint"]["files"]
    assert [item["path"] for item in files] == ["model_index.json", "transformer/weights.safetensors"]
    assert files[1]["sha256"] == hashlib.sha256(b"weight-bytes").hexdigest()

    old_content_sha = source["content_fingerprint"]["content_sha256"]
    (model_dir / "transformer" / "weights.safetensors").write_bytes(b"changed-weight-bytes")
    _, changed_source = gpu_evidence._resolve_snapshot(str(model_dir), "A" * 40, False)
    assert changed_source["content_fingerprint"]["content_sha256"] != old_content_sha


def test_resolve_snapshot_preserves_huggingface_content_address_metadata(tmp_path) -> None:
    revision = "b" * 40
    snapshot = tmp_path / "models--stabilityai--stable-diffusion-3.5-medium" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_bytes(b"{}")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    resolved, source = gpu_evidence._resolve_snapshot(
        gpu_evidence.MODEL_REPO,
        revision,
        True,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert resolved == snapshot.resolve()
    assert calls == [{"repo_id": gpu_evidence.MODEL_REPO, "revision": revision, "local_files_only": True}]
    assert source["source_kind"] == "huggingface_snapshot"
    assert source["requested_source"] == gpu_evidence.MODEL_REPO
    assert source["requested_revision"] == revision
    fingerprint = source["content_address_fingerprint"]
    assert fingerprint["algorithm"] == "sha256(relative_path,size,resolved_content_address)-v1"
    assert fingerprint["files"][0]["path"] == "model_index.json"
    assert "resolved_blob" in fingerprint["files"][0]


def test_completion_rehashes_local_model_but_trusts_huggingface_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "weights.safetensors"
    weights.write_bytes(b"original")
    _, local_source = gpu_evidence._resolve_snapshot(str(model_dir), "a" * 40, False)

    gpu_evidence._validate_model_source_unchanged(model_dir, local_source)
    weights.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="Local model content changed"):
        gpu_evidence._validate_model_source_unchanged(model_dir, local_source)

    monkeypatch.setattr(
        gpu_evidence,
        "_local_model_content_fingerprint",
        lambda _: pytest.fail("Hugging Face snapshots must trust content-address cache metadata"),
    )
    gpu_evidence._validate_model_source_unchanged(
        model_dir,
        {"source_kind": "huggingface_snapshot"},
    )


def test_coordinator_atomically_publishes_eligible_manifest_after_all_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from tests.diffusion import batch_invariance_support

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"weights")
    expected_head = "a" * 40
    events: list[str] = []

    recorded_head = expected_head

    def fake_git_state(*, expected_head, phase):
        events.append(phase)
        return expected_head or recorded_head

    monkeypatch.setattr(gpu_evidence, "_require_clean_git_state", fake_git_state)

    def fake_gpu_queue(gpu_id, queue, *, run_dir, **kwargs):
        paths = []
        for repetition in range(3):
            path = run_dir / f"result_{repetition}.json"
            gpu_evidence._json_dump(
                path,
                {
                    "repetition": repetition,
                    "environment": {"cuda_visible_devices": gpu_id},
                },
            )
            paths.append(path)
        return paths

    monkeypatch.setattr(gpu_evidence, "_run_gpu_queue", fake_gpu_queue)
    monkeypatch.setattr(
        gpu_evidence,
        "_validate_frozen_stack",
        lambda *args, **kwargs: events.append("frozen-stack"),
    )

    def fake_compare(results):
        events.append("tensor-compare")
        return {"status": "pass"}

    monkeypatch.setattr(batch_invariance_support, "compare_matrix_results", fake_compare)
    monkeypatch.setattr(gpu_evidence, "_performance_summary", lambda results: {})

    original_model_validation = gpu_evidence._validate_model_source_unchanged

    def tracking_model_validation(model_snapshot, model_source):
        events.append("model-post-check")
        original_model_validation(model_snapshot, model_source)

    monkeypatch.setattr(gpu_evidence, "_validate_model_source_unchanged", tracking_model_validation)

    initial_manifests = []
    final_manifests = []
    original_json_dump = gpu_evidence._json_dump
    original_json_dump_atomic = gpu_evidence._json_dump_atomic

    def tracking_json_dump(path, value):
        if path.name == "run_manifest.json":
            initial_manifests.append(copy.deepcopy(value))
        original_json_dump(path, value)

    def tracking_json_dump_atomic(path, value):
        events.append("atomic-pass-publish")
        final_manifests.append(copy.deepcopy(value))
        original_json_dump_atomic(path, value)

    monkeypatch.setattr(gpu_evidence, "_json_dump", tracking_json_dump)
    monkeypatch.setattr(gpu_evidence, "_json_dump_atomic", tracking_json_dump_atomic)

    args = SimpleNamespace(
        model=str(model_dir),
        model_revision="b" * 40,
        local_files_only=False,
        evidence_dir=str(tmp_path / "evidence"),
        gpu_ids="0",
        request_batch_max_wait_ms=250.0,
    )
    assert gpu_evidence._run_coordinator(args) == 0

    assert len(initial_manifests) == 1
    assert initial_manifests[0]["status"] == "running"
    assert initial_manifests[0]["merge_evidence_eligible"] is False
    assert len(final_manifests) == 1
    assert final_manifests[0]["status"] == "pass"
    assert final_manifests[0]["merge_evidence_eligible"] is True
    assert events.index("frozen-stack") < events.index("tensor-compare")
    assert events.index("tensor-compare") < events.index("model-post-check")
    assert events.index("model-post-check") < events.index("coordinator completion")
    assert events.index("coordinator completion") < events.index("atomic-pass-publish")
    manifest_path = next((tmp_path / "evidence").glob("*/run_manifest.json"))
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["merge_evidence_eligible"] is True


def test_require_clean_git_state_rejects_dirty_or_changed_head(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_head = "c" * 40

    def clean_run(command, cwd=None):
        return expected_head if "rev-parse" in command else ""

    monkeypatch.setattr(gpu_evidence, "_run_text", clean_run)
    assert gpu_evidence._require_clean_git_state(expected_head=expected_head, phase="test clean") == expected_head

    monkeypatch.setattr(
        gpu_evidence,
        "_run_text",
        lambda command, cwd=None: expected_head if "rev-parse" in command else "?? untracked.py",
    )
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        gpu_evidence._require_clean_git_state(expected_head=expected_head, phase="test dirty")

    changed_head = "d" * 40
    monkeypatch.setattr(
        gpu_evidence,
        "_run_text",
        lambda command, cwd=None: changed_head if "rev-parse" in command else "",
    )
    with pytest.raises(RuntimeError, match="Git HEAD changed"):
        gpu_evidence._require_clean_git_state(expected_head=expected_head, phase="test changed")


def test_frozen_stack_rejects_identically_dirty_children() -> None:
    head = "e" * 40
    results = [
        {
            "repetition": repetition,
            "environment": {
                "vllm_omni_commit": head,
                "vllm_omni_status": "?? same-untracked-file",
                "cuda_visible_devices": str(repetition),
            },
        }
        for repetition in range(3)
    ]
    with pytest.raises(AssertionError, match="every child must run from a clean worktree"):
        gpu_evidence._validate_frozen_stack(results, expected_git_head=head)


def test_module_file_fingerprint_binds_path_and_content(tmp_path) -> None:
    module_file = tmp_path / "batch_invariant.py"
    module_file.write_bytes(b"def init_batch_invariance(): pass\n")
    fingerprint = gpu_evidence._module_file_fingerprint(SimpleNamespace(__file__=str(module_file)))
    assert fingerprint == {
        "path": str(module_file.resolve()),
        "sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
    }


def test_frozen_stack_compares_native_batch_invariance_module_fingerprint() -> None:
    head = "f" * 40
    common_environment = {
        "vllm_omni_commit": head,
        "vllm_omni_status": "",
        "native_batch_invariance_module": {"path": "/wheel/batch_invariant.py", "sha256": "1" * 64},
    }
    results = [
        {
            "repetition": repetition,
            "environment": {**common_environment, "cuda_visible_devices": str(repetition)},
        }
        for repetition in range(3)
    ]
    results[2]["environment"]["native_batch_invariance_module"] = {
        "path": "/wheel/batch_invariant.py",
        "sha256": "2" * 64,
    }
    with pytest.raises(AssertionError, match="native_batch_invariance_module"):
        gpu_evidence._validate_frozen_stack(results, expected_git_head=head)


def test_validate_case_evidence_requires_one_real_forward_and_mapping() -> None:
    record = _case("bs2_forward", ["A", "B"])
    assert validate_case_evidence("bs2_forward", ["A", "B"], {"records": [record]}) is record

    with pytest.raises(AssertionError, match="exactly one pipeline.forward"):
        validate_case_evidence("bs2_forward", ["A", "B"], {"records": [record, record]})

    wrong = copy.deepcopy(record)
    wrong["request_ids"] = ["B", "A"]
    with pytest.raises(AssertionError, match="scheduled/forward IDs"):
        validate_case_evidence("bs2_forward", ["A", "B"], {"records": [wrong]})


def test_worker_probe_captures_one_forward_and_all_tensor_layers(tmp_path) -> None:
    class FakeVAE(torch.nn.Module):
        def decode(self, tensor, return_dict=False):
            return (tensor + 2,)

    class FakePipeline:
        def __init__(self):
            self.vae = FakeVAE()

        def prepare_latents(self, batch_size):
            return torch.arange(batch_size * 2, dtype=torch.float32).reshape(batch_size, 2)

        def diffuse(self, latents):
            return latents + 1

        def forward(self, batch):
            initial = self.prepare_latents(len(batch.request_ids))
            final = self.diffuse(initial)
            decoded = self.vae.decode(final, return_dict=False)[0]
            return [SimpleNamespace(output=decoded[index : index + 1]) for index in range(len(batch.request_ids))]

    extension = object.__new__(SD3BatchInvarianceProbeExtension)
    extension.model_runner = SimpleNamespace(pipeline=FakePipeline())
    extension.install_sd3_batch_invariance_probe(str(tmp_path))
    extension.arm_sd3_batch_invariance_probe("bs2_forward", ["A", "B"])
    batch = SimpleNamespace(
        request_ids=["A", "B"],
        requests=[SimpleNamespace(seed_was_explicit=True), SimpleNamespace(seed_was_explicit=True)],
        prompts=[REQUESTS["A"]["prompt"], REQUESTS["B"]["prompt"]],
        sampling_params_list=[SimpleNamespace(seed=REQUESTS[request_id]["seed"]) for request_id in ("A", "B")],
    )
    extension.model_runner.pipeline.forward(batch)
    result = extension.finish_sd3_batch_invariance_probe()
    record = validate_case_evidence("bs2_forward", ["A", "B"], result)
    assert set(record["tensors"]) == set(REQUIRED_TENSOR_LAYERS)
    assert len(list(tmp_path.rglob("*.pt"))) == 9


def test_compare_matrix_results_accepts_three_fresh_repetitions() -> None:
    results = json.loads(json.dumps(_matrix_results(), sort_keys=True))
    summary = compare_matrix_results(results, require_tensor_artifacts=False)
    assert summary["status"] == "pass"
    assert summary["fresh_process_repetitions"] == [0, 1, 2]
    assert summary["tensor_comparisons"] > 0


def test_compare_matrix_results_detects_neighbor_or_index_dependence() -> None:
    results = _matrix_results()
    target = next(result for result in results if result["repetition"] == 2 and result["batch_size"] == 4)
    target["cases"]["bs4_neighbor_mutation"]["tensors"]["final_latent"]["by_request"]["A"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="BS1-vs-BS4"):
        compare_matrix_results(results, require_tensor_artifacts=False)


def test_compare_matrix_results_rejects_a_missing_declared_case() -> None:
    results = _matrix_results()
    target = next(result for result in results if result["repetition"] == 1 and result["batch_size"] == 4)
    target["cases"].pop("bs4_shuffle")
    with pytest.raises(AssertionError, match="cases"):
        compare_matrix_results(results, require_tensor_artifacts=False)


def test_tensor_evidence_uses_torch_equal_and_checks_artifact_integrity(tmp_path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()
    left = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    right = left.clone()
    torch.save(left, left_root / "value.pt")
    torch.save(right, right_root / "value.pt")
    metadata = {**tensor_metadata(left), "artifact": "value.pt"}
    left_result = {"tensor_artifact_root": str(left_root)}
    right_result = {"tensor_artifact_root": str(right_root)}
    assert_tensor_evidence_equal(
        left_result,
        metadata,
        right_result,
        metadata,
        context="equal tensors",
        require_tensor_artifacts=True,
    )

    torch.save(torch.tensor([[1.0, 3.0]], dtype=torch.float32), right_root / "value.pt")
    with pytest.raises(AssertionError, match="artifact integrity"):
        assert_tensor_evidence_equal(
            left_result,
            metadata,
            right_result,
            metadata,
            context="changed tensor",
            require_tensor_artifacts=True,
        )

    nan_tensor = torch.tensor([[float("nan")]], dtype=torch.float32)
    torch.save(nan_tensor, left_root / "value.pt")
    torch.save(nan_tensor, right_root / "value.pt")
    nan_metadata = {**tensor_metadata(nan_tensor), "artifact": "value.pt"}
    with pytest.raises(AssertionError, match="torch.equal returned False"):
        assert_tensor_evidence_equal(
            left_result,
            nan_metadata,
            right_result,
            nan_metadata,
            context="NaN tensors",
            require_tensor_artifacts=True,
        )
