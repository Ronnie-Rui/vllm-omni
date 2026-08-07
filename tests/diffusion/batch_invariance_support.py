# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only evidence helpers for the SD3 batch-invariance GPU gate."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch

REQUIRED_TENSOR_LAYERS = ("initial_latent", "final_latent", "decoded_pixels")

REQUESTS: dict[str, dict[str, str | int]] = {
    "A": {"prompt": "a red ceramic cup on a plain white table", "seed": 101},
    "B": {"prompt": "a blue bicycle beside a brick wall", "seed": 202},
    "C": {"prompt": "a green glass bottle under studio light", "seed": 303},
    "D": {"prompt": "a yellow umbrella on a rainy street", "seed": 404},
    "B2": {"prompt": "a silver train crossing a snow field", "seed": 1202},
    "C2": {"prompt": "a wooden chair in an empty gallery", "seed": 1303},
    "D2": {"prompt": "a paper boat floating on a quiet pond", "seed": 1404},
}

CASES_BY_BATCH_SIZE: dict[int, list[tuple[str, list[str]]]] = {
    1: [(f"bs1_{request_id}", [request_id]) for request_id in REQUESTS],
    2: [
        ("bs2_forward", ["A", "B"]),
        ("bs2_reverse", ["B", "A"]),
        ("bs2_neighbor_mutation", ["A", "B2"]),
    ],
    3: [
        ("bs3_forward", ["A", "B", "C"]),
        ("bs3_reverse", ["C", "B", "A"]),
        ("bs3_neighbor_mutation", ["B2", "A", "C2"]),
    ],
    4: [
        ("bs4_forward", ["A", "B", "C", "D"]),
        ("bs4_reverse", ["D", "C", "B", "A"]),
        ("bs4_shuffle", ["B", "A", "D", "C"]),
        ("bs4_index3", ["C", "B", "D", "A"]),
        ("bs4_neighbor_mutation", ["B2", "A", "D2", "C2"]),
    ],
}


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    """Describe and hash exact contiguous CPU storage bytes."""
    cpu_tensor = tensor.detach().contiguous().cpu()
    raw_bytes = cpu_tensor.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "numel": cpu_tensor.numel(),
        "nbytes": len(raw_bytes),
        "byte_order": sys.byteorder,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def write_tensor_evidence(
    tensor: torch.Tensor,
    output_dir: Path,
    *,
    case_id: str,
    layer: str,
    request_ids: list[str],
) -> dict[str, Any]:
    """Write a batch tensor plus request-row tensors and return typed hashes."""
    if tensor.ndim == 0:
        raise AssertionError(f"{layer} must expose a batch dimension")
    if tensor.shape[0] != len(request_ids):
        raise AssertionError(f"{layer} batch dimension is {tensor.shape[0]}, expected {len(request_ids)}")

    case_dir = output_dir / _safe_component(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    batch_path = case_dir / f"{_safe_component(layer)}.pt"
    cpu_tensor = tensor.detach().contiguous().cpu()
    torch.save(cpu_tensor, batch_path)

    by_request: dict[str, dict[str, Any]] = {}
    for index, request_id in enumerate(request_ids):
        request_tensor = cpu_tensor[index : index + 1].contiguous()
        request_path = case_dir / (f"{_safe_component(layer)}__{index:02d}__{_safe_component(request_id)}.pt")
        torch.save(request_tensor, request_path)
        by_request[request_id] = {
            **tensor_metadata(request_tensor),
            "batch_index": index,
            "artifact": str(request_path.relative_to(output_dir)),
        }

    return {
        **tensor_metadata(cpu_tensor),
        "artifact": str(batch_path.relative_to(output_dir)),
        "batch_dim": int(cpu_tensor.shape[0]),
        "by_request": by_request,
    }


def validate_case_evidence(
    case_id: str,
    expected_ids: list[str],
    probe_result: dict[str, Any],
) -> dict[str, Any]:
    """Enforce the hard co-batch assertions for one scheduler wave."""
    records = probe_result.get("records", [])
    if len(records) != 1:
        raise AssertionError(f"{case_id}: expected exactly one pipeline.forward call, observed {len(records)}")
    record = records[0]
    if record.get("case_id") != case_id:
        raise AssertionError(f"{case_id}: probe returned evidence for {record.get('case_id')!r}")
    if record.get("request_ids") != expected_ids:
        raise AssertionError(f"{case_id}: scheduled/forward IDs {record.get('request_ids')} != {expected_ids}")
    if record.get("expected_ids") != expected_ids:
        raise AssertionError(f"{case_id}: probe was armed for the wrong request-ID set")
    if record.get("error"):
        raise AssertionError(f"{case_id}: pipeline.forward failed: {record['error']}")

    expected_seeds = [int(REQUESTS[request_id]["seed"]) for request_id in expected_ids]
    if record.get("seeds") != expected_seeds:
        raise AssertionError(f"{case_id}: request-to-seed mapping {record.get('seeds')} != {expected_seeds}")
    if record.get("seed_was_explicit") != [True] * len(expected_ids):
        raise AssertionError(
            f"{case_id}: every request must preserve explicit-seed provenance, got {record.get('seed_was_explicit')}"
        )
    if record.get("output_count") != len(expected_ids):
        raise AssertionError(
            f"{case_id}: pipeline returned {record.get('output_count')} request outputs, expected {len(expected_ids)}"
        )

    tensors = record.get("tensors", {})
    for layer in REQUIRED_TENSOR_LAYERS:
        layer_evidence = tensors.get(layer)
        if layer_evidence is None:
            raise AssertionError(f"{case_id}: probe did not observe {layer}")
        if layer_evidence.get("batch_dim") != len(expected_ids):
            raise AssertionError(
                f"{case_id}: {layer} batch dimension {layer_evidence.get('batch_dim')} != {len(expected_ids)}"
            )
        if list(layer_evidence.get("by_request", {})) != expected_ids:
            raise AssertionError(f"{case_id}: {layer} row-to-request mapping is not scheduler order")

    request_outputs = record.get("request_outputs", {})
    if list(request_outputs) != expected_ids:
        raise AssertionError(f"{case_id}: split output mapping is not scheduler order")
    for request_id in expected_ids:
        decoded = tensors["decoded_pixels"]["by_request"][request_id]
        assert_tensor_metadata_equal(
            request_outputs[request_id],
            decoded,
            context=f"{case_id}: split output for {request_id} vs decoded tensor row",
        )
    return record


def load_and_verify_tensor_artifact(result: dict[str, Any], metadata: dict[str, Any]) -> torch.Tensor:
    root_value = result.get("tensor_artifact_root")
    if not isinstance(root_value, str):
        raise AssertionError("matrix result is missing tensor_artifact_root")
    root = Path(root_value).resolve()
    artifact_value = metadata.get("artifact")
    if not isinstance(artifact_value, str):
        raise AssertionError("tensor evidence is missing its artifact path")
    artifact = (root / artifact_value).resolve()
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise AssertionError(f"tensor artifact escapes evidence root: {artifact}") from exc
    if not artifact.is_file():
        raise AssertionError(f"tensor artifact does not exist: {artifact}")
    tensor = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(f"tensor artifact has unexpected type {type(tensor).__name__}: {artifact}")
    assert_tensor_metadata_equal(tensor_metadata(tensor), metadata, context=f"artifact integrity {artifact}")
    return tensor


def assert_tensor_evidence_equal(
    left_result: dict[str, Any],
    left: dict[str, Any],
    right_result: dict[str, Any],
    right: dict[str, Any],
    *,
    context: str,
    require_tensor_artifacts: bool,
) -> None:
    assert_tensor_metadata_equal(left, right, context=context)
    if not require_tensor_artifacts:
        return
    left_tensor = load_and_verify_tensor_artifact(left_result, left)
    right_tensor = load_and_verify_tensor_artifact(right_result, right)
    if not torch.equal(left_tensor, right_tensor):
        raise AssertionError(f"{context}: torch.equal returned False")


def compare_matrix_results(
    results: list[dict[str, Any]],
    *,
    require_tensor_artifacts: bool = True,
) -> dict[str, Any]:
    """Compare three fresh-process BS1/BS2/BS3/BS4 result sets bitwise."""
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for result in results:
        key = (int(result["repetition"]), int(result["batch_size"]))
        if key in grouped:
            raise AssertionError(f"duplicate result for repetition/batch {key}")
        grouped[key] = result

    repetitions = sorted({key[0] for key in grouped})
    if len(repetitions) != 3:
        raise AssertionError(f"expected exactly three fresh repetitions, got {repetitions}")
    for repetition in repetitions:
        missing = [batch_size for batch_size in (1, 2, 3, 4) if (repetition, batch_size) not in grouped]
        if missing:
            raise AssertionError(f"repetition {repetition} is missing batch sizes {missing}")

    for (repetition, batch_size), result in grouped.items():
        expected_cases = CASES_BY_BATCH_SIZE[batch_size]
        expected_case_ids = [case_id for case_id, _ in expected_cases]
        actual_cases = result.get("cases", {})
        if sorted(actual_cases) != sorted(expected_case_ids):
            raise AssertionError(
                f"repetition {repetition} BS{batch_size} cases {list(actual_cases)} != {expected_case_ids}"
            )
        for case_id, expected_ids in expected_cases:
            if actual_cases[case_id].get("request_ids") != expected_ids:
                raise AssertionError(
                    f"repetition {repetition} {case_id} request IDs "
                    f"{actual_cases[case_id].get('request_ids')} != {expected_ids}"
                )

    for batch_size in (1, 2, 3, 4):
        process_identities: set[tuple[int, int]] = set()
        for repetition in repetitions:
            result = grouped[(repetition, batch_size)]
            expected_capacity = 4 if batch_size == 3 else batch_size
            if result.get("engine_max_num_seqs") != expected_capacity:
                raise AssertionError(
                    f"repetition {repetition} BS{batch_size} used max_num_seqs="
                    f"{result.get('engine_max_num_seqs')}, expected {expected_capacity}"
                )
            worker_pids = {case.get("worker_pid") for case in result["cases"].values()}
            if len(worker_pids) != 1 or None in worker_pids:
                raise AssertionError(
                    f"repetition {repetition} BS{batch_size} did not record one worker process: {worker_pids}"
                )
            outer_pid = result.get("outer_pid")
            if not isinstance(outer_pid, int):
                raise AssertionError(f"repetition {repetition} BS{batch_size} is missing outer_pid")
            process_identities.add((outer_pid, worker_pids.pop()))
        if len(process_identities) != 3:
            raise AssertionError(
                f"BS{batch_size} requires three fresh outer/worker processes, got {process_identities}"
            )

    reference: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for request_id in REQUESTS:
        evidence_by_repetition: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for repetition in repetitions:
            result = grouped[(repetition, 1)]
            case = result["cases"][f"bs1_{request_id}"]
            evidence_by_repetition.append(
                (
                    result,
                    {layer: case["tensors"][layer]["by_request"][request_id] for layer in REQUIRED_TENSOR_LAYERS},
                )
            )
        first_result, first = evidence_by_repetition[0]
        for other_result, other in evidence_by_repetition[1:]:
            for layer in REQUIRED_TENSOR_LAYERS:
                assert_tensor_evidence_equal(
                    first_result,
                    first[layer],
                    other_result,
                    other[layer],
                    context=f"fresh-process BS1 {request_id}/{layer}",
                    require_tensor_artifacts=require_tensor_artifacts,
                )
        reference[request_id] = (first_result, first)

    comparisons = 0
    for (repetition, batch_size), result in grouped.items():
        if batch_size == 1:
            continue
        for case_id, case in result["cases"].items():
            for request_id in case["request_ids"]:
                for layer in REQUIRED_TENSOR_LAYERS:
                    reference_result, reference_tensors = reference[request_id]
                    assert_tensor_evidence_equal(
                        reference_result,
                        reference_tensors[layer],
                        result,
                        case["tensors"][layer]["by_request"][request_id],
                        context=(f"repetition {repetition} {case_id} {request_id}/{layer} BS1-vs-BS{batch_size}"),
                        require_tensor_artifacts=require_tensor_artifacts,
                    )
                    comparisons += 1

    return {
        "status": "pass",
        "fresh_process_repetitions": repetitions,
        "batch_sizes": [1, 2, 3, 4],
        "tensor_comparisons": comparisons,
        "layers": list(REQUIRED_TENSOR_LAYERS),
    }


def assert_tensor_metadata_equal(left: dict[str, Any], right: dict[str, Any], *, context: str) -> None:
    for key in ("dtype", "shape", "numel", "nbytes", "byte_order", "sha256"):
        if left.get(key) != right.get(key):
            raise AssertionError(f"{context}: tensor {key} differs: {left.get(key)!r} != {right.get(key)!r}")


class SD3BatchInvarianceProbeExtension:
    """Worker extension that observes SD3 without changing production APIs."""

    def install_sd3_batch_invariance_probe(self, evidence_dir: str) -> dict[str, Any]:
        if getattr(self, "_sd3_bic_probe", None) is not None:
            raise RuntimeError("SD3 batch-invariance probe is already installed")
        if self.model_runner is None or self.model_runner.pipeline is None:
            raise RuntimeError("SD3 batch-invariance probe requires a loaded pipeline")

        pipeline = self.model_runner.pipeline
        output_dir = Path(evidence_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "output_dir": output_dir,
            "armed": None,
            "active": None,
            "records": [],
        }
        self._sd3_bic_probe = state

        original_forward = pipeline.forward
        original_prepare_latents = pipeline.prepare_latents
        original_diffuse = pipeline.diffuse
        original_decode = pipeline.vae.decode

        def capture(layer: str, tensor: torch.Tensor) -> None:
            record = state["active"]
            if record is None:
                return
            if layer in record["tensors"]:
                raise AssertionError(f"{record['case_id']}: observed {layer} more than once")
            record["tensors"][layer] = write_tensor_evidence(
                tensor,
                output_dir,
                case_id=record["case_id"],
                layer=layer,
                request_ids=record["request_ids"],
            )

        def forward_wrapper(_pipeline: Any, request_batch: Any) -> Any:
            armed = state["armed"]
            if armed is None:
                return original_forward(request_batch)
            if state["active"] is not None:
                raise AssertionError("nested SD3 pipeline.forward probe invocation")

            request_ids = list(request_batch.request_ids)
            record: dict[str, Any] = {
                "case_id": armed["case_id"],
                "worker_pid": os.getpid(),
                "expected_ids": list(armed["expected_ids"]),
                "request_ids": request_ids,
                "seeds": [sampling.seed for sampling in request_batch.sampling_params_list],
                "seed_was_explicit": [request.seed_was_explicit for request in request_batch.requests],
                "prompts": list(request_batch.prompts),
                "started_ns": time.time_ns(),
                "tensors": {},
                "request_outputs": {},
                "error": None,
            }
            state["active"] = record
            try:
                outputs = original_forward(request_batch)
                record["output_count"] = len(outputs) if isinstance(outputs, list) else None
                if isinstance(outputs, list):
                    for request_id, output in zip(request_ids, outputs, strict=False):
                        if isinstance(output.output, torch.Tensor):
                            record["request_outputs"][request_id] = tensor_metadata(output.output)
                return outputs
            except BaseException as exc:  # noqa: BLE001 - preserve evidence before re-raising
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["finished_ns"] = time.time_ns()
                state["records"].append(record)
                state["active"] = None

        def prepare_latents_wrapper(_pipeline: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
            tensor = original_prepare_latents(*args, **kwargs)
            capture("initial_latent", tensor)
            return tensor

        def diffuse_wrapper(_pipeline: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
            tensor = original_diffuse(*args, **kwargs)
            capture("final_latent", tensor)
            return tensor

        def decode_wrapper(_vae: Any, *args: Any, **kwargs: Any) -> Any:
            decoded = original_decode(*args, **kwargs)
            tensor = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
            if not isinstance(tensor, torch.Tensor):
                raise AssertionError(f"VAE decode returned unsupported type {type(tensor).__name__}")
            capture("decoded_pixels", tensor)
            return decoded

        pipeline.forward = MethodType(forward_wrapper, pipeline)
        pipeline.prepare_latents = MethodType(prepare_latents_wrapper, pipeline)
        pipeline.diffuse = MethodType(diffuse_wrapper, pipeline)
        pipeline.vae.decode = MethodType(decode_wrapper, pipeline.vae)
        from vllm.model_executor.layers import batch_invariant

        return {
            "status": "installed",
            "pipeline": type(pipeline).__name__,
            "worker_pid": os.getpid(),
            "native_batch_invariant_mode": bool(batch_invariant._batch_invariant_MODE),
        }

    def arm_sd3_batch_invariance_probe(self, case_id: str, expected_ids: list[str]) -> dict[str, Any]:
        state = getattr(self, "_sd3_bic_probe", None)
        if state is None:
            raise RuntimeError("SD3 batch-invariance probe is not installed")
        if state["active"] is not None:
            raise RuntimeError("cannot arm probe during pipeline.forward")
        state["records"] = []
        state["armed"] = {"case_id": case_id, "expected_ids": list(expected_ids)}
        return {"status": "armed", "case_id": case_id, "expected_ids": list(expected_ids)}

    def finish_sd3_batch_invariance_probe(self) -> dict[str, Any]:
        state = getattr(self, "_sd3_bic_probe", None)
        if state is None:
            raise RuntimeError("SD3 batch-invariance probe is not installed")
        if state["active"] is not None:
            raise RuntimeError("cannot finish probe during pipeline.forward")
        result = {
            "armed": state["armed"],
            "records": state["records"],
        }
        state["armed"] = None
        state["records"] = []
        return result
