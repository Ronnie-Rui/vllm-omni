# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Produce BS1/BS2/BS3/BS4 bitwise evidence for the supported SD3 BIC tuple.

The coordinator resolves a Hugging Face revision or fingerprints a local model
directory, then launches three fresh outer processes for each batch size. Each
child starts one real diffusion engine and uses trusted test instrumentation to
observe the single pipeline forward and its initial latent, final latent, and
decoded tensor. Passing evidence validates only the recorded hardware/software
stack; it is not a claim for every GPU with the same compute capability.
This script imports ``tests.diffusion.batch_invariance_support`` and therefore
must be run from a source checkout rather than an installed wheel.

Example::

    VLLM_BATCH_INVARIANT=1 DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA python \
      examples/offline_inference/text_to_image/sd3_batch_invariance_gpu.py \
      --model /root/public-storage/model/AI-ModelScope/stable-diffusion-3.5-medium \
      --model-revision <40-hex-source-label> --gpu-ids 0,1,2,3 \
      --evidence-dir /tmp/sd3-bic-evidence
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_REPO = "stabilityai/stable-diffusion-3.5-medium"
PROBE_EXTENSION = "tests.diffusion.batch_invariance_support.SD3BatchInvarianceProbeExtension"
REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_CAPACITY_BY_BATCH_SIZE = {1: 1, 2: 2, 3: 4, 4: 4}
EVIDENCE_SOURCE_FILES = (
    "examples/offline_inference/text_to_image/sd3_batch_invariance_gpu.py",
    "tests/diffusion/batch_invariance_support.py",
    "vllm_omni/diffusion/diffusion_engine.py",
    "vllm_omni/diffusion/request.py",
    "vllm_omni/diffusion/worker/diffusion_model_runner.py",
    "vllm_omni/diffusion/worker/diffusion_worker.py",
    "vllm_omni/diffusion/models/sd3/pipeline_sd3.py",
)


def _require_process_environment() -> None:
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise RuntimeError(
            "Start this process with VLLM_BATCH_INVARIANT=1; native BIC is process-global "
            "and cannot be enabled after imports."
        )
    if os.environ.get("DIFFUSION_ATTENTION_BACKEND") != "TORCH_SDPA":
        raise RuntimeError(
            "Start this process with DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA so the evidence stack "
            "uses the supported attention backend."
        )


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_dump_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_text(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _require_clean_git_state(*, expected_head: str | None, phase: str) -> str:
    head = _run_text(["git", "rev-parse", "HEAD"], REPO_ROOT)
    status = _run_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        REPO_ROOT,
    )
    if head is None or status is None:
        raise RuntimeError(f"{phase}: could not read Git HEAD/status for {REPO_ROOT}")
    if expected_head is not None and head != expected_head:
        raise RuntimeError(f"{phase}: Git HEAD changed from {expected_head} to {head}")
    if status:
        raise RuntimeError(f"{phase}: evidence requires a clean committed worktree:\n{status}")
    return head


def _snapshot_fingerprint(snapshot: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        stat = path.stat()
        resolved = path.resolve()
        entries.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size": stat.st_size,
                "resolved_blob": resolved.name if resolved != path else None,
            }
        )
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "algorithm": "sha256(relative_path,size,resolved_content_address)-v1",
        "content_address_metadata_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "files": entries,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_model_content_fingerprint(model_dir: Path) -> dict[str, Any]:
    paths: list[tuple[str, Path]] = []
    for path in model_dir.rglob("*"):
        relative_path = path.relative_to(model_dir)
        if path.is_file() and not any(part.startswith(".") for part in relative_path.parts):
            paths.append((relative_path.as_posix(), path))
    files = [
        {
            "path": relative_path,
            "num_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for relative_path, path in sorted(paths)
    ]
    return {
        "algorithm": "sha256-per-file+canonical-json-v1",
        "content_sha256": _canonical_sha256(files),
        "file_count": len(files),
        "total_bytes": sum(item["num_bytes"] for item in files),
        "files": files,
    }


def _evidence_source_fingerprint() -> dict[str, Any]:
    files = {
        relative_path: hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        for relative_path in EVIDENCE_SOURCE_FILES
    }
    aggregate = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"algorithm": "sha256", "aggregate_sha256": aggregate, "files": files}


def _module_file_fingerprint(module: Any) -> dict[str, str]:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError(f"Cannot bind module {module!r}: __file__ is unavailable")
    path = Path(module_file).resolve()
    if not path.is_file():
        raise RuntimeError(f"Cannot bind module {module!r}: source file does not exist at {path}")
    return {"path": str(path), "sha256": _file_sha256(path)}


def _resolve_snapshot(
    model: str,
    revision: str,
    local_files_only: bool,
    *,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("--model-revision must be a full 40-character commit or local-source label")
    normalized_revision = revision.lower()

    local_path = Path(model).expanduser()
    if local_path.is_dir():
        resolved = local_path.resolve()
        return resolved, {
            "source_kind": "local_directory",
            "requested_source": model,
            "requested_revision_label": normalized_revision,
            "resolved_path": str(resolved),
            "content_fingerprint": _local_model_content_fingerprint(resolved),
        }

    if model != MODEL_REPO:
        raise ValueError(f"The first supported remote-model matrix requires source {MODEL_REPO!r}")
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    resolved = Path(
        snapshot_download_fn(
            repo_id=model,
            revision=normalized_revision,
            local_files_only=local_files_only,
        )
    ).resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Resolved model snapshot does not exist: {resolved}")
    if resolved.parent.name != "snapshots" or resolved.name.lower() != normalized_revision:
        raise RuntimeError(
            "Resolved Hugging Face snapshot path does not match the requested revision: "
            f"expected snapshots/{normalized_revision}, got {resolved}"
        )
    return resolved, {
        "source_kind": "huggingface_snapshot",
        "requested_source": model,
        "requested_revision": normalized_revision,
        "resolved_path": str(resolved),
        "content_address_fingerprint": _snapshot_fingerprint(resolved),
    }


def _validate_model_source_unchanged(model_snapshot: Path, model_source: dict[str, Any]) -> None:
    source_kind = model_source.get("source_kind")
    if source_kind == "huggingface_snapshot":
        return
    if source_kind != "local_directory":
        raise RuntimeError(f"Unsupported model source kind in evidence: {source_kind!r}")

    expected = model_source.get("content_fingerprint")
    actual = _local_model_content_fingerprint(model_snapshot)
    if actual != expected:
        raise RuntimeError(
            f"Local model content changed while the evidence matrix was running: expected {expected}, got {actual}"
        )


def _child_environment_manifest(
    *,
    expected_git_head: str,
    model_snapshot: Path,
    model_source: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from vllm.model_executor.layers import batch_invariant

    props = torch.cuda.get_device_properties(0)
    git_head = _require_clean_git_state(expected_head=expected_git_head, phase="child completion")
    return {
        "vllm_omni_commit": git_head,
        "vllm_omni_status": "",
        "evidence_source_fingerprint": _evidence_source_fingerprint(),
        "resolved_model_path": str(model_snapshot),
        "model_source": model_source,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "vllm": _package_version("vllm"),
        "vllm_omni": _package_version("vllm-omni"),
        "native_batch_invariance_module": _module_file_fingerprint(batch_invariant),
        "gpu": {
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory": props.total_memory,
        },
        "driver": _run_text(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        "DIFFUSION_ATTENTION_BACKEND": os.environ.get("DIFFUSION_ATTENTION_BACKEND"),
        "worker_extension_cls": PROBE_EXTENSION,
        "instrumentation_trust": "exact worker extension qualname plus evidence source SHA-256",
        "validation_scope": "only the recorded GPU model, capability, driver, and software stack",
        "execution": "eager",
        "dtype": "torch.bfloat16",
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        module_name = distribution.replace("-", "_")
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))


async def _collect_request(engine: Any, request: Any) -> dict[str, Any]:
    started = time.perf_counter()
    final_output = None
    async for output in engine.async_add_req_and_stream_response(request):
        final_output = output
    elapsed = time.perf_counter() - started
    if final_output is None:
        raise RuntimeError(f"{request.request_id}: engine returned no output")
    if final_output.error:
        raise RuntimeError(f"{request.request_id}: {final_output.error}")
    return {
        "request_id": request.request_id,
        "latency_s": elapsed,
        "peak_memory_mb": final_output.peak_memory_mb,
    }


async def _run_child_cases(engine: Any, batch_size: int, evidence_dir: Path) -> dict[str, Any]:
    from tests.diffusion.batch_invariance_support import (
        CASES_BY_BATCH_SIZE,
        REQUESTS,
        validate_case_evidence,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    installed = engine.collective_rpc(
        "install_sd3_batch_invariance_probe",
        args=(str(evidence_dir / "tensors"),),
        unique_reply_rank=0,
    )
    if installed.get("pipeline") != "StableDiffusion3Pipeline":
        raise AssertionError(f"probe attached to unexpected pipeline: {installed}")
    if installed.get("native_batch_invariant_mode") is not True:
        raise AssertionError(f"native vLLM BIC was not initialized in the worker: {installed}")

    cases: dict[str, Any] = {}
    for case_id, request_ids in CASES_BY_BATCH_SIZE[batch_size]:
        await engine.async_collective_rpc(
            "arm_sd3_batch_invariance_probe",
            args=(case_id, request_ids),
            unique_reply_rank=0,
        )
        requests = [
            OmniDiffusionRequest(
                request_id=request_id,
                prompt=str(REQUESTS[request_id]["prompt"]),
                sampling_params=OmniDiffusionSamplingParams(
                    seed=int(REQUESTS[request_id]["seed"]),
                    height=512,
                    width=512,
                    num_inference_steps=8,
                    num_outputs_per_prompt=1,
                    guidance_scale=1.0,
                    output_type="pt",
                ),
            )
            for request_id in request_ids
        ]
        started = time.perf_counter()
        request_metrics = await asyncio.gather(*(_collect_request(engine, request) for request in requests))
        wall_s = time.perf_counter() - started
        probe_result = await engine.async_collective_rpc(
            "finish_sd3_batch_invariance_probe",
            unique_reply_rank=0,
        )
        record = validate_case_evidence(case_id, request_ids, probe_result)
        record["performance"] = {
            "wall_s": wall_s,
            "requests_per_s": len(request_ids) / wall_s,
            "images_per_s": len(request_ids) / wall_s,
            "request_metrics": request_metrics,
        }
        cases[case_id] = record
        _json_dump(evidence_dir / "cases" / f"{case_id}.json", record)
    return cases


def _run_child(args: argparse.Namespace) -> int:
    mp.set_start_method("spawn", force=True)
    if args.batch_size not in ENGINE_CAPACITY_BY_BATCH_SIZE:
        raise ValueError("child batch size must be 1, 2, 3, or 4")
    _require_clean_git_state(expected_head=args.expected_git_head, phase="child startup")
    model_snapshot = Path(args.resolved_model_path).resolve()
    if not model_snapshot.is_dir():
        raise RuntimeError(f"model snapshot does not exist: {model_snapshot}")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SD3 BIC evidence matrix")
    props = torch.cuda.get_device_properties(0)
    if props.major < 8:
        raise RuntimeError(f"GPU compute capability must be >= 8.0, got {props.major}.{props.minor}")

    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    evidence_dir = Path(args.child_evidence_dir).resolve()
    model_source = json.loads(args.model_source_json)
    config = OmniDiffusionConfig(
        model=str(model_snapshot),
        model_class_name="StableDiffusion3Pipeline",
        dtype=torch.bfloat16,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        num_gpus=1,
        output_type="latent",
        vae_use_slicing=True,
        enforce_eager=True,
        # max_num_seqs=4 can naturally admit a partial BS3 wave. Proving BS3
        # closes that otherwise-unvalidated runtime shape.
        max_num_seqs=ENGINE_CAPACITY_BY_BATCH_SIZE[args.batch_size],
        request_batch_max_wait_ms=args.request_batch_max_wait_ms,
        worker_extension_cls=PROBE_EXTENSION,
        master_port=args.master_port,
    )
    config.enrich_config()
    engine = DiffusionEngine.make_engine(config)
    try:
        cases = asyncio.run(_run_child_cases(engine, args.batch_size, evidence_dir))
    finally:
        engine.close()

    result = {
        "status": "pass",
        "repetition": args.repetition,
        "batch_size": args.batch_size,
        "engine_max_num_seqs": ENGINE_CAPACITY_BY_BATCH_SIZE[args.batch_size],
        "request_batch_max_wait_ms": args.request_batch_max_wait_ms,
        "worker_extension_cls": PROBE_EXTENSION,
        "master_port": config.master_port,
        "outer_pid": os.getpid(),
        "tensor_artifact_root": str((evidence_dir / "tensors").resolve()),
        "environment": _child_environment_manifest(
            expected_git_head=args.expected_git_head,
            model_snapshot=model_snapshot,
            model_source=model_source,
        ),
        "cases": cases,
    }
    _json_dump(evidence_dir / "result.json", result)
    print(json.dumps({"status": "pass", "result": str(evidence_dir / "result.json")}, sort_keys=True))
    return 0


def _run_child_job(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    model_snapshot: Path,
    model_source: dict[str, Any],
    expected_git_head: str,
    repetition: int,
    batch_size: int,
    gpu_id: str,
) -> Path:
    child_dir = run_dir / "processes" / f"repetition_{repetition}" / f"bs{batch_size}"
    child_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--evidence-dir",
        str(run_dir),
        "--model-revision",
        args.model_revision,
        "--resolved-model-path",
        str(model_snapshot),
        "--model-source-json",
        json.dumps(model_source, sort_keys=True),
        "--expected-git-head",
        expected_git_head,
        "--child-evidence-dir",
        str(child_dir),
        "--repetition",
        str(repetition),
        "--batch-size",
        str(batch_size),
        "--request-batch-max-wait-ms",
        str(args.request_batch_max_wait_ms),
        "--master-port",
        str(24000 + repetition * 100 + batch_size * 10),
    ]
    env = os.environ.copy()
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(REPO_ROOT), env.get("PYTHONPATH")) if part)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (child_dir / "process.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"repetition={repetition} BS{batch_size} on CUDA_VISIBLE_DEVICES={gpu_id} failed "
            f"after {time.perf_counter() - started:.1f}s:\n{tail}"
        )
    result_path = child_dir / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"child passed without writing {result_path}")
    return result_path


def _run_gpu_queue(
    gpu_id: str,
    jobs: list[tuple[int, int]],
    *,
    args: argparse.Namespace,
    run_dir: Path,
    model_snapshot: Path,
    model_source: dict[str, Any],
    expected_git_head: str,
) -> list[Path]:
    return [
        _run_child_job(
            args=args,
            run_dir=run_dir,
            model_snapshot=model_snapshot,
            model_source=model_source,
            expected_git_head=expected_git_head,
            repetition=repetition,
            batch_size=batch_size,
            gpu_id=gpu_id,
        )
        for repetition, batch_size in jobs
    ]


def _assign_repetitions_to_gpus(gpu_ids: list[str]) -> list[list[tuple[int, int]]]:
    queues: list[list[tuple[int, int]]] = [[] for _ in gpu_ids]
    for repetition in range(3):
        # Keep every comparison within a repetition on one physical GPU.
        # Repetitions may run in parallel, but device identity cannot become
        # a BS1-vs-BSN variable.
        queues[repetition % len(gpu_ids)].extend((repetition, batch_size) for batch_size in (1, 2, 3, 4))
    return queues


def _validate_frozen_stack(results: list[dict[str, Any]], *, expected_git_head: str) -> None:
    keys = (
        "vllm_omni_commit",
        "vllm_omni_status",
        "evidence_source_fingerprint",
        "model_source",
        "python",
        "torch",
        "torch_cuda",
        "cudnn",
        "vllm",
        "vllm_omni",
        "native_batch_invariance_module",
        "gpu",
        "driver",
        "VLLM_BATCH_INVARIANT",
        "DIFFUSION_ATTENTION_BACKEND",
        "worker_extension_cls",
        "instrumentation_trust",
        "validation_scope",
        "execution",
        "dtype",
    )
    if not results:
        raise AssertionError("frozen-stack validation received no child results")
    for result in results:
        environment = result.get("environment", {})
        if environment.get("vllm_omni_commit") != expected_git_head:
            raise AssertionError(
                "child Git HEAD does not match coordinator start: "
                f"{environment.get('vllm_omni_commit')!r} != {expected_git_head!r}"
            )
        if environment.get("vllm_omni_status") != "":
            raise AssertionError(
                f"every child must run from a clean worktree; got status {environment.get('vllm_omni_status')!r}"
            )
    baseline = results[0]["environment"]
    for result in results[1:]:
        environment = result["environment"]
        for key in keys:
            if environment.get(key) != baseline.get(key):
                raise AssertionError(
                    f"frozen-stack mismatch for {key}: {environment.get(key)!r} != {baseline.get(key)!r}"
                )

    for repetition in range(3):
        visible_devices = {
            result["environment"].get("cuda_visible_devices")
            for result in results
            if int(result["repetition"]) == repetition
        }
        if len(visible_devices) != 1 or None in visible_devices:
            raise AssertionError(
                f"repetition {repetition} must keep BS1/2/3/4 on one physical GPU, got {visible_devices}"
            )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _performance_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        batch_size = int(result["batch_size"])
        grouped.setdefault(batch_size, []).extend(case["performance"] for case in result["cases"].values())

    by_batch_size: dict[str, Any] = {}
    for batch_size, cases in sorted(grouped.items()):
        latencies = [float(metric["latency_s"]) for case in cases for metric in case["request_metrics"]]
        peak_memory = [float(metric["peak_memory_mb"]) for case in cases for metric in case["request_metrics"]]
        by_batch_size[str(batch_size)] = {
            "forward_count": len(cases),
            "mean_requests_per_s": sum(float(case["requests_per_s"]) for case in cases) / len(cases),
            "mean_images_per_s": sum(float(case["images_per_s"]) for case in cases) / len(cases),
            "latency_p50_s": _percentile(latencies, 0.50),
            "latency_p95_s": _percentile(latencies, 0.95),
            "peak_memory_mb": max(peak_memory),
        }
    return {
        "correctness_gate": False,
        "effective_batch_size_distribution": {
            str(batch_size): len(cases) for batch_size, cases in sorted(grouped.items())
        },
        "by_batch_size": by_batch_size,
        "serialized_reference_batch_size": 1,
    }


def _run_coordinator(args: argparse.Namespace) -> int:
    from tests.diffusion.batch_invariance_support import compare_matrix_results

    expected_git_head = _require_clean_git_state(expected_head=None, phase="coordinator startup")
    model_snapshot, model_source = _resolve_snapshot(args.model, args.model_revision, args.local_files_only)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    evidence_root = Path(args.evidence_dir).resolve()
    if evidence_root == REPO_ROOT or REPO_ROOT in evidence_root.parents:
        raise RuntimeError(
            "--evidence-dir must be outside the Git worktree; child artifacts would otherwise make "
            "the frozen-stack status change while the matrix is running."
        )
    run_dir = evidence_root / f"sd3_bic_{args.model_revision[:12]}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one CUDA device index")
    jobs = [(repetition, batch_size) for repetition in range(3) for batch_size in (1, 2, 3, 4)]
    queues = _assign_repetitions_to_gpus(gpu_ids)

    manifest = {
        "status": "running",
        "expected_vllm_omni_commit": expected_git_head,
        "requested_model_source": args.model,
        "requested_model_revision_or_label": args.model_revision.lower(),
        "model_source_kind": model_source["source_kind"],
        "resolved_model_path": str(model_snapshot),
        "model_source": model_source,
        "gpu_ids": gpu_ids,
        "fresh_process_repetitions": 3,
        "batch_sizes": [1, 2, 3, 4],
        "request_batch_max_wait_ms": args.request_batch_max_wait_ms,
        "worker_extension_cls": PROBE_EXTENSION,
        "instrumentation_trust": "exact worker extension qualname plus evidence source SHA-256",
        "validation_scope": "only the GPU model, capability, driver, and software stack recorded by each child",
        "merge_evidence_eligible": False,
        "jobs": [{"repetition": repetition, "batch_size": batch_size} for repetition, batch_size in jobs],
    }
    _json_dump(run_dir / "run_manifest.json", manifest)

    result_paths: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [
            pool.submit(
                _run_gpu_queue,
                gpu_id,
                queue,
                args=args,
                run_dir=run_dir,
                model_snapshot=model_snapshot,
                model_source=model_source,
                expected_git_head=expected_git_head,
            )
            for gpu_id, queue in zip(gpu_ids, queues, strict=True)
            if queue
        ]
        for future in concurrent.futures.as_completed(futures):
            result_paths.extend(future.result())

    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    _validate_frozen_stack(results, expected_git_head=expected_git_head)
    summary = compare_matrix_results(results)
    summary["worker_extension_cls"] = PROBE_EXTENSION
    summary["cuda_visible_device_by_repetition"] = {
        str(repetition): next(
            result["environment"]["cuda_visible_devices"]
            for result in results
            if int(result["repetition"]) == repetition
        )
        for repetition in range(3)
    }
    summary["performance"] = _performance_summary(results)
    summary["result_files"] = sorted(str(path.relative_to(run_dir)) for path in result_paths)
    _validate_model_source_unchanged(model_snapshot, model_source)
    _require_clean_git_state(expected_head=expected_git_head, phase="coordinator completion")
    summary["merge_evidence_eligible"] = True
    _json_dump(run_dir / "comparison_summary.json", summary)
    manifest["status"] = "pass"
    manifest["merge_evidence_eligible"] = True
    manifest["comparison_summary"] = "comparison_summary.json"
    _json_dump_atomic(run_dir / "run_manifest.json", manifest)
    print(json.dumps({"status": "pass", "evidence_dir": str(run_dir), **summary}, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_REPO, help="Hugging Face repo ID or local model directory")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="40-hex HF commit or source label for a content-hashed local directory",
    )
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated GPUs; each child still uses exactly one GPU")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--request-batch-max-wait-ms", type=float, default=250.0)
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resolved-model-path", help=argparse.SUPPRESS)
    parser.add_argument("--model-source-json", help=argparse.SUPPRESS)
    parser.add_argument("--expected-git-head", help=argparse.SUPPRESS)
    parser.add_argument("--child-evidence-dir", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--master-port", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    _require_process_environment()
    args = _parse_args()
    if args.request_batch_max_wait_ms <= 0:
        raise ValueError("--request-batch-max-wait-ms must be greater than zero for true co-batch evidence")
    return _run_child(args) if args.child_run else _run_coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
