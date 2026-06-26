#!/usr/bin/env python3
"""Benchmark Qwen3-TTS decode preprocess: scalar per-request loop vs batched hook.

Weight-free: uses a dependency-light mirror of the Qwen3-TTS decode-only
preprocess branch (token embedding stubbed with a local ``torch.nn.Embedding``)
fed synthetic decode-state tensors. Scalar path calls ``model.preprocess`` once
per request; batched path calls ``model.preprocess_decode_batch`` once.

    python benchmarks/tts/bench_qwen3_tts_decode_preprocess.py \\
        --device cuda --batch-sizes 1 2 4 8 16 32 64 \\
        --output-json /tmp/decode_preprocess.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TRAILING_TEXT_COMPACT_MIN_FRAMES = 64


@dataclass
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float


@dataclass
class BenchmarkResult:
    batch_size: int
    scalar: TimingStats
    batched: TimingStats
    speedup: float
    scalar_us_per_request: float
    batched_us_per_request: float
    saved_ms: float
    saved_pct: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_times(batch_size: int, scalar_times_ms: list[float], batched_times_ms: list[float]) -> BenchmarkResult:
    scalar = TimingStats(
        mean_ms=statistics.fmean(scalar_times_ms),
        median_ms=statistics.median(scalar_times_ms),
        min_ms=min(scalar_times_ms),
        max_ms=max(scalar_times_ms),
        p90_ms=_percentile(scalar_times_ms, 0.90),
    )
    batched = TimingStats(
        mean_ms=statistics.fmean(batched_times_ms),
        median_ms=statistics.median(batched_times_ms),
        min_ms=min(batched_times_ms),
        max_ms=max(batched_times_ms),
        p90_ms=_percentile(batched_times_ms, 0.90),
    )
    saved_ms = scalar.mean_ms - batched.mean_ms
    return BenchmarkResult(
        batch_size=batch_size,
        scalar=scalar,
        batched=batched,
        speedup=scalar.mean_ms / batched.mean_ms if batched.mean_ms > 0 else float("inf"),
        scalar_us_per_request=scalar.mean_ms * 1000.0 / batch_size,
        batched_us_per_request=batched.mean_ms * 1000.0 / batch_size,
        saved_ms=saved_ms,
        saved_pct=(saved_ms / scalar.mean_ms * 100.0) if scalar.mean_ms > 0 else 0.0,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize()


class SyntheticQwen3TTSDecodePreprocess:
    """Dependency-light mirror of Qwen3-TTS decode preprocess.

    The benchmark cares about the current runner shape: calling this logic once
    per request vs once per decode batch.  Keeping this mirror local makes the
    microbenchmark runnable before the full vLLM stack is installed.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.talker_config = SimpleNamespace(codec_pad_id=7, num_code_groups=16)
        self._tts_pad_embed = torch.zeros((1, hidden_size), dtype=torch.bfloat16)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self._embedding = torch.nn.Embedding(vocab_size, hidden_size, device=device, dtype=torch.float32)
        with torch.no_grad():
            self._embedding.weight.uniform_(-0.02, 0.02, generator=generator)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self._embedding(input_ids.to(device=self._embedding.weight.device, dtype=torch.long))

    def _decode_one(
        self,
        *,
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
        require_last_hidden: bool,
        compute_embed: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        hs = info_dict.get("hidden_states", {})
        meta = info_dict.get("meta", {})

        text_list = info_dict.get("text")
        if not isinstance(text_list, list) or not text_list or not text_list[0]:
            raise ValueError("Missing additional_information.text for Qwen3-TTS AR talker.")

        task_type = (info_dict.get("task_type") or ["CustomVoice"])[0]
        codec_streaming_val = meta.get("codec_streaming")
        if isinstance(codec_streaming_val, list):
            codec_streaming_raw = codec_streaming_val[0] if codec_streaming_val else None
        else:
            codec_streaming_raw = codec_streaming_val
        if isinstance(codec_streaming_raw, bool):
            codec_streaming = codec_streaming_raw
        else:
            codec_streaming = task_type == "Base"

        device = input_ids.device
        dtype = torch.bfloat16
        tts_pad_embed = self._tts_pad_embed.to(device=device, dtype=dtype).reshape(1, -1)

        tail = hs.get("trailing_text")
        text_offset = max(0, int(meta.get("talker_text_offset", 0) or 0))
        trailing_text_update = None
        if isinstance(tail, torch.Tensor) and tail.ndim == 2:
            tail_len = int(tail.shape[0])
            if text_offset < tail_len:
                text_step = tail[text_offset : text_offset + 1].to(device=device, dtype=dtype).reshape(1, -1)
                next_text_offset = text_offset + 1
                should_compact_tail = next_text_offset >= tail_len or (
                    next_text_offset >= _TRAILING_TEXT_COMPACT_MIN_FRAMES and next_text_offset * 2 >= tail_len
                )
                if should_compact_tail:
                    if next_text_offset >= tail_len:
                        trailing_text_update = torch.empty((0, tail.shape[1]), device=tail.device, dtype=tail.dtype)
                    else:
                        trailing_text_update = tail[next_text_offset:].contiguous()
                    next_text_offset = 0
            else:
                text_step = tts_pad_embed
                next_text_offset = 0
                if tail.numel() > 0:
                    trailing_text_update = torch.empty((0, tail.shape[1]), device=tail.device, dtype=tail.dtype)
        else:
            text_step = tts_pad_embed
            next_text_offset = text_offset

        last_hidden = hs.get("last")
        if isinstance(last_hidden, torch.Tensor):
            past_hidden = last_hidden.to(device=device, dtype=dtype).reshape(1, -1)
        elif require_last_hidden:
            raise RuntimeError("Missing hidden_states['last'] in additional_information; postprocess must run.")
        else:
            past_hidden = torch.zeros_like(text_step)

        if compute_embed:
            last_id_hidden = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long)).to(device=device, dtype=dtype)
            inputs_embeds_out = last_id_hidden.reshape(1, -1)
        else:
            inputs_embeds_out = None

        info_update: dict[str, Any] = {
            "meta": {
                "talker_text_offset": int(next_text_offset),
                "codec_streaming": codec_streaming,
            },
        }
        if trailing_text_update is not None:
            info_update["hidden_states"] = {"trailing_text": trailing_text_update.detach()}
        return inputs_embeds_out, past_hidden, text_step, info_update

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del input_embeds
        inputs_embeds_out, past_hidden, text_step, info_update = self._decode_one(
            input_ids=input_ids,
            info_dict=info_dict,
            require_last_hidden=False,
            compute_embed=True,
        )
        assert inputs_embeds_out is not None
        return input_ids, inputs_embeds_out, {"mtp_inputs": (past_hidden, text_step), **info_update}

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any] | None]:
        input_ids_flat = input_ids.reshape(-1)
        if int(input_ids_flat.numel()) != len(req_infos):
            raise ValueError(
                f"preprocess_decode_batch expected {len(req_infos)} input ids, got {int(input_ids_flat.numel())}"
            )

        past_hidden_out: list[torch.Tensor] = []
        text_step_out: list[torch.Tensor] = []
        updates_out: list[dict[str, Any]] = []
        for idx, req_info in enumerate(req_infos):
            _req_embeds, past_hidden, text_step, update = self._decode_one(
                input_ids=input_ids_flat[idx : idx + 1],
                info_dict=req_info,
                require_last_hidden=True,
                compute_embed=False,
            )
            past_hidden_out.append(past_hidden)
            text_step_out.append(text_step)
            updates_out.append(update)

        inputs_embeds_out = self.embed_input_ids(input_ids_flat.reshape(-1, 1).to(torch.long)).to(
            device=input_ids_flat.device,
            dtype=torch.bfloat16,
        )
        inputs_embeds_out = inputs_embeds_out.reshape(len(req_infos), -1)

        # Generic contract: base return is (ids, embeds, updates); MTP tensors
        # ride along in the optional batch-level ``extras`` dict.
        extras = {
            "mtp_inputs": (
                torch.cat(past_hidden_out, dim=0),
                torch.cat(text_step_out, dim=0),
            )
        }
        return (
            input_ids_flat,
            inputs_embeds_out,
            updates_out,
            extras,
        )


def _make_talker(
    *,
    hidden_size: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
):
    return SyntheticQwen3TTSDecodePreprocess(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        seed=seed,
    )


def _make_req_infos(
    *,
    batch_size: int,
    hidden_size: int,
    tail_frames: int,
    text_offset: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    trailing_text = torch.randn((tail_frames, hidden_size), device=device, dtype=torch.float32)
    last_hidden = torch.randn((hidden_size,), device=device, dtype=torch.float32)
    return [
        {
            "text": [f"synthetic request {idx}"],
            "task_type": ["Base" if idx % 2 == 0 else "CustomVoice"],
            "hidden_states": {
                "trailing_text": trailing_text,
                "last": last_hidden,
            },
            "meta": {"talker_text_offset": text_offset},
            "_omni_is_prefill": False,
            "_omni_num_computed_tokens": 2,
            "_omni_prompt_len": 2,
        }
        for idx in range(batch_size)
    ]


def _run_scalar_once(
    model: Any,
    input_ids: torch.Tensor,
    req_infos: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    ids_out: list[torch.Tensor] = []
    embeds_out: list[torch.Tensor] = []
    past_hidden_out: list[torch.Tensor] = []
    text_step_out: list[torch.Tensor] = []
    updates_out: list[dict[str, Any]] = []

    for idx, req_info in enumerate(req_infos):
        req_ids, req_embeds, update = model.preprocess(
            input_ids=input_ids[idx : idx + 1],
            input_embeds=None,
            **req_info,
        )
        update = dict(update)
        past_hidden, text_step = update.pop("mtp_inputs")
        ids_out.append(req_ids.reshape(-1))
        embeds_out.append(req_embeds.reshape(1, -1))
        past_hidden_out.append(past_hidden.reshape(1, -1))
        text_step_out.append(text_step.reshape(1, -1))
        updates_out.append(update)

    return (
        torch.cat(ids_out, dim=0),
        torch.cat(embeds_out, dim=0),
        torch.cat(past_hidden_out, dim=0),
        torch.cat(text_step_out, dim=0),
        updates_out,
    )


def _run_batched_once(
    model: Any,
    input_ids: torch.Tensor,
    req_infos: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    # Generic 4-tuple contract: (ids, embeds, updates, extras|None). Unpack the
    # MTP batch-level tensors out of ``extras`` so parity can compare against the
    # scalar path, which carries the same tensors per-request via ``mtp_inputs``.
    req_input_ids, req_embeds, updates, extras = model.preprocess_decode_batch(input_ids=input_ids, req_infos=req_infos)
    mtp_inputs = extras.get("mtp_inputs") if isinstance(extras, dict) else None
    if mtp_inputs is None:
        raise AssertionError("benchmark expects an MTP stage returning extras['mtp_inputs']")
    past_hidden, text_step = mtp_inputs
    return req_input_ids, req_embeds, past_hidden, text_step, updates


def _assert_parity(
    scalar_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]],
    batched_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]],
) -> None:
    scalar_ids, scalar_embeds, scalar_past_hidden, scalar_text_step, scalar_updates = scalar_out
    batched_ids, batched_embeds, batched_past_hidden, batched_text_step, batched_updates = batched_out
    torch.testing.assert_close(batched_ids, scalar_ids)
    torch.testing.assert_close(batched_embeds, scalar_embeds)
    torch.testing.assert_close(batched_past_hidden, scalar_past_hidden)
    torch.testing.assert_close(batched_text_step, scalar_text_step)
    if batched_updates != scalar_updates:
        raise AssertionError(f"update mismatch: batched={batched_updates!r}, scalar={scalar_updates!r}")


def _measure(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmups):
            fn()
        _sync(device)
        times_ms: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            _sync(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = _make_talker(
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        device=device,
        seed=args.seed,
    )

    results: list[BenchmarkResult] = []
    for batch_size in args.batch_sizes:
        input_ids = torch.arange(100, 100 + batch_size, device=device, dtype=torch.long) % args.vocab_size
        req_infos = _make_req_infos(
            batch_size=batch_size,
            hidden_size=args.hidden_size,
            tail_frames=args.tail_frames,
            text_offset=args.text_offset,
            device=device,
        )

        def scalar_fn():
            return _run_scalar_once(model, input_ids, req_infos)

        def batched_fn():
            return _run_batched_once(model, input_ids, req_infos)

        if args.verify:
            _assert_parity(scalar_fn(), batched_fn())
            _sync(device)

        if args.collect_gc:
            gc.collect()
            if device.type == "cuda":
                torch.accelerator.empty_cache()

        scalar_times = _measure(scalar_fn, device=device, warmups=args.warmups, repeats=args.repeats)
        batched_times = _measure(batched_fn, device=device, warmups=args.warmups, repeats=args.repeats)
        results.append(summarize_times(batch_size, scalar_times, batched_times))

    payload = {
        "benchmark": "qwen3_tts_decode_preprocess",
        "description": "Scalar per-request decode preprocess loop vs preprocess_decode_batch.",
        "device": str(device),
        "torch_version": torch.__version__,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "tail_frames": args.tail_frames,
        "text_offset": args.text_offset,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "verify": args.verify,
        "results": [asdict(result) for result in results],
    }
    return payload


def format_results_table(results: list[BenchmarkResult]) -> str:
    header = (
        f"{'batch':>7} {'scalar mean ms':>15} {'batch mean ms':>14} "
        f"{'speedup':>9} {'scalar us/req':>14} {'batch us/req':>13} {'saved %':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.batch_size:7d} "
            f"{result.scalar.mean_ms:15.4f} "
            f"{result.batched.mean_ms:14.4f} "
            f"{result.speedup:9.2f} "
            f"{result.scalar_us_per_request:14.2f} "
            f"{result.batched_us_per_request:13.2f} "
            f"{result.saved_pct:8.1f}"
        )
    return "\n".join(lines)


def _results_from_payload(payload: dict[str, Any]) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for item in payload["results"]:
        scalar = TimingStats(**item["scalar"])
        batched = TimingStats(**item["batched"])
        results.append(
            BenchmarkResult(
                batch_size=item["batch_size"],
                scalar=scalar,
                batched=batched,
                speedup=item["speedup"],
                scalar_us_per_request=item["scalar_us_per_request"],
                batched_us_per_request=item["batched_us_per_request"],
                saved_ms=item["saved_ms"],
                saved_pct=item["saved_pct"],
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tail-frames", type=int, default=128)
    parser.add_argument("--text-offset", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--collect-gc", action="store_true", help="Run gc/empty_cache before each batch-size pair.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.set_defaults(verify=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(args)
    results = _results_from_payload(payload)
    print(format_results_table(results))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {args.output_json}")


if __name__ == "__main__":
    main()
