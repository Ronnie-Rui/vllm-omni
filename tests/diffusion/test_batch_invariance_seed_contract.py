# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from collections.abc import Iterator

import pytest
import torch
import vllm.envs as envs
from pytest_mock import MockerFixture

from vllm_omni.diffusion.batch_invariance import (
    DIFFUSION_BATCH_INVARIANT_ENV,
    MAX_TORCH_MANUAL_SEED,
    MIN_TORCH_MANUAL_SEED,
    diffusion_batch_invariant_enabled,
    validate_batch_invariant_diffusion_seed,
)
from vllm_omni.diffusion.worker import diffusion_worker as diffusion_worker_module
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _restore_native_batch_invariant_env() -> Iterator[None]:
    """Undo the bootstrap's process-level env write.

    _initialize_batch_invariance() sets VLLM_BATCH_INVARIANT so upstream's own
    re-check sees it. That is permanent by design in a worker process, but in a
    test process it would leak into later tests, where an unset diffusion switch
    follows the global one and silently arms the seed contract.
    """
    sentinel = object()
    before = os.environ.get("VLLM_BATCH_INVARIANT", sentinel)
    try:
        yield
    finally:
        if before is sentinel:
            os.environ.pop("VLLM_BATCH_INVARIANT", None)
        else:
            os.environ["VLLM_BATCH_INVARIANT"] = before
        # drop any value pinned into envs.__dict__ by monkeypatch.setattr, whose
        # undo writes the previously computed value back and defeats the lazy read
        envs.__dict__.pop("VLLM_BATCH_INVARIANT", None)


@pytest.mark.parametrize("seed", [MIN_TORCH_MANUAL_SEED, -2, 0, 42, MAX_TORCH_MANUAL_SEED])
def test_batch_invariant_mode_accepts_full_torch_seed_range(monkeypatch, seed):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(seed=seed),
        request_id="request-test",
    )


@pytest.mark.parametrize(
    "seed",
    [None, True, False, 1.5, "1", MIN_TORCH_MANUAL_SEED - 1, MAX_TORCH_MANUAL_SEED + 1],
)
def test_batch_invariant_mode_rejects_missing_invalid_or_out_of_range_seed(monkeypatch, seed):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    with pytest.raises(ValueError, match="seed"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(seed=seed),
            request_id="request-test",
        )


@pytest.mark.parametrize("seed", [None, 7])
def test_batch_invariant_mode_rejects_generator_input(monkeypatch, seed):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    generator = torch.Generator(device="cpu").manual_seed(7)

    with pytest.raises(ValueError, match="does not accept generator"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(seed=seed, generator=generator),
            request_id="request-test",
        )


def test_feature_off_preserves_generator_and_missing_seed_compatibility(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(generator=torch.Generator(device="cpu")),
        request_id="request-test",
    )
    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(),
        request_id="request-test",
    )


def test_diffusion_switch_unset_follows_global_batch_invariant(monkeypatch):
    monkeypatch.delenv(DIFFUSION_BATCH_INVARIANT_ENV, raising=False)

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert diffusion_batch_invariant_enabled() is True

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert diffusion_batch_invariant_enabled() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_diffusion_switch_enables_while_global_is_off(monkeypatch, raw):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, raw)

    assert diffusion_batch_invariant_enabled() is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " Off "])
def test_diffusion_switch_disables_while_global_is_on(monkeypatch, raw):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, raw)

    assert diffusion_batch_invariant_enabled() is False


@pytest.mark.parametrize("raw", ["", "maybe", "2", "none"])
def test_diffusion_switch_rejects_unparsable_values(monkeypatch, raw):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, raw)

    with pytest.raises(ValueError, match=DIFFUSION_BATCH_INVARIANT_ENV):
        diffusion_batch_invariant_enabled()


def test_seed_contract_follows_the_diffusion_switch_not_the_global_one(monkeypatch):
    """The switch must gate the seed validator itself, not just the helper."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, "1")

    with pytest.raises(ValueError, match="seed"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(),
            request_id="request-test",
        )

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, "0")
    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(),
        request_id="request-test",
    )


def test_worker_bootstrap_is_noop_when_batch_invariance_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: pytest.fail())

    diffusion_worker_module._initialize_batch_invariance(torch.device("cpu"))


def test_worker_bootstrap_skips_non_cuda_device_silently(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: pytest.fail())

    with caplog.at_level(logging.DEBUG):
        diffusion_worker_module._initialize_batch_invariance(torch.device("cpu"))

    assert caplog.records == []


def test_worker_bootstrap_skips_rocm_cuda_device_silently(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(torch.version, "hip", "6.0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: pytest.fail())

    with caplog.at_level(logging.DEBUG):
        diffusion_worker_module._initialize_batch_invariance(torch.device("cuda", 0))

    assert caplog.records == []


def test_worker_bootstrap_rejects_below_sm80_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

    with pytest.raises(RuntimeError, match=">= 8.0"):
        diffusion_worker_module._initialize_batch_invariance(torch.device("cuda", 0))


def test_worker_bootstrap_calls_native_vllm_on_supported_cuda(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    from vllm.model_executor.layers import batch_invariant

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    init_batch_invariance = mocker.patch.object(batch_invariant, "init_batch_invariance")

    diffusion_worker_module._initialize_batch_invariance(torch.device("cuda", 0))

    init_batch_invariance.assert_called_once_with()


def test_worker_bootstrap_aligns_native_env_for_diffusion_only_switch(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """The diffusion-only switch must reach upstream, which re-reads the env itself.

    init_batch_invariance() opens with ``if envs.VLLM_BATCH_INVARIANT``, so without
    the alignment every gate passes and the op replacement is a silent no-op.
    """
    from vllm.model_executor.layers import batch_invariant

    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))

    # vllm.envs serves this name from a module-level __getattr__. An earlier
    # monkeypatch.setattr(envs, ...) in this file pins the computed value into
    # envs.__dict__ and its undo leaves it there, which would silently turn the
    # assertion below into a check against a frozen constant. Drop the pin and
    # assert the lazy path is live, so ordering damage fails loudly here.
    envs.__dict__.pop("VLLM_BATCH_INVARIANT", None)
    assert envs.VLLM_BATCH_INVARIANT is False, "lazy env read is not live"

    seen: list[bool] = []
    mocker.patch.object(
        batch_invariant,
        "init_batch_invariance",
        side_effect=lambda: seen.append(envs.VLLM_BATCH_INVARIANT),
    )

    diffusion_worker_module._initialize_batch_invariance(torch.device("cuda", 0))

    # upstream's own re-check has to observe an enabled switch at call time
    assert seen == [True]
    # bool(int(...)) is upstream's parser, so a non-numeric truthy value would raise
    assert int(os.environ["VLLM_BATCH_INVARIANT"]) == 1
