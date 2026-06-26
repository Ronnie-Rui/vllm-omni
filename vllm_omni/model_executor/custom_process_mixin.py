from collections.abc import Callable
from typing import Any

import torch


class CustomProcessMixin:
    """
    Mixin class for all stages in the Omni model.
    """

    def set_custom_preprocess(self, preprocess_fn: Callable) -> None:
        """
        Set a preprocess function for the stage.
        Args:
            preprocess_fn: The preprocess function to register.
        """
        self.preprocess = preprocess_fn

    def set_custom_postprocess(self, postprocess_fn: Callable) -> None:
        """
        Set a postprocess function for the stage.
        Args:
            postprocess_fn: The postprocess function to register.
        """
        self.postprocess = postprocess_fn

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **input_dict: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Process the input_ids and input_embeds for the given input_dict.
        Returns the processed input_ids, input_embeds, and the input_dict.
        If the stage don't applicable, return the original input_ids, input_embeds, and an empty dict.
        """
        raise NotImplementedError("Preprocess is not implemented for this stage.")

    def preprocess_decode_batch(
        self,
        *,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict], dict[str, Any] | None]:
        """Batched, decode-only counterpart of :meth:`preprocess`.

        This is the generic, model-agnostic fast path for steady-state decode
        (every request contributes exactly one token, ``span_len == 1`` and not
        prefill). Implementing it lets a stage process a whole decode batch in
        one shot instead of paying the per-request Python ``preprocess`` loop in
        the runner. Stages that do not implement it keep going through the
        scalar :meth:`preprocess` path; the runner also keeps using the scalar
        path for prefill and mixed spans. Seed-dependent talker-MTP sampling is
        scalarized later in the MTP forward path; decode preprocess itself is
        seed-independent.

        Args:
            input_ids: 1-D tensor of shape ``[B]`` holding one decode token per
                request, ordered to match ``req_infos``.
            req_infos: per-request info dicts (the same payloads the scalar
                path receives as ``**input_dict``), one per request in ``B``.

        Returns a 4-tuple:
            - ``req_input_ids``: ``[B]`` processed input ids.
            - ``req_embeds``: ``[B, hidden_size]`` processed input embeddings.
            - ``updates``: length-``B`` list of per-request state-update dicts,
              merged into the runner's intermediate buffer and persisted across
              steps. This is the model-agnostic base return; it must NOT carry
              transient per-step compute inputs.
            - ``extras``: optional, model-specific batch-level extension, or
              ``None`` when the stage has none. It carries transient per-step
              compute inputs that are consumed immediately (not persisted). For
              talker-MTP models this is ``{"mtp_inputs": (last_talker_hidden,
              text_step)}`` where both are batch-level ``[B, hidden_size]``
              tensors. Kept separate from ``updates`` because (1) it has a
              different lifecycle (immediate vs persisted) and (2) it keeps the
              base three-element return clean for non-MTP stages, which return
              ``extras=None``.

        Implementations must keep output parity with the scalar decode branch
        of :meth:`preprocess`. In particular the per-request state they must
        reproduce exactly (see Qwen3-TTS for the reference implementation):
            - ``hidden_states['last']``: required for the next decode step's
              code predictor; missing it must fail fast, not be silently zeroed.
            - ``hidden_states['trailing_text']`` / ``meta['talker_text_offset']``:
              the trailing-text frame is consumed by offset and compacted past a
              threshold; the offset must advance identically to the scalar path
              or later tokens desync.
            - ``meta['codec_streaming']``: the Base-vs-CustomVoice default must
              be unchanged.
        """
        raise NotImplementedError("preprocess_decode_batch is not implemented for this stage.")

    def postprocess(self, model_output, **info_dict: object):
        """
        Postprocess the model output.
        Returns the postprocessed model output and the save dictionary.
        Args:
            model_output: The model output to postprocess.
        """
        raise NotImplementedError("Postprocess is not implemented for this stage.")
