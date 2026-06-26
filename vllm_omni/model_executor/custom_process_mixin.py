from collections.abc import Callable

import torch

from vllm_omni.data_entry_keys import OmniPayload


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
        req_infos: list[dict[str, object]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[OmniPayload], dict[str, object] | None]:
        """Batched, decode-only counterpart of :meth:`preprocess`.

        Args:
            input_ids: 1-D tensor of one decode token per request.
            req_infos: per-request runtime payload entries collected from the
                runner's ``model_intermediate_buffer``, ordered to match
                ``input_ids``. Entries contain :class:`OmniPayload` fields plus
                runner-injected scheduling keys such as ``request_id``,
                ``_omni_prompt_len``, ``_omni_num_computed_tokens``, and
                ``_omni_is_prefill``.

        Returns:
            ``(req_input_ids, req_embeds, updates, extras)``. ``updates`` are
            per-request :class:`OmniPayload` state updates persisted by the
            runner. ``extras`` is an optional batch-level extension for
            transient compute inputs, such as ``{"mtp_inputs": (...)}`` for
            talker-MTP models.

        Implementations must match the scalar decode branch of
        :meth:`preprocess` for the same inputs.
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
