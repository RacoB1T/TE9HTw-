"""
Custom SimPO Trainer for LOGO.

Provides the base trainer interface that SimORPOTrainer and LOGOTrainer
in logo_train.py inherit from. Built on top of Hugging Face Trainer with
SimPO-specific utilities: log-probability computation, device movement,
dropout control, and cache support for pre-tokenized datasets.

Key design note:
  The LOGO codebase passes load_from_cache_path=True and cache_path=None
  to the Trainer constructor. These are custom kwargs that skip tokenization
  when the dataset already contains input_ids / labels / attention_mask /
  position_ids fields — the SimPODataCollator only stacks and truncates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput

logger = logging.getLogger(__name__)


class SimPOTrainer(Trainer):
    """Base trainer providing the utilities that LOGO/SimORPO subclasses rely on.

    This is NOT a drop-in copy of the Princeton SimPO trainer. It is a
    re-implementation that matches the LOGO call signature, including:

        - load_from_cache_path / cache_path kwargs
        - move_to_device
        - get_batch_logps (used by concatenated_forward in subclasses)
        - disable_dropout
    """

    def __init__(
        self,
        model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        **kwargs,
    ):
        # --- absorb LOGO-specific kwargs so they don't reach HF Trainer ---
        load_from_cache_path = kwargs.pop("load_from_cache_path", False)
        cache_path = kwargs.pop("cache_path", None)

        # --- Transformers 5.x compat: keep custom data columns alive ---
        if getattr(args, "remove_unused_columns", True):
            logger.warning(
                "Setting remove_unused_columns=False — LOGO datasets carry "
                "custom fields (chosen_input_ids, reject_1_*, …) that are "
                "not in the model forward() signature and would otherwise "
                "be dropped."
            )
            args.remove_unused_columns = False

        # --- Transformers 5.x compat: Trainer uses processing_class ---
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        # Keep a .tokenizer alias for subclasses that reference it directly.
        self.tokenizer = tokenizer

        # --- SimPO hyperparameters ---
        self.beta = getattr(args, "beta", 2.0)
        self.gamma_beta_ratio = getattr(args, "gamma_beta_ratio", 0.25)
        self.loss_type = getattr(args, "loss_type", "sigmoid")
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)
        self.sft_weight = getattr(args, "sft_weight", 0.0)

        # --- sequence / token configuration ---
        self.max_target_length = getattr(args, "max_target_length", 512)
        self.label_pad_token_id = getattr(args, "label_pad_token_id", -100)
        self.is_encoder_decoder = getattr(args, "is_encoder_decoder", None)

        # --- cache support ---
        self.load_from_cache_path = load_from_cache_path
        self.cache_path = cache_path

        # --- reference model (unused in LOGO but may be set via kwargs) ---
        self.ref_model = kwargs.pop("ref_model", None)
        self.reference_free = getattr(args, "reference_free", False)

        if self.is_encoder_decoder is None:
            self.is_encoder_decoder = getattr(
                model.config, "is_encoder_decoder", False
            )

        if hasattr(args, "disable_dropout") and args.disable_dropout:
            self.disable_dropout(model)

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def move_to_device(
        batch: Dict[str, Any],
        device: Union[str, torch.device],
    ) -> Dict[str, Any]:
        """Recursively move all tensors in a (possibly nested) batch dict to
        *device*. Non-tensor values are passed through unchanged."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)

        if isinstance(batch, dict):
            return {
                key: SimPOTrainer.move_to_device(value, device)
                for key, value in batch.items()
            }

        if isinstance(batch, (list, tuple)):
            return type(batch)(
                SimPOTrainer.move_to_device(item, device) for item in batch
            )

        return batch

    @staticmethod
    def disable_dropout(model: torch.nn.Module) -> None:
        """Permanently set dropout probability to 0 for all Dropout modules.

        Using ``module.eval()`` is **not** sufficient — the Trainer calls
        ``model.train()`` at the start of training, which recursively
        re-enables every sub-module's training mode (and thus dropout).

        Setting ``module.p = 0.0`` survives ``model.train()``.
        """
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

        logger.info(
            "Dropout probability has been set to 0.0 "
            "for all Dropout modules."
        )

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute per-token log-probabilities for causal LMs.

        For decoder-only models the standard shift is applied:
            logits[t]  predicts  labels[t+1]
        so we drop the last logit position and the first label position.

        Parameters
        ----------
        logits : (batch, seq_len, vocab)
        labels : (batch, seq_len)
        average_log_prob : bool
            If True, return the *average* log-probability across non-padding
            tokens. If False, return the sum.
        is_encoder_decoder : bool
            If True, the shift is skipped (encoder-decoder models produce
            logits that are already aligned with labels).
        label_pad_token_id : int
            Token id used for padding / masking (default -100).

        Returns
        -------
        logps : (batch,)  — averaged or summed per sequence.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape[:-1]} "
                f"vs labels {labels.shape}"
            )

        labels = labels.to(logits.device)

        # ---- causal LM shift ----
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        # ---- build a mask BEFORE touching the labels ----
        loss_mask = labels != label_pad_token_id

        # Replace pad tokens with a safe index (0) so gather never sees -100.
        safe_labels = labels.masked_fill(~loss_mask, 0)

        per_token_logps = torch.gather(
            logits.log_softmax(dim=-1),
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)  # (batch, seq_len-1)

        # Zero-out positions that are padding
        per_token_logps = per_token_logps * loss_mask

        sequence_logps = per_token_logps.sum(dim=-1)

        if average_log_prob:
            token_counts = loss_mask.sum(dim=-1).clamp(min=1)
            sequence_logps = sequence_logps / token_counts

        return sequence_logps

    # ------------------------------------------------------------------
    #  HuggingFace Trainer hooks
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Delegates to ``self.get_batch_loss_metrics``, which must be
        implemented by subclasses (SimORPOTrainer / LOGOTrainer)."""
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Store metrics so they are logged by HF Trainer callbacks
        if hasattr(self, "store_metrics"):
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)

        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Evaluation step — delegates to ``get_batch_loss_metrics``."""
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="eval"
            )

        # No generated predictions — loss-only eval
        if prediction_loss_only:
            return (loss, None, None)

        # Collect batch-level metrics as "predictions"
        return (loss, metrics, None)

    def store_metrics(
        self, metrics: Dict[str, Any], train_eval: str = "train"
    ) -> None:
        """Accumulate metrics for logging.

        Subclasses may override; default simply logs via self.log()."""
        self.log(metrics)
