"""
diffusion_action_decoder.py

Discrete diffusion-style (mask-predict) decoder for action tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class DiffusionActionDecoder:
    """Iterative mask-predict decoder for discrete action tokens."""

    mask_token_id: int

    def _keep_ratio(self, step: int, steps: int, schedule: str) -> float:
        if steps <= 0:
            raise ValueError("steps must be >= 1")
        if schedule == "linear":
            return float(step) / float(steps)
        if schedule == "cosine":
            # 0 -> 1 with cosine ease-in
            return 0.5 * (1.0 - math.cos(math.pi * float(step) / float(steps)))
        raise ValueError(f"Unknown mask schedule: {schedule}")

    @torch.no_grad()
    def decode(
        self,
        model: nn.Module,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        action_dim: int,
        steps: int,
        schedule: str,
        action_vocab_start: int,
        action_vocab_end: int,
    ) -> torch.LongTensor:
        """
        Args:
            model: Causal LM model with a forward returning `logits`.
            input_ids: [B, L] prompt tokens.
            attention_mask: [B, L] attention mask (optional).
            pixel_values: image inputs (optional, forwarded to model).
            action_dim: number of action tokens to decode.
            steps: diffusion steps K.
            schedule: mask schedule ("linear" or "cosine").
            action_vocab_start/end: inclusive/exclusive range for action tokens.
        Returns:
            action_token_ids: [B, action_dim] discrete action tokens.
        """
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if action_vocab_end <= action_vocab_start:
            raise ValueError("Invalid action vocab range")
        if steps < 1:
            raise ValueError("steps must be >= 1")

        device = input_ids.device
        batch_size = input_ids.shape[0]
        mask_token = int(self.mask_token_id)

        # Initialize all action tokens as [MASK]
        action_ids = torch.full(
            (batch_size, action_dim), mask_token, dtype=torch.long, device=device
        )
        frozen = torch.zeros((batch_size, action_dim), dtype=torch.bool, device=device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)

        for step in range(1, steps + 1):
            # Append current action tokens to prompt
            curr_input_ids = torch.cat([input_ids, action_ids], dim=1)
            action_attn = torch.ones((batch_size, action_dim), dtype=attention_mask.dtype, device=device)
            curr_attention = torch.cat([attention_mask, action_attn], dim=1)

            outputs = model(
                input_ids=curr_input_ids,
                attention_mask=curr_attention,
                pixel_values=pixel_values,
                use_cache=False,
                return_dict=True,
            )

            logits = outputs.logits[:, -action_dim:, action_vocab_start:action_vocab_end]
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            pred_tokens = idx + action_vocab_start

            keep_ratio = self._keep_ratio(step, steps, schedule)
            keep_num = max(1, int(round(keep_ratio * action_dim)))

            # Ensure monotonic freezing (never unfreeze positions)
            frozen_count = frozen.sum(dim=1)
            target_keep = torch.maximum(
                frozen_count, torch.full_like(frozen_count, keep_num, dtype=frozen_count.dtype)
            )

            new_frozen = torch.zeros_like(frozen)
            select_scores = conf.clone()
            # Force already-frozen positions to be kept
            select_scores[frozen] = float("inf")

            for b in range(batch_size):
                k = int(target_keep[b].item())
                topk_idx = torch.topk(select_scores[b], k=k, dim=-1).indices
                new_frozen[b, topk_idx] = True

            # Update action ids: keep old for frozen positions, set new for newly frozen
            mask_fill = torch.full_like(action_ids, mask_token)
            action_ids = torch.where(
                frozen,
                action_ids,
                torch.where(new_frozen, pred_tokens, mask_fill),
            )
            frozen = new_frozen

        return action_ids
