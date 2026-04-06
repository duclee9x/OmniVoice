#!/usr/bin/env python3
# Copyright    2026  (authors: Distillation module)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hidden-state CFG (Classifier-Free Guidance) for OmniVoice.

How OmniVoice CFG works
-----------------------
During **inference**, OmniVoice runs the backbone twice per step:

- **Conditional** input: ``[style | text | ref_audio? | target(MASK)]``
- **Unconditional** input: ``[target(MASK)]`` only (NO style, NO text, NO ref_audio)
  - audio_mask = all True (every position treated as audio embedding)

During **training**, unconditional conditioning is implemented via
``drop_cond_ratio=0.1`` in the data processor:
- When ``drop_cond=True``: ``input_ids = audio_inputs`` only (no style/text prefix)
  and ``audio_mask = all ones``

The CFG blend happens on **log-softmax outputs** (output-space CFG)::

    log_probs = softmax(c_log + cfg_scale * (c_log - u_log))

Hidden-state CFG alternative
-----------------------------
We replace output-space CFG with hidden-state CFG::

    H_cfg = H_uncond + cfg_scale * (H_cond - H_uncond)
    logits = audio_heads(H_cfg)  [runs audio_heads ONCE]
"""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _build_uncond_from_training_batch(batch: dict, audio_mask: torch.Tensor):
    """Extract unconditional inputs from a training batch.

    Matches OmniVoice's actual uncond mechanism:
    - Strip all text/style prefix tokens
    - Keep ONLY audio region tokens
    - Set audio_mask = all True

    Args:
        batch: Training batch with input_ids [B, C, S] and audio_mask [B, S].
        audio_mask: Audio mask from the batch [B, S].

    Returns:
        uncond_input_ids: [B, C, S_audio] — only the audio region.
        uncond_audio_mask: [B, S_audio] — all True.
    """
    B = batch["input_ids"].size(0)
    uncond_parts = []

    for i in range(B):
        # Find where audio starts (first True in audio_mask)
        mask_i = audio_mask[i]  # [S]
        audio_positions = mask_i.nonzero(as_tuple=True)[0]
        if len(audio_positions) == 0:
            # No audio? Shouldn't happen, but handle gracefully
            uncond_parts.append(batch["input_ids"][i:i+1])
            continue
        audio_start = audio_positions[0].item()
        # Extract only audio region tokens
        uncond_parts.append(batch["input_ids"][i:i+1, :, audio_start:])

    # All uncond sequences should be padded to same length
    max_u_len = max(p.size(2) for p in uncond_parts)

    device = batch["input_ids"].device
    uncond_input_ids = torch.full(
        (B, batch["input_ids"].size(1), max_u_len),
        fill_value=1024,  # mask_id as padding
        dtype=torch.long,
        device=device,
    )
    uncond_audio_mask = torch.ones(B, max_u_len, dtype=torch.bool, device=device)

    for i, part in enumerate(uncond_parts):
        u_len = part.size(2)
        uncond_input_ids[i, :, :u_len] = part[0]

    return uncond_input_ids, uncond_audio_mask


def get_teacher_hidden_states_cfg(
    teacher,
    batch: dict,
    cfg_scale: float = 1.5,
) -> torch.Tensor:
    """Extract CFG-blended hidden states from the teacher model.

    Correctly implements OmniVoice's unconditional conditioning:
    - Uncond = ONLY audio tokens (strip text/style prefix entirely)
    - audio_mask = all True for uncond

    The CFG blend happens in hidden-state space, not output space.

    Args:
        teacher: The frozen teacher OmniVoice model.
        batch: Training batch dict with: input_ids [B, C, S], audio_mask [B, S].
        cfg_scale: CFG guidance scale.

    Returns:
        H_cfg: CFG-blended hidden states [B, S, hidden_size].
            Full-sequence length matching the CONDITIONAL input.
    """
    with torch.no_grad():
        # ---- Conditional forward (full input) ----
        cond_embeds = teacher._prepare_embed_inputs(
            batch["input_ids"], batch["audio_mask"]
        )
        cond_out = teacher.llm(
            inputs_embeds=cond_embeds,
            attention_mask=batch.get("attention_mask"),
            return_dict=True,
            position_ids=batch.get("position_ids"),
        )
        H_cond = cond_out[0]  # [B, S, hidden]

        if cfg_scale == 0:
            return H_cond

        # ---- Unconditional forward ----
        # Strip text/style prefix, keep ONLY audio tokens
        uncond_input_ids, uncond_audio_mask = _build_uncond_from_training_batch(
            batch, batch["audio_mask"]
        )

        uncond_embeds = teacher._prepare_embed_inputs(
            uncond_input_ids, uncond_audio_mask
        )
        uncond_out = teacher.llm(
            inputs_embeds=uncond_embeds,
            return_dict=True,
        )
        H_uncond = uncond_out[0]  # [B, S_uncond, hidden]

        # ---- CFG blend ----
        # H_cond is [B, S_full, hidden], H_uncond is [B, S_audio, hidden]
        # We need to blend only the audio region of H_cond with H_uncond
        B = H_cond.size(0)
        H_cfg = H_cond.clone()

        for i in range(B):
            audio_positions = batch["audio_mask"][i].nonzero(as_tuple=True)[0]
            if len(audio_positions) == 0:
                continue
            audio_start = audio_positions[0].item()
            audio_len = len(audio_positions)
            u_len = min(H_uncond.size(1), audio_len)

            # Blend: H_cfg_audio = H_uncond + cfg * (H_cond_audio - H_uncond)
            H_cond_audio = H_cond[i, audio_start:audio_start+u_len, :]
            H_uncond_i = H_uncond[i, :u_len, :]
            H_cfg[i, audio_start:audio_start+u_len, :] = (
                H_uncond_i + cfg_scale * (H_cond_audio - H_uncond_i)
            )

    return H_cfg


def hidden_state_cfg_forward(
    model,
    cond_input_ids: torch.Tensor,
    cond_audio_mask: torch.Tensor,
    cond_attention_mask: torch.Tensor,
    uncond_input_ids: torch.Tensor,
    uncond_audio_mask: torch.Tensor,
    uncond_attention_mask: torch.Tensor,
    cfg_scale: float = 1.5,
) -> torch.Tensor:
    """Compute CFG-blended hidden states from conditional/unconditional inputs.

    For use during inference. The caller is responsible for constructing
    the unconditional inputs correctly (only target tokens, no prefix).

    Returns:
        H_cfg: [B, T, hidden_size] where T = target audio length.
    """
    if cfg_scale == 0:
        cond_embeds = model._prepare_embed_inputs(cond_input_ids, cond_audio_mask)
        llm_out = model.llm(
            inputs_embeds=cond_embeds,
            attention_mask=cond_attention_mask,
            return_dict=True,
        )
        return llm_out[0]

    # Conditional forward
    cond_embeds = model._prepare_embed_inputs(cond_input_ids, cond_audio_mask)
    cond_out = model.llm(
        inputs_embeds=cond_embeds,
        attention_mask=cond_attention_mask,
        return_dict=True,
    )
    H_cond = cond_out[0]

    # Unconditional forward
    uncond_embeds = model._prepare_embed_inputs(uncond_input_ids, uncond_audio_mask)
    uncond_out = model.llm(
        inputs_embeds=uncond_embeds,
        attention_mask=uncond_attention_mask,
        return_dict=True,
    )
    H_uncond = uncond_out[0]

    # CFG blend on target region only
    # uncond_input_ids contains only target tokens, so H_uncond is [B, T, hidden]
    # Extract matching region from H_cond (last T positions)
    T = H_uncond.size(1)
    H_cond_target = H_cond[:, -T:, :]
    H_cfg = H_uncond + cfg_scale * (H_cond_target - H_uncond)

    return H_cfg


def hidden_state_cfg_inference(
    model,
    cond_input_ids: torch.Tensor,
    cond_audio_mask: torch.Tensor,
    cond_attention_mask: torch.Tensor,
    uncond_input_ids: torch.Tensor,
    uncond_audio_mask: torch.Tensor,
    uncond_attention_mask: torch.Tensor,
    target_len: int,
    cond_len: int,
    cfg_scale: float = 1.5,
    class_temperature: float = 0.0,
    layer_penalty_factor: float = 5.0,
    position_temperature: float = 5.0,
    num_step: int = 8,
    t_shift: float = 0.1,
) -> torch.Tensor:
    """Iterative masked decoding with hidden-state CFG.

    Mirrors OmniVoice._generate_iterative() but uses hidden-state CFG.

    Important: uncond_input_ids should contain ONLY the target audio tokens
    (no text/style prefix), matching OmniVoice's actual uncond mechanism.

    Returns:
        Generated audio tokens [1, C, target_len].
    """
    from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample, _filter_top_k

    C = model.config.num_audio_codebook
    V = model.config.audio_vocab_size
    mask_id = model.config.audio_mask_id
    device = model.device

    tokens = torch.full(
        (1, C, target_len), mask_id, dtype=torch.long, device=device
    )

    # Compute schedule
    timesteps = _get_time_steps(0.0, 1.0, num_step + 1, t_shift).tolist()
    total_mask = target_len * C
    rem = total_mask
    schedule = []
    for step in range(num_step):
        num = (
            rem
            if step == num_step - 1
            else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
        )
        schedule.append(int(num))
        rem -= int(num)

    layer_ids = torch.arange(C, device=device).view(1, -1, 1)

    for step in range(num_step):
        k = schedule[step]
        if k <= 0:
            continue

        # Update target tokens in both cond and uncond inputs
        cond_input_ids[0, :, cond_len - target_len : cond_len] = tokens[0]
        uncond_input_ids[0, :, :target_len] = tokens[0]

        # Run backbone for both
        cond_embeds = model._prepare_embed_inputs(cond_input_ids, cond_audio_mask)
        cond_out = model.llm(
            inputs_embeds=cond_embeds,
            attention_mask=cond_attention_mask,
            return_dict=True,
        )
        H_cond = cond_out[0]

        uncond_embeds = model._prepare_embed_inputs(uncond_input_ids, uncond_audio_mask)
        uncond_out = model.llm(
            inputs_embeds=uncond_embeds,
            attention_mask=uncond_attention_mask,
            return_dict=True,
        )
        H_uncond = uncond_out[0]

        # Extract target region
        H_cond_target = H_cond[:, cond_len - target_len : cond_len, :]
        H_uncond_target = H_uncond[:, :target_len, :]

        # CFG blend
        if cfg_scale != 0:
            H_cfg = H_uncond_target + cfg_scale * (H_cond_target - H_uncond_target)
        else:
            H_cfg = H_cond_target

        # Project through hidden_proj if available (student model)
        if hasattr(model, "hidden_proj"):
            H_cfg = model.hidden_proj(H_cfg)

        # Run audio_heads ONCE
        logits_flat = model.audio_heads(H_cfg)
        logits = logits_flat.view(1, target_len, C, V).permute(0, 2, 1, 3)

        # Predict tokens
        log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
        log_probs[..., mask_id] = -float("inf")

        if class_temperature > 0.0:
            filtered = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(filtered, class_temperature).argmax(dim=-1)
        else:
            pred_tokens = log_probs.argmax(dim=-1)

        scores = log_probs.max(dim=-1)[0]
        scores = scores - (layer_ids * layer_penalty_factor)

        if position_temperature > 0.0:
            scores = _gumbel_sample(scores, position_temperature)

        scores.masked_fill_(tokens != mask_id, -float("inf"))
        _, topk_idx = torch.topk(scores.flatten(), k)
        flat_tokens = tokens.flatten()
        flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
        tokens.copy_(flat_tokens.view_as(tokens))

    return tokens
