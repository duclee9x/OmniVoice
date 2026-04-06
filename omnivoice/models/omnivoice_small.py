#!/usr/bin/env python3
# Copyright    2026  (authors: Distillation module)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""OmniVoice-Small: Shrunk backbone for knowledge distillation.

Architecture overview::

    Input tokens → audio_embeddings (teacher_dim=1024)
                 → embed_proj (1024 → student_dim)
                 → Student LLM backbone (student_dim)
                 → hidden_proj (student_dim → 1024)
                 → audio_heads (1024 → 8*1025)

- ``audio_embeddings`` and ``audio_heads`` are at teacher dimension (1024)
  and can be copied directly from the teacher model.
- ``embed_proj`` projects embeddings down from teacher → student dimension.
- ``hidden_proj`` projects hidden states back up from student → teacher dimension.
- Both projections are learned during distillation.

Usage::

    from omnivoice.models.omnivoice_small import OmniVoiceSmall

    student = OmniVoiceSmall.from_small_config("small")
    print(student.count_total_params() / 1e6, "M params")
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small model configurations (Qwen3 GQA architecture)
# ---------------------------------------------------------------------------

SMALL_CONFIGS = {
    "nano": {
        "num_hidden_layers": 6,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 1364,
    },
    "small": {
        "num_hidden_layers": 12,
        "hidden_size": 640,
        "num_attention_heads": 10,
        "num_key_value_heads": 5,
        "intermediate_size": 1706,
    },
    "medium": {
        "num_hidden_layers": 16,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_key_value_heads": 6,
        "intermediate_size": 2048,
    },
}

TEACHER_HIDDEN_SIZE = 1024


@dataclass
class OmniVoiceSmallOutput(ModelOutput):
    """Output of OmniVoiceSmall.forward()."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states_projected: Optional[torch.Tensor] = None


class OmniVoiceSmall(OmniVoice):
    """Shrunk OmniVoice with projection layers for distillation.

    The student's LLM backbone operates at a smaller hidden dimension.
    Two learned projections bridge the gap:
      - embed_proj: teacher_dim → student_dim (before LLM)
      - hidden_proj: student_dim → teacher_dim (after LLM)

    audio_embeddings and audio_heads remain at teacher dimension (1024)
    so they can be copied from the teacher and shared.
    """

    def __init__(
        self,
        config: OmniVoiceConfig,
        llm: Optional[PreTrainedModel] = None,
        teacher_hidden_size: int = TEACHER_HIDDEN_SIZE,
    ):
        # Parent __init__ creates audio_embeddings at config.llm_config.hidden_size
        # but we want it at teacher_hidden_size. We'll override after super().__init__.
        super().__init__(config, llm=llm)

        student_hidden = self.config.llm_config.hidden_size
        self._teacher_hidden_size = teacher_hidden_size

        if student_hidden != teacher_hidden_size:
            # Override audio_embeddings to use teacher dimension
            self.audio_embeddings = nn.Embedding(
                config.num_audio_codebook * config.audio_vocab_size,
                teacher_hidden_size,
            )
            # Override audio_heads to use teacher dimension
            self.audio_heads = nn.Linear(
                teacher_hidden_size,
                config.num_audio_codebook * config.audio_vocab_size,
                bias=False,
            )
            # Projection: teacher_dim → student_dim (for LLM input)
            self.embed_proj = nn.Linear(teacher_hidden_size, student_hidden)
            # Projection: student_dim → teacher_dim (for audio_heads)
            self.hidden_proj = nn.Linear(student_hidden, teacher_hidden_size)
        else:
            self.embed_proj = nn.Identity()
            self.hidden_proj = nn.Identity()

    @classmethod
    def from_small_config(
        cls,
        config_name: str = "small",
        teacher_hidden_size: int = TEACHER_HIDDEN_SIZE,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        audio_codebook_weights: Optional[list] = None,
        attn_implementation: str = "sdpa",
    ) -> "OmniVoiceSmall":
        """Create a student model from a predefined small config."""
        if config_name not in SMALL_CONFIGS:
            raise ValueError(
                f"Unknown config_name '{config_name}'. "
                f"Choose from: {list(SMALL_CONFIGS.keys())}"
            )

        small_cfg = SMALL_CONFIGS[config_name]

        # Use Qwen3-0.6B config as template
        base_llm_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")

        # Override with small dims
        base_llm_config.num_hidden_layers = small_cfg["num_hidden_layers"]
        base_llm_config.hidden_size = small_cfg["hidden_size"]
        base_llm_config.num_attention_heads = small_cfg["num_attention_heads"]
        base_llm_config.num_key_value_heads = small_cfg["num_key_value_heads"]
        base_llm_config.intermediate_size = small_cfg["intermediate_size"]

        # Force float32 torch_dtype for CPU/MPS compatibility
        base_llm_config.torch_dtype = "float32"

        ov_config = OmniVoiceConfig(
            audio_vocab_size=audio_vocab_size,
            audio_mask_id=audio_mask_id,
            num_audio_codebook=num_audio_codebook,
            audio_codebook_weights=audio_codebook_weights,
            llm_config=base_llm_config,
        )

        llm = AutoModel.from_config(
            base_llm_config,
            attn_implementation=attn_implementation,
        )
        # Ensure float32 weights
        llm = llm.float()

        model = cls(
            config=ov_config,
            llm=llm,
            teacher_hidden_size=teacher_hidden_size,
        )
        model = model.float()  # Ensure all params are float32

        logger.info(
            "Created OmniVoiceSmall (%s): "
            "layers=%d, hidden=%d, heads=%d/%d, ffn=%d, "
            "total_params=%.1fM",
            config_name,
            small_cfg["num_hidden_layers"],
            small_cfg["hidden_size"],
            small_cfg["num_attention_heads"],
            small_cfg["num_key_value_heads"],
            small_cfg["intermediate_size"],
            sum(p.numel() for p in model.parameters()) / 1e6,
        )

        return model

    def _prepare_embed_inputs(self, input_ids, audio_mask):
        """Override to add embed_proj projection.

        Matches parent OmniVoice._prepare_embed_inputs exactly, but adds
        embed_proj to project from teacher_dim → student_dim.
        """
        # Text embedding (from LLM word embeddings at student_dim)
        text_embeds = self.llm.get_input_embeddings()(input_ids[:, 0, :])

        # Audio embedding: shift IDs by codebook offsets (same as parent)
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1)
        ) + self.codebook_layer_offsets.view(1, -1, 1)

        # Embed at teacher_dim, sum across codebook layers (same as parent)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)  # [B, S, 1024]

        # Project from teacher_dim → student_dim
        audio_embeds = self.embed_proj(audio_embeds)  # [B, S, student_dim]

        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> OmniVoiceSmallOutput:
        """Forward pass returning hidden_states_projected for distillation."""

        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)

        if attention_mask is None and document_ids is not None:
            from torch.nn.attention.flex_attention import create_block_mask
            from omnivoice.models.omnivoice import _get_packed_mask

            attention_mask = create_block_mask(
                _get_packed_mask(document_ids[0].to(inputs_embeds.device)),
                B=None, H=None,
                Q_LEN=input_ids.size(-1), KV_LEN=input_ids.size(-1),
                _compile=True, device=inputs_embeds.device,
            )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=position_ids,
        )
        hidden_states = llm_outputs[0]  # [B, S, student_dim]

        # Project to teacher dimension
        hidden_states_projected = self.hidden_proj(hidden_states)  # [B, S, 1024]

        # Audio logits via audio_heads (operates at teacher dim)
        batch_size, seq_len, _ = hidden_states_projected.shape
        logits_flat = self.audio_heads(hidden_states_projected)
        audio_logits = logits_flat.view(
            batch_size, seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)  # [B, C, S, V]

        # Loss
        loss = None
        if labels is not None:
            per_token_loss = torch.nn.functional.cross_entropy(
                audio_logits.permute(0, 3, 1, 2),
                labels,
                reduction="none",
                ignore_index=-100,
            )
            valid_mask = (labels != -100).float()
            layer_means = (per_token_loss * valid_mask).sum(
                dim=(0, 2)
            ) / valid_mask.sum(dim=(0, 2)).clamp(min=1.0)
            weights = torch.tensor(
                self.normalized_audio_codebook_weights,
                device=audio_logits.device,
            )
            loss = (layer_means * weights).sum()

        return OmniVoiceSmallOutput(
            loss=loss,
            logits=audio_logits,
            hidden_states_projected=hidden_states_projected,
        )

    def copy_shared_weights_from_teacher(self, teacher: OmniVoice):
        """Copy audio_embeddings and audio_heads from teacher."""
        self.audio_embeddings.load_state_dict(
            teacher.audio_embeddings.state_dict()
        )
        self.audio_heads.load_state_dict(teacher.audio_heads.state_dict())
        self.codebook_layer_offsets.copy_(teacher.codebook_layer_offsets)
        logger.info("Copied audio_embeddings, audio_heads from teacher.")

    def freeze_shared_weights(self):
        """Freeze audio_embeddings and audio_heads (Phase A)."""
        for p in self.audio_embeddings.parameters():
            p.requires_grad = False
        for p in self.audio_heads.parameters():
            p.requires_grad = False
        logger.info("Froze audio_embeddings and audio_heads.")

    def unfreeze_shared_weights(self):
        """Unfreeze audio_embeddings and audio_heads (Phase B)."""
        for p in self.audio_embeddings.parameters():
            p.requires_grad = True
        for p in self.audio_heads.parameters():
            p.requires_grad = True
        logger.info("Unfroze audio_embeddings and audio_heads.")

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
