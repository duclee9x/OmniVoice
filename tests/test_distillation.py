#!/usr/bin/env python3
"""Tests for OmniVoice distillation pipeline.

Uses tiny models with float32 for CPU testing.

Usage::

    cd /Users/duclee/OmniVoice
    python -m pytest tests/test_distillation.py -v
"""

import pytest
import torch
import torch.nn as nn

# Register OmniVoice Auto classes
import omnivoice.models.omnivoice  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def teacher_model():
    """Create a tiny 2-layer teacher model in float32."""
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
    from transformers import AutoConfig, AutoModel

    llm_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    llm_config.num_hidden_layers = 2
    llm_config.torch_dtype = "float32"

    ov_config = OmniVoiceConfig(
        audio_vocab_size=1025,
        audio_mask_id=1024,
        num_audio_codebook=8,
        llm_config=llm_config,
    )

    llm = AutoModel.from_config(llm_config, attn_implementation="sdpa")
    llm = llm.float()

    model = OmniVoice(config=ov_config, llm=llm)
    model = model.float()
    model.eval()
    return model


@pytest.fixture(scope="module")
def dummy_batch():
    """Dummy training batch [B=2, C=8, S=32]."""
    B, C, S = 2, 8, 32
    V = 1025
    input_ids = torch.randint(0, V, (B, C, S))
    audio_mask = torch.zeros(B, S, dtype=torch.bool)
    audio_mask[:, S // 2 :] = True
    labels = torch.randint(0, V - 1, (B, C, S))
    labels[:, :, : S // 2] = -100
    return {"input_ids": input_ids, "audio_mask": audio_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Test 1: Parameter counts
# ---------------------------------------------------------------------------


class TestStudentParamCount:

    def test_nano_param_count(self):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        model = OmniVoiceSmall.from_small_config("nano")
        total = model.count_total_params() / 1e6
        print(f"nano: {total:.1f}M params")
        assert total < 120, f"Nano should be < 120M, got {total:.1f}M"

    def test_small_param_count(self):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        model = OmniVoiceSmall.from_small_config("small")
        total = model.count_total_params() / 1e6
        print(f"small: {total:.1f}M params")
        assert total < 200, f"Small should be < 200M, got {total:.1f}M"

    def test_medium_param_count(self):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        model = OmniVoiceSmall.from_small_config("medium")
        total = model.count_total_params() / 1e6
        print(f"medium: {total:.1f}M params")
        assert total < 300, f"Medium should be < 300M, got {total:.1f}M"

    def test_configs_exist(self):
        from omnivoice.models.omnivoice_small import SMALL_CONFIGS
        assert "nano" in SMALL_CONFIGS
        assert "small" in SMALL_CONFIGS
        assert "medium" in SMALL_CONFIGS

    def test_invalid_config_raises(self):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        with pytest.raises(ValueError, match="Unknown config_name"):
            OmniVoiceSmall.from_small_config("nonexistent")


# ---------------------------------------------------------------------------
# Test 2: Forward pass shapes
# ---------------------------------------------------------------------------


class TestForwardShape:

    def test_student_logits_shape(self, dummy_batch):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        model = OmniVoiceSmall.from_small_config("nano")
        model.eval()
        with torch.no_grad():
            out = model(**dummy_batch)
        B, C, S = dummy_batch["input_ids"].shape
        V = model.config.audio_vocab_size
        assert out.logits.shape == (B, model.config.num_audio_codebook, S, V)

    def test_student_teacher_logits_match_shape(self, teacher_model, dummy_batch):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        student = OmniVoiceSmall.from_small_config("nano")
        student.eval()
        with torch.no_grad():
            teacher_out = teacher_model(**dummy_batch)
            student_out = student(**dummy_batch)
        assert teacher_out.logits.shape == student_out.logits.shape


# ---------------------------------------------------------------------------
# Test 3: Hidden projection
# ---------------------------------------------------------------------------


class TestHiddenProj:

    def test_hidden_proj_shape(self, dummy_batch):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        model = OmniVoiceSmall.from_small_config("nano")
        model.eval()
        with torch.no_grad():
            out = model(**dummy_batch)
        B, C, S = dummy_batch["input_ids"].shape
        assert out.hidden_states_projected.shape == (B, S, 1024)

    def test_hidden_proj_is_linear(self):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        model = OmniVoiceSmall.from_small_config("nano")
        assert isinstance(model.hidden_proj, nn.Linear)
        assert model.hidden_proj.in_features == 512   # nano
        assert model.hidden_proj.out_features == 1024  # teacher

    def test_embed_proj_exists(self):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        model = OmniVoiceSmall.from_small_config("nano")
        assert isinstance(model.embed_proj, nn.Linear)
        assert model.embed_proj.in_features == 1024  # teacher
        assert model.embed_proj.out_features == 512   # nano

    def test_hidden_proj_gradient_flows(self, dummy_batch):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        model = OmniVoiceSmall.from_small_config("nano")
        model.train()
        out = model(**dummy_batch)
        loss = out.hidden_states_projected.mean()
        loss.backward()
        assert model.hidden_proj.weight.grad is not None
        assert model.hidden_proj.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Test 4: Distillation step
# ---------------------------------------------------------------------------


class TestDistillationStep:

    def test_distill_loss_finite(self, teacher_model, dummy_batch):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        from omnivoice.inference.latent_cfg import get_teacher_hidden_states_cfg
        import torch.nn.functional as F

        student = OmniVoiceSmall.from_small_config("nano")
        student.copy_shared_weights_from_teacher(teacher_model)
        student.freeze_shared_weights()
        student.train()

        with torch.no_grad():
            H_teacher = get_teacher_hidden_states_cfg(
                teacher_model, dummy_batch, cfg_scale=1.5
            )

        out = student(**dummy_batch)
        loss = F.mse_loss(out.hidden_states_projected, H_teacher.detach())
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss should be > 0, got {loss.item()}"

    def test_gradient_flows_to_backbone(self, teacher_model, dummy_batch):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        from omnivoice.inference.latent_cfg import get_teacher_hidden_states_cfg
        import torch.nn.functional as F

        student = OmniVoiceSmall.from_small_config("nano")
        student.copy_shared_weights_from_teacher(teacher_model)
        student.freeze_shared_weights()
        student.train()

        with torch.no_grad():
            H_teacher = get_teacher_hidden_states_cfg(
                teacher_model, dummy_batch, cfg_scale=1.5
            )

        out = student(**dummy_batch)
        loss = F.mse_loss(out.hidden_states_projected, H_teacher.detach())
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.llm.parameters()
        )
        assert has_grad, "No gradients flowed to student backbone"

    def test_frozen_weights_no_grad(self, teacher_model, dummy_batch):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall
        from omnivoice.inference.latent_cfg import get_teacher_hidden_states_cfg
        import torch.nn.functional as F

        student = OmniVoiceSmall.from_small_config("nano")
        student.copy_shared_weights_from_teacher(teacher_model)
        student.freeze_shared_weights()
        student.train()

        with torch.no_grad():
            H_teacher = get_teacher_hidden_states_cfg(
                teacher_model, dummy_batch, cfg_scale=1.5
            )

        out = student(**dummy_batch)
        loss = F.mse_loss(out.hidden_states_projected, H_teacher.detach())
        loss.backward()

        for p in student.audio_heads.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


# ---------------------------------------------------------------------------
# Test 5: CFG forward
# ---------------------------------------------------------------------------


class TestCFGForward:

    def test_cfg_output_shape(self, teacher_model, dummy_batch):
        from omnivoice.inference.latent_cfg import get_teacher_hidden_states_cfg

        H_cfg = get_teacher_hidden_states_cfg(teacher_model, dummy_batch, cfg_scale=1.5)
        B, C, S = dummy_batch["input_ids"].shape
        hidden = teacher_model.config.llm_config.hidden_size
        assert H_cfg.shape == (B, S, hidden)

    def test_cfg_scale_zero_equals_conditional(self, teacher_model, dummy_batch):
        from omnivoice.inference.latent_cfg import get_teacher_hidden_states_cfg

        H_cfg0 = get_teacher_hidden_states_cfg(teacher_model, dummy_batch, cfg_scale=0)
        with torch.no_grad():
            embeds = teacher_model._prepare_embed_inputs(
                dummy_batch["input_ids"], dummy_batch["audio_mask"]
            )
            out = teacher_model.llm(inputs_embeds=embeds, return_dict=True)
            H_cond = out[0]
        assert torch.allclose(H_cfg0, H_cond, atol=1e-5)

    def test_cfg_differs_from_conditional(self, teacher_model, dummy_batch):
        from omnivoice.inference.latent_cfg import get_teacher_hidden_states_cfg

        H_cfg15 = get_teacher_hidden_states_cfg(teacher_model, dummy_batch, cfg_scale=1.5)
        H_cfg0 = get_teacher_hidden_states_cfg(teacher_model, dummy_batch, cfg_scale=0)
        assert not torch.allclose(H_cfg15, H_cfg0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: Weight sharing
# ---------------------------------------------------------------------------


class TestWeightSharing:

    def test_copy_weights(self, teacher_model):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall

        student = OmniVoiceSmall.from_small_config("nano")
        student.copy_shared_weights_from_teacher(teacher_model)

        assert torch.equal(
            student.audio_heads.weight.float(),
            teacher_model.audio_heads.weight.float(),
        )
        assert torch.equal(
            student.audio_embeddings.weight.float(),
            teacher_model.audio_embeddings.weight.float(),
        )

    def test_freeze_unfreeze(self):
        from omnivoice.models.omnivoice_small import OmniVoiceSmall

        model = OmniVoiceSmall.from_small_config("nano")
        model.freeze_shared_weights()
        for p in model.audio_heads.parameters():
            assert not p.requires_grad
        for p in model.audio_embeddings.parameters():
            assert not p.requires_grad

        model.unfreeze_shared_weights()
        for p in model.audio_heads.parameters():
            assert p.requires_grad
        for p in model.audio_embeddings.parameters():
            assert p.requires_grad
