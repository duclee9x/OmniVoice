#!/usr/bin/env python3
# Copyright    2026  (authors: Distillation module)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Distillation trainer: shrink OmniVoice backbone from 0.6B → ~150M.

Strategy:
  Phase A (steps 0 → PHASE_A_STEPS):
    - Teacher backbone frozen entirely
    - Student audio_heads + audio_embeddings frozen (copied from teacher)
    - Only train: student LLM backbone + hidden_proj
    - Loss = MSE(student_hidden_proj(H_student), H_teacher_cfg)

  Phase B (steps PHASE_A_STEPS → total_steps):
    - Teacher still frozen
    - Unfreeze student audio_heads + audio_embeddings
    - Train: student LLM + hidden_proj + audio_heads + audio_embeddings
    - Loss = alpha * MSE(H_student_proj, H_teacher_cfg)
           + beta * weighted_CE(logits_student, ground_truth_tokens)

Usage::

    python omnivoice/train_distill.py \\
        --teacher-checkpoint drbaph/OmniVoice-bf16 \\
        --student-config small \\
        --data-dir /path/to/data \\
        --output-dir checkpoints/distilled \\
        --steps 70000
"""

import argparse
import copy
import logging
import math
import os
import sys
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datetime import timedelta
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from omnivoice.inference.latent_cfg import get_teacher_hidden_states_cfg
from omnivoice.models.omnivoice import OmniVoice
from omnivoice.models.omnivoice_small import OmniVoiceSmall
from omnivoice.training.checkpoint import TrainLogger, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distillation hyperparameters
# ---------------------------------------------------------------------------

PHASE_A_STEPS = 50_000   # Backbone-only distillation
CFG_ALPHA = 0.0           # CFG scale for teacher hidden states (0 = no CFG, saves VRAM)
DISTILL_WEIGHT = 1.0      # Weight of MSE distillation loss
RECON_WEIGHT = 0.1         # Weight of CE reconstruction loss (Phase B)


class DistillationTrainer:
    """Two-phase distillation trainer for OmniVoice.

    Phase A: Pure hidden-state distillation (backbone + hidden_proj only)
    Phase B: Joint fine-tuning (backbone + hidden_proj + audio_heads + embeddings)
    """

    def __init__(
        self,
        teacher_path: str,
        student_config: str = "small",
        output_dir: str = "checkpoints/distilled",
        data_config: Optional[str] = None,
        steps: int = 70_000,
        phase_a_steps: int = PHASE_A_STEPS,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        cfg_scale: float = CFG_ALPHA,
        distill_weight: float = DISTILL_WEIGHT,
        recon_weight: float = RECON_WEIGHT,
        warmup_ratio: float = 0.03,
        seed: int = 42,
        mixed_precision: str = "bf16",
        logging_steps: int = 100,
        save_steps: int = 10000,
        batch_tokens: int = 8192,
        num_workers: int = 4,
        resume_from: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.steps = steps
        self.phase_a_steps = phase_a_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.cfg_scale = cfg_scale
        self.distill_weight = distill_weight
        self.recon_weight = recon_weight
        self.warmup_ratio = warmup_ratio
        self.seed = seed
        self.mixed_precision = mixed_precision
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.batch_tokens = batch_tokens
        self.num_workers = num_workers
        self.data_config = data_config
        self.resume_from = resume_from
        self.student_config_name = student_config

        self.phase = "A"
        self.global_step = 0

        # ----- 1. Load Teacher (frozen) -----
        logger.info("Loading teacher model from %s ...", teacher_path)

        # Determine device-appropriate dtype
        if torch.cuda.is_available():
            teacher_dtype = torch.float16
        else:
            teacher_dtype = torch.float32

        self.teacher = OmniVoice.from_pretrained(
            teacher_path,
            torch_dtype=teacher_dtype,
            train=True,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        teacher_hidden = self.teacher.config.llm_config.hidden_size
        teacher_params = sum(p.numel() for p in self.teacher.parameters()) / 1e6
        logger.info(
            "Teacher loaded: hidden=%d, params=%.1fM", teacher_hidden, teacher_params
        )

        # ----- 2. Build Student -----
        logger.info("Building student model (config=%s) ...", student_config)
        self.student = OmniVoiceSmall.from_small_config(
            config_name=student_config,
            teacher_hidden_size=teacher_hidden,
        )

        # Copy audio_heads + audio_embeddings from teacher
        self.student.copy_shared_weights_from_teacher(self.teacher)

        # Freeze shared weights for Phase A
        self.student.freeze_shared_weights()

        student_params = sum(p.numel() for p in self.student.parameters()) / 1e6
        trainable_params = self.student.count_trainable_params() / 1e6
        logger.info(
            "Student: total=%.1fM, trainable=%.1fM (Phase A)",
            student_params,
            trainable_params,
        )

        # Normalized codebook weights for CE loss
        raw_weights = self.teacher.config.audio_codebook_weights
        total_w = sum(raw_weights)
        self.normalized_codebook_weights = [w / total_w for w in raw_weights]

        # ----- 3. Tokenizer -----
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_path)

    def build_optimizer_and_scheduler(self):
        """Create optimizer (only for trainable params) and LR scheduler."""
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        warmup_steps = max(1, int(self.steps * self.warmup_ratio))
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.steps,
        )

    def rebuild_optimizer_for_phase_b(self):
        """Rebuild optimizer when transitioning to Phase B (more params trainable)."""
        unwrapped = self.student.module if hasattr(self.student, "module") else self.student
        trainable = [p for p in unwrapped.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.learning_rate * 0.5,  # Lower LR for phase B
            weight_decay=self.weight_decay,
        )
        remaining = self.steps - self.global_step
        warmup = max(1, int(remaining * 0.05))
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=remaining,
        )
        logger.info(
            "Rebuilt optimizer for Phase B: %d trainable params, LR=%.2e",
            sum(p.numel() for p in trainable),
            self.learning_rate * 0.5,
        )

    def training_step(self, batch: dict) -> dict:
        """Execute one training step.

        Args:
            batch: Dict with keys: input_ids, audio_mask, labels,
                attention_mask, document_ids, position_ids.

        Returns:
            Dict with loss values for logging.
        """
        # ----- Teacher forward (frozen, no grad) -----
        H_teacher_cfg = get_teacher_hidden_states_cfg(
            self.teacher, batch, cfg_scale=self.cfg_scale
        )

        # ----- Student forward -----
        student_output = self.student(
            input_ids=batch["input_ids"],
            audio_mask=batch["audio_mask"],
            labels=batch.get("labels"),
            attention_mask=batch.get("attention_mask"),
            document_ids=batch.get("document_ids"),
            position_ids=batch.get("position_ids"),
        )

        H_student_proj = student_output.hidden_states_projected  # [B, S, 1024]

        # ----- Distillation loss: MSE on hidden states -----
        # Cast teacher hidden states to match student dtype (teacher=fp16, student=fp32)
        loss_distill = F.mse_loss(
            H_student_proj, H_teacher_cfg.detach().to(H_student_proj.dtype)
        )

        metrics = {"distill_loss": loss_distill.item()}

        if self.phase == "A":
            loss = self.distill_weight * loss_distill

        elif self.phase == "B":
            # Reconstruction loss: weighted CE on audio logits
            labels = batch.get("labels")
            if labels is not None and student_output.logits is not None:
                audio_logits = student_output.logits  # [B, C, S, V]
                per_token_loss = F.cross_entropy(
                    audio_logits.permute(0, 3, 1, 2),  # [B, V, C, S]
                    labels,
                    reduction="none",
                    ignore_index=-100,
                )
                valid_mask = (labels != -100).float()
                layer_means = (per_token_loss * valid_mask).sum(
                    dim=(0, 2)
                ) / valid_mask.sum(dim=(0, 2)).clamp(min=1.0)
                weights = torch.tensor(
                    self.normalized_codebook_weights,
                    device=audio_logits.device,
                )
                loss_recon = (layer_means * weights).sum()
                metrics["recon_loss"] = loss_recon.item()
            else:
                loss_recon = torch.tensor(0.0, device=H_student_proj.device)

            loss = self.distill_weight * loss_distill + self.recon_weight * loss_recon

        metrics["total_loss"] = loss.item()
        metrics["phase"] = self.phase

        return loss, metrics

    def _enter_phase_b(self):
        """Transition from Phase A to Phase B."""
        self.phase = "B"
        # Unwrap DDP to access underlying model methods
        unwrapped = self.student.module if hasattr(self.student, "module") else self.student
        unwrapped.unfreeze_shared_weights()
        self.rebuild_optimizer_for_phase_b()
        trainable = unwrapped.count_trainable_params() / 1e6
        logger.info(
            "[Step %d] Entering Phase B: joint fine-tuning, trainable=%.1fM",
            self.global_step,
            trainable,
        )

    def train(self, train_dataloader: DataLoader):
        """Main training loop with Accelerate."""
        set_seed(self.seed)

        # Initialize accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))

        accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
            project_dir=self.output_dir,
            kwargs_handlers=[ddp_kwargs, init_kwargs],
        )

        if accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(os.path.join(self.output_dir, "distill.log")),
                ],
            )

        # Move teacher to device
        self.teacher = self.teacher.to(accelerator.device)

        # Build optimizer
        self.build_optimizer_and_scheduler()

        # Prepare student + optimizer + scheduler with accelerator
        self.student, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.student, self.optimizer, self.lr_scheduler
        )

        # Resume if needed
        if self.resume_from:
            step = load_checkpoint(accelerator, self.resume_from)
            self.global_step = step
            if step >= self.phase_a_steps:
                self._enter_phase_b()
            logger.info("Resumed from step %d (Phase %s)", step, self.phase)

        # Training loop
        train_logger = TrainLogger(accelerator, self.steps, self.logging_steps)
        train_logger.start(self.global_step)
        accelerator.init_trackers("distillation")

        self.student.train()
        train_iterator = iter(train_dataloader)

        logging_start_time = time.time()
        logging_start_step = self.global_step
        tr_loss = torch.tensor(0.0, device=accelerator.device)
        logging_loss_scalar = 0.0

        while self.global_step < self.steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)

            # Move batch to device
            batch = {
                k: v.to(accelerator.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward + backward
            loss, metrics = self.training_step(batch)
            tr_loss += loss.detach()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = 0.0
                if self.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(
                        self.student.parameters(), self.max_grad_norm
                    )
                    grad_norm = grad_norm.item() if grad_norm is not None else 0.0

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Phase transition
                if self.phase == "A" and self.global_step >= self.phase_a_steps:
                    self._enter_phase_b()

                # Logging
                current_lr = self.lr_scheduler.get_last_lr()[0]
                train_logger.update(
                    step=self.global_step, loss=loss.item(), lr=current_lr
                )

                if self.global_step % self.logging_steps == 0:
                    elapsed = time.time() - logging_start_time
                    steps_per_sec = (
                        (self.global_step - logging_start_step) / elapsed
                        if elapsed > 0 else 0
                    )
                    tr_loss_scalar = accelerator.gather(tr_loss).mean().item()
                    current_loss = tr_loss_scalar - logging_loss_scalar
                    avg_loss = current_loss / self.logging_steps
                    logging_loss_scalar = tr_loss_scalar

                    logs = {
                        "distill/total_loss": avg_loss,
                        "distill/learning_rate": current_lr,
                        "distill/grad_norm": grad_norm,
                        "distill/phase": 0 if self.phase == "A" else 1,
                        "distill/steps_per_sec": steps_per_sec,
                    }
                    logs.update({f"distill/{k}": v for k, v in metrics.items()
                                 if isinstance(v, (int, float))})
                    train_logger.log_metrics(step=self.global_step, metrics=logs)

                    logging_start_time = time.time()
                    logging_start_step = self.global_step

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    save_checkpoint(
                        accelerator,
                        self.student,
                        self.tokenizer,
                        self.output_dir,
                        self.global_step,
                        keep_last_n=5,
                    )

        # Final save
        save_checkpoint(
            accelerator,
            self.student,
            self.tokenizer,
            self.output_dir,
            self.global_step,
            keep_last_n=5,
        )
        train_logger.close()
        accelerator.end_training()
        logger.info("Distillation training complete at step %d.", self.global_step)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OmniVoice distillation: shrink backbone from 0.6B to ~150M"
    )
    parser.add_argument(
        "--teacher-checkpoint", required=True,
        help="Path or HuggingFace model ID of the teacher OmniVoice model",
    )
    parser.add_argument(
        "--student-config", default="small", choices=["nano", "small", "medium"],
        help="Student model size configuration",
    )
    parser.add_argument("--output-dir", default="checkpoints/distilled")
    parser.add_argument("--data-config", default=None, help="Path to data config JSON")
    parser.add_argument("--steps", type=int, default=70_000)
    parser.add_argument("--phase-a-steps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=0.1)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=10000)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override --steps (for quick testing)")
    args = parser.parse_args()

    if args.max_steps is not None:
        args.steps = args.max_steps

    trainer = DistillationTrainer(
        teacher_path=args.teacher_checkpoint,
        student_config=args.student_config,
        output_dir=args.output_dir,
        data_config=args.data_config,
        steps=args.steps,
        phase_a_steps=args.phase_a_steps,
        learning_rate=args.learning_rate,
        cfg_scale=args.cfg_scale,
        distill_weight=args.distill_weight,
        recon_weight=args.recon_weight,
        batch_tokens=args.batch_tokens,
        num_workers=args.num_workers,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        resume_from=args.resume_from,
    )

    # Build dataloaders
    if args.data_config:
        from omnivoice.training.builder import build_dataloaders
        from omnivoice.training.config import TrainingConfig

        data_cfg = TrainingConfig(
            data_config=args.data_config,
            batch_tokens=args.batch_tokens,
            num_workers=args.num_workers,
        )
        train_loader, _ = build_dataloaders(data_cfg, trainer.tokenizer)
    else:
        logger.warning(
            "No --data-config provided. Creating a dummy dataloader for testing."
        )
        train_loader = _create_dummy_dataloader(trainer.student, batch_size=2, seq_len=64)

    trainer.train(train_loader)


def _create_dummy_dataloader(model, batch_size=2, seq_len=64):
    """Create a dummy dataloader for testing without real data."""
    C = model.config.num_audio_codebook
    V = model.config.audio_vocab_size
    mask_id = model.config.audio_mask_id

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            input_ids = torch.randint(0, V, (C, seq_len))
            audio_mask = torch.zeros(seq_len, dtype=torch.bool)
            audio_mask[seq_len // 2 :] = True
            labels = torch.randint(0, V - 1, (C, seq_len))
            labels[:, : seq_len // 2] = -100
            return {
                "input_ids": input_ids,
                "audio_mask": audio_mask,
                "labels": labels,
            }

    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    main()
