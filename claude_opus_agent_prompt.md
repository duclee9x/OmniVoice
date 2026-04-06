# Agent Prompt: OmniVoice Distillation — Shrink to 100M–200M Parameters

## Mục tiêu

Bạn là một ML engineer expert. Nhiệm vụ của bạn là implement **Latent Distillation** để shrink model OmniVoice từ 0.8B xuống còn ~150M–200M tham số, giữ chất lượng TTS multilingual tốt nhất có thể và đạt inference speed cao nhất có thể.

Tôi sẽ cung cấp:
- **Source code OmniVoice** (thư mục `omnivoice/`)
- **Source code Pocket TTS** (thư mục `pocket_tts/` hoặc tương đương)

---

## Bước 0 — Scan codebase trước khi làm bất cứ điều gì

Trước khi viết bất kỳ dòng code nào, hãy đọc và hiểu toàn bộ cấu trúc của cả hai repo. Cụ thể:

### Với OmniVoice, bạn CẦN xác định chính xác:

1. **Backbone LLM**: File nào, class nào? Số layers, hidden_size, num_heads, ffn_dim cụ thể là bao nhiêu?
2. **Flow Matching head**: File nào, class nào? Input shape là gì? Conditioning vector `Z` có shape gì (batch, seq, dim)?
3. **Điểm kết nối**: Ở đâu trong forward pass, backbone trả ra vector `Z` rồi đưa vào Flow head? Đây là điểm cần "cắm" distillation.
4. **CFG (Classifier-Free Guidance)**: Được implement ở đâu? Output-space hay latent-space?
5. **Training loop**: File `train.py` hay tương đương — loss function là gì, dataloader format ra sao?
6. **Speaker conditioning**: Speaker embedding được tạo ra từ đâu, inject vào backbone ở layer nào?
7. **Tokenizer + codec**: Dùng RVQ codebook nào? Bao nhiêu codebooks? Vocab size?

### Với Pocket TTS, bạn CẦN xác định chính xác:

1. **Latent-space CFG**: Hàm/đoạn code nào thực hiện CFG trên Z thay vì trên output? Extract nguyên logic này.
2. **Distillation loss**: Nếu có sẵn distillation code — copy nguyên cách compute `L2(Z_student, Z_teacher_cfg)`.
3. **Model split pattern**: Pocket TTS split backbone thành `main` + `flow` như thế nào? Dùng pattern này cho ONNX export sau này.
4. **Config system**: Pocket TTS define model size (layers, hidden) ở đâu? Tham khảo để tạo `SmallConfig` cho OmniVoice.

### Sau khi scan xong, báo cáo:

```
=== SCAN REPORT ===
OmniVoice backbone: [class name, file, layers=X, hidden=X, heads=X, ffn=X, total_params=~XM]
OmniVoice flow head: [class name, file, input_dim=X, total_params=~XM]
OmniVoice Z vector shape: [batch, seq_len, dim=X]
OmniVoice CFG location: [file:line, type: output/latent-space]
OmniVoice training loss: [loss function names]
---
Pocket TTS latent CFG: [file:line, YES/NO có sẵn]
Pocket TTS distill loss: [file:line, YES/NO có sẵn]
Pocket TTS model split: [pattern mô tả]
=== END REPORT ===
```

Chỉ sau khi có report này, mới bắt đầu implement.

---

## Bước 1 — Tạo SmallConfig và OmniVoice-Small backbone

### Target architecture (~150M params):

Dựa trên hidden_size thực tế từ scan, chọn một trong các config sau (tính toán số params chính xác):

| Config | Layers | Hidden | Heads | FFN | ~Params |
|--------|--------|--------|-------|-----|---------|
| nano   | 6      | 512    | 8     | 1364| ~80M    |
| small  | 12     | 640    | 10    | 1706| ~150M   |
| medium | 16     | 768    | 12    | 2048| ~250M   |

Tạo file `omnivoice/models/omnivoice_small.py`:

```python
# omnivoice/models/omnivoice_small.py

SMALL_CONFIGS = {
    "nano": {
        "num_hidden_layers": 6,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "intermediate_size": 1364,
    },
    "small": {
        "num_hidden_layers": 12,
        "hidden_size": 640,
        "num_attention_heads": 10,
        "intermediate_size": 1706,
    },
    "medium": {
        "num_hidden_layers": 16,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
    },
}

class OmniVoiceSmall(nn.Module):
    """
    Shrunk backbone for OmniVoice distillation.
    Kiến trúc giữ nguyên như OmniVoice gốc,
    chỉ thay đổi số layer và hidden dim.
    """
    def __init__(self, config_name="small", teacher_hidden_size=None):
        super().__init__()
        cfg = SMALL_CONFIGS[config_name]
        
        # --- COPY kiến trúc backbone từ OmniVoice gốc ---
        # (Điền vào đây sau khi đã scan file omnivoice.py)
        self.backbone = ...  # same class, different config
        
        # Projection để match teacher Z dimension
        # Chỉ cần nếu student hidden != teacher hidden
        if teacher_hidden_size and teacher_hidden_size != cfg["hidden_size"]:
            self.z_proj = nn.Linear(cfg["hidden_size"], teacher_hidden_size)
        else:
            self.z_proj = nn.Identity()
    
    def forward(self, *args, **kwargs):
        Z = self.backbone(*args, **kwargs)
        return self.z_proj(Z)
```

**Quan trọng**: Đảm bảo `OmniVoiceSmall` có cùng interface (input/output signature) với backbone gốc, đặc biệt là phần speaker conditioning và text token conditioning.

---

## Bước 2 — Implement Latent-space CFG (mượn từ Pocket TTS)

Kiểm tra xem OmniVoice hiện tại dùng output-space hay latent-space CFG. Nếu đang là **output-space**, chuyển sang **latent-space** theo pattern của Pocket TTS:

Tạo file `omnivoice/inference/latent_cfg.py`:

```python
# omnivoice/inference/latent_cfg.py

def latent_cfg_forward(backbone, text_tokens, speaker_embedding, cfg_scale=1.5):
    """
    Thay thế output-space CFG bằng latent-space CFG.
    
    Output-space CFG (CŨ, chậm):
        out = cfg * cond_audio + (1-cfg) * uncond_audio  # chạy flow head 2 lần
    
    Latent-space CFG (MỚI, nhanh hơn ~30%):
        Z_cfg = Z_uncond + cfg * (Z_cond - Z_uncond)   # blend TRÊN Z
        audio = flow_head(Z_cfg)                        # chạy flow head 1 lần
    """
    # Conditional forward
    Z_cond = backbone(
        text_tokens=text_tokens,
        speaker=speaker_embedding,
    )
    
    # Unconditional forward (null speaker)
    null_speaker = torch.zeros_like(speaker_embedding)
    Z_uncond = backbone(
        text_tokens=text_tokens,
        speaker=null_speaker,
    )
    
    # CFG blend trong latent space
    Z_cfg = Z_uncond + cfg_scale * (Z_cond - Z_uncond)
    
    return Z_cfg


def single_step_inference(backbone, flow_head, text_tokens, speaker_embedding,
                          cfg_scale=1.5, num_steps=1):
    """
    Kết hợp latent CFG + few-step flow matching.
    Mục tiêu: num_steps=1 hoặc 4 cho inference nhanh nhất.
    """
    Z_cfg = latent_cfg_forward(backbone, text_tokens, speaker_embedding, cfg_scale)
    
    # Chạy flow head với ít steps
    audio_tokens = flow_head(Z_cfg, num_steps=num_steps)
    return audio_tokens
```

**Chú ý**: Adapt đúng với actual API của backbone và flow_head sau khi đã scan code. Đây chỉ là template — tên arguments có thể khác nhau.

---

## Bước 3 — Viết Distillation Training Loop

Tạo file `omnivoice/train_distill.py`:

```python
# omnivoice/train_distill.py
"""
Distillation trainer: shrink OmniVoice backbone từ 0.8B → ~150M.

Strategy:
  Phase A (steps 0 → PHASE_A_STEPS):
    - Teacher backbone frozen, flow head frozen
    - Chỉ train student backbone
    - Loss = L2(Z_student_projected, Z_teacher_cfg)
    
  Phase B (steps PHASE_A_STEPS → PHASE_A_STEPS + PHASE_B_STEPS):
    - Teacher frozen
    - Student backbone + flow head đều train (flow head copy từ teacher, unfreeze)
    - Loss = alpha * L2(Z_student, Z_teacher) + beta * reconstruction_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PHASE_A_STEPS = 50_000
PHASE_B_STEPS = 20_000
CFG_ALPHA = 1.5       # CFG scale cho teacher
DISTILL_WEIGHT = 1.0  # weight của distillation loss
RECON_WEIGHT = 0.1    # weight của reconstruction loss

class DistillationTrainer:
    def __init__(self, teacher_path, student_config="small"):
        # Load teacher (frozen)
        self.teacher = load_omnivoice(teacher_path)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Init student backbone
        teacher_hidden = self.teacher.backbone.config.hidden_size
        self.student = OmniVoiceSmall(
            config_name=student_config,
            teacher_hidden_size=teacher_hidden,
        )
        
        # Copy flow head từ teacher sang student, freeze ban đầu
        self.flow_head = copy.deepcopy(self.teacher.flow_head)
        for p in self.flow_head.parameters():
            p.requires_grad = False
        
        self.phase = "A"
        self.global_step = 0
    
    def get_teacher_z_cfg(self, batch):
        """Lấy conditioning vector Z của teacher với CFG."""
        with torch.no_grad():
            Z_cond = self.teacher.backbone(
                text_tokens=batch["text_tokens"],
                speaker=batch["speaker_embedding"],
            )
            Z_uncond = self.teacher.backbone(
                text_tokens=batch["text_tokens"],
                speaker=torch.zeros_like(batch["speaker_embedding"]),
            )
            Z_cfg = Z_uncond + CFG_ALPHA * (Z_cond - Z_uncond)
        return Z_cfg
    
    def training_step(self, batch):
        # Get teacher Z (ground truth cho distillation)
        Z_teacher = self.get_teacher_z_cfg(batch)  # [B, T, D_teacher]
        
        # Student forward
        Z_student = self.student(
            text_tokens=batch["text_tokens"],
            speaker=batch["speaker_embedding"],
        )  # [B, T, D_teacher] sau projection
        
        # Distillation loss (L2 trên conditioning vectors)
        loss_distill = F.mse_loss(Z_student, Z_teacher)
        
        if self.phase == "A":
            loss = DISTILL_WEIGHT * loss_distill
        
        elif self.phase == "B":
            # Reconstruction loss (dùng flow head để decode và compare)
            audio_pred = self.flow_head(Z_student)
            loss_recon = compute_flow_matching_loss(audio_pred, batch["audio_tokens"])
            loss = DISTILL_WEIGHT * loss_distill + RECON_WEIGHT * loss_recon
        
        # Phase transition
        self.global_step += 1
        if self.global_step == PHASE_A_STEPS:
            self._enter_phase_b()
        
        return loss
    
    def _enter_phase_b(self):
        """Unfreeze flow head cho joint fine-tuning."""
        self.phase = "B"
        for p in self.flow_head.parameters():
            p.requires_grad = True
        print(f"[Step {self.global_step}] Entering Phase B: joint fine-tuning")


# === ENTRY POINT ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-config", default="small",
                        choices=["nano", "small", "medium"])
    parser.add_argument("--output-dir", default="checkpoints/distilled")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    
    trainer = DistillationTrainer(
        teacher_path=args.teacher_checkpoint,
        student_config=args.student_config,
    )
    # TODO: thay bằng actual training loop của OmniVoice gốc
    # (copy từ omnivoice/train.py, replace model với trainer.student)
```

**Quan trọng**: Hàm `compute_flow_matching_loss` phải được copy/import từ training loop gốc của OmniVoice để đảm bảo consistency.

---

## Bước 4 — ONNX Export cho inference nhanh

Sau khi distillation xong, export sang ONNX để chạy không cần PyTorch:

Tạo file `omnivoice/export_onnx.py`:

```python
# omnivoice/export_onnx.py
"""
Export OmniVoice-Small thành 2 ONNX files tách biệt:
  - backbone.onnx  (phần nặng nhất, chạy 1 lần)
  - flow_head.onnx (có thể chạy nhiều steps)

Tham khảo pattern từ:
  https://github.com/KevinAHM/pocket-tts-onnx-export
"""

def export_backbone(model, output_path, example_inputs):
    """Export backbone với dynamic batch/seq axes."""
    torch.onnx.export(
        model.student_backbone,
        example_inputs,
        f"{output_path}/backbone.onnx",
        opset_version=17,
        input_names=["text_tokens", "speaker_embedding"],
        output_names=["Z"],
        dynamic_axes={
            "text_tokens": {0: "batch", 1: "seq_len"},
            "speaker_embedding": {0: "batch"},
            "Z": {0: "batch", 1: "seq_len"},
        },
    )
    print(f"Saved backbone.onnx")

def export_flow_head(model, output_path, example_z):
    """Export flow head."""
    torch.onnx.export(
        model.flow_head,
        example_z,
        f"{output_path}/flow_head.onnx",
        opset_version=17,
        input_names=["Z"],
        output_names=["audio_tokens"],
        dynamic_axes={
            "Z": {0: "batch", 1: "seq_len"},
            "audio_tokens": {0: "batch", 1: "seq_len"},
        },
    )
    print(f"Saved flow_head.onnx")
```

---

## Checklist cho agent

Sau khi scan xong, implement theo thứ tự và tick từng bước:

- [ ] **Scan report** hoàn thành — biết chính xác hidden_size, num_layers, Z shape, CFG location
- [ ] **Tính toán params** — xác nhận target config nào cho ~150M total (backbone + head)
- [ ] `omnivoice/models/omnivoice_small.py` — backbone nhỏ với đúng architecture của OmniVoice
- [ ] `omnivoice/inference/latent_cfg.py` — latent-space CFG function
- [ ] `omnivoice/train_distill.py` — distillation trainer với Phase A và Phase B
- [ ] `omnivoice/export_onnx.py` — ONNX export chia 2 phần
- [ ] **Test nhanh**: Load teacher + student, chạy 1 batch distillation step, assert loss giảm

---

## Những điều agent KHÔNG được làm

1. **Không giả định số cụ thể** (hidden_size, num_layers...) nếu chưa đọc code — hãy scan trước
2. **Không thay đổi interface** của tokenizer, codec, hay dataloader — chỉ thay backbone
3. **Không xóa/sửa code gốc** — tạo file mới, import từ gốc khi cần
4. **Không skip Phase A** — Phase A (backbone-only distill) là critical để warm up student trước khi joint fine-tune
5. **Không dùng KL divergence** thay cho L2 trên Z — L2 trực tiếp trên conditioning vector hoạt động tốt hơn trong continuous latent space

---

## Câu hỏi cần trả lời sau khi scan

Sau khi đọc xong cả hai codebase, hãy trả lời những câu hỏi này TRƯỚC khi viết code:

1. OmniVoice backbone có phải là Qwen3 transformer không, hay là custom transformer? Config cụ thể?
2. Z vector được trả ra ở vị trí nào trong forward pass — trước hay sau LayerNorm cuối?
3. Flow Matching head nhận Z dưới dạng gì — cả sequence hay chỉ một vector?
4. Pocket TTS có sẵn distillation code không? Nếu có, copy nguyên đoạn đó.
5. Speaker embedding được inject vào backbone bằng cross-attention, concatenation, hay additive conditioning?

Trả lời xong 5 câu này rồi mới bắt đầu code.

