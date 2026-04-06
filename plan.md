Bạn đã scan xong và có implementation plan chính xác. Bây giờ hãy implement 
đầy đủ 4 file theo đúng thứ tự, dựa trên code thực tế đã đọc:

=== CONSTRAINTS (KHÔNG ĐƯỢC VI PHẠM) ===
1. OmniVoice KHÔNG có flow matching — chỉ có audio_heads = nn.Linear(1024, 8*1025)
2. Speaker conditioning là IN-CONTEXT (prepend ref tokens), không phải explicit embedding
3. CFG hiện tại là output-space (log_softmax blending) — cần chuyển sang hidden-state CFG
4. hidden_states shape = [batch, seq_len, 1024] tại omnivoice.py:401
5. Student cần một hidden_proj: nn.Linear(student_hidden, 1024) để match teacher dim
6. audio_heads và audio_embeddings được SHARE từ teacher (copy + freeze trong Phase A)

=== FILE 1: omnivoice/models/omnivoice_small.py ===
Yêu cầu:
- Copy TOÀN BỘ class OmniVoice từ omnivoice.py, chỉ thay phần khởi tạo Qwen3 config
- SMALL_CONFIGS phải dùng đúng Qwen3 GQA architecture (num_key_value_heads khác num_attention_heads)
  - nano:   layers=6,  hidden=512,  heads=8,  kv_heads=4,  ffn=1364
  - small:  layers=12, hidden=640,  heads=10, kv_heads=5,  ffn=1706
  - medium: layers=16, hidden=768,  heads=12, kv_heads=6,  ffn=2048
- Thêm self.hidden_proj = nn.Linear(student_hidden, 1024) SAU backbone
- Forward pass phải return BOTH hidden_states (để distill) VÀ logits (để reconstruct)
- Đảm bảo _generate_iterative() hoạt động với student size (vì kế thừa từ OmniVoice)

=== FILE 2: omnivoice/inference/latent_cfg.py ===
Yêu cầu:
- Implement hidden_state_cfg_forward(model, inputs, uncond_inputs, cfg_scale=1.5)
  * Chạy backbone 2 lần để lấy H_cond và H_uncond
  * Return H_cfg = H_uncond + cfg_scale * (H_cond - H_uncond)
  * "uncond_inputs" = inputs nhưng thay ref_audio_tokens bằng zeros/empty
- Implement adapted_inference(model, text_tokens, ref_audio_tokens, cfg_scale, num_iter=8)
  * Dùng hidden-state CFG thay cho output-space CFG hiện tại
  * Giữ nguyên iterative masked unmasking logic từ _generate_iterative()
  * Mục tiêu: giảm num_iter từ default xuống 4-8 steps mà vẫn giữ quality
- Xác định chính xác format của "uncond" condition từ code _generate_iterative() tại line 1117

=== FILE 3: omnivoice/train_distill.py ===
Yêu cầu:
- Import và kế thừa OmniTrainer từ omnivoice/training/trainer.py
- DistillationTrainer(teacher_path, student_config="small"):
  * Load teacher = full OmniVoice checkpoint, freeze hoàn toàn
  * Init student = OmniVoiceSmall(student_config)
  * COPY audio_heads và audio_embeddings từ teacher sang student (share weights)
  * Freeze audio_heads trong Phase A, unfreeze trong Phase B
- training_step(batch):
  Phase A: loss = MSE(student.hidden_proj(H_student), H_teacher_cfg.detach())
  Phase B: loss = alpha*MSE(H_student_proj, H_teacher_cfg) + beta*CE(logits_student, tokens_gt)
  * CE loss dùng đúng weighted cross_entropy với weights=[8,8,6,6,4,4,2,2] như teacher
- Thêm --student-config argument vào existing training CLI

=== FILE 4: tests/test_distillation.py ===
Yêu cầu:
- test_student_param_count(): verify nano<100M, small<200M, medium<300M
- test_forward_shape(): student và teacher output cùng shape logits
- test_hidden_proj(): H_student sau projection = shape [B, S, 1024]
- test_one_distill_step(): loss finite, gradient flows, loss > 0
- test_cfg_forward(): hidden_state_cfg_forward trả ra tensor shape đúng
Sử dụng pytest + torch với dummy inputs (không cần real checkpoint)

=== SAU KHI VIẾT XONG 4 FILES ===
Chạy test và báo cáo kết quả:
  python -m pytest tests/test_distillation.py -v

Nếu test fail, debug và fix ngay trong cùng lượt.