Môi trường
Platform: Kaggle, 2x GPU ~14.56GB mỗi card

Framework: PyTorch DDP, torchrun --nproc_per_node=2

Teacher: drbaph/OmniVoice-bf16 (Qwen3-based, 0.6B)

Student: OmniVoiceSmall nano config (6 layers, hidden=512)

Script: /kaggle/working/OmniVoice/omnivoice/train_distill.py

Lỗi 1 — int16 overflow (đã fix)
Triệu chứng: vectorized_gather_kernel: Assertion ind >= 0 && ind < ind_dim_size ngay khi training bắt đầu.

Nguyên nhân: dataset.py dòng 234 load .npy bằng torch.from_numpy(sample["npy"]) giữ nguyên dtype int16. Khi model cộng offset (text_vocab_size + codebook_idx * audio_vocab_size) vào token IDs để map vào LLM embedding table (Qwen3 vocab ~151K), phép cộng bị int16 overflow (max=32767) → sinh ra giá trị âm → assertion fail.

Fix: sed patch thêm .long() → torch.from_numpy(sample["npy"]).long() — đã xác nhận thành công.

Data: .npy shape (8, T), dtype int16, values trong [0, 1023] — hoàn toàn hợp lệ.

Lỗi 2 — CUDA OOM + CUBLAS_STATUS_EXECUTION_FAILED (chưa fix)
Triệu chứng: rank0 báo CUDA out of memory: Tried to allocate 1024MB, rank1 báo CUBLAS_STATUS_EXECUTION_FAILED do CUDA state bị corrupt từ OOM của rank0. --batch-tokens 1024 vẫn bị OOM.

Nguyên nhân nghi ngờ: Teacher model forward 2 lần/step (conditional + unconditional CFG) trong get_teacher_hidden_states_cfg() → quá tải VRAM.

Chưa giải quyết dứt điểm vì lỗi 3 xuất hiện đồng thời.

Lỗi 3 — index out of bounds từ _prepare_embed_inputs (nghi ngờ root cause chính)
Triệu chứng: vectorized_gather_kernel assertion fail vẫn tiếp tục dù đã fix int16. Lỗi đến từ latent_cfg.py:124 và latent_cfg.py:144 (cond và uncond forward), gọi teacher._prepare_embed_inputs(input_ids, audio_mask).

Root cause phân tích: Trong omnivoice-2.py (file model chính), hàm _prepare_embed_inputs:

python
shiftedids = inputids + audiomask.unsqueeze(1) * self.codebooklayeroffsets.view(1,-1,1)
audioembeds = self.audioembeddings(shiftedids).sum(dim=1)  # ← BUG
self.audioembeddings có size (num_codebook * audio_vocab_size, hidden) = (8*1025=8200, 1024). Nhưng hàm gọi audioembeddings(shiftedids) trên toàn bộ sequence bao gồm cả text positions. Text token IDs (~151643 cho Qwen3) >> 8200 → index out of bounds. torch.where(audiomask...) chỉ được dùng sau lookup, quá muộn.

Fix đề xuất (chưa xác nhận):

python
safe_ids = torch.where(
    audiomask.unsqueeze(1).expand_as(inputids),
    shiftedids,
    torch.zeros_like(shiftedids)  # dummy index hợp lệ
)
audioembeds = self.audioembeddings(safe_ids).sum(dim=1)
Config model
text
audio_vocab_size: 1025
audio_mask_id: 1024
num_audio_codebook: 8
audio_codebook_weights: [8, 8, 6, 6, 4, 4, 2, 2]
eos_token_id: 151645
pad_token_id: 151643
Files quan trọng
omnivoice/data/dataset.py — load .npy, đã patch .long()

omnivoice/inference/latent_cfg.py — get_teacher_hidden_states_cfg(), điểm crash

omnivoice/models/omnivoice.py (tên thực tế cần xác nhận) — _prepare_embed_inputs, nghi ngờ bug

omnivoice/models/omnivoice_small.py — student model, override _prepare_embed_inputs nhưng cùng bug