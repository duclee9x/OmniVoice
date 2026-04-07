# Notebook script: Arrow dataset → RVQ audio tokens → WebDataset shards
# pip install soundfile tqdm datasets transformers torch torchaudio
#
# Dùng trực tiếp trên Colab notebook — không cần multi-process.
# Đọc HuggingFace Arrow dataset, tokenize audio bằng HiggsAudioV2,
# ghi ra tar + jsonl shards cho OmniVoice training.

import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

# === CONFIG ===
ARROW_DIR    = "/content/drive/MyDrive/dataset/train"
OUTPUT_DIR   = "/content/drive/MyDrive/dataset/tokens"
TOKENIZER_ID = "eustlb/higgs-audio-v2-tokenizer"
SHARD_SIZE   = 2000   # samples per shard
DEV_SIZE     = 1000
SAMPLE_RATE  = 24000  # Higgs tokenizer requires 24kHz

# Create output directories
for split in ("train", "dev"):
    Path(f"{OUTPUT_DIR}/{split}/audios").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/{split}/txts").mkdir(parents=True, exist_ok=True)

# === Load audio tokenizer ===
print("Loading audio tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = AutoFeatureExtractor.from_pretrained(TOKENIZER_ID)
audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
    TOKENIZER_ID, device_map=device
)
print(f"Tokenizer loaded on {device}")

# === Load Arrow dataset ===
print("Loading arrow dataset...")
dataset = load_from_disk(ARROW_DIR)
print(f"Total: {len(dataset)} samples")


def tokenize_audio(audio_array: np.ndarray, sr: int) -> np.ndarray:
    """Convert audio numpy array → RVQ tokens [8, T] using HiggsAudioV2.

    Args:
        audio_array: Raw audio waveform (1D numpy array).
        sr: Original sample rate.

    Returns:
        Audio tokens as int16 numpy array with shape [8, T].
    """
    # Convert to torch tensor
    waveform = torch.tensor(audio_array, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]

    # Resample to 24kHz if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Normalize audio
    waveform = (waveform / (waveform.abs().max() + 1e-7)) * 0.9

    # Use feature extractor (expects 1D numpy at 24kHz)
    inputs = feature_extractor(
        raw_audio=waveform.squeeze(0).numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).to(device)

    # Encode → audio_codes shape: [1, 8, T]
    with torch.inference_mode():
        audio_codes = audio_tokenizer.encode(
            inputs["input_values"]
        ).audio_codes.squeeze(0)  # [8, T]

    assert audio_codes.dim() == 2 and audio_codes.size(0) == 8, \
        f"Expected shape [8, T], got {audio_codes.shape}"

    return audio_codes.to(torch.int16).cpu().numpy()


def process_split(samples, split_name: str):
    """Process a dataset split into WebDataset shards."""
    shard_idx = 0
    tar_file = None
    jsonl_file = None
    data_lst_lines = []
    shard_count = 0
    total_duration = 0.0

    for i, sample in enumerate(tqdm(samples, desc=split_name)):
        # Open new shard if needed
        if i % SHARD_SIZE == 0:
            if tar_file:
                tar_file.close()
                jsonl_file.close()
                data_lst_lines.append(
                    f"{tar_path} {jsonl_path} {shard_count} {total_duration:.1f}\n"
                )
                shard_count = 0
                total_duration = 0.0

            tar_path   = f"{OUTPUT_DIR}/{split_name}/audios/shard-{shard_idx:06d}.tar"
            jsonl_path = f"{OUTPUT_DIR}/{split_name}/txts/shard-{shard_idx:06d}.jsonl"
            tar_file   = tarfile.open(tar_path, "w")
            jsonl_file = open(jsonl_path, "w", encoding="utf-8")
            shard_idx += 1

        # Extract audio and metadata
        audio_array = sample["audio"]["array"]
        sr          = sample["audio"]["sampling_rate"]
        text        = sample["transcription"].strip()
        utt_id      = f"vlsp_{i:06d}"

        try:
            # Tokenize audio
            audio_tokens = tokenize_audio(audio_array, sr)  # [8, T]
            num_tokens = audio_tokens.shape[1]

            # Write tokens to tar as .npy
            token_bytes = io.BytesIO()
            np.save(token_bytes, audio_tokens)
            token_bytes.seek(0)
            tarinfo = tarfile.TarInfo(name=f"{utt_id}.npy")
            tarinfo.size = len(token_bytes.getvalue())
            tar_file.addfile(tarinfo, token_bytes)

            # Write metadata to jsonl
            duration = len(audio_array) / sr
            jsonl_file.write(json.dumps({
                "id": utt_id,
                "text": text,
                "language_id": "vi",
                "duration": round(duration, 3),
                "num_tokens": num_tokens,
            }, ensure_ascii=False) + "\n")

            shard_count += 1
            total_duration += duration

        except Exception as e:
            print(f"⚠️ Skipping {utt_id}: {e}")
            continue

    # Close last shard
    if tar_file:
        tar_file.close()
        jsonl_file.close()
        data_lst_lines.append(
            f"{tar_path} {jsonl_path} {shard_count} {total_duration:.1f}\n"
        )

    # Write data.lst manifest
    with open(f"{OUTPUT_DIR}/{split_name}/data.lst", "w") as f:
        f.writelines(data_lst_lines)
    print(f"✅ {split_name}: {len(samples)} samples → {shard_idx} shards")


# === Run ===
process_split(dataset.select(range(DEV_SIZE)), "dev")
process_split(dataset.select(range(DEV_SIZE, len(dataset))), "train")
print("🎉 All done!")