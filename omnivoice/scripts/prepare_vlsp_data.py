# prepare_vlsp_data.py
# Chạy: python prepare_vlsp_data.py
# pip install datasets soundfile tqdm

from datasets import load_dataset
import soundfile as sf
import json, os
from tqdm import tqdm
from pathlib import Path

OUTPUT_AUDIO_DIR = "data/audio/vlsp2020"
OUTPUT_TRAIN_JSONL = "data/train.jsonl"
OUTPUT_DEV_JSONL   = "data/dev.jsonl"
DEV_RATIO = 0.02  # 2% làm dev set (~1100 mẫu)

Path(OUTPUT_AUDIO_DIR).mkdir(parents=True, exist_ok=True)

print("Loading dataset (streaming)...")
dataset = load_dataset(
    "doof-ferb/vlsp2020_vinai_100h",
    split="train",
    streaming=False  # tải toàn bộ về disk (~11.6GB)
)

total = len(dataset)
dev_count = int(total * DEV_RATIO)
train_records, dev_records = [], []

print(f"Processing {total} samples...")
for i, sample in enumerate(tqdm(dataset)):
    audio_array = sample["audio"]["array"]
    sample_rate  = sample["audio"]["sampling_rate"]
    text         = sample["transcription"].strip()
    utt_id       = f"vlsp_{i:06d}"
    wav_path     = os.path.abspath(f"{OUTPUT_AUDIO_DIR}/{utt_id}.wav")

    # Lưu audio ra file .wav
    sf.write(wav_path, audio_array, sample_rate)

    record = {
        "id": utt_id,
        "audio_path": wav_path,
        "text": text,
        "language_id": "vi"
    }

    if i < dev_count:
        dev_records.append(record)
    else:
        train_records.append(record)

# Ghi JSONL
with open(OUTPUT_TRAIN_JSONL, "w", encoding="utf-8") as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(OUTPUT_DEV_JSONL, "w", encoding="utf-8") as f:
    for r in dev_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nDone!")
print(f"Train: {len(train_records)} samples → {OUTPUT_TRAIN_JSONL}")
print(f"Dev:   {len(dev_records)} samples  → {OUTPUT_DEV_JSONL}")