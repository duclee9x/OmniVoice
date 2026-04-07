from datasets import load_from_disk
import soundfile as sf, json, os
from pathlib import Path
from tqdm import tqdm

# Đường dẫn đến folder chứa các file .arrow trên Drive
# (folder chứa data-00000-of-00024.arrow, state.json, dataset_info.json)
ARROW_DIR     = "/content/drive/MyDrive/dataset/train"

# Output: wav ghi vào disk Colab (nhanh), JSONL ghi vào Drive (an toàn)
OUTPUT_AUDIO  = "/content/drive/MyDrive/dataset"          # disk Colab ~80GB
OUTPUT_TRAIN  = "/content/drive/MyDrive/dataset/train.jsonl"
OUTPUT_DEV    = "/content/drive/MyDrive/dataset/dev.jsonl"
DEV_SIZE      = 1000

Path(OUTPUT_AUDIO).mkdir(parents=True, exist_ok=True)

print("Loading dataset from Drive (arrow format)...")
dataset = load_from_disk(ARROW_DIR)  # đọc thẳng, không cần re-download
print(f"Total samples: {len(dataset)}")

train_f = open(OUTPUT_TRAIN, "w", encoding="utf-8")
dev_f   = open(OUTPUT_DEV,   "w", encoding="utf-8")

for i, sample in enumerate(tqdm(dataset)):
    audio_array = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]
    text        = sample["transcription"].strip()
    utt_id      = f"vlsp_{i:06d}"
    wav_path    = f"{OUTPUT_AUDIO}/{utt_id}.wav"

    sf.write(wav_path, audio_array, sample_rate)

    record = json.dumps({
        "id": utt_id,
        "audio_path": wav_path,
        "text": text,
        "language_id": "vi"
    }, ensure_ascii=False)

    (dev_f if i < DEV_SIZE else train_f).write(record + "\n")

train_f.close()
dev_f.close()
print(f"Done! Train: {len(dataset)-DEV_SIZE}, Dev: {DEV_SIZE}")
