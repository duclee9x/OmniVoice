import io, json, os, queue, tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

SAMPLE_RATE   = 24000
PREFETCH_SIZE = 4


def tokenize_audio(audio_array, sr, fe, tok, device):
    waveform = torch.tensor(audio_array, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = (waveform / (waveform.abs().max() + 1e-7)) * 0.9

    inputs = fe(
        raw_audio=waveform.squeeze(0).numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        codes = tok.encode(inputs["input_values"]).audio_codes.squeeze(0)

    assert codes.dim() == 2 and codes.size(0) == 8
    return codes.to(torch.int16).cpu().numpy()


def make_prefetch_loader(samples, batch_size):
    q = queue.Queue(maxsize=PREFETCH_SIZE)

    def producer():
        batch = []
        for s in samples:
            batch.append(s)
            if len(batch) == batch_size:
                q.put(batch); batch = []
        if batch: q.put(batch)
        q.put(None)

    ThreadPoolExecutor(max_workers=1).submit(producer)
    while True:
        batch = q.get()
        if batch is None: break
        yield batch


def process_split(samples, split_name, fe, tok, device,
                  output_dir, shard_size=2000, batch_size=1, global_offset=0):
    audios_dir = Path(f"{output_dir}/{split_name}/audios")
    txts_dir   = Path(f"{output_dir}/{split_name}/txts")
    audios_dir.mkdir(parents=True, exist_ok=True)
    txts_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    done_shards = sorted([
        int(p.stem.replace("shard-", ""))
        for p in audios_dir.glob("shard-*.tar")
        if (txts_dir / p.name.replace(".tar", ".jsonl")).exists()
    ])
    completed = done_shards[:-1] if done_shards else []
    samples_done = len(completed) * shard_size

    if samples_done > 0:
        print(f"↩️  Resume [{split_name}]: {len(completed)} shards ({samples_done} done)")
        if done_shards and done_shards[-1] not in completed:
            bad = done_shards[-1]
            for p in [audios_dir / f"shard-{bad:06d}.tar",
                      txts_dir   / f"shard-{bad:06d}.jsonl"]:
                if p.exists(): p.unlink()

    data_lst = []
    for s in completed:
        tp = str(audios_dir / f"shard-{s:06d}.tar")
        jp = str(txts_dir   / f"shard-{s:06d}.jsonl")
        lines = open(jp).readlines()
        dur = sum(json.loads(l)["duration"] for l in lines)
        data_lst.append(f"{tp} {jp} {len(lines)} {dur:.1f}\n")

    shard_idx = len(completed)
    remaining = samples.select(range(samples_done, len(samples)))
    tar_file = jsonl_file = tar_path = jsonl_path = None
    shard_count = total_dur = processed = skipped = 0

    pbar = tqdm(total=len(samples), initial=samples_done, desc=f"[{split_name}]")

    for batch in make_prefetch_loader(remaining, batch_size):
        for sample in batch:
            utt_id = f"vlsp_{global_offset + samples_done + processed:06d}"

            if processed % shard_size == 0:
                if tar_file:
                    tar_file.close(); jsonl_file.close()
                    data_lst.append(f"{tar_path} {jsonl_path} {int(shard_count)} {total_dur:.1f}\n")
                    shard_count = total_dur = 0
                tar_path   = str(audios_dir / f"shard-{shard_idx:06d}.tar")
                jsonl_path = str(txts_dir   / f"shard-{shard_idx:06d}.jsonl")
                tar_file   = tarfile.open(tar_path, "w")
                jsonl_file = open(jsonl_path, "w", encoding="utf-8")
                shard_idx += 1

            try:
                tokens   = tokenize_audio(sample["audio"]["array"], sample["audio"]["sampling_rate"], fe, tok, device)
                duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]

                tb = io.BytesIO()
                np.save(tb, tokens); tb.seek(0)
                info = tarfile.TarInfo(name=f"{utt_id}.npy")
                info.size = len(tb.getvalue())
                tar_file.addfile(info, tb)

                jsonl_file.write(json.dumps({
                    "id": utt_id, "text": sample["transcription"].strip(),
                    "language_id": "vi",
                    "duration": round(duration, 3),
                    "num_tokens": tokens.shape[1],
                }, ensure_ascii=False) + "\n")

                shard_count += 1; total_dur += duration

            except Exception as e:
                print(f"\n⚠️  Skip {utt_id}: {e}")
                skipped += 1

            processed += 1; pbar.update(1)

    pbar.close()
    if tar_file:
        tar_file.close(); jsonl_file.close()
        data_lst.append(f"{tar_path} {jsonl_path} {int(shard_count)} {total_dur:.1f}\n")

    with open(f"{output_dir}/{split_name}/data.lst", "w") as f:
        f.writelines(data_lst)

    print(f"✅ [{split_name}] {len(samples)-skipped} samples → {shard_idx} shards")
    return f"{output_dir}/{split_name}/data.lst"


def run_on_gpu(gpu_id, indices, split_name, global_offset, output_dir,
               arrow_dir, tokenizer_id, shard_size, batch_size):
    """Entry point cho subprocess — load mọi thứ từ đầu trong process mới."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    print(f"[GPU {gpu_id}] Loading tokenizer...")
    fe  = AutoFeatureExtractor.from_pretrained(tokenizer_id)
    tok = HiggsAudioV2TokenizerModel.from_pretrained(tokenizer_id, device_map=device)
    print(f"[GPU {gpu_id}] Ready — {len(indices)} samples")

    dataset = load_from_disk(arrow_dir)
    subset  = dataset.select(indices)

    process_split(subset, split_name, fe, tok, device,
                  output_dir=output_dir, shard_size=shard_size,
                  batch_size=batch_size, global_offset=global_offset)