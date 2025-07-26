import os
from pathlib import Path
from datetime import datetime
import gzip
import json

from pathlib import Path
import os
import torch
import torchaudio
from lhotse import SupervisionSet
from f5_tts.api import F5TTS

def load_utterances(supervisions_manifest: str, session_id: str):
    """Load and return all SupervisionSegments for a single session ID."""
    all_supervisions = list(SupervisionSet.from_jsonl(supervisions_manifest))
    return sorted(
        [s for s in all_supervisions if s.recording_id.split("-", 1)[0] == session_id],
        key=lambda s: int(s.id.split("-")[-1])
    )

def generate_each_utterance(supervisions, ckpt_file, vocab_file, output_dir):
    chunks_path = os.path.join(output_dir, "generated_chunks")
    os.makedirs(chunks_path, exist_ok=True)

    tts_api = F5TTS(ckpt_file=ckpt_file, vocab_file=vocab_file, use_ema=True)

    for sup in supervisions:
        utt_id = sup.id
        prefix = utt_id.split("-")[0]
        chan = sup.recording_id.rsplit("-", 1)[1]

        ref_map = {
            'A': f"/work/users/r/p/rphadke/JSALT/fisher_chunks/wavs/{prefix}_0000_A.wav",
            'B': f"/work/users/r/p/rphadke/JSALT/fisher_chunks/wavs/{prefix}_0000_B.wav",
        }

        gen_text = sup.text.strip()
        if not gen_text.endswith("."):
            gen_text += "."

        wav, sr, _ = tts_api.infer(
            ref_file=ref_map[chan],
            ref_text="",
            gen_text=gen_text
        )

        wav_out_path = Path(chunks_path) / f"{utt_id}.wav"
        wav = torch.from_numpy(wav).float()
        target_sr = 16000
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        sr = target_sr

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(str(wav_out_path), wav, sr)
        print(f"Saved utterance: {wav_out_path}")


def concatenate_chunks(output_dir, prefix="fe_03_00001"):
    combined_out_path = os.path.join(output_dir, "concatenated_generations")
    os.makedirs(combined_out_path, exist_ok=True)

    chunks_path = os.path.join(output_dir, "generated_chunks")
    wav_files = sorted(Path(chunks_path).glob(f"{prefix}-*.wav"))
    tensors = []
    for path in wav_files:
        print(f"Processing {path}")
        wav, sr = torchaudio.load(str(path))
        tensors.append(wav)
    
    combined = torch.cat(tensors, dim=1)
    combined_out_path = Path(combined_out_path)
    combined_out_path = combined_out_path / f"{prefix}.wav"
    torchaudio.save(str(combined_out_path), combined, sr)
    print(f"Saved combined audio to {combined_out_path}")


r_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/recordings.jsonl.gz"
s_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/supervisions.jsonl.gz"
recording_ids = ["fe_03_00001", "fe_03_00002", "fe_03_00003", "fe_03_00004"]

ckpt_file = "/work/users/r/p/rphadke/JSALT/ckpts/pretrained_model_1250000.safetensors"
vocab_file = "/work/users/r/p/rphadke/JSALT/vocab_files/vocab_v1.txt"
output_dir = "/work/users/r/p/rphadke/JSALT/eval/ablation_sequential_uttwise"



for prefix in recording_ids:
    utterances = load_utterances(s_man, prefix)
    print(f"Loaded {len(utterances)} utterances for session {prefix}")
    generate_each_utterance(utterances, ckpt_file, vocab_file, output_dir)
    concatenate_chunks(output_dir, prefix=prefix)
