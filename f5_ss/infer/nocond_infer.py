from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    tempfile_kwargs,
)
from f5_tts.model import DiT, UNetT
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import transcribe

import torch
import torchaudio

import json
import argparse
from pathlib import Path
import random
import os
from datetime import datetime

parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Inference"
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
args = parser.parse_args()

data_root = "/work/users/r/p/rphadke/JSALT/fisher_chunks_singlespeaker"
ckpt_file = "/work/users/r/p/rphadke/JSALT/ckpts/pretrained_model_1250000.safetensors"
vocab_file = "/work/users/r/p/rphadke/JSALT/vocab_files/vocab_v1.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not data_root:
    raise ValueError("When using --random_ref you must also pass --data_root")
meta_csv = os.path.join(data_root, "metadata.csv")
if not os.path.exists(meta_csv):
    raise FileNotFoundError(f"metadata.csv not found at {meta_csv}")
lines = [l.strip() for l in open(meta_csv, "r", encoding="utf-8") if l.strip()]

choice = random.choice(lines)
parts = choice.split(",")
fname = parts[0]
text = ",".join(parts[1:-1])
ref_audio = os.path.join(os.path.dirname(meta_csv), "wavs", fname)
ref_text  = text
print(f"→ Random reference: {fname} → “{text}”")

gen_text = "Where in baltimore are you from? I have a friend somewhere there. "



tts_api = F5TTS(
            ckpt_file=ckpt_file, vocab_file=vocab_file, use_ema=True
        )

wav, sr, _ = tts_api.infer(
            ref_file=ref_audio,
            ref_text=ref_text.strip(),
            gen_text=gen_text.strip()
        )

output_file = f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
os.makedirs(args.output_dir, exist_ok=True)
output_path = Path(args.output_dir) / output_file
print(wav.shape)
# if wav_np is 1D (mono):
wav = torch.from_numpy(wav).unsqueeze(0)      # [1, time]
print(wav.shape)
torchaudio.save(output_path, wav, sr)