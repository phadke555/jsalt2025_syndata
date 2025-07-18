import os
from pathlib import Path
from datetime import datetime
import gzip
import json

import torch
import torchaudio
from lhotse import RecordingSet, SupervisionSet, CutSet
from f5_tts.api import F5TTS

def build_groups_from_supervisions(supervisions_manifest: str, session_id: str):
    """
    Load supervisions as Lhotse SupervisionSegment objects, sort by their numeric ID suffix,
    and group into turn-taking blocks whenever the channel (A/B) flips.
    Returns: List of (channel: str, List[SupervisionSegment])
    """
    # Load all SupervisionSegment objects
    supervisions = list(SupervisionSet.from_jsonl(supervisions_manifest))

    # Filter to this session (strip off channel suffix)
    sel = [
        s
        for s in supervisions
        if s.recording_id.split("-", 1)[0] in session_id
    ]
    if not sel:
        raise ValueError(f"No supervisions found for session {session_id}")

    # Sort by the integer suffix of each supervision's ID (e.g. "-006" → 6)
    sel_sorted = sorted(
        sel,
        key=lambda s: int(s.id.rsplit("-", 1)[1])
    )

    # Group into blocks on channel change (recording_id ends with "-A" or "-B")
    groups = []
    first_channel = sel_sorted[0].recording_id.rsplit("-", 1)[1]
    current_channel = first_channel
    current_block = [sel_sorted[0]]

    for sup in sel_sorted[1:]:
        channel = sup.recording_id.rsplit("-", 1)[1]
        if channel == current_channel:
            current_block.append(sup)
        else:
            groups.append((current_channel, current_block))
            current_channel = channel
            current_block = [sup]

    # Append the final block
    groups.append((current_channel, current_block))
    return groups

def seq_infer(groups, ckpt_file, vocab_file, output_dir, ref_root=None):
    # directories
    combined_out_path = os.path.join(output_dir, "concatenated_generations")
    os.makedirs(combined_out_path, exist_ok=True)

    chunks_path = os.path.join(output_dir, "generated_chunks")
    os.makedirs(chunks_path, exist_ok=True)

    # api
    tts_api = F5TTS(
            ckpt_file=ckpt_file, vocab_file=vocab_file, use_ema=True
        )
    

    # generate?
    for idx, (chan, block) in enumerate(groups):
        session_id = [s.id for s in block][0]
        prefix = session_id.rsplit("-", 1)[0]
        ref_map = {
            'A': f"/work/users/r/p/rphadke/JSALT/fisher_chunks/wavs/{prefix}_0000_A.wav",
            'B': f"/work/users/r/p/rphadke/JSALT/fisher_chunks/wavs/{prefix}_0000_B.wav",
        }
        gen_text = ' '.join(s.text.strip() for s in block)
        gen_text += ". "



        wav, sr, _ = tts_api.infer(
                    ref_file=ref_map[chan],
                    ref_text="",
                    gen_text=gen_text
                )
        fname = f"{session_id}.wav"
        wav_out_path = Path(chunks_path) / fname
        wav = torch.from_numpy(wav)
        if wav.ndim==1: wav = wav.unsqueeze(0)
        torchaudio.save(str(wav_out_path), wav, sr)
        print(f"Saved Chunk: {wav_out_path}")


def concatenate_chunks(output_dir, prefix="fe_03_00001"):
    combined_out_path = os.path.join(output_dir, "concatenated_generations")
    os.makedirs(combined_out_path, exist_ok=True)

    chunks_path = os.path.join(output_dir, "generated_chunks")
    wav_files = sorted(Path(chunks_path).glob(f"{prefix}-*.wav"))
    tensors = []
    for path in wav_files:
        print(f"Processing {path}")
        wav, sr = torchaudio.load(path)
        tensors.append(wav)
    
    combined = torch.cat(tensors, dim=1)
    combined_out_path = Path(combined_out_path)
    combined_out_path = combined_out_path / f"{prefix}.wav"
    torchaudio.save(combined_out_path, combined, sr)
    print(f"Saved combined audio to {combined_out_path}")




r_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/recordings.jsonl.gz"
s_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/supervisions.jsonl.gz"
recording_ids = ["fe_03_00001", "fe_03_00002", "fe_03_00003", "fe_03_00004"]

ckpt_file = "/work/users/r/p/rphadke/JSALT/ckpts/pretrained_model_1250000.safetensors"
vocab_file = "/work/users/r/p/rphadke/JSALT/vocab_files/vocab_v1.txt"
output_dir = "/work/users/r/p/rphadke/JSALT/eval/ablation_sequential"



for prefix in recording_ids:
    groups = build_groups_from_supervisions(s_man, prefix)
    print(len(groups))
    seq_infer(groups, ckpt_file, vocab_file, output_dir, ref_root=None)
    concatenate_chunks(output_dir, prefix=prefix)


from f5_tac.eval.diarize import process_folder
from tqdm import tqdm
from pyannote.audio import Pipeline

# Load pyannote pipeline
print("Loading Pyannote diarization pipeline...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Set up output folders
generated_output_dir = os.path.join(output_dir, "diarization_results", "ablation-seq")
process_folder(pipeline, os.path.join(output_dir,"concatenated_generations"), generated_output_dir)


from f5_tac.eval.compute_speech_metrics import process_directory
generated_json_dir = os.path.join(output_dir, "diarization_results", "ablation-seq")
dur_gen, occ_gen, mean_gen = process_directory(generated_json_dir, "ablation-seq")
# Save combined results
combined_output_dir = os.path.join(output_dir, "diarization_results", "metrics")
os.makedirs(combined_output_dir, exist_ok=True)

dur_gen.to_csv(os.path.join(combined_output_dir, "total_duration.csv"), index=False)
occ_gen.to_csv(os.path.join(combined_output_dir, "occurrences.csv"), index=False)
mean_gen.to_csv(os.path.join(combined_output_dir, "mean_duration.csv"), index=False)
print(f"Metrics saved to {combined_output_dir}")