import os
import torch
import torchaudio
import torch.multiprocessing as mp
from lhotse import SupervisionSet, RecordingSet
from f5_tts.api import F5TTS
from pathlib import Path

def load_utterances(supervisions_manifest: str, session_id: str):
    all_supervisions = list(SupervisionSet.from_jsonl(supervisions_manifest))
    return sorted(
        [s for s in all_supervisions if s.recording_id.split("-", 1)[0] == session_id],
        key=lambda s: int(s.id.split("-")[-1])
    )

def generate_for_prefix(rank, world_size, prefixes, s_man, ckpt, vocab, output_dir):
    """Each process (rank) will handle 1/world_size of prefixes."""
    # pin this process to a single GPU
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # instantiate your TTS once per process, on its GPU
    tts_api = F5TTS(
        ckpt_file=ckpt,
        vocab_file=vocab,
        use_ema=True,
    )

    # work only on the subset assigned to this rank
    for i, prefix in enumerate(prefixes):
        if i % world_size != rank:
            continue

        supervisions = load_utterances(s_man, prefix)
        print(f"[GPU{rank}] {prefix}: {len(supervisions)} utts")

        # output dirs
        chunks = Path(output_dir) / "generated_chunks"
        chunks.mkdir(parents=True, exist_ok=True)
        concat = Path(output_dir) / "concatenated_generations"
        concat.mkdir(parents=True, exist_ok=True)

        # generate utterances
        for sup in supervisions:
            utt_id = sup.id
            chan = sup.recording_id.rsplit("-", 1)[1]
            ref_map = {
                'A': f"/work/users/r/p/rphadke/JSALT/fisher_chunks_1K/wavs/{prefix}_0000_A.wav",
                'B': f"/work/users/r/p/rphadke/JSALT/fisher_chunks_1K/wavs/{prefix}_0000_B.wav",
            }
            text = sup.text.strip()
            if not text.endswith("."):
                text += "."

            # infer on this GPU
            wav_np, sr, _ = tts_api.infer(
                ref_file=ref_map[chan],
                ref_text="",
                gen_text=text
            )
            wav = torch.from_numpy(wav_np).float().to(device)
            if wav.ndim == 1:
                # mono
                wav = wav.unsqueeze(0)
            elif wav.ndim == 3:
                # maybe [batch, ch, time] → drop the batch
                wav = wav[0]
            # resample
            wav = torchaudio.transforms.Resample(sr, 16000).to(device)(wav)
            out = chunks / f"{utt_id}.wav"
            torchaudio.save(str(out), wav.cpu(), 16000)
            print(f"[GPU{rank}] saved {out}")

        # # concatenate
        # files = sorted(chunks.glob(f"{prefix}-*.wav"))
        # tensors = []
        # for f in files:
        #     w, sr = torchaudio.load(str(f))
        #     tensors.append(w)
        # combined = torch.cat(tensors, dim=1)
        # out = concat / f"{prefix}.wav"
        # torchaudio.save(str(out), combined, sr)
        # print(f"[GPU{rank}] combined -> {out}")

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

def main():
    # manifests, ckpt, vocab, output_dir as before…
    r_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/recordings.jsonl.gz"
    s_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/supervisions.jsonl.gz"
    recordings = RecordingSet.from_jsonl(r_man)
    prefixes = []
    for rec in recordings.recordings:
        cid = rec.id.rsplit("-",1)[0]
        if cid not in prefixes:
            prefixes.append(cid)
        if len(prefixes) >= 500:
            break

    ckpt_file = "/work/users/r/p/rphadke/JSALT/ckpts/pretrained_model_1250000.safetensors"
    vocab_file = "/work/users/r/p/rphadke/JSALT/vocab_files/vocab_v1.txt"
    output_dir = "/work/users/r/p/rphadke/JSALT/eval/ablation_sequential_uttwise"

    world_size = torch.cuda.device_count()
    mp.spawn(
        generate_for_prefix,
        args=(world_size, prefixes, s_man, ckpt_file, vocab_file, output_dir),
        nprocs=world_size,
        join=True
    )

    for prefix in prefixes:
        concatenate_chunks(output_dir, prefix=prefix)

if __name__ == "__main__":
    main()
