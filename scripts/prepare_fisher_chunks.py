#!/usr/bin/env python3
"""
segment_fisher_with_cuts.py

Use Lhotse CutSet to:
  - Resample audio from 8 kHz → 24 kHz (lazy, on-the-fly).
  - Cut into fixed-length windows (20 s by default).
  - Process only the first few conversations (configurable via --max-conversations).
  - Write out per-window WAVs and metadata (incl. speaker A/B WAV paths & texts).
"""
import argparse
import shutil
import logging
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import soundfile as sf
from lhotse import CutSet, RecordingSet, SupervisionSet
from datasets.arrow_writer import ArrowWriter

import torch
import torchaudio


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--recordings-jsonl", type=Path, required=True,
                   help="Lhotse recordings.jsonl")
    p.add_argument("--supervisions-jsonl", type=Path, required=True,
                   help="Lhotse supervisions.jsonl")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save chunks + metadata")
    p.add_argument("--chunk-length", type=float, default=20.0,
                   help="Length of each window in seconds")
    p.add_argument("--min-chunk-length", type=float, default=1.0,
                   help="Discard windows shorter than this")
    p.add_argument("--target-sample-rate", type=int, default=24000,
                   help="Resample (lazy) to this sampling rate")
    p.add_argument("--max-conversations", type=int, default=None,
                   help="Process only the first N conversations (for testing)")
    p.add_argument("--vocab_path", type=str, default=None)
    p.add_argument("--eval-ratio", type=float, default=0.1,
                   help="Fraction of conversations to use for dev/eval set")
    return p.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# CHANGE: helper to build per-speaker texts with [spkchange] only on real speaker changes
def build_spkchange_texts(cut_A, cut_B) -> tuple[str, str]:
    """
    Merge A/B supervisions by their (cut-relative) start times and create two strings:
    - text_A: A's utterances concatenated; append [spkchange] only when the next utterance is by B
    - text_B: B's utterances concatenated; append [spkchange] only when the next utterance is by A
    """
    # Collect (speaker, start_time, text)
    merged = []
    for s in cut_A.supervisions:
        merged.append(("A", s.start, s.text))
    for s in cut_B.supervisions:
        merged.append(("B", s.start, s.text))

    # Sort by time; ties keep input order
    merged.sort(key=lambda x: x[1])

    a_parts, b_parts = [], []
    for i, (spk, _, txt) in enumerate(merged):
        next_spk = merged[i + 1][0] if i + 1 < len(merged) else None

        if spk == "A":
            if txt:
                a_parts.append(txt)
            # Add boundary token only if the *next* utterance is from the other speaker
            if next_spk == "B":
                a_parts.append("[spkchange]")
        else:  # spk == "B"
            if txt:
                b_parts.append(txt)
            if next_spk == "A":
                b_parts.append("[spkchange]")

    # Join with single spaces; no trailing token if last utterance has no change after it
    text_A = " ".join(a_parts).strip()
    text_B = " ".join(b_parts).strip()
    return text_A, text_B


def main():
    args = parse_args()
    setup_logging()

    # Prepare output dirs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_out = args.output_dir / "wavs"
    wav_out.mkdir(exist_ok=True)

    logging.info("Loading Lhotse manifests…")
    recordings = RecordingSet.from_jsonl(str(args.recordings_jsonl))
    supervisions = SupervisionSet.from_jsonl(str(args.supervisions_jsonl))

    logging.info("Building CutSet from recordings + supervisions…")
    cuts = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions
    )

    logging.info(f"Resampling CutSet to {args.target_sample_rate} Hz (lazy)…")
    cuts = cuts.resample(args.target_sample_rate)

    # Determine first N conversation IDs
    unique_conv_ids = []
    for rec in recordings.recordings:
        conv_id = rec.id.rsplit("-", 1)[0]
        if conv_id not in unique_conv_ids:
            unique_conv_ids.append(conv_id)
        if args.max_conversations and len(unique_conv_ids) >= args.max_conversations:
            break
    # unique_conv_ids = unique_conv_ids[5000:args.max_conversations]
    logging.info(f"Limiting to first {len(unique_conv_ids)} conversations: {unique_conv_ids}")
    cuts = cuts.filter(lambda c: c.recording_id.rsplit("-", 1)[0] in unique_conv_ids)

    logging.info(f"Slicing into {args.chunk_length}s windows…")
    cuts = cuts.cut_into_windows(duration=args.chunk_length)

    logging.info(f"Filtering out windows shorter than {args.min_chunk_length}s…")
    cuts = cuts.filter(lambda c: c.duration >= args.min_chunk_length)

    # Eagerize so we can iterate multiple times
    cuts = cuts.to_eager()

    logging.info("Writing per-window WAVs and assembling metadata…")
    rows = []
    durations: Dict[str, float] = {}
    grouped = defaultdict(dict)
    for cut in cuts:
        conv_id, speaker = cut.recording_id.rsplit("-", 1)
        window_idx = int(cut.start // args.chunk_length)
        grouped[(conv_id, window_idx)][speaker] = cut

    for (conv_id, idx), parts in sorted(grouped.items()):
        print(conv_id, idx)
        print(parts)
        if "A" not in parts or "B" not in parts:
            logging.warning(f"Skipping incomplete window: {conv_id} @ idx {idx}")
            continue

        cut_A, cut_B = parts["A"], parts["B"]
        out_A = wav_out / f"{conv_id}_{idx:04d}_A.wav"
        out_B = wav_out / f"{conv_id}_{idx:04d}_B.wav"

        # Load audio (returns a NumPy array); convert to torch Tensor
        samples_A = torch.from_numpy(cut_A.load_audio())
        samples_B = torch.from_numpy(cut_B.load_audio())
        sr = cut_A.sampling_rate
        # torchaudio expects shape [channels, time]; if 1D, add channel dim
        if samples_A.dim() == 1:
            samples_A = samples_A.unsqueeze(0)
            samples_B = samples_B.unsqueeze(0)

        # torchaudio.save(
        #     str(out_A),
        #     samples_A,
        #     sr
        # )
        # torchaudio.save(
        #     str(out_B),
        #     samples_B,
        #     sr
        # )

        durations[str(out_A)] = cut_A.duration
        durations[str(out_B)] = cut_B.duration

        # add a trailing “-” to each supervision utterance
        # text_A = " ".join(f"{s.text} -" for s in cut_A.supervisions)
        # text_B = " ".join(f"{s.text} -" for s in cut_B.supervisions)
        text_A, text_B = build_spkchange_texts(cut_A, cut_B)


        rows.append({
            "recording_id":   conv_id,
            "speaker_A_wav":  str(out_A),
            "speaker_A_text": text_A,
            "speaker_B_wav":  str(out_B),
            "speaker_B_text": text_B,
        })

    # Save metadata.csv
    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Wrote metadata.csv ({len(df)} rows) to {csv_path}")


    arrow_path = args.output_dir / "raw.arrow"
    logging.info(f"Writing raw arrow via ArrowWriter to {arrow_path}")
    with ArrowWriter(
        path=str(arrow_path),
        writer_batch_size=1  # flush on every example
    ) as writer:
        for rec in rows:
            writer.write(rec)
    # when the context exits, it finalizes the file automatically
    logging.info("Finished writing raw.arrow")

    # Build a list of durations in row‐order, so durations[i] matches rows[i]
    durations_list = [
        # pick either speaker’s cut duration; they’re identical
        durations[row["speaker_A_wav"]]
        for row in rows
    ]
    dur_path = args.output_dir / "duration.json"
    with open(dur_path, "w") as jf:
        json.dump({"durations": durations_list}, jf, indent=2)
    logging.info(f"Wrote duration.json to {dur_path} (with {len(durations_list)} entries)")

    if args.vocab_path:
        vocab_source = Path(args.vocab_path)
        if vocab_source.is_file():
            shutil.copy(vocab_source, Path(args.output_dir) / vocab_source.name)
            print(f"Copied vocab file to {args.output_dir}/{vocab_source.name}")
        else:
            print(f"Warning: Vocab file not found at {args.vocab_path}")


if __name__ == "__main__":
    main()
