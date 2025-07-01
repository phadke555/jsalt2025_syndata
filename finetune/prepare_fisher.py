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
    p.add_argument("--chunk-length", type=float, default=10.0,
                   help="Length of each window in seconds")
    p.add_argument("--min-chunk-length", type=float, default=1.0,
                   help="Discard windows shorter than this")
    p.add_argument("--target-sample-rate", type=int, default=24000,
                   help="Resample (lazy) to this sampling rate")
    p.add_argument("--max-conversations", type=int, default=2,
                   help="Process only the first N conversations (for testing)")
    p.add_argument("--vocab_path", type=str, default=None)
    return p.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


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
        if len(unique_conv_ids) >= args.max_conversations:
            break
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

    rows = []
    durations = {}

    for cut in cuts:
        conv_id = cut.recording_id.rsplit("-", 1)[0]  # Just the convo ID
        window_idx = int(cut.start // args.chunk_length)
        out_path = wav_out / f"{conv_id}_{window_idx:04d}.wav"

        samples = torch.from_numpy(cut.load_audio())
        sr = cut.sampling_rate
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)

        torchaudio.save(str(out_path), samples, sr)
        durations[str(out_path)] = cut.duration

        text = " ".join(s.text for s in cut.supervisions)

        rows.append({
            "audio_path": str(out_path),
            "text": text,
            "duration": cut.duration
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
    durations_list = [durations[row["wav"]] for row in rows]    
    dur_path = args.output_dir / "duration.json"
    with open(dur_path, "w") as jf:
        json.dump({"duration": durations_list}, jf, indent=2)
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
