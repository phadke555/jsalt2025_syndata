import argparse
import os
import json
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.audio import AudioSource

def get_args():
    parser = argparse.ArgumentParser(description="Convert split-speaker WAV files into Lhotse manifests using existing supervisions.")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing speaker-separated WAV files.")
    parser.add_argument("--supervisions_path", type=str, required=True, help="Path to existing supervision manifest (JSONL or JSONL.GZ).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save new recordings and updated supervisions manifests.")
    return parser.parse_args()

def main():
    args = get_args()

    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading existing supervisions...")
    supervision_set = SupervisionSet.from_file(args.supervisions_path)

    print("Indexing supervisions by recording_id and channel...")
    grouped = {}
    for sup in supervision_set:
        key = (sup.recording_id, sup.channel)
        grouped.setdefault(key, []).append(sup)

    print("Searching for WAV files in", wav_dir)
    recordings = []
    new_supervisions = []

    for wav_path in sorted(wav_dir.glob("*.wav")):
        fname = wav_path.stem  # e.g., fe_03_00001-A
        if '-' not in fname:
            print(f"Skipping malformed filename: {fname}")
            continue

        base_id, suffix = fname.rsplit('-', 1)
        channel = 0 if suffix.upper() == 'A' else 1
        recording_id = f"{base_id}-{suffix.upper()}"

        # Create new Recording object for the WAV
        source = AudioSource(type='file', channels=[0], source=str(wav_path))
        recording = Recording.from_file(wav_path, recording_id=recording_id)
        recordings.append(recording)

        # Find and fix supervision segments for this split
        original_id = base_id
        matched_segs = grouped.get((original_id, channel), [])
        for sup in matched_segs:
            new_sup = SupervisionSegment(
                id=sup.id,
                recording_id=recording_id,
                start=sup.start,
                duration=sup.duration,
                channel=0,
                text=sup.text,
                speaker=sup.speaker,
                language=sup.language
            )
            new_supervisions.append(new_sup)

    print(f"Created {len(recordings)} new Recording objects and {len(new_supervisions)} SupervisionSegments.")

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(new_supervisions)

    recordings_path = output_dir / "recordings.jsonl.gz"
    supervisions_path = output_dir / "supervisions.jsonl.gz"

    print(f"Writing recordings to {recordings_path}")
    recording_set.to_file(recordings_path)

    print(f"Writing updated supervisions to {supervisions_path}")
    supervision_set.to_file(supervisions_path)

    print("✅ Finished generating new manifests.")

if __name__ == "__main__":
    main()
