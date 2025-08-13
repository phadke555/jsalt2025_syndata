"""
Preprocessing script for Fisher dataset:
- Resamples all WAV files to 24 kHz
- Organizes data into per-speaker folders (speaker_A, speaker_B, etc.)
- Saves `wav/` folder, `metadata.csv`, `raw.arrow`, and `duration.json` for each speaker

Usage:
    python preprocess.py \
        --wav_dir /path/to/fisher_wavs \
        --lhotse_manifest /path/to/cuts.jsonl \
        --output_dir /path/to/output \
        --speakers A B

Requirements:
    pip install lhotse librosa soundfile pandas pyarrow
"""

# import os
# import argparse
# import json
# import soundfile as sf
# import librosa
# import pandas as pd
# import pyarrow as pa
# from lhotse import CutSet


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Preprocess Fisher dataset: resample and organize per speaker"
#     )
#     parser.add_argument(
#         "--wav_dir", required=True,
#         help="Path to directory containing original WAV files"
#     )
#     parser.add_argument(
#         "--lhotse_manifest", required=True,
#         help="Path to Lhotse cuts manifest (JSONL)"
#     )
#     parser.add_argument(
#         "--output_dir", required=True,
#         help="Directory where processed data will be saved"
#     )
#     parser.add_argument(
#         "--speakers", nargs='+', default=["A", "B"],
#         help="Speaker labels to process (e.g. A B)"
#     )
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     cuts = CutSet.from_jsonl(args.lhotse_manifest)
#     # Map cut IDs to Cut objects for transcript lookup
#     cuts_map = {cut.id: cut for cut in cuts}
#     target_sr = 24000

#     # Prepare per-speaker structures
#     speakers = [s.upper() for s in args.speakers]
#     metadata = {s: {"id": [], "audio_filepath": [], "text": [], "duration": []}
#                 for s in speakers}

#     # Process each WAV file in wav_dir
#     for fname in os.listdir(args.wav_dir):
#         if not fname.lower().endswith('.wav'):
#             continue
#         utt_full = os.path.splitext(fname)[0]  # e.g. fe_03_00347-B
#         if '-' not in utt_full:
#             print(f"Skipping file without speaker suffix: {fname}")
#             continue
#         utt_base, sp = utt_full.rsplit('-', 1)
#         sp = sp.upper()
#         if sp not in speakers:
#             continue

#         in_path = os.path.join(args.wav_dir, fname)
#         # Load & resample audio
#         audio, _ = librosa.load(in_path, sr=target_sr)

#         # Create speaker-specific output dirs
#         sp_dir = os.path.join(args.output_dir, f"speaker_{sp}")
#         wav_out = os.path.join(sp_dir, 'wav')
#         os.makedirs(wav_out, exist_ok=True)

#         # Write resampled WAV
#         out_wav = os.path.join(wav_out, f"{utt_full}.wav")
#         sf.write(out_wav, audio, target_sr)

#         # Lookup transcript from cuts_map
#         cut = cuts_map.get(utt_full) or cuts_map.get(utt_base)
#         text = ""
#         if cut and cut.supervisions:
#             text = " ".join(sup.text for sup in cut.supervisions)

#         duration = len(audio) / target_sr

#         # Collect metadata
#         meta = metadata[sp]
#         meta['id'].append(utt_full)
#         meta['audio_filepath'].append(out_wav)
#         meta['text'].append(text)
#         meta['duration'].append(duration)

#     # Save per-speaker metadata, arrow, and durations
#     for sp in speakers:
#         sp_dir = os.path.join(args.output_dir, f"speaker_{sp}")
#         meta = metadata[sp]
#         # Save metadata.csv
#         df = pd.DataFrame(meta)
#         df.to_csv(os.path.join(sp_dir, 'metadata.csv'), index=False)
#         # Save raw.arrow
#         table = pa.Table.from_pandas(df)
#         with pa.OSFile(os.path.join(sp_dir, 'raw.arrow'), 'wb') as sink:
#             with pa.ipc.RecordBatchFileWriter(sink, table.schema) as writer:
#                 writer.write_table(table)
#         # Save duration.json
#         durations = dict(zip(meta['id'], meta['duration']))
#         with open(os.path.join(sp_dir, 'duration.json'), 'w') as f:
#             json.dump(durations, f, indent=2)

#     print(f"Preprocessing complete. Data saved under {args.output_dir}.")




# if __name__ == '__main__':
#     main()


import argparse
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.cut import append_cuts

def get_args():
    parser = argparse.ArgumentParser(description="Prepare the Fisher dataset for two-speaker TTS training.")
    parser.add_argument("--fisher_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--target_sample_rate", type=int, default=24000)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--supervisions_path", type=str, default=None)
    return parser.parse_args()

def prepare_fisher(fisher_path, output_path, target_sample_rate, num_samples=None, supervisions_path=None):
    print("--- Starting Fisher Dataset Preparation with CutSet ---")

    fisher_path = Path(fisher_path)
    output_path = Path(output_path)
    output_wav_path = output_path / "wavs"
    output_path.mkdir(parents=True, exist_ok=True)
    output_wav_path.mkdir(parents=True, exist_ok=True)

    if supervisions_path:
        supervisions_file = Path(supervisions_path)
        recordings_file = supervisions_file.parent / "recordings.jsonl.gz"
    else:
        supervisions_file = fisher_path / "supervisions.jsonl.gz"
        recordings_file = fisher_path / "recordings.jsonl.gz"

    print(f"Loading manifests from: {supervisions_file}, {recordings_file}")
    recordings = RecordingSet.from_file(recordings_file)
    supervisions = SupervisionSet.from_file(supervisions_file)
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)

    conversations = defaultdict(list)
    for cut in cuts:
        base_id = cut.recording_id.rsplit('-', 1)[0]  # removes -A or -B
        conversations[base_id].append(cut)
        if num_samples is not None and len(conversations) >= num_samples:
            break 


    conversation_items = list(conversations.items())
    if num_samples is not None:
        conversation_items = conversation_items[:num_samples]

    metadata = []
    for recording_id, cuts_per_convo in tqdm(conversation_items, desc="Processing Conversations"):
        print(f"Processing: {recording_id}")
        speaker_cuts = defaultdict(list)
        for cut in cuts_per_convo:
            for sup in cut.supervisions:
                speaker_cuts[sup.speaker].append(cut)
            
        print(f"Found {len(speaker_cuts)} speakers in {recording_id}")
        if len(speaker_cuts) != 2:
            continue

        spk_A, spk_B = sorted(speaker_cuts)
        cuts_A = append_cuts(speaker_cuts[spk_A])
        cuts_B = append_cuts(speaker_cuts[spk_B])

        audio_A = torch.from_numpy(cuts_A.load_audio()).float()
        audio_B = torch.from_numpy(cuts_B.load_audio()).float()

        source_sr = cuts_A.sampling_rate
        if source_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(source_sr, target_sample_rate)
            audio_A = resampler(audio_A)
            audio_B = resampler(audio_B)

        duration = max(audio_A.shape[1], audio_B.shape[1]) / target_sample_rate

        wav_A_path = output_wav_path / f"{recording_id}_A.wav"
        wav_B_path = output_wav_path / f"{recording_id}_B.wav"
        torchaudio.save(str(wav_A_path), audio_A, target_sample_rate)
        torchaudio.save(str(wav_B_path), audio_B, target_sample_rate)

        text_A = " ".join([sup.text for c in speaker_cuts[spk_A] for sup in c.supervisions])
        text_B = " ".join([sup.text for c in speaker_cuts[spk_B] for sup in c.supervisions])

        metadata.append({
            "id": recording_id,
            "path_A": str(wav_A_path),
            "text_A": text_A,
            "path_B": str(wav_B_path),
            "text_B": text_B,
            "duration": duration
        })

    if not metadata:
        print("No valid conversations were processed.")
        return

    df = pd.DataFrame(metadata)
    df.to_csv(output_path / "metadata.csv", index=False)
    print(f"Saved metadata to {output_path / 'metadata.csv'}")

    file_raw = output_path / "raw.arrow"
    with ArrowWriter(path=file_raw, writer_batch_size=1) as writer:
        for rec in tqdm(metadata, desc="Writing raw.arrow"):
            writer.write(rec)
    print(f"Saved dataset to {file_raw}")

    with open(output_path / "duration.json", "w") as f:
        json.dump({"duration": df["duration"].tolist()}, f)
    print(f"Saved durations to {output_path / 'duration.json'}")
    print("\n--- Dataset preparation complete! ---")

def main():
    args = get_args()
    prepare_fisher(
        fisher_path=args.fisher_path,
        output_path=args.output_path,
        target_sample_rate=args.target_sample_rate,
        num_samples=args.num_samples,
        supervisions_path=args.supervisions_path
    )

    if args.vocab_path:
        vocab_source = Path(args.vocab_path)
        if vocab_source.is_file():
            shutil.copy(vocab_source, Path(args.output_path) / vocab_source.name)
            print(f"Copied vocab file to {args.output_path}/{vocab_source.name}")
        else:
            print(f"Warning: Vocab file not found at {args.vocab_path}")

if __name__ == "__main__":
    main()