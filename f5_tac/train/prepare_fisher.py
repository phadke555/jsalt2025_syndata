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

import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from lhotse import SupervisionSet
from tqdm import tqdm

def get_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the Fisher dataset for two-speaker TTS training."
    )
    parser.add_argument(
        "--fisher_path",
        type=str,
        required=True,
        help="Path to the root of the Fisher dataset directory, containing lhotse manifests.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory where prepared files will be saved.",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Optional: Path to the vocab.txt file to copy to the output directory.",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=24000,
        help="The target sample rate to resample the audio to.",
    )
    return parser.parse_args()


def prepare_fisher(fisher_path: str, output_path: str, target_sample_rate: int):
    """
    Main function to process the dataset.

    This function reads lhotse manifests, processes each conversation to extract
    paired speaker data, resamples and saves the audio, and creates the
    necessary metadata, arrow, and duration files for training.
    """
    print("--- Starting Fisher Dataset Preparation ---")
    
    # 1. Setup paths
    fisher_path = Path(fisher_path)
    output_path = Path(output_path)
    output_wav_path = output_path / "wavs"
    
    print(f"Fisher Manifests Path: {fisher_path}")
    print(f"Output Path: {output_path}")
    
    os.makedirs(output_wav_path, exist_ok=True)

    # 2. Load Lhotse manifests
    print("Loading Lhotse manifests...")
    try:
        supervisions = SupervisionSet.from_file(fisher_path / "supervisions.jsonl.gz")
        assert (fisher_path / "recordings.jsonl.gz").exists(), \
            "recordings.jsonl.gz not found in the fisher_path."
    except Exception as e:
        print(f"Error loading manifests: {e}")
        print("Please ensure 'supervisions.jsonl.gz' and 'recordings.jsonl.gz' are in the specified fisher_path.")
        return

    # 3. Group supervisions by conversation (recording_id)
    print("Grouping supervisions by conversation...")
    conversations = supervisions.group_by_recording_id()
    
    metadata = []
    
    print(f"Found {len(conversations)} conversations. Processing...")
    for recording_id, conv_supervisions in tqdm(conversations.items(), desc="Processing Conversations"):
        
        speaker_ids = sorted(list(conv_supervisions.speakers))
        if len(speaker_ids) != 2:
            continue
            
        spk_A_id, spk_B_id = speaker_ids[0], speaker_ids[1]

        # 4. Extract and concatenate audio and text for each speaker
        text_A, text_B = [], []
        audio_A, audio_B = [], []

        for sup in conv_supervisions:
            audio_segment = torch.from_numpy(sup.load_audio()).float()
            
            if sup.speaker == spk_A_id:
                text_A.append(sup.text)
                audio_A.append(audio_segment)
            else:
                text_B.append(sup.text)
                audio_B.append(audio_segment)

        if not audio_A or not audio_B:
            continue
        
        full_audio_A = torch.cat(audio_A, dim=1)
        full_audio_B = torch.cat(audio_B, dim=1)
        
        # Resample audio
        source_sr = conv_supervisions[0].sampling_rate
        if source_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(source_sr, target_sample_rate)
            full_audio_A = resampler(full_audio_A)
            full_audio_B = resampler(full_audio_B)
            
        # 5. Calculate durations. No padding is applied here.
        # The duration for the DynamicBatchSampler will be the *longer* of the two.
        duration_A_sec = full_audio_A.shape[1] / target_sample_rate
        duration_B_sec = full_audio_B.shape[1] / target_sample_rate
        max_duration_sec = max(duration_A_sec, duration_B_sec)
        
        # 6. Save the prepared audio files
        wav_A_path = output_wav_path / f"{recording_id}_A.wav"
        wav_B_path = output_wav_path / f"{recording_id}_B.wav"
        
        torchaudio.save(wav_A_path, full_audio_A, target_sample_rate)
        torchaudio.save(wav_B_path, full_audio_B, target_sample_rate)
        
        # 7. Collate metadata
        metadata.append({
            "id": recording_id,
            "path_A": str(wav_A_path),
            "text_A": " ".join(text_A),
            "path_B": str(wav_B_path),
            "text_B": " ".join(text_B),
            "duration": max_duration_sec
        })

    if not metadata:
        print("No valid conversations were processed. Please check your manifests and data.")
        return
        
    # 8. Create and save the final output files
    print("\nCreating final output files...")
    
    df = pd.DataFrame(metadata)
    df.to_csv(output_path / "metadata.csv", index=False)
    print(f"Saved metadata to {output_path / 'metadata.csv'}")

    # Create raw.arrow as a single file
    dataset = Dataset.from_pandas(df.drop(columns=['id']))
    dataset.to_file(str(output_path / "raw.arrow"))
    print(f"Saved dataset to {output_path / 'raw.arrow'}")

    # Create duration.json
    duration_data = {"duration": df["duration"].tolist()}
    with open(output_path / "duration.json", "w") as f:
        json.dump(duration_data, f)
    print(f"Saved durations to {output_path / 'duration.json'}")
    
    print("\n--- Dataset preparation complete! ---")


if __name__ == "__main__":
    args = get_args()
    
    # Run the main preparation function
    prepare_fisher(
        fisher_path=args.fisher_path, 
        output_path=args.output_path, 
        target_sample_rate=args.target_sample_rate
    )

    # Copy vocab file if path is provided
    if args.vocab_path:
        vocab_source = Path(args.vocab_path)
        if vocab_source.is_file():
            vocab_dest = Path(args.output_path) / vocab_source.name
            shutil.copy(vocab_source, vocab_dest)
            print(f"Copied vocab file to {vocab_dest}")
        else:
            print(f"Warning: Vocab file not found at {args.vocab_path}")