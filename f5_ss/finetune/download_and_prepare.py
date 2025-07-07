#!/usr/bin/env python3
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Download AMI and prepare data for F5-TTS"
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/work/users/r/p/rphadke/JSALT",
        help="Base directory under which my_speak/ will be created"
    )
    return p.parse_args()
    
    
import shutil
import json
from collections import Counter
from datasets import load_dataset, Audio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

# import your own version of F5-TTS helpers
from f5_tts.train.finetune_gradio import (
    get_correct_audio_path,
    get_audio_duration,
    format_seconds_to_hms,
)

from f5_tts.model.utils import convert_char_to_pinyin



# Base data directory (one level above this script)


def download_and_prepare():
    # Define paths
    path_project   = os.path.join(path_data, project_name)
    path_wavs      = os.path.join(path_project, "wavs")
    file_metadata  = os.path.join(path_project, "metadata.csv")

    # Ensure directories
    os.makedirs(path_wavs, exist_ok=True)

    # 1) Download AMI “ihm” train split
    print("→ Loading AMI 'ihm' train split…")
    ds = load_dataset("edinburghcstr/ami", "ihm", split="train", trust_remote_code=True)

    # 2) Copy WAVs & write metadata.csv
    print(f"→ Copying WAVs into {path_wavs} and writing {file_metadata}…")
    with open(file_metadata, "w", encoding="utf-8") as meta_f:
        for ex in ds:
            mid, aid = ex["meeting_id"], ex["audio_id"]
            src      = ex["audio"]["path"]
            fname    = f"{mid}_{aid}.wav"
            dst      = os.path.join(path_wavs, fname)
            shutil.copy(src, dst)

            # sanitize transcript
            txt = ex["text"].replace("|", " ")
            meta_f.write(f"{fname}|{txt}\n")

    # 3) Process metadata: raw.arrow, durations, vocab
    print("→ Creating metadata (raw.arrow, duration.json, vocab.txt)…")
    info, vocab_txt = create_metadata_local(project_name, ch_tokenizer=True)
    print(info)
    if vocab_txt:
        print("New vocab entries:\n", vocab_txt)


def create_metadata_local(name_project, ch_tokenizer):
    # Paths inside project
    path_project      = os.path.join(path_data, name_project)
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata     = os.path.join(path_project, "metadata.csv")
    file_raw          = os.path.join(path_project, "raw.arrow")
    file_duration     = os.path.join(path_project, "duration.json")
    file_vocab        = os.path.join(path_project, "vocab.txt")

    # Check metadata exists
    if not os.path.isfile(file_metadata):
        return f"ERROR: metadata.csv not found at {file_metadata}", ""

    # Read lines
    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]

    # Containers
    result, durations, errors, vocab_set = [], [], [], set()

    # Iterate and filter
    for line in tqdm(lines, desc="Processing metadata"):
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        name_audio, text = parts

        file_audio = get_correct_audio_path(name_audio, path_project_wavs)
        if not os.path.isfile(file_audio):
            errors.append(f"{file_audio} not found")
            continue

        try:
            dur = get_audio_duration(file_audio)
        except Exception as e:
            errors.append(f"{file_audio} duration error: {e}")
            continue

        # Duration & length filter
        if not (1.0 <= dur <= 30.0) or len(text.strip()) < 3:
            errors.append(f"{name_audio} skipped (dur={dur:.2f}, len={len(text)})")
            continue

        # Convert text (e.g. to pinyin)
        text = convert_char_to_pinyin([text.strip()], polyphone=True)[0]

        result.append({"audio_path": file_audio, "text": text, "duration": dur})
        durations.append(dur)
        if ch_tokenizer:
            vocab_set.update(list(text))

    if not result:
        return "No valid files found in wavs/", ""

    # Write raw.arrow
    with ArrowWriter(path=file_raw, writer_batch_size=1) as writer:
        for rec in tqdm(result, desc="Writing raw.arrow"):
            writer.write(rec)

    # Save durations
    with open(file_duration, "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False, indent=2)

    # Prepare vocab list
    if ch_tokenizer:
        vocab_list = sorted(vocab_set)
    else:
        if not os.path.isfile(file_vocab):
            return "Error: no existing vocab.txt found", ""
        with open(file_vocab, "r", encoding="utf-8-sig") as f:
            vocab_list = [line.strip() for line in f]

    # Write vocab.txt
    new_vocab = ""
    with open(file_vocab, "w", encoding="utf-8") as f:
        for tok in vocab_list:
            f.write(tok + "\n")
            new_vocab += tok + "\n"

    # Summary
    total_dur = sum(durations)
    summary = (
        f"Prepared {len(result)} samples\n"
        f"Total audio time: {format_seconds_to_hms(total_dur)}\n"
        f"Min dur: {min(durations):.2f}s, Max dur: {max(durations):.2f}s\n"
        f"raw.arrow → {file_raw}\n"
        f"vocab size → {len(vocab_list)}\n"
    )
    if errors:
        summary += "Errors:\n" + "\n".join(errors)

    return summary, new_vocab


if __name__ == "__main__":
    args = parse_args()
    path_data    = args.data_root
    project_name = "my_speak"
    download_and_prepare()
