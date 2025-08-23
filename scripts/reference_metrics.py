import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from lhotse import RecordingSet, SupervisionSet, CutSet
import torch
from transformers import pipeline
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Whisper on Fisher Test Set Real Individual Stream Recordings to Measure Base WER. Generated quality should try to match this.")

    # --------------------- Experiment Configuration --------------------- #
    parser.add_argument("--data_root", type=str, default=None, help="Base directory to datasets, overrides defaults.")
    parser.add_argument("--max_conversations", type=int, default=233, help="Number of conversations to use in the test set")
    parser.add_argument("--conversation_offset", type=int, default=11465, help="Number of conversations to use in the test set")

    return parser.parse_args()

import re
def normalize_text(s: str):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    args = parse_args()
    max_conversations = args.max_conversations
    conversation_offset = args.conversation_offset
    target_sample_rate = 16000
    recs = Path(args.data_root) / "recordings.jsonl.gz"
    sups = Path(args.data_root) / "supervisions.jsonl.gz"

    recordings = RecordingSet.from_jsonl(str(recs))
    supervisions = SupervisionSet.from_jsonl(str(sups))

    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)

    unique_conv_ids: List[str] = []
    for rec in recordings.recordings:
        conv_id = rec.id.rsplit("-", 1)[0]
        if conv_id not in unique_conv_ids:
            unique_conv_ids.append(conv_id)
    if max_conversations is not None:
        start = conversation_offset
        end = conversation_offset + max_conversations
        unique_conv_ids = unique_conv_ids[start:end]
        cuts = cuts.filter(lambda c: c.recording_id.rsplit("-", 1)[0] in unique_conv_ids)
    print(len(unique_conv_ids))

    cuts = cuts.resample(target_sample_rate)
    cuts = cuts.trim_to_supervisions().cut_into_windows(duration=30.0)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        chunk_length_s=30,
        batch_size=4,
        device=device
    )

    # Run inference per chunk
    hypotheses = []
    references = []
    scores = []
    print_every = 5  # how often to print WER updates
    import meeteval

    for i, cut in enumerate(cuts):
        # import pdb; pdb.set_trace()
        audio = cut.load_audio()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)  # ensure mono
        try:
            text_ref = cut.supervisions[0].text
            id_ = cut.supervisions[0].recording_id

            # Whisper output
            text_hyp = whisper_pipe(audio, return_timestamps=False)["text"]
            text_hyp = normalize_text(text_hyp)
            score = meeteval.wer.wer.siso.siso_word_error_rate(text_ref, text_hyp)

            hypotheses.append(text_hyp)
            references.append(text_ref)
            scores.append(score.error_rate)
        except:
            continue

        partial_score = meeteval.wer.wer.siso.siso_word_error_rate(text_ref, text_hyp)
        if partial_score.error_rate > 0.50:
            print(f"{i} | {id_} | Partial WER: {partial_score}")
            print(f"Ref Text {text_ref} | \n Hyp Text {text_hyp}")
    
    import numpy as np
    print("Average SISO WER:", round(np.mean(scores), 2))

    final_score = meeteval.wer.wer.siso.siso_word_error_rate("".join(references), "".join(hypotheses))
    print("Final SISO WER:", final_score)

if __name__ == "__main__":
    main()