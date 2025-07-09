import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import evaluate
from jiwer import compute_measures
from glob import glob
import os
import pandas as pd
from difflib import ndiff


def main():
    parser = argparse.ArgumentParser(description="Batch ASR of WAV files.")
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    ref_dir = os.path.join(args.data_root, "concatenated_real")
    gen_dir = os.path.join(args.data_root, "concatenated_generations")

    ref_files = sorted(glob(os.path.join(ref_dir, "*.wav")))
    gen_files = sorted(glob(os.path.join(gen_dir, "*.wav")))

    assert len(ref_files) == len(gen_files), "Reference/gen counts differ!"

    combined_output_dir = os.path.join(args.data_root, "asr_results")
    os.makedirs(combined_output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,  # batch size for inference - set based on your device
        torch_dtype=torch_dtype,
        device=device,
    )

    wer = evaluate.load("wer")

    results = []
    for ref_path, gen_path in zip(ref_files, gen_files):
        conv_id = os.path.splitext(os.path.basename(ref_path))[0]
        # 4.1 Transcribe
        real_out = pipe(ref_path)
        gen_out  = pipe(gen_path)
        ref_text = real_out["text"]
        hyp_text = gen_out["text"]

        baseline = wer.compute(predictions=[ref_text], references=[ref_text])
        score  = wer.compute(predictions=[hyp_text], references=[ref_text])

        print(f"WER(ref vs. ref)       = {baseline:.3f}") 
        print(f"WER(generated vs. ref) = {score:.3f}")
        print(f"ΔWER                  = {score - baseline:.3f}")

        measures = compute_measures(ref_text, hyp_text)

        print("Hits:          ", measures["hits"])
        print("Substitutions: ", measures["substitutions"])
        print("Deletions:     ", measures["deletions"])
        print("Insertions:    ", measures["insertions"])

        results.append({
            "conversation_id": conv_id,
            "wer":            score,
            "hits":           measures["hits"],
            "substitutions":  measures["substitutions"],
            "deletions":      measures["deletions"],
            "insertions":     measures["insertions"],
        })

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(combined_output_dir, "tts_asr_stats.csv"), index=False)
        print(f"Saved {len(df)} rows to tts_asr_stats.csv")

        diff_tokens = ndiff(ref_text.split(), hyp_text.split())
        diff_file = os.path.join(combined_output_dir, f"{conv_id}_diff.txt")
        with open(diff_file, "w", encoding="utf-8") as f:
            for token in diff_tokens:
                f.write(token + "\n")
        print(f"Wrote diff to {diff_file}")
        print("============================================================")



    df = pd.DataFrame(results)
    df.to_csv(os.path.join(combined_output_dir, "tts_asr_stats.csv"), index=False)
    print(f"Saved {len(df)} rows to tts_asr_stats.csv")



if __name__ == "__main__":
    main()