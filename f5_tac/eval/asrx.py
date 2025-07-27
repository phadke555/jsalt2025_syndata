import whisperx
import torch
import json
import argparse
import os

def transcribe_dialogue(audio_path, model_size="large-v2", device="cuda", output="whisperx_output", diarize=False):
    audio = whisperx.load_audio(audio_path)
    
    print("Loading Whisper model...")
    model = whisperx.load_model(model_size, device=device, compute_type="float16" if "cuda" in device else "float32")

    print("Transcribing...")
    result = model.transcribe(audio, language="en")

    print("Aligning words...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device=device)

    print(f"Saving to {output}.json and {output}.segments.json")
    with open(output + ".json", "w") as f:
        json.dump(aligned_result, f, indent=2)

    with open(output + ".segments.json", "w") as f:
        json.dump(result["segments"], f, indent=2)

    # 3. Assign speaker labels
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="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", device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs
    print("Done.")

recording_wav_path = "/work/users/r/p/rphadke/JSALT/eval/BASELINESingle-3.0Reproduce/concatenated_generations/fe_03_00001_full_generated.wav"
output_path_no_json = "/work/users/r/p/rphadke/JSALT/eval/BASELINESingle-3.0Reproduce/supervisions/output_comb"

transcribe_dialogue(
            audio_path=recording_wav_path,
            model_size="large-v2",
            device="cuda",
            output=output_path_no_json,
        )