import argparse
import os
import json
import torch
import torchaudio
from tqdm import tqdm
from pyannote.audio import Pipeline

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--output_path", type=str, required=True)
#     parser.add_argument("--generated_audio_path", type=str, required=True)
#     parser.add_argument("--real_audio_path", type=str, required=True)

#     args = parser.parse_args()
#     os.makedirs(args.output_path, exist_ok=True)

#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token="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")

#     # send pipeline to GPU (when available)
#     pipeline.to(torch.device("cuda"))

#     # Load audio file and resample to 16kHz
#     waveform, sample_rate = torchaudio.load("/work/users/r/p/rphadke/JSALT/outputs/fisher_verysmallbatch/concatenated_generations/fe_03_00001_full_generated.wav")
#     resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#     waveform_16k = resampler(waveform)



#     diarization = pipeline({"waveform": waveform_16k, "sample_rate": 16000})


#     # # apply pretrained pipeline
#     # diarization = pipeline("/work/users/r/p/rphadke/JSALT/outputs/fisher_verysmallbatch/concatenated_generations/fe_03_00001_full_generated.wav")

#     # Print result
#     print("================================================================")
#     print("Diarization of Generated Conversation 1")
#     results = []
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
#         results.append({
#             "speaker": speaker,
#             "start_time": turn.start,
#             "end_time": turn.end
#         })

#     # Save results as JSON
#     output_json = "/work/users/r/p/rphadke/JSALT/outputs/fisher_verysmallbatch/diarization_results/generated/fe_03_00001_full_generated.json"
#     with open(output_json, "w") as f:
#         json.dump(results, f, indent=2)

#     print(f"Saved diarization results to {output_json}")



# if __name__ == "__main__":
#     main()


# # # apply pretrained pipeline
# # diarization = pipeline("/work/users/r/p/rphadke/JSALT/outputs/fisher_verysmallbatch/concatenated_real/fe_03_00001_full_real.wav")

# # # print the result
# # print("===========================================================")
# # print("Diarization of Real Conversation 1")
# # for turn, _, speaker in diarization.itertracks(yield_label=True):
# #     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")




def load_and_resample(path, target_sr=16000):
    """Load and resample audio, ensuring shape (1, samples)."""
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, target_sr


def diarize_and_save(pipeline, audio_path, output_json):
    """Run diarization and save result to JSON."""
    waveform, sample_rate = load_and_resample(audio_path)

    # Run pyannote pipeline
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Convert to list of dicts
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "speaker": speaker,
            "start_time": turn.start,
            "end_time": turn.end
        })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


def process_folder(pipeline, input_dir, output_dir):
    """Process all WAV files in a directory."""
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    print(f"Found {len(wav_files)} audio files in {input_dir}.")

    for wav_file in tqdm(wav_files, desc=f"Processing {input_dir}"):
        input_path = os.path.join(input_dir, wav_file)
        output_filename = os.path.splitext(wav_file)[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)
        diarize_and_save(pipeline, input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Batch diarization of WAV files.")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--generated_audio_path", type=str, required=True)
    parser.add_argument("--real_audio_path", type=str, required=True)
    parser.add_argument("--auth_token", type=str, required=True, help="HuggingFace token for pyannote model")
    args = parser.parse_args()

    # Load pyannote pipeline
    print("Loading Pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=args.auth_token)
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Set up output folders
    generated_output_dir = os.path.join(args.output_path, "generated")
    real_output_dir = os.path.join(args.output_path, "real")

    # Process generated and real audio
    process_folder(pipeline, args.generated_audio_path, generated_output_dir)
    process_folder(pipeline, args.real_audio_path, real_output_dir)


if __name__ == "__main__":
    main()
