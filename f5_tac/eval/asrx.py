import os
import glob
import json
import whisperx
from pathlib import Path
from lhotse import SupervisionSegment, SupervisionSet
import argparse

def transcribe_dialogue(audio_path, model_size="large-v2", device="cuda", output="whisperx_output", diarize=False, name="supervisions"):
    
    print("Loading Whisper model...")
    model = whisperx.load_model(
        model_size,
        device=device,
        compute_type="float16" if "cuda" in device else "float32"
    )
    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code="en",
        device=device
    )

    all_segments = []
    
    # 2) Iterate over every WAV in the given path
    for wav_file in glob.glob(os.path.join(audio_path, "*.wav")):
        recording_id = os.path.splitext(os.path.basename(wav_file))[0]
        print(f"\n→ Processing {wav_file!r} as recording '{recording_id}'")
        
        # 2a) Load & transcribe
        audio = whisperx.load_audio(wav_file)
        result = model.transcribe(audio, language="en")
        
        # 2b) Time‑align words
        aligned = whisperx.align(
            result["segments"], model_a, metadata, audio, device=device
        )

        # 2d) Turn each segment into a SupervisionSegment
        for i, seg in enumerate(result["segments"], start=1):
            start = seg["start"]
            end = seg["end"]
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker", None)
            
            sup = SupervisionSegment(
                id=f"{recording_id}_{i}",
                recording_id=recording_id,
                start=start,
                duration=end - start,
                channel=0,
                text=text,
                speaker=speaker
            )
            all_segments.append(sup)

    # 3) Build & save the combined SupervisionSet
    sup_set = SupervisionSet.from_segments(all_segments)
    output_supervisions_path = Path(output) / f"{name}.jsonl.gz"
    print(f"\nSaving Lhotse SupervisionSet to {output_supervisions_path}")
    sup_set.to_file(str(output_supervisions_path))

 
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

# ""
recording_wav_path = args.data_root
output_path_no_json = "/work/users/r/p/rphadke/data/simfisher-layered/manifests"

transcribe_dialogue(
            audio_path=recording_wav_path,
            model_size="large-v2",
            device="cuda",
            output=output_path_no_json,
            name=args.name
        )