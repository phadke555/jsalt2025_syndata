import os
from lhotse import SupervisionSegment, SupervisionSet
from lhotse import Recording, RecordingSet
from lhotse.audio import info
import torchaudio
import json

# Your existing supervision file
old_supervisions_path = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/supervisions.jsonl.gz"
output_supervisions_path = "/work/users/r/p/rphadke/JSALT/eval/ablation_sequential_uttwise/manifests/supervisions.jsonl.gz"
generated_wav_dir = "/work/users/r/p/rphadke/JSALT/eval/ablation_sequential_uttwise/generated_chunks"
concat_generations_dir = "/work/users/r/p/rphadke/JSALT/eval/ablation_sequential_uttwise/concatenated_generations"
output_recordings_path = "/work/users/r/p/rphadke/JSALT/eval/ablation_sequential_uttwise/manifests/recordings.jsonl.gz"

# Load old supervision segments
old_supervisions = SupervisionSet.from_file(old_supervisions_path)

# Sort by ID to match order (e.g., fe_03_00001-000.wav, -001.wav, etc.)
old_segments = sorted(old_supervisions, key=lambda s: s.id)


# Helper to get duration from wav
def get_duration(wav_path):
    info = torchaudio.info(wav_path)
    return info.num_frames / info.sample_rate


new_segments = []
start_time = 0.0
prev_rec_id = None

recording_cutoff = "fe_03_00501"  # Stop after this

for old_segment in old_segments:
    utt_id = old_segment.id
    speaker = old_segment.speaker
    text = old_segment.text
    recording_id = old_segment.recording_id  # Or you can map it to a conversation ID instead
    recording_id = recording_id.split("-")[0]

    if recording_id != prev_rec_id:
        start_time = 0.0
        prev_rec_id = recording_id


    # Early stopping condition
    if recording_id >= recording_cutoff:
        print(f"Stopping early: reached recording ID {recording_id} beyond cutoff {recording_cutoff}.")
        break

    print(f"Processing {utt_id}")
    wav_path = os.path.join(generated_wav_dir, f"{utt_id}.wav")
    if not os.path.exists(wav_path):
        print(f"Warning: {wav_path} not found, skipping.")
        continue

    duration = get_duration(wav_path)

    segment = SupervisionSegment(
        id=utt_id,
        recording_id=recording_id,
        start=start_time,
        duration=duration,
        channel=0,
        text=text,
        speaker=speaker,
        language=old_segment.language
    )

    new_segments.append(segment)
    start_time += duration  # + optional_pause if you want padding

r_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/recordings.jsonl.gz"
recordings = RecordingSet.from_jsonl(r_man)
recording_ids = []
max_conversations = 500
for rec in recordings.recordings:
    conv_id = rec.id.rsplit("-", 1)[0]
    if conv_id not in recording_ids:
        recording_ids.append(conv_id)
    if max_conversations and len(recording_ids) >= max_conversations:
        break

print(len(recording_ids))

new_recordings = []
for rid in recording_ids:
    path = f"/work/users/r/p/rphadke/JSALT/eval/ablation_sequential_uttwise/concatenated_generations/{rid}.wav"
    recording = Recording.from_file(str(path), recording_id=rid)
    new_recordings.append(recording)

# # Create and save SupervisionSet
new_supervisions = SupervisionSet.from_segments(new_segments)
new_supervisions.to_file(output_supervisions_path)

new_recordings = RecordingSet.from_recordings(new_recordings)
new_recordings.to_file(output_recordings_path)
