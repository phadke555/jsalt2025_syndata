import os
import json
import random
import torchaudio
from lhotse import RecordingSet, SupervisionSet, Recording, SupervisionSegment

# Paths (change to your paths)
old_wav_dir = "/work/users/r/p/rphadke/JSALT/fisher/fisher_wavs"
old_supervisions_path = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/supervisions.jsonl.gz"
old_recordings_path = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/recordings.jsonl.gz"
new_data_dir = "/work/users/r/p/rphadke/data/fisher"
new_wav_dir = os.path.join(new_data_dir, "wavs")
new_manifest_dir = os.path.join(new_data_dir, "manifests")

os.makedirs(new_wav_dir, exist_ok=True)
os.makedirs(new_manifest_dir, exist_ok=True)

# Read old manifests
old_recordings = RecordingSet.from_file(old_recordings_path)
old_supervisions = SupervisionSet.from_file(old_supervisions_path)

# Map recordings for quick access
recording_map = {}
for rec in old_recordings:
    rec_id = rec.id
    recording_map[rec_id] = rec

# Split unique conversation IDs into train/dev/test
unique_ids = list(set([rec_id.rsplit("-", 1)[0] for rec_id in recording_map.keys()]))
unique_ids.sort()
unique_ids = unique_ids[:500]
# random.seed(42)  # For reproducibility
# random.shuffle(unique_ids)

n_total = len(unique_ids)
n_train = int(1.0 * n_total)
n_dev = int(0 * n_total)

train_ids = set(unique_ids[:n_train])
dev_ids = set(unique_ids[n_train:n_train + n_dev])
test_ids = set(unique_ids[n_train + n_dev:])

print(f"Total: {n_total} | Train: {len(train_ids)} | Dev: {len(dev_ids)} | Test: {len(test_ids)}")

# Initialize split dictionaries
splits = {
    "train": {"recordings": [], "supervisions": []},
    "dev": {"recordings": [], "supervisions": []},
    "test": {"recordings": [], "supervisions": []}
}

# Process each conversation
for base_id in unique_ids:
    print(f"Processing {base_id}")

    wav_a_path = os.path.join(old_wav_dir, f"{base_id}-A.wav")
    wav_b_path = os.path.join(old_wav_dir, f"{base_id}-B.wav")
    wav_mix_path = os.path.join(new_wav_dir, f"{base_id}.wav")

    # # Load and mix audio
    # new_freq=16000
    # audio_a, sr_a = torchaudio.load(wav_a_path)
    # audio_a = torchaudio.transforms.Resample(orig_freq=sr_a, new_freq=new_freq)(audio_a)
    # audio_b, sr_b = torchaudio.load(wav_b_path)
    # audio_b = torchaudio.transforms.Resample(orig_freq=sr_b, new_freq=new_freq)(audio_b)
    # assert sr_a == sr_b, f"Sample rates differ for {base_id}"

    # min_len = min(audio_a.shape[1], audio_b.shape[1])
    # mixture = audio_a[:, :min_len] + audio_b[:, :min_len]

    # # Save mixed audio
    # torchaudio.save(wav_mix_path, mixture, new_freq)

    # Create Lhotse Recording
    recording = Recording.from_file(wav_mix_path, recording_id=base_id)

    # Determine which split this base_id belongs to
    if base_id in train_ids:
        split = "train"
    elif base_id in dev_ids:
        split = "dev"
    else:
        split = "test"

    splits[split]["recordings"].append(recording)

    # Adjust and add supervisions
    for sup in old_supervisions:
        if sup.recording_id.startswith(base_id):
            new_sup = SupervisionSegment(
                id=sup.id,
                recording_id=base_id,
                start=sup.start,
                duration=sup.duration,
                channel=0,
                text=sup.text,
                language=sup.language,
                speaker=sup.speaker
            )
            splits[split]["supervisions"].append(new_sup)

# Save split manifests
for split_name, data in splits.items():
    print(f"Saving {split_name} manifests with {len(data['recordings'])} recordings")
    rec_set = RecordingSet.from_recordings(data["recordings"])
    sup_set = SupervisionSet.from_segments(data["supervisions"])

    rec_set.to_file(os.path.join(new_manifest_dir, f"fisher_mix500_recordings_{split_name}.jsonl.gz"))
    sup_set.to_file(os.path.join(new_manifest_dir, f"fisher_mix500_supervisions_{split_name}.jsonl.gz"))

print("✅ Finished creating train/dev/test manifests and mixed WAVs.")
