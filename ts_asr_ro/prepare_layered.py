from lhotse import SupervisionSet, Recording, RecordingSet, CutSet

# 1) Load your two per‑speaker supervision sets
sups_A = SupervisionSet.from_jsonl("/work/users/r/p/rphadke/data/simfisher-layered/manifests/supervisionsA.jsonl.gz")
sups_B = SupervisionSet.from_jsonl("/work/users/r/p/rphadke/data/simfisher-layered/manifests/supervisionsB.jsonl.gz")

# 2) Rewrite each set: add speaker label and disambiguate the ID
def tag_and_rename(sup_set: SupervisionSet, speaker: str):
    tagged = []
    for sup in sup_set:
        # attach the speaker field ("A" or "B")
        sup.speaker = speaker
        # make the ID unique by appending the speaker tag
        sup.id = f"{sup.id}_{speaker}"
        tagged.append(sup)
    return tagged

tagged_A = tag_and_rename(sups_A, "A")
tagged_B = tag_and_rename(sups_B, "B")

# 3) Combine and sort by start time
all_sups = tagged_A + tagged_B
mix_supervisions = (
    SupervisionSet.from_segments(all_sups)
)

# 4) (Optional) write out the merged set
mix_supervisions.to_file("/work/users/r/p/rphadke/data/simfisher-layered/manifests/supervisions_all_convs_mix.jsonl.gz")

from pathlib import Path
input_path = "/work/users/r/p/rphadke/data/simfisher-layered/concatenated_generations"
input_path = Path(input_path)
recordings = []
# Walk through directory recursively
for file_path in input_path.rglob("*"):
    # Skip directories
    if not file_path.is_file():
        continue

    # Use file stem as recording ID
    recording_id = file_path.stem

    # Create a Recording object
    rec = Recording.from_file(
        path=str(file_path),
        recording_id=recording_id
    )
    recordings.append(rec)

# Assemble into a RecordingSet
recording_set = RecordingSet.from_recordings(recordings)
recording_set.to_file("/work/users/r/p/rphadke/data/simfisher-layered/manifests/recordings.jsonl.gz")


cuts = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=mix_supervisions
    )

for cut in cuts:
    for sup in cut.supervisions:
        sup.duration = min(sup.duration, cut.duration - sup.start)

cuts.to_file("/work/users/r/p/rphadke/data/simfisher-layered/manifests/cutset.jsonl.gz")