import pandas as pd
from pathlib import Path
import pyarrow as pa
from datasets.arrow_writer import ArrowWriter
from datasets import Features, Value

#fs06/rphadke1/data/fisher_chunks_0.1K_v2.1
# fs06/rphadke1/data/fisher_chunks_0.1K_v3.0

output_dir   = Path("/export/fs06/rphadke1/data/fisher_chunks_v2.1")
metadata_path = output_dir / "metadata.csv"
arrow_path   = output_dir / "raw.arrow"

# 2. Load & fix the CSV
old_prefix = "/work/users/r/p/rphadke/JSALT"
new_prefix = "/export/fs06/rphadke1/data"

df = pd.read_csv(metadata_path)
df["speaker_A_wav"] = df["speaker_A_wav"].str.replace(
    old_prefix, new_prefix, regex=False
)
df["speaker_B_wav"] = df["speaker_B_wav"].str.replace(
    old_prefix, new_prefix, regex=False
)

# 3. (Optional) inspect / save a fixed CSV
df.to_csv(output_dir / "metadata.csv", index=False)

features = Features({
    "recording_id": Value("string"),
    "speaker_A_wav": Value("string"),
    "speaker_A_text": Value("string"),
    "speaker_B_wav": Value("string"),
    "speaker_B_text": Value("string"),
})

# 4. Write out raw.arrow with ArrowWriter
print(f"Writing {len(df)} records to {arrow_path} …")
with ArrowWriter(
    path=str(arrow_path),
    writer_batch_size=1,  # flush on every example; bump this up for speed if you like
    features=features  # 🔥 Force schema
) as writer:
    for record in df.to_dict(orient="records"):
        writer.write(record)

print("✅ Finished writing raw.arrow")