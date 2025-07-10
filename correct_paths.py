import pandas as pd
from pathlib import Path
import pyarrow as pa
from datasets.arrow_writer import ArrowWriter


output_dir   = Path("/export/fs06/rphadke1/data/fisher_chunks")
metadata_path = "/export/fs06/rphadke1/data/fisher_chunks/metadata.csv"
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

# 4. Write out raw.arrow with ArrowWriter
print(f"Writing {len(df)} records to {arrow_path} …")
with ArrowWriter(
    path=str(arrow_path),
    writer_batch_size=1  # flush on every example; bump this up for speed if you like
) as writer:
    for record in df.to_dict(orient="records"):
        writer.write(record)

print("✅ Finished writing raw.arrow")