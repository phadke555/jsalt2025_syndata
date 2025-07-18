import os
from pathlib import Path
from datetime import datetime
import gzip
import json

import torch
import torchaudio
from lhotse import RecordingSet, SupervisionSet, CutSet


# def infer_blocks_for_session(
#     session_id: str,
#     recordings_manifest: str,
#     supervisions_manifest: str
#     # vocab_file: str,
#     # ckpt_file: str,
#     # output_dir: str,
#     # device: torch.device = None,
# ):
#     """
#     For a session (e.g. fisher interaction), group contiguous speaker segments across both channels,
#     run one inference per speaker block, and save each generated WAV.
#     """
#     # Load manifests
#     print(f"Loading recordings from {recordings_manifest} …")
#     recordings = RecordingSet.from_jsonl(recordings_manifest)
#     print(f"Loading supervisions from {supervisions_manifest} …")
#     supervisions = SupervisionSet.from_jsonl(supervisions_manifest)

#     # Combine into CutSet
#     print("Combining into CutSet …")
#     cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)

#     # Filter by session prefix (before channel suffix)
#     cuts = cuts.filter(lambda cut: cut.recording.id.startswith(session_id))
#     if len(cuts) == 0:
#         raise ValueError(f"No cuts found for session {session_id}")

#     # ‘cuts’ is your CutSet filtered to this session
#     def extract_supervision_index(cut):
#         # e.g. cut.supervisions[0].id == "fe_03_00001-006"
#         idx_str = cut.supervisions[0].id.rsplit("-", 1)[-1]
#         return int(idx_str)

#     # 1. Sort purely by the supervision-ID suffix
#     cuts_ord = sorted(cuts, key=extract_supervision_index)

#     # 2. Now group on channel switches
#     groups = []
#     first_chan = cuts_ord[0].recording.id.rsplit("-", 1)[-1]  # “A” or “B”
#     current_chan = first_chan
#     current_group = [cuts_ord[0]]

#     for cut in cuts_ord[1:]:
#         chan = cut.recording.id.rsplit("-", 1)[-1]
#         if chan == current_chan:
#             current_group.append(cut)
#         else:
#             groups.append((current_chan, current_group))
#             current_chan = chan
#             current_group = [cut]
#     # final group
#     groups.append((current_chan, current_group))
#     return groups
def build_groups_from_supervisions(supervisions_manifest: str, session_id: str):
    """
    Load supervisions as Lhotse SupervisionSegment objects, sort by their numeric ID suffix,
    and group into turn-taking blocks whenever the channel (A/B) flips.
    Returns: List of (channel: str, List[SupervisionSegment])
    """
    # Load all SupervisionSegment objects
    supervisions = list(SupervisionSet.from_jsonl(supervisions_manifest))

    # Filter to this session (strip off channel suffix)
    sel = [
        s
        for s in supervisions
        if s.recording_id.split("-", 1)[0] == session_id
    ]
    if not sel:
        raise ValueError(f"No supervisions found for session {session_id}")

    # Sort by the integer suffix of each supervision's ID (e.g. "-006" → 6)
    sel_sorted = sorted(
        sel,
        key=lambda s: int(s.id.rsplit("-", 1)[1])
    )

    # Group into blocks on channel change (recording_id ends with "-A" or "-B")
    groups = []
    first_channel = sel_sorted[0].recording_id.rsplit("-", 1)[1]
    current_channel = first_channel
    current_block = [sel_sorted[0]]

    for sup in sel_sorted[1:]:
        channel = sup.recording_id.rsplit("-", 1)[1]
        if channel == current_channel:
            current_block.append(sup)
        else:
            groups.append((current_channel, current_block))
            current_channel = channel
            current_block = [sup]

    # Append the final block
    groups.append((current_channel, current_block))
    return groups


r_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/recordings.jsonl.gz"
s_man = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/supervisions.jsonl.gz"
recording_id = "fe_03_00001"

groups = build_groups_from_supervisions(s_man, recording_id)
print(len(groups))
# Display first 10 groups for unit testing / inspection
# 4. Print out the first 10 for inspection
for idx, (chan, block) in enumerate(groups[:10], start=1):
    ids    = [s.id for s in block]
    starts = [round(s.start, 2) for s in block]
    texts  = [s.text for s in block]
    print(f"Group {idx:02d}: channel={chan}, items={len(block)}")
    print(f"  IDs    = {ids}")
    print(f"  starts = {starts}")
    print(f"  texts  = {texts}\n")