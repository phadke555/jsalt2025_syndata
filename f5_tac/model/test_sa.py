import numpy as np
import torch
from lhotse import RecordingSet, SupervisionSet, CutSet
from f5_tac.model.utils import get_sa

# Import your class that has `get_sa`; adjust import as needed:
# from your_module import YourClass

def extract_window_supervisions(supervisions, window_start, window_end):
    """
    Return a list of supervisions overlapping [window_start, window_end),
    with start times shifted to be relative to window_start,
    and durations clipped to the window.
    """
    window_sups = []
    for sup in supervisions:
        sup_start = sup.start
        sup_end = sup.start + sup.duration
        # overlap?
        if sup_end > window_start and sup_start < window_end:
            # compute clipped start/duration
            new_start = max(sup_start, window_start) - window_start
            new_end   = min(sup_end,   window_end)   - window_start
            clipped = sup.with_offsets(start=new_start, duration=new_end - new_start)
            window_sups.append(clipped)
    return window_sups

def main():
    duration = 20.0
    # Paths to your manifests
    rec_manifest = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/recordings.jsonl.gz"
    sup_manifest = "/work/users/r/p/rphadke/JSALT/fisher/lhotse_manifests/fixed/supervisions.jsonl.gz"

    # Load Lhotse manifests
    recordings = RecordingSet.from_jsonl(rec_manifest)
    supervisions = SupervisionSet.from_jsonl(sup_manifest)

    # Grab the first recording's ID
    first_rec_id = recordings[0].id

    # 2. Build 20 s cuts (no overlap)
    cuts = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions
    )
    cuts = cuts.filter(lambda cut: cut.recording_id == first_rec_id)
    cuts = cuts.cut_into_windows(duration=duration)


    # 3. Mel‐spec aligned settings
    fs         = 8000    # target_sample_rate
    resolution = 256      # hop_length
    max_spk    = 2

    # 4. Iterate cuts and print masks
    for cut in cuts:
        # Lhotse supervisions are already clipped & shifted
        # into cut.start–cut.end, and cut.duration == 20 s (or shorter at end).
        sa = get_sa(
            supervisions=cut.supervisions,
            segment=cut.duration,
            max_spk=max_spk,
            resolution=resolution,
            overlap_f=0,
            fs=fs,
            order_by_most_active=True
        )
        print(f"Cut {cut.id} | duration {cut.duration:.1f}s | mask {sa.shape}")
        print(sa.int())
        print("-" * 80)

if __name__ == "__main__":
    main()
