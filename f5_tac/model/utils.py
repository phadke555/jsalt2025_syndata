import torch
from collections import OrderedDict
import numpy as np

def get_sa(
        supervisions,
        segment=20,
        max_spk=2,
        resolution=160,
        overlap_f=0,
        fs=24000,
        order_by_most_active=False
    ):

        duration = segment * fs
        n_frames = int(np.ceil((duration - overlap_f * resolution) / resolution))
        if len(supervisions) == 0:
            return torch.zeros((max_spk, n_frames), dtype=torch.bool)

        if order_by_most_active:
            spk2duration = {}
            for x in supervisions:
                if x.speaker not in spk2duration:
                    spk2duration[x.speaker] = 0
                spk2duration[x.speaker] += x.duration

            # Sort speakers by total duration (most active first)
            sorted_speakers = sorted(spk2duration.keys(), key=lambda spk: spk2duration[spk], reverse=True)
            # Create index mapping
            spk2indx = {spk: idx for idx, spk in enumerate(sorted_speakers)}
        else:
            spk2indx = list(OrderedDict.fromkeys([x.speaker for x in supervisions]))
            spk2indx = {k: indx for indx, k in enumerate(spk2indx)}

        sa = torch.zeros((len(spk2indx.keys()), n_frames), dtype=torch.bool)

        for utt in supervisions:
            start = max(utt.start, 0.0)
            stop = min(utt.start + utt.duration, segment)
            start = int(start / (resolution / fs))
            stop = int(stop / (resolution / fs))
            sa[spk2indx[utt.speaker], start:stop] = 1

        if len(sa) < max_spk:
            # pad to max local speaker dimension.
            sa = torch.nn.functional.pad(
                sa, (0, 0, 0, max_spk - len(sa)), mode="constant", value=0
            )
        return sa

from lhotse.cut import Cut
def align_text_to_frames_from_cut(
    cut: Cut,
    hop_length: int,
    pad_token: str = " ",
    boundary_token: str = "-",
) -> list[str]:
    """
    Given a Lhotse Cut, produce a list of length n_frames,
    where each element is the character (from any supervision
    covering that frame) or pad_token if silent.
    """
    fs = cut.sampling_rate
    segment = cut.duration  # in seconds

    # Total frames in this cut
    n_frames = int(np.ceil((segment * fs) / hop_length))
    char_seq = [pad_token] * n_frames

    # Seconds per frame
    frame_shift = hop_length / fs

    for sup in cut.supervisions:
        # sup.start and sup.duration are already relative to cut
        start = max(0.0, sup.start)
        stop  = min(segment, sup.start + sup.duration)
        if stop <= start:
            continue

        s_frame = int(start / frame_shift)
        e_frame = int(stop  / frame_shift)
        n_utt_frames = e_frame - s_frame
        if n_utt_frames <= 0:
            continue

        # Break the supervision text into characters
        chars = list(sup.text)
        n_chars = len(chars)
        if n_chars == 0:
            continue

        for i in range(n_utt_frames):
            char_idx = int(i * n_chars / n_utt_frames)
            char_seq[s_frame + i] = chars[char_idx]

        # now mark the boundary frame (if it exists)
        if e_frame < n_frames:
            char_seq[e_frame] = boundary_token

    return char_seq



def align_text_once_to_frames_from_cut(
    cut: Cut,
    hop_length: int,
    pad_token: str = " ",
    boundary_token: str = "<utt>",
) -> list[str]:
    """
    Given a Lhotse Cut, produce a list of length n_frames,
    where each character in each supervision is placed on exactly
    one frame, in order, and any leftover frames in that utterance
    span are filled with pad_token. Optionally mark the boundary
    frame with boundary_token.
    """
    sr = cut.sampling_rate
    segment = cut.duration   # seconds
    n_frames = int(np.ceil((segment * sr) / hop_length))
    seq = [pad_token] * n_frames

    frame_shift = hop_length / sr  # seconds per frame

    for sup in cut.supervisions:
        # Clip utterance to [0, segment]
        start = max(0.0, sup.start)
        stop  = min(segment, sup.start + sup.duration)
        if stop <= start:
            continue

        s_f = int(start / frame_shift)
        e_f = int(stop  / frame_shift)
        span = e_f - s_f
        if span <= 0:
            continue

        chars = list(sup.text)
        # Place each character on its own frame, up to the span
        for i, ch in enumerate(chars):
            if i >= span:
                break
            seq[s_f + i] = ch

        # Any remaining frames in [s_f+len(chars) : e_f) stay as pad_token

        # Optionally mark the exact boundary frame
        if boundary_token is not None and e_f < n_frames:
            seq[e_f] = boundary_token

    return seq


import torch
from torch.nn.utils.rnn import pad_sequence

def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
):
    list_idx_tensors = []
    for t in text:
        idxs = []
        i = 0
        while i < len(t):
            if t[i : i + 5] == "<utt>":  # Detect "<utt>"
                idxs.append(vocab_char_map.get("<utt>", 0))
                i += 5  # Skip over "<utt>"
            elif t[i : i + 5] == "<sil>":
                idxs.append(vocab_char_map.get("<sil>", 0))
                i += 5  # Skip over "<sil>"
            else:
                idxs.append(vocab_char_map.get(t[i], 0))
                i += 1
        list_idx_tensors.append(torch.tensor(idxs))
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text