from __future__ import annotations
from lhotse import RecordingSet, SupervisionSet, CutSet
from lhotse.cut import Cut
from typing import Dict, Optional, Iterable, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default
from pathlib import Path

recordings_manifest = "/export/fs06/rphadke1/data/fisher/lhotse_manifests/fixed/recordings.jsonl.gz"
supervisions_manifest = "/export/fs06/rphadke1/data/fisher/lhotse_manifests/fixed/supervisions.jsonl.gz"

def build_cutset(recordings_manifest, supervisions_manifest, max_conversations, conversation_offset):
    def _base_id(rec_or_cut_id: str) -> str:
        """Strip trailing '-A' or '-B' to get the conversation id."""
        if "-A" in rec_or_cut_id or "-B" in rec_or_cut_id:
            return rec_or_cut_id.split("-")[0]
        return rec_or_cut_id

    def sample_cuts(cuts, recordings, max_conversations, conversation_offset):
        unique_conv_ids: List[str] = []
        for rec in recordings.recordings:
            conv_id = rec.id.rsplit("-", 1)[0]
            if conv_id not in unique_conv_ids:
                unique_conv_ids.append(conv_id)
        if max_conversations is not None:
            start = conversation_offset
            end = conversation_offset + max_conversations
            unique_conv_ids = unique_conv_ids[start:end]
            cuts = cuts.filter(lambda c: c.recording_id.rsplit("-", 1)[0] in unique_conv_ids)
        return cuts

    # import pdb; pdb.set_trace()
    # 1) Load
    recs = RecordingSet.from_jsonl(recordings_manifest)
    sups = SupervisionSet.from_jsonl(supervisions_manifest)

    # 2) Split A/B by id suffix
    recs_A = RecordingSet.from_recordings(r for r in recs if r.id.endswith("-A"))
    recs_B = RecordingSet.from_recordings(r for r in recs if r.id.endswith("-B"))
    sups_A = sups.filter(lambda s: s.recording_id.endswith("-A"))
    sups_B = sups.filter(lambda s: s.recording_id.endswith("-B"))

    # 3) Make *recording-spanning* cuts that carry all supervisions
    cuts_A = CutSet.from_manifests(
        recordings=recs_A, supervisions=sups_A
    )
    cuts_A = sample_cuts(cuts_A, recs_A, max_conversations, conversation_offset)
    cuts_B = CutSet.from_manifests(
        recordings=recs_B, supervisions=sups_B
    )
    cuts_B = sample_cuts(cuts_B, recs_B, max_conversations, conversation_offset)

    # Index by base conversation id (fe_03_00001)
    a_by_base: Dict[str, Cut] = {_base_id(c.id): c for c in cuts_A}
    b_by_base: Dict[str, Cut] = {_base_id(c.id): c for c in cuts_B}

    shared = sorted(set(a_by_base) & set(b_by_base))
    if not shared:
        # Fallback: include whichever side exists
        shared = sorted(set(a_by_base) | set(b_by_base))

    drop_unpaired = True
    mixed: list[Cut] = []
    for base in shared:
        a = a_by_base.get(base)
        b = b_by_base.get(base)
        if a is None or b is None:
            if drop_unpaired:
                continue
            m = a or b  # single side only
            m = m.with_id(base)  # normalize id
        else:
            # 4) Mix A+B (this *adds/sums* waveforms; default offsets=0)
            m = a.mix(b).with_id(base)
        mixed.append(m)

    mono_mixed = CutSet.from_cuts(mixed).resample(24000).cut_into_windows(duration=20.0)
    return mono_mixed


class LhotseDataset(Dataset):
    """
    For a CutSet of 20s MixedCuts (A+B), returns:
        {
            "mel_spec": Tensor [n_mels, n_frames],
            "text":     str,   # merged dialog with [speakerA]/[speakerB] tokens only on speaker changes
        }
    """
    def __init__(
        self,
        cuts: CutSet,
        durations=None,
        target_sample_rate=24000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        # text-generation knobs
        speakerA_token: str = "[speakerA]",
        speakerB_token: str = "[speakerB]",
        drop_if_no_supervisions: bool = False,
    ):
        super().__init__()
        self.cuts = cuts

        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type

        # Mel front-end
        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )
        else:
            self.mel_spectrogram = mel_spec_module

        self.durations = [c.duration for c in cuts]

        self.spkA = speakerA_token
        self.spkB = speakerB_token
        self.drop_if_no_supervisions = drop_if_no_supervisions

    # ---------- audio helpers ----------

    def __len__(self) -> int:
        return len(self.cuts.to_eager())
    
    def get_frame_len(self, index: int) -> float:
        # Keep same semantics as your old dataset
        return self.durations[index] * self.target_sample_rate / self.hop_length

    # ---------- dialog text builder ----------

    @staticmethod
    def _side_from_recording_id(rec_id: str) -> Optional[str]:
        if rec_id.endswith("-A"):
            return "A"
        if rec_id.endswith("-B"):
            return "B"
        return None  # fallback if needed

    def _window_dialog_text(self, cut: Cut) -> str:
        """
        Merge neighboring utterances from the same side.
        Insert [speakerA] for first speaker, and [speakerB] (or [speakerA]) only when side changes.
        """
        sups = list(cut.supervisions)
        if not sups:
            return ""

        # Sort by start time; tie-break by side so A is stable before B if equal start
        def side_rank(s):
            side = self._side_from_recording_id(s.recording_id)
            return 0 if side == "A" else 1

        sups.sort(key=lambda s: (s.start, side_rank(s)))

        parts: List[str] = []
        prev_side: Optional[str] = None
        buffer: List[str] = []

        def flush():
            nonlocal buffer
            if buffer:
                parts.append(" ".join(buffer))
                buffer = []

        for s in sups:
            side = self._side_from_recording_id(s.recording_id)
            token = self.spkA if side == "A" else self.spkB if side == "B" else None

            # On speaker change, flush previous text and emit the new speaker token.
            if prev_side is None or side != prev_side:
                flush()
                if token is not None:
                    parts.append(token)
                prev_side = side

            # Accumulate neighboring text for the same speaker
            txt = (s.text or "").strip()
            if txt:
                buffer.append(txt)

        flush()
        # Join everything with single spaces
        out = " ".join(p for p in parts if p)
        # Optionally collapse any double spaces
        out = " ".join(out.split())
        return out
    
    def _wav_to_mel(self, wav_np) -> torch.Tensor:
        wav = torch.from_numpy(wav_np)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]
        mel = self.mel_spectrogram(wav)  # -> [1, n_mels, frames] (implementation-dependent)
        return mel.squeeze(0)           # -> [n_mels, frames]

    # ---------- main fetch ----------

    def __getitem__(self, idx: int):
        # import pdb; pdb.set_trace()
        cut = self.cuts[idx]

        # 1) Build dialog text for this 20s window
        text = self._window_dialog_text(cut)
        if self.drop_if_no_supervisions and len(text) == 0:
            # If you prefer to never return empty text windows, pick the next non-empty one.
            # Simple linear probe; you can tailor this behavior or raise instead.
            j = (idx + 1) % len(self)
            while j != idx:
                alt_cut = self.cuts[j]
                alt_text = self._window_dialog_text(alt_cut)
                if alt_text:
                    cut, text = alt_cut, alt_text
                    break
                j = (j + 1) % len(self)

        # 2) Load mixed mono audio
        wav = cut.load_audio()
        mel = self._wav_to_mel(wav)  # [1, n_mels, n_frames]
        return {"mel_spec": mel, "text": text}


from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from torch.utils.data import DataLoader
def load_lhotse_dataset(
    dataset_path,
    max_conversations=5,
    conversation_offset=0
):
    recordings_manifest = Path(dataset_path) / "recordings.jsonl.gz"
    supervisions_manifest = Path(dataset_path) / "supervisions.jsonl.gz"
    cuts = build_cutset(recordings_manifest, supervisions_manifest, max_conversations, conversation_offset)
    return LhotseDataset(
            cuts=cuts, 
            speakerA_token="[speakerA]",
            speakerB_token="[speakerB]",
            drop_if_no_supervisions=True,  
        )
