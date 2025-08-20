import os
import torch
import torchaudio
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from f5_tts.model.utils import default
from f5_tts.model.modules import MelSpec
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from lhotse import RecordingSet, SupervisionSet, CutSet
from pathlib import Path

def build_spkchange_texts(cut_A, cut_B) -> tuple[str, str]:
    """
    Merge A/B supervisions by their (cut-relative) start times and create two strings:
    - text_A: A's utterances concatenated; append [spkchange] only when the next utterance is by B
    - text_B: B's utterances concatenated; append [spkchange] only when the next utterance is by A
    """
    # Collect (speaker, start_time, text)
    merged = []
    for s in cut_A.supervisions:
        merged.append(("A", s.start, s.text))
    for s in cut_B.supervisions:
        merged.append(("B", s.start, s.text))

    # Sort by time; ties keep input order
    merged.sort(key=lambda x: x[1])

    a_parts, b_parts = [], []
    for i, (spk, _, txt) in enumerate(merged):
        next_spk = merged[i + 1][0] if i + 1 < len(merged) else None

        if spk == "A":
            if txt:
                a_parts.append(txt)
            # Add boundary token only if the *next* utterance is from the other speaker
            if next_spk == "B":
                a_parts.append("[spkchange]")
        else:  # spk == "B"
            if txt:
                b_parts.append(txt)
            if next_spk == "A":
                b_parts.append("[spkchange]")

    # Join with single spaces; no trailing token if last utterance has no change after it
    text_A = " ".join(a_parts).strip()
    text_B = " ".join(b_parts).strip()
    return text_A, text_B

class LhotseFisherDataset(Dataset):
    """
    Drop-in replacement for your FisherDataset that *directly* loads from Lhotse manifests.
    Returns dicts with the SAME keys:
        { "mel_A", "text_A", "mel_B", "text_B" }

    It pairs A/B windows lazily, computes mels on the fly (or loads features if available),
    and inserts [spkchange] only when the speaker actually changes.

    NOTE: If you precompute Lhotse features (e.g., mels), set `preprocessed_mel=True`
    and make sure both A/B cuts have features.
    """

    def __init__(
        self,
        recordings_jsonl: Path | str,
        supervisions_jsonl: Path | str,
        *,
        chunk_length: float = 20.0,
        min_chunk_length: float = 1.0,
        target_sample_rate: int = 24000,
        max_conversations: int | None = None,
        conversation_offset: int = 0,  # like your slice [5000:...]
        hop_length: int = 256,
        n_mel_channels: int = 100,
        n_fft: int = 1024,
        win_length: int = 1024,
        mel_spec_type: str = "vocos",
        preprocessed_mel: bool = False,  # if your Lhotse cuts already have features
        mel_spec_module: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.preprocessed_mel = preprocessed_mel

        # Build CutSet pipeline (all lazy until load_audio/features())
        recordings = RecordingSet.from_jsonl(str(recordings_jsonl))
        supervisions = SupervisionSet.from_jsonl(str(supervisions_jsonl))
        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        cuts = cuts.resample(target_sample_rate)
        cuts = cuts.cut_into_windows(duration=chunk_length)
        cuts = cuts.filter(lambda c: c.duration >= min_chunk_length)

        # Limit to first N conversation IDs (optional/testing)
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

        # We need to pair A/B cuts per (conversation, window-idx)
        # Eagerize so we can iterate and store references
        cuts = cuts.to_eager()

        grouped: Dict[Tuple[str, int], Dict[str, object]] = defaultdict(dict)
        for cut in cuts:
            conv_id, speaker = cut.recording_id.rsplit("-", 1)
            window_idx = int(cut.start // chunk_length)
            grouped[(conv_id, window_idx)][speaker] = cut

        # Build a list of valid (A_cut, B_cut) pairs and their duration
        self.pairs: List[Tuple[object, object]] = []
        self.durations: List[float] = []
        self.text_cache: List[Tuple[str, str]] = []  # optional small cache

        for (conv_id, idx), parts in sorted(grouped.items()):
            if "A" not in parts or "B" not in parts:
                # Skip incomplete windows
                continue
            cut_A, cut_B = parts["A"], parts["B"]
            self.pairs.append((cut_A, cut_B))
            # they should be identical (same window)
            self.durations.append(cut_A.duration)

            # Prebuild texts to keep __getitem__ fast & deterministic
            tA, tB = build_spkchange_texts(cut_A, cut_B)
            self.text_cache.append((tA, tB))

        # Mel front-end
        if not preprocessed_mel:
            self.mel_spectrogram = MelSpec(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def get_frame_len(self, index: int) -> float:
        # Keep same semantics as your old dataset
        return self.durations[index] * self.target_sample_rate / self.hop_length

    def _wav_to_mel(self, wav_np) -> torch.Tensor:
        wav = torch.from_numpy(wav_np)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]
        mel = self.mel_spectrogram(wav)  # -> [1, n_mels, frames] (implementation-dependent)
        return mel.squeeze(0)           # -> [n_mels, frames]

    def __getitem__(self, index: int) -> dict:
        cut_A, cut_B = self.pairs[index]
        text_A, text_B = self.text_cache[index]

        if self.preprocessed_mel:
            # Requires Lhotse features present on cuts
            if not (cut_A.has_features and cut_B.has_features):
                raise RuntimeError(
                    "preprocessed_mel=True, but cuts do not have features. "
                    "Either precompute Lhotse features or set preprocessed_mel=False."
                )
            mel_A = torch.from_numpy(cut_A.load_features())
            mel_B = torch.from_numpy(cut_B.load_features())
        else:
            wav_A = cut_A.load_audio()  # np.ndarray [C, T] (resampled lazily)
            wav_B = cut_B.load_audio()
            mel_A = self._wav_to_mel(wav_A)
            mel_B = self._wav_to_mel(wav_B)

        return {
            "mel_A": mel_A,
            "text_A": text_A,
            "mel_B": mel_B,
            "text_B": text_B,
        }

class FisherDataset(Dataset):
    def __init__(
        self,
        train_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = train_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

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

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def _process_audio(self, path):
        wav, sr = torchaudio.load(path)
        mel = self.mel_spectrogram(wav)
        return mel.squeeze(0)

    def __getitem__(self, index: int) -> dict:
        """
        Loads, preprocesses audio, and returns a single conversation pair with raw text.
        """
        item = self.data[index]

        if self.preprocessed_mel:
            mel_A = torch.tensor(item["mel_A"])
            mel_B = torch.tensor(item["mel_B"])
        else:
            mel_A = self._process_audio(item["speaker_A_wav"])
            mel_B = self._process_audio(item["speaker_B_wav"])

        return {
            "mel_A": mel_A,
            "text_A": item["speaker_A_text"],
            "mel_B": mel_B,
            "text_B": item["speaker_B_text"],
        }

def load_lhotse_dataset(
    dataset_path: str | Path | None = None,
    *,
    recordings_jsonl: str | Path | None = None,
    supervisions_jsonl: str | Path | None = None,
    chunk_length: float = 20.0,
    min_chunk_length: float = 1.0,
    max_conversations: int | None = None,
    conversation_offset: int = 0,
    preprocessed_mel: bool = False,         # set True only if Lhotse features are present on cuts
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: Optional[Dict] = None,
):
    # Resolve manifest paths
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        recs = dataset_path / "recordings.jsonl.gz"
        sups = dataset_path / "supervisions.jsonl.gz"
        if not recs.is_file() or not sups.is_file():
            raise FileNotFoundError(
                f"Expected {dataset_path}/recordings.jsonl.gz and {dataset_path}/supervisions.jsonl.gz"
            )
        
    else:
        if recordings_jsonl is None or supervisions_jsonl is None:
            raise ValueError(
                "Provide either `dataset_path` or both `recordings_jsonl` and `supervisions_jsonl`."
            )
        recs = Path(recordings_jsonl)
        sups = Path(supervisions_jsonl)
        if not recs.is_file():
            raise FileNotFoundError(f"recordings_jsonl not found: {recs}")
        if not sups.is_file():
            raise FileNotFoundError(f"supervisions_jsonl not found: {sups}")

    return LhotseFisherDataset(
        recordings_jsonl=str(recs),
        supervisions_jsonl=str(sups),
        chunk_length=chunk_length,
        min_chunk_length=min_chunk_length,
        max_conversations=max_conversations,        # or an int for faster dev cycles
        conversation_offset=conversation_offset,          # if you want to skip the first N convs
        **mel_spec_kwargs
    )


def load_conversation_dataset(
    dataset_path: str,
    tokenizer: str = "custom",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = None,
) -> FisherDataset:
    """
    Loads a pre-processed conversation dataset from disk.
    For Custom Dataset Path
    """
    print("Loading conversation dataset...")
    raw_arrow = os.path.join(dataset_path, "raw.arrow")
    duration_path = os.path.join(dataset_path, "duration.json")
    
    # Try loading a HuggingFace dataset directory first; otherwise read the raw.arrow directly:
    if os.path.isdir(os.path.join(dataset_path, "raw")):
        train_dataset = load_from_disk(os.path.join(dataset_path, "raw"))
    else:
        train_dataset = Dataset_.from_file(raw_arrow)

    with open(duration_path, "r", encoding="utf-8") as f:
        durations = json.load(f)["durations"]
        
    mel_spec_module = MelSpec(**mel_spec_kwargs)

    return FisherDataset(
        train_dataset=train_dataset,
        durations=durations,
        hop_length=mel_spec_kwargs.get("hop_length"),
        target_sample_rate=mel_spec_kwargs.get("target_sample_rate"),
        mel_spec_module=mel_spec_module
    )

def conversation_collate_fn(batch: list[dict]) -> dict:
    """
    A standalone collate function that batches paired conversation data.
    It manually pads mel spectrograms and returns raw text, mirroring the
    structure of the user-provided single-speaker collate function.
    Tokenization is explicitly NOT handled here.
    """
    # --- Process Speaker A ---
    mel_specs_A = [item["mel_A"] for item in batch]
    mel_lengths_A = torch.LongTensor([spec.shape[-1] for spec in mel_specs_A])
    
    # --- Process Speaker B ---
    mel_specs_B = [item["mel_B"] for item in batch]
    mel_lengths_B = torch.LongTensor([spec.shape[-1] for spec in mel_specs_B])

    # Since wavs are the same length, max length will be the same
    max_mel_length = mel_lengths_A.amax()

    # --- Pad Mel Spectrograms Manually ---
    padded_mel_specs_A = []
    for spec in mel_specs_A:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs_A.append(padded_spec)
    
    padded_mel_specs_B = []
    for spec in mel_specs_B:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs_B.append(padded_spec)

    final_mel_specs_A = torch.stack(padded_mel_specs_A)
    final_mel_specs_B = torch.stack(padded_mel_specs_B)

    # --- Process Raw Text ---
    texts_A = [item["text_A"] for item in batch]
    text_lengths_A = torch.LongTensor([len(text) for text in texts_A])

    texts_B = [item["text_B"] for item in batch]
    text_lengths_B = torch.LongTensor([len(text) for text in texts_B])

    return {
        "mel_A": final_mel_specs_A,
        "mel_lengths_A": mel_lengths_A,
        "text_A": texts_A,
        "text_lengths_A": text_lengths_A,
        "mel_B": final_mel_specs_B,
        "mel_lengths_B": mel_lengths_B,
        "text_B": texts_B,
        "text_lengths_B": text_lengths_B,
    }