import torch
import torchaudio
import json
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as Dataset_
from f5_tts.model.utils import default

class FisherDataset(Dataset):
    def __init__(
        self,
        fisher_dataset: Dataset,
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
        self.data = fisher_dataset
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
        while True:
            try:
                item = self.data[index]
                duration = self.durations[index]

                if not (0.3 <= duration <= 30.0):
                    index = (index + 1) % len(self)
                    continue

                if self.preprocessed_mel:
                    # Assuming precomputed mels are stored in the arrow file
                    mel_A = torch.tensor(item["mel_A"])
                    mel_B = torch.tensor(item["mel_B"])
                else:
                    mel_A = self._process_audio(item['path_A'])
                    mel_B = self._process_audio(item['path_B'])

                text_A = item['text_A']
                text_B = item['text_B']
                
                return {
                    "mel_A": mel_A,
                    "text_A": text_A,
                    "mel_B": mel_B,
                    "text_B": text_B,
                }

            except Exception as e:
                print(f"Warning: Skipping index {index} due to error: {e}")
                index = (index + 1) % len(self)

def load_conversation_dataset(
    dataset_path: str,
    mel_spec_kwargs: dict = None,
) -> FisherDataset:
    """
    Loads a pre-processed conversation dataset from disk.
    For Custom Dataset Path
    """
    print("Loading conversation dataset...")
    raw_path = os.path.join(dataset_path, "raw")
    duration_path = os.path.join(dataset_path, "duration.json")
    
    try:
        train_dataset = load_from_disk(raw_path)
    except FileNotFoundError:
        train_dataset = Dataset_.from_file(f"{raw_path}.arrow")

    with open(duration_path, "r", encoding="utf-8") as f:
        durations = json.load(f)["duration"]
        
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