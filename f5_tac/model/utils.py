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
            if t[i : i + 11] == "[spkchange]":  # Detect "<utt>"
                idxs.append(vocab_char_map.get("[spkchange]", 0))
                i += 11  # Skip over "<utt>"
            elif t[i : i + 9] == "[silence]":
                idxs.append(-1)
                i += 9  # Skip over "<sil>"
            else:
                idxs.append(vocab_char_map.get(t[i], 0))
                i += 1
        list_idx_tensors.append(torch.tensor(idxs))
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
def load_whisper_pipeline(device, model_id="openai/whisper-large-v3-turbo"):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=1,  # batch size for inference - set based on your device
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=device,
    )
    return pipe

import re
def normalize_text(s: str):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s