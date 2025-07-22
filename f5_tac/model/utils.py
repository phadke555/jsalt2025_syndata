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
                idxs.append(-1)
                i += 5  # Skip over "<sil>"
            else:
                idxs.append(vocab_char_map.get(t[i], 0))
                i += 1
        list_idx_tensors.append(torch.tensor(idxs))
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text