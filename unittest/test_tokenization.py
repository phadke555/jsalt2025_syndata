import torch
from torch.nn.utils.rnn import pad_sequence
from f5_tts.model.utils import get_tokenizer

# --- 2. Setup Tokenizer ---
tokenizer_path = "/export/fs06/rphadke1/data/fisher_chunks_0.1K_v2.1/vocab.txt"
vocab_char_map, vocab_size = get_tokenizer(tokenizer_path,"custom")
print(f"Loaded tokenizer with vocab size: {vocab_size}")

# def list_str_to_idx(
#     text: list[str] | list[list[str]],
#     vocab_char_map: dict[str, int],  # {char: idx}
#     padding_value=-1,
# ):
#     list_idx_tensors = []
#     for t in text:
#         idxs = []
#         i = 0
#         while i < len(t):
#             if t[i : i + 5] == "<utt>":  # Detect "<utt>"
#                 idxs.append(vocab_char_map.get("<utt>", 0))
#                 i += 5  # Skip over "<utt>"
#             if t[i : i + 5] == "<sil>":  # Detect "<utt>"
#                 idxs.append(-1)
#                 i += 5  # Skip over "<utt>"
#             else:
#                 idxs.append(vocab_char_map.get(t[i], 0))
#                 i += 1
#         list_idx_tensors.append(torch.tensor(idxs))
#     text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
#     return text

# text = "and i generally<sil><sil> prefer <utt>  eating at home <utt>  hello andy <utt>  how are you <utt>  good <utt>  do you have any idea what's going on <utt> "
# print(list_str_to_idx([text], vocab_char_map))