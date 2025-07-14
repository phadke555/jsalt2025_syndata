from torch.utils.data import DataLoader, SequentialSampler
import torch
import torchaudio

from f5_tac.model.dataset import load_conversation_dataset, conversation_collate_fn
from f5_tac.configs.model_kwargs import mel_spec_kwargs
from f5_tac.infer.utils import load_model_and_vocoder

import argparse
import os

# parser = argparse.ArgumentParser()
# parser.add_argument("--data_root", type=str, required=True)
# args = parser.parse_args()

# 1) Load your eval dataset from disk
eval_dataset = load_conversation_dataset(
    dataset_path="/work/users/r/p/rphadke/JSALT/fisher_chunks_v2",
    mel_spec_kwargs=mel_spec_kwargs,
)

print("Len Eval Dataset")
print(len(eval_dataset))


# 2) Create a DataLoader for inference
eval_loader = DataLoader(
    eval_dataset,
    sampler=SequentialSampler(eval_dataset),
    batch_size=1,                     # tune to your GPU memory
    collate_fn=conversation_collate_fn,
    num_workers=8,                    # tune up to #CPU cores
    pin_memory=True,
    persistent_workers=True,
)

print("Len Dataloader")
print(len(eval_loader))

# 3) Rewrite generate_sample to accept *batch* inputs
def generate_batch(model, vocoder, mel_A, mel_B, lengths_A, lengths_B, texts_A, texts_B, device):
    """
    wavs_A: (B,1,T_A) on GPU, wavs_B: (B,1,T_B)
    texts_A/B: list[str] of length B
    Returns:
      mels_out: (B*2, n_mel, T_out)
      T_As:   Tensor[B] of original mel lengths
      T_Bs:   Tensor[B]
    """
    # compute mel specs in parallel
    B = mel_A.size(0)
    # flatten to (B*2, n_mel, T)
    conds = torch.cat([mel_A, mel_B], dim=0)
    print("conds", conds.shape)
    T_As = lengths_A
    T_Bs = lengths_B

    # prepare texts and durations
    full_texts = []
    durations  = []
    for i in range(B):
        ta, tb = len(texts_A[i]), len(texts_B[i])
        ratio_A = ta / max(ta, 1)
        ratio_B = tb / max(tb, 1)
        full_texts += [texts_A[i], texts_B[i]]
        durations  += [
            int(T_As[i] * (1.1 + ratio_A)),
            int(T_Bs[i] * (1.1 + ratio_B)),
        ]

    # single call to sample_joint
    mels_out, _ = model.sample_joint(
        texts=full_texts,
        conds=conds,
        durations=durations,
        vocoder=vocoder,
        steps=32,
        cfg_strength=1.5,
        sway_sampling_coef=-1.0,
        seed=42,
        max_duration=4096,
        use_epss=True,
    )
    return mels_out, T_As, T_Bs

def generate_sample(model, vocoder, mel_A, mel_B, text_A, text_B, device):
    """Generate TTS samples for two speakers."""
    # mel_A = model.mel_spec(wav_A)
    # mel_B = model.mel_spec(wav_B)
    conds = torch.cat([mel_A, mel_B], 0)

    T_A = mel_A.size(2)
    T_B = mel_B.size(2)

    if not isinstance(text_A, str):
        text_A = ""
    if not isinstance(text_B, str):
        text_B = ""

    gen_text_A = text_A
    # gen_text_A = ""
    gen_text_B = text_B
    # gen_text_B = ""

    full_texts = [
        text_A + gen_text_A,   
        text_B + gen_text_B,   
    ]

    ratio_A = len(text_A) / max(len(text_A), 1)
    ratio_B = len(text_B) / max(len(text_B), 1)

    durations = [
        int(T_A * (1.1 + ratio_A)),
        int(T_B * (1.1 + ratio_B))
    ]

    mels_out, _ = model.sample_joint(
        texts=full_texts,
        conds=conds,
        durations=durations,
        vocoder=vocoder,
        steps=32,
        cfg_strength=1.5,
        sway_sampling_coef=-1.0,
        seed=42,
        max_duration=4096,
        use_epss=True
    )

    return mels_out, T_A, T_B


import torch
import torch.nn.functional as F

def decode_and_mix(
    m_out: torch.Tensor,
    T_A: int,
    T_B: int,
    vocoder: torch.nn.Module
) -> torch.Tensor:
    """
    Decode generated mel-spectrograms for two speakers and mix into a mono waveform.

    Args:
        m_out: Tensor of shape (2, n_mel, T_total) containing mel outputs for A and B.
        T_A:   int, original mel length for speaker A.
        T_B:   int, original mel length for speaker B.
        vocoder: nn.Module with a `.decode()` method that maps (B, T, n_mel) -> waveform.

    Returns:
        torch.Tensor of shape (1, L) on CPU, the mixed mono audio.
    """
    # --- Speaker A ---
    gen_mel_A = m_out[0:1, T_A:, :]     # (n_mel, T_gen_A)
    # gen_mel_A = gen_mel_A.permute(0,2,1)
    if gen_mel_A.numel() == 0:
        wav_A = torch.zeros(1, 0, device="cpu")
    else:
        # Vocoder expects shape (batch, T, n_mel)
        gen_mel_A = gen_mel_A.permute(0, 2, 1)  # (1, T_gen_A, n_mel)
        wav_A = vocoder.decode(gen_mel_A).detach().cpu()    # (1, L_A)

    # --- Speaker B ---
    gen_mel_B = m_out[1:2, T_B:, :]     # (n_mel, T_gen_A)
    # gen_mel_B = gen_mel_B.permute(0,2,1)
    if gen_mel_B.numel() == 0:
        wav_B = torch.zeros(1, 0, device="cpu")
    else:
        gen_mel_B = gen_mel_B.permute(0, 2, 1)
        wav_B = vocoder.decode(gen_mel_B).detach().cpu()    # (1, L_B)

    # --- Pad to equal length and mix ---
    len_A, len_B = wav_A.shape[-1], wav_B.shape[-1]
    max_len = max(len_A, len_B)
    wav_A = F.pad(wav_A, (0, max_len - len_A))
    wav_B = F.pad(wav_B, (0, max_len - len_B))
    mono = wav_A + wav_B

    return mono


# 4) Inference loop
def run_inference_with_arrow(
    eval_loader,
    model, vocoder,
    device,
    out_dir,
    sr=24000
):
    model.eval()
    vocoder.eval()
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for batch in eval_loader:
            # Move everything to GPU in one go
            mel_A = batch["mel_A"].to(device)  # actually raw audio spectrogram input?
            mel_B = batch["mel_B"].to(device)
            # If your FisherDataset returns preprocessed mel, switch Dataset to return raw wavs
            lengths_A = batch["mel_lengths_A"]
            lengths_B = batch["mel_lengths_B"]
            texts_A = batch["text_A"]
            texts_B = batch["text_B"]
            clip_ids = batch["clip_id"]

            print(texts_A)
            print(texts_B)
            print(lengths_A, lengths_B)

            # generate in batch
            mels_out, T_A, T_B = generate_sample(
                model, vocoder, mel_A, mel_B, texts_A, texts_B, device
            )

            # decode + save
            for i, cid in enumerate(clip_ids):
                m_out  = mels_out[2*i:2*i+2]
                wav_gen = decode_and_mix(m_out, T_A, T_B, vocoder)
                path = os.path.join(out_dir, f"{cid}_generated.wav")
                torchaudio.save(path, wav_gen, sr)
                # save_audio(wav_gen, path, sr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vocoder = load_model_and_vocoder("/work/users/r/p/rphadke/JSALT/ckpts/fisher_chunks_1K_LoRAv3.0/model_120000.pt", "/work/users/r/p/rphadke/JSALT/fisher_chunks_1Kv2/vocab.txt", device=device, lora=True)

run_inference_with_arrow(
    eval_loader=eval_loader,
    model = model,
    vocoder=vocoder,
    device=device,
    out_dir="/work/users/r/p/rphadke/JSALT/outputs/batch_inferTEST"
)