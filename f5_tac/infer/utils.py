from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from f5_tts.infer.utils_infer import chunk_text
from f5_tac.model.cfm import CFMWithTAC
from f5_tac.configs.model_kwargs import mel_spec_kwargs, dit_cfg
import torch


import os
import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd

# from f5_tac.model.cfm import CFMWithTAC
from f5_tac.model.reccfm import CFMWithTACRecon
from f5_tac.model.backbones.dittac import DiTWithTAC
from f5_tts.model.cfm import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.model.utils import get_tokenizer
import logging
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model
from f5_tac.configs.model_kwargs import lora_configv1, lora_configv2, lora_configv3, mel_spec_kwargs, dit_cfg

cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
target_sample_rate = 24000
steps=32
max_duration = 4096

def load_model_and_vocoder(ckpt_path, vocab_file, device, lora=False):
    """Load model and vocoder."""
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    transformer = DiTWithTAC(
        **dit_cfg,
        num_speakers=2,
        text_num_embeds=vocab_size,
        mel_dim=mel_spec_kwargs["n_mel_channels"]
    )
    model = CFMWithTACRecon(
        transformer=transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if lora:
        model = get_peft_model(model, lora_configv2)

    # TODO: load ema_model_state_dict 
    model.load_state_dict(ckpt["model_state_dict"])

    model.to(device).eval()

    vocoder = load_vocoder().to(device)
    return model, vocoder

def prep_wav(wav, orig_sr, target_sr, device):
    """Make wav mono, resample, move to device."""
    wav = wav.mean(0, keepdim=True)  # mono
    if orig_sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_sr, target_sr)(wav)
    return wav.to(device)




def generate_sample(model, vocoder, wav_A, wav_B, text_A, text_B, gen_text_A=None, gen_text_B=None, device=None, steps=32, cfg_strength=1.5, sway_sampling_coef=-1.0, seed=42, max_duration=4096, use_epss=True):
    """Generate TTS samples for two speakers."""
    mel_A = model.mel_spec(wav_A)
    mel_B = model.mel_spec(wav_B)
    conds = torch.cat([mel_A, mel_B], 0)

    T_A = mel_A.size(2)
    T_B = mel_B.size(2)

    if not isinstance(text_A, str):
        text_A = ""
    if not isinstance(text_B, str):
        text_B = ""

    gen_text_A = text_A if gen_text_A is None else gen_text_A
    # gen_text_A = ""
    gen_text_B = text_B if gen_text_B is None else gen_text_B
    # gen_text_B = ""

    full_texts = [
        text_A + gen_text_A,   
        text_B + gen_text_B,   
    ]

    ratio_A = len(gen_text_A) / max(len(text_A), 1)
    ratio_B = len(gen_text_B) / max(len(text_B), 1)

    durations = [
        int(T_A * (1.1 + ratio_A)),
        int(T_B * (1.1 + ratio_B))
    ]

    mels_out, _ = model.sample_joint(
        texts=full_texts,
        conds=conds,
        durations=durations,
        vocoder=vocoder,
        steps=steps,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        seed=seed,
        max_duration=max_duration,
        use_epss=True
    )

    return mels_out, T_A, T_B


def process_row(row, model, vocoder, out_dir, device, sr=24000):
    """Process one metadata row: generate & save outputs."""
    # import pdb; pdb.set_trace()
    base_name_A = os.path.basename(row["speaker_A_wav"])
    clip_id = base_name_A.replace("_A.wav", "")

    # --- Load & prepare reference wavs ---
    ref_wav_A, orig_sr_A = torchaudio.load(row["speaker_A_wav"])
    ref_wav_B, orig_sr_B = torchaudio.load(row["speaker_B_wav"])
    wav_A = prep_wav(ref_wav_A, orig_sr_A, sr, device)
    wav_B = prep_wav(ref_wav_B, orig_sr_B, sr, device)
    ref_text_A = row["speaker_A_text"]
    ref_text_B = row["speaker_B_text"]

    # Generate
    mels_out, T_A, T_B = generate_sample(
        model, vocoder, wav_A, wav_B,
        ref_text_A, ref_text_B,
        device=device
    )

    # Decode generated portion only
    gen_mel_A = mels_out[0:1, T_A:, :]
    gen_mel_B = mels_out[1:2, T_B:, :]

    gen_mel_A = gen_mel_A.permute(0, 2, 1)
    if gen_mel_A.numel() == 0:
        # produce an “empty” waveform: shape [1, 0]
        wav_A = torch.zeros(1, 0, device="cpu")
    else:
        wav_A = vocoder.decode(gen_mel_A).detach().cpu()

    gen_mel_B = gen_mel_B.permute(0, 2, 1)
    if gen_mel_B.numel() == 0:
        wav_B = torch.zeros(1, 0, device="cpu")
    else:
        wav_B = vocoder.decode(gen_mel_B).detach().cpu()

    max_len = max(wav_A.shape[-1], wav_B.shape[-1])
    A_pad = F.pad(wav_A, (0, max_len - wav_A.shape[-1]))
    B_pad = F.pad(wav_B, (0, max_len - wav_B.shape[-1]))
    mono = A_pad + B_pad

    # --- Save combined generated audio ---
    out_wav_path = os.path.join(out_dir, f"{clip_id}_generated.wav")
    torchaudio.save(out_wav_path, mono, sr)




def process_all(metadata_path, out_dir, model, vocoder, device):
    """Loop over all metadata rows and process them."""
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    for idx, row in df.iterrows():
        conversation_id = f"{row['recording_id']}_{idx}"
        print(f"Processing {conversation_id}...")
        process_row(row, model, vocoder, out_dir, device)