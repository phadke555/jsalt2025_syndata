from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from f5_tts.infer.utils_infer import chunk_text
from f5_tac.model.cfm import CFMWithTAC
from f5_tac.configs.model_kwargs import mel_spec_kwargs, dit_cfg
import torch

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "xpu"
#     if torch.xpu.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

import os
import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd

# from f5_tac.model.cfm import CFMWithTAC
# from f5_tac.model.reccfm import CFMWithTACRecon
# from f5_tac.model.backbones.dittac import DiTWithTAC
from f5_tts.model.cfm import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tac.model.dlcfm import CFMDD
from f5_tac.model.backbones.doubledit import DoubleDiT
from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.model.utils import get_tokenizer
import logging
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model
from f5_tac.configs.model_kwargs import lora_configv1, lora_configv2, lora_configv3, mel_spec_kwargs, dit_cfg


def load_base_model_and_vocoder(ckpt_path, vocab_file, device, lora=False):
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    old_transformer = DiT(
        **dit_cfg,
        text_num_embeds=vocab_size,
        mel_dim=mel_spec_kwargs["n_mel_channels"]
    )

    old_model = CFM(
        transformer=old_transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt = load_file(ckpt_path, device="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state = (
        ckpt.get("ema_model_state_dict", 
        ckpt.get("model_state_dict", ckpt))
    )
    state = {k.replace("ema_model.", ""): v for k, v in state.items()}
    incomplete = old_model.load_state_dict(state, strict = False)
    print(incomplete)
    transformer_backbone = DoubleDiT(
        **dit_cfg,
        text_num_embeds=vocab_size,
        mel_dim=mel_spec_kwargs["n_mel_channels"]
    )
    print(transformer_backbone.time_embed.load_state_dict(old_transformer.time_embed.state_dict()))
    print(transformer_backbone.text_embed.load_state_dict(old_transformer.text_embed.state_dict()))
    print(transformer_backbone.input_embed.load_state_dict(old_transformer.input_embed.state_dict()))
    print(transformer_backbone.rotary_embed.load_state_dict(old_transformer.rotary_embed.state_dict()))
    print(transformer_backbone.norm_out.load_state_dict(old_transformer.norm_out.state_dict()))
    print(transformer_backbone.proj_out.load_state_dict(old_transformer.proj_out.state_dict()))

    for i, old_block in enumerate(old_transformer.transformer_blocks):
        # block A
        transformer_backbone.blocks_A[i].load_state_dict(old_block.state_dict(), strict=True)
        # block B
        transformer_backbone.blocks_B[i].load_state_dict(old_block.state_dict(), strict=True)

    model = CFMDD(
        transformer=transformer_backbone,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    model.to(device).eval()

    vocoder = load_vocoder().to(device)
    return model, vocoder



def load_model_and_vocoder(ckpt_path, vocab_file, device, lora=False):
    """Load model and vocoder."""
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    transformer_backbone = DoubleDiT(
        **dit_cfg,
        text_num_embeds=vocab_size,
        mel_dim=mel_spec_kwargs["n_mel_channels"]
    )
    model = CFMDD(
        transformer=transformer_backbone,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if lora:
        model = get_peft_model(model, lora_configv2)
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


def save_transcript(transcript_path, ref_text_A, gen_text_A, ref_text_B, gen_text_B, conversation_id):
    """Append transcript to file."""
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write(f"=== Conversation {conversation_id} ===\n")
        f.write(f"SPEAKER A\nREF: {ref_text_A}\nGEN: {gen_text_A}\n\n")
        f.write(f"SPEAKER B\nREF: {ref_text_B}\nGEN: {gen_text_B}\n\n")


def save_audio(wav, path, sr):
    """Save audio file."""
    torchaudio.save(path, wav, sr)


def generate_sample(model, vocoder, wav_A, wav_B, text_A, text_B, device):
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


def process_row(row, model, vocoder, out_dir, device, transcript_file):
    """Process one metadata row: generate & save outputs."""

    sr = 24000  # target sample rate

    # --- Extract identifiers from wav path ---
    # e.g. /path/to/fe_03_00001_0000_A.wav → fe_03_00001_0000
    base_name_A = os.path.basename(row["speaker_A_wav"])

    # Take common prefix (before _A.wav or _B.wav)
    clip_id = base_name_A.replace("_A.wav", "")

    # --- Load & prepare reference wavs ---
    ref_wav_A, orig_sr_A = torchaudio.load(row["speaker_A_wav"])
    ref_wav_B, orig_sr_B = torchaudio.load(row["speaker_B_wav"])

    wav_A = prep_wav(ref_wav_A, orig_sr_A, sr, device)
    wav_B = prep_wav(ref_wav_B, orig_sr_B, sr, device)

    # Handle potential missing text
    ref_text_A = row["speaker_A_text"]
    ref_text_B = row["speaker_B_text"]

    # Generate
    mels_out, T_A, T_B = generate_sample(
        model, vocoder, wav_A, wav_B,
        ref_text_A, ref_text_B,
        device
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

    # gen_waveforms = vocoder.decode(
    #     torch.cat([gen_mel_A, gen_mel_B], dim=0).permute(0, 2, 1)
    # ).detach().cpu()

    # --- Save combined generated audio ---
    out_wav_path = os.path.join(out_dir, f"{clip_id}_generated.wav")
    save_audio(mono, out_wav_path, sr)




def process_all(metadata_path, out_dir, model, vocoder, device):
    """Loop over all metadata rows and process them."""
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    for idx, row in df.iterrows():
        conversation_id = f"{row['recording_id']}_{idx}"
        print(f"Processing {conversation_id}...")
        process_row(row, model, vocoder, out_dir, device, conversation_id)


import os
import argparse
import torchaudio
import torch


def find_generated_groups(input_dir):
    """
    Group all files in input_dir by conversation ID (prefix before "_xxxx_generated.wav").
    """
    groups = {}
    for filename in os.listdir(input_dir):
        if filename.endswith("_generated.wav"):
            # Extract conversation ID (e.g., fe_03_00001 from fe_03_00001_0000_generated.wav)
            base = filename.split("_")[0:3]  # ['fe', '03', '00001']
            conversation_id = "_".join(base)

            if conversation_id not in groups:
                groups[conversation_id] = []
            groups[conversation_id].append(os.path.join(input_dir, filename))

    # Sort each group by subpart index (ensure _0000, _0001, ...)
    for conv_id in groups:
        groups[conv_id] = sorted(groups[conv_id])

    return groups

def find_original_groups(input_dir):
    """
    Group all *_A.wav and *_B.wav files by conversation ID.
    """
    groups = {}

    for filename in os.listdir(input_dir):
        if filename.endswith("_A.wav") or filename.endswith("_B.wav"):
            # Extract conversation ID (e.g., fe_03_00001)
            parts = filename.split("_")
            conversation_id = "_".join(parts[0:3])  # fe_03_00001
            if conversation_id not in groups:
                groups[conversation_id] = {"A": [], "B": []}
            if filename.endswith("_A.wav"):
                groups[conversation_id]["A"].append(os.path.join(input_dir, filename))
            elif filename.endswith("_B.wav"):
                groups[conversation_id]["B"].append(os.path.join(input_dir, filename))

    # Sort each speaker’s file list by subpart index (_0000, _0001, ...)
    for conv_id in groups:
        groups[conv_id]["A"] = sorted(groups[conv_id]["A"])
        groups[conv_id]["B"] = sorted(groups[conv_id]["B"])

    return groups

def concatenate_clips(file_list):
    """
    Load and concatenate a list of wav files along the time axis.
    """
    waveforms = []
    sample_rate = None

    for file in file_list:
        waveform, sr = torchaudio.load(file)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {file} has {sr}Hz instead of {sample_rate}Hz")
        waveforms.append(waveform)

    # Concatenate along time dimension
    combined = torch.cat(waveforms, dim=1)
    return combined, sample_rate


def process_directory(input_dir, output_dir, mode=None):
    """
    Combine all subpart clips in input_dir and save them as full recordings in output_dir.
    mode: 'generated' or 'original'
    """
    os.makedirs(output_dir, exist_ok=True)

    if mode == "generated":
        groups = find_generated_groups(input_dir)
        print(f"Found {len(groups)} conversation groups to combine (generated).")

        for conv_id, files in groups.items():
            print(f"Combining {len(files)} generated clips for {conv_id}...")
            combined_audio, sr = concatenate_clips(files)

            output_path = os.path.join(output_dir, f"{conv_id}_full_generated.wav")
            torchaudio.save(output_path, combined_audio, sr)
            print(f"Saved combined file: {output_path}")

    elif mode == "original":
        groups = find_original_groups(input_dir)
        print(f"Found {len(groups)} conversation groups to combine (original).")

        for conv_id, speakers in groups.items():
            print(f"Combining original chunks for {conv_id}...")

            # Combine speaker A
            combined_A, sr_A = concatenate_clips(speakers["A"])
            # out_path_A = os.path.join(output_dir, f"{conv_id}_full_A.wav")
            # # torchaudio.save(out_path_A, combined_A, sr_A)
            # print(f"Saved: {out_path_A}")

            # Combine speaker B
            combined_B, sr_B = concatenate_clips(speakers["B"])
            # out_path_B = os.path.join(output_dir, f"{conv_id}_full_B.wav")
            # # torchaudio.save(out_path_B, combined_B, sr_B)
            # # print(f"Saved: {out_path_B}")

            # Combine into stereo: A → left, B → right
            max_len = max(combined_A.shape[-1], combined_B.shape[-1])
            A_pad = torch.nn.functional.pad(combined_A, (0, max_len - combined_A.shape[-1]))
            B_pad = torch.nn.functional.pad(combined_B, (0, max_len - combined_B.shape[-1]))
            # stereo = torch.cat([A_pad, B_pad], dim=0)  # [2, max_len]
            mono = A_pad + B_pad
            mono = mono / mono.abs().max() 
            # Ensure mono is 2D: [channels, samples]
            if mono.ndim == 1:
                mono = mono.unsqueeze(0)  # Add channel dim
                
            out_path_mono = os.path.join(output_dir, f"{conv_id}_full_real.wav")
            torchaudio.save(out_path_mono, mono, sr_A)
            print(f"Saved: {out_path_mono}")

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'generated' or 'original'.")
