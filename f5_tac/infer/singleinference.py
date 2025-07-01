# reference audio for A and B
# reference transcript for A and B???

# text to generate for A and B


import os
import torch
import torchaudio
import torch.nn.functional as F

print(os.getcwd())

# -----------------------------------------------------------------------------
# 1) Setup
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = ""
out_dir = "/work/users/r/p/rphadke/JSALT/outputs/fisher_jointspeaker/inference_3"
os.makedirs(out_dir, exist_ok=True)

# your model import
from f5_tac.model.cfm import CFMWithTAC
from f5_tac.model.backbones.dittac import DiTWithTAC

from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.model.utils import get_tokenizer

# (a) instantiate backbone & CFMWithTAC exactly as you did in training
vocab_char_map, vocab_size = get_tokenizer("/work/users/r/p/rphadke/JSALT/fisher_chunks/vocab.txt", "custom")
mel_spec_kwargs = dict(
        n_fft=1024, hop_length=256, win_length=1024,
        n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos",
    )
dit_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
    )
transformer = DiTWithTAC(
    **dit_cfg,
    num_speakers=2, # Critical for TAC blocks
    text_num_embeds=vocab_size,
    mel_dim=mel_spec_kwargs["n_mel_channels"]
    )
model = CFMWithTAC(
    transformer=transformer,
    mel_spec_kwargs=mel_spec_kwargs,
    vocab_char_map=vocab_char_map)
ckpt = torch.load("/work/users/r/p/rphadke/JSALT/ckpts/fisher_chunks/f5_tac_finetune/model_last.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# (b) load your neural vocoder here
# e.g. from f5_tts.vocoder import BigVGAN
vocoder = load_vocoder().to(device)

# -----------------------------------------------------------------------------
# 2) Load references + transcripts
# -----------------------------------------------------------------------------
ref_wav_A, sr = torchaudio.load("/work/users/r/p/rphadke/JSALT/fisher_chunks/wavs/fe_03_00001_0000_A.wav")  # shape [1, N]
ref_wav_B, _  = torchaudio.load("/work/users/r/p/rphadke/JSALT/fisher_chunks/wavs/fe_03_00001_0000_B.wav")  # assume same sr

# make sure mono, correct sr, and move to device
def prep_wav(wav, orig_sr):
    wav = wav.mean(0, keepdim=True)                    # to mono
    if orig_sr != sr:
        wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
    return wav.to(device)

wav_A = prep_wav(ref_wav_A, sr)
wav_B = prep_wav(ref_wav_B, sr)

ref_text_A = "and i generally prefer eating at home. hello andy. how are you? good do you have any idea what's going on yeah. "
gen_text_A = " I'm looking forward to the rest of my day. "
ref_text_B = "hi my name is andy. good, how are you doing? no, i guess we're supposed to talk about food now. "
gen_text_B = " That is good, what you have planned? "

full_texts = [
  ref_text_A + gen_text_A,   
  ref_text_B + gen_text_B,   
]


# -----------------------------------------------------------------------------
# 3) Build the conditioning mels
# -----------------------------------------------------------------------------
# your model already has a .mel_spec module
mel_A = model.mel_spec(wav_A)        # [1, n_mels, T]
mel_B = model.mel_spec(wav_B)        # [1, n_mels, T]
conds = torch.cat([mel_A, mel_B], 0) # [2, n_mels, T]

T_A = mel_A.size(2)
T_B = mel_B.size(2)

# Suppose you want length ∝ ratio of gen_text to ref_text
ratio_A = len(gen_text_A) / max(len(ref_text_A), 1)
ratio_B = len(gen_text_B) / max(len(ref_text_B), 1)

durations = [
    int(T_A * (1 + ratio_A)), 
    int(T_B * (1 + ratio_B))
]


# -----------------------------------------------------------------------------
# 4) Sample jointly
# -----------------------------------------------------------------------------
# if you’d like you can pass durations=[...], or let it default
mels_out, _ = model.sample_joint(
    texts=full_texts, 
    conds=conds, 
    durations=durations,
    steps=32,
    cfg_strength=1.5,
    sway_sampling_coef=-1.0,
    seed=42,
    max_duration=4096,
    vocoder=vocoder,   # returns waveform [2, waveform_length]
    use_epss=True,
)

mel_A = mels_out[0:1]; mel_B = mels_out[1:2]
mel_A = mel_A.permute(0, 2, 1)
wav_A = vocoder.decode(mel_A).detach().cpu()   # shape [1, T_A]
mel_B = mel_B.permute(0, 2, 1)
wav_B = vocoder.decode(mel_B).detach().cpu()   # shape [1, T_B]

# 1) Save Speaker A
# wav_A is [1, T_A], so this writes a mono file
torchaudio.save(os.path.join(out_dir, "speakerA.wav"), wav_A, sr)

# 2) Save Speaker B
torchaudio.save(os.path.join(out_dir, "speakerB.wav"), wav_B, sr)

# 3) Build & save combined stereo
# Pad to same length
max_len = max(wav_A.shape[-1], wav_B.shape[-1])
A_pad = F.pad(wav_A, (0, max_len - wav_A.shape[-1]))
B_pad = F.pad(wav_B, (0, max_len - wav_B.shape[-1]))
stereo = torch.cat([A_pad, B_pad], dim=0)  # [2, max_len], channels=(A,B)
torchaudio.save(os.path.join(out_dir, "combined.wav"), stereo, sr)

# # gen_wavs: Tensor[2, L] on device → move to CPU
# gen_A = gen_wavs[0].cpu().unsqueeze(0)  # [1, L]
# gen_B = gen_wavs[1].cpu().unsqueeze(0)  # [1, L]

# # -----------------------------------------------------------------------------
# # 5) Save out
# # -----------------------------------------------------------------------------
# # (a) Speaker A
# torchaudio.save(os.path.join(out_dir, "speakerA.wav"), gen_A, sr)
# print("saved speaker A wav")

# # (b) Speaker B
# torchaudio.save(os.path.join(out_dir, "speakerB.wav"), gen_B, sr)
# print("saved speaker B wav")

# # (c) Combined stereo (A on left, B on right)
# max_len = max(gen_A.shape[-1], gen_B.shape[-1])
# # pad to same length
# pad = lambda x: torch.nn.functional.pad(x, (0, max_len - x.shape[-1]))
# stereo = torch.cat([pad(gen_A), pad(gen_B)], dim=0)  # [2, max_len]
# torchaudio.save(os.path.join(out_dir, "combined.wav"), stereo, sr)

# print(f"Saved → {out_dir}/speakerA.wav, speakerB.wav, combined.wav")
