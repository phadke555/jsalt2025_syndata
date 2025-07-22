import torch
import torchaudio
from f5_tac.infer.utils import generate_sample, load_doublemodel_and_vocoder, save_audio
import os
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_file = "/work/users/r/p/rphadke/JSALT/vocab_files/vocab_v2.1.txt"
ckpt_path = "/work/users/r/p/rphadke/JSALT/ckpts/BASELINEDouble-SpkChg/model_last.pt"

model, vocoder = load_doublemodel_and_vocoder(ckpt_path, vocab_file, device, lora=True)

out_dir="/work/users/r/p/rphadke/JSALT/outputs/fisher_jointspeaker/BaselineDoubleSpk-Chg"

wav_A = "/work/users/r/p/rphadke/JSALT/fisher_chunks_v2.1/wavs/fe_03_00001_0000_A.wav"
wav_B = "/work/users/r/p/rphadke/JSALT/fisher_chunks_v2.1/wavs/fe_03_00001_0000_B.wav"

# --- Load & prepare reference wavs ---
wav_A, sr = torchaudio.load(wav_A)
wav_B, _ = torchaudio.load(wav_B)

ref_text_A = "and i generally prefer <utt> eating at home <utt> hello andy <utt> how are you <utt> good <utt> do you have any idea what's going on <utt> yeah <utt>"
ref_text_B = "hi my name is andy <utt> good how are you doing <utt> no i guess we're supposed to talk about food now <utt>"

speaker_A = [
    "Did you see the email about the meeting tomorrow? <utt> I think we need to prepare some slides. <utt>",
    "That’s a good point. <utt> Do you think we should ask for more data first? <utt> It might save us time later. <utt>",
    "I’m not sure if this solution will scale well. <utt> Maybe we should test it with a larger dataset. <utt>",
    "Could you send me the updated report later today? <utt> I want to review it before the deadline. <utt>",
    "Yeah, I’ve been to that restaurant before. <utt> It’s amazing and their pasta is the best I’ve ever had. <utt>",
    "Wait <utt>, are we supposed to submit this by Friday or Monday? <utt> I don’t want to miss the deadline. <utt>",
    "I think we need a backup plan <utt> just in case. <utt> Things could change at the last minute. <utt>",
    "How was your weekend? <utt> Did you go hiking <utt> like you planned? <utt>",
    "Let’s meet at the coffee shop <utt> around 3 PM. <utt> I’ll grab us a table near the window. <utt>",
    "Do you remember what the professor said about this topic? <utt> I didn’t get to write it down. <utt>"
]
speaker_B = [
    "Yeah, I saw it. <utt> It’s at 10 AM in the main conference room. <utt> We should get there early to set up. <utt>",
    "Probably. <utt> We can draft an email and get their input first. <utt> That way we won’t waste any effort. <utt>",
    "Maybe not, <utt> but we could optimize it later if needed. <utt> Let’s keep it simple for now. <utt>",
    "Sure, <utt> I’ll send it over in an hour. <utt> Let me know if you need any changes after that. <utt>",
    "Really? <utt> I’ve been meaning to try it. <utt> Is the atmosphere good for a group dinner? <utt>",
    "It’s due Monday, <utt> but earlier would be better. <utt> We’ll have time for revisions that way. <utt>",
    "Agreed. <utt> Let’s brainstorm some options this afternoon. <utt> I’ll block some time on my calendar. <utt>",
    "Yeah, I did! <utt> It was beautiful up in the mountains. <utt> You should come next time. <utt>",
    "Sounds good, <utt> I’ll see you there. <utt> Do you want me to bring my laptop? <utt>",
    "He mentioned we need to focus on the key assumptions. <utt> I think that’s critical for the exam. <utt>"
]

# gen_text_A = "hello, do enjoy cooking? <utt> wow thats very cool. "
# gen_text_B = "yes, i enjoy cooking. the best dish i can cook is buffalo chicken pasta. <utt>"

for i in range(len(speaker_A)):
    gen_text_A = speaker_A[i]
    gen_text_B = speaker_B[i]
    mels_out, T_A, T_B = generate_sample(
        model, vocoder, wav_A, wav_B,
        ref_text_A, ref_text_B,
        gen_text_A=gen_text_A, gen_text_B=gen_text_B,
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

    out_wav_path = os.path.join(out_dir, f"generated{i}.wav")
    save_audio(mono, out_wav_path, sr)