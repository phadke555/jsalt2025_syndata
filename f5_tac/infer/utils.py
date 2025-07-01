from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from f5_tts.infer.utils_infer import chunk_text
from f5_tac.model.cfm import CFMWithTAC

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def joint_infer_process(
    ref_audio_A, ref_text_A,
    ref_audio_B, ref_text_B,
    gen_textA, gen_text_B,           # full conversation prompt: e.g. "A: …; B: …"
    model: CFMWithTAC,
    vocoder: Callable,
    *,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Split the input text into batches
    audio_A, sr_A = torchaudio.load(ref_audio_A)
    audio_B, sr_B = torchaudio.load(ref_audio_B)

    # 2) Build conds
    mel_A = model.mel_spec(wav_A.to(device)).permute(0,2,1)  # [1, T, D]
    mel_B = model.mel_spec(wav_B.to(device)).permute(0,2,1)
    conds = torch.cat([mel_A, mel_B], dim=0)                 # [2, T, D]


    chunks_A = chunk_text(gen_text_A, max_chars=chunk_chars)
    chunks_B = chunk_text(gen_text_B, max_chars=chunk_chars)



