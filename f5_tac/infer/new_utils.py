import os
import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd

from f5_tts.model.utils import get_tokenizer
from f5_tac.configs.model_kwargs import mel_spec_kwargs, dit_cfg, lora_configv2
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model
from f5_tts.model.cfm import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tac.model.dlcfm import CFMDD
from f5_tac.model.backbones.doubledit import DoubleDiT
from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.model.utils import get_tokenizer

def load_doublemodel_and_vocoder(ckpt_path, vocab_file, dataset_path, device, lora=False):
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

    state = (
        ckpt.get("ema_model_state_dict", 
        ckpt.get("model_state_dict", ckpt))
    )
    state = {k.replace("ema_model.", ""): v for k, v in state.items()}

    incomplete = model.load_state_dict(state, strict=False)
    print(incomplete)

    model.to(device).eval()

    vocoder = load_vocoder().to(device)

    from f5_tac.model.dataset import load_conversation_dataset, conversation_collate_fn
    from torch.utils.data import DataLoader, Dataset
    inference_dataset = load_conversation_dataset(
        dataset_path=dataset_path,
        mel_spec_kwargs=mel_spec_kwargs
    )
    dataloader = DataLoader(
        inference_dataset,
        batch_size=1,  # Process one row at a time
        shuffle=False,  # Keep original order
        collate_fn=conversation_collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    return model, vocoder, dataloader

def generate_sample(batch, model, gen_text_A = None, gen_text_B = None, device = None):
    cond_A = batch["mel_A"].permute(0, 2, 1)
    cond_B = batch["mel_B"].permute(0, 2, 1)

    T = cond_A.shape[1]

    if gen_text_A is None or gen_text_B is None:
        gen_text_A = batch["text_A"][0]
        gen_text_B = batch["text_B"][0]

    ratio_A = len(gen_text_A) / max(len(batch["text_A"][0]), 1)
    ratio_B = len(gen_text_B) / max(len(batch["text_B"][0]), 1)
    max_ratio = max(ratio_A, ratio_B)
    duration = int(T * (1+max_ratio))

    text_A = [batch["text_A"][0] + " " + gen_text_A]
    text_B = [batch["text_B"][0] + " " + gen_text_B]

    out_A, _, out_B, _ = model.new_sample_joint(
        cond_A, text_A, cond_B, text_B, duration,
        cfg_strength = 1.5, seed=42
    )

    return out_A, out_B, T



def process_all(dataloader, out_dir, model, vocoder, device):
    os.makedirs(out_dir, exist_ok=True)
    
    id_set = set()
    i = 0
    for batch in dataloader:
        if i >= 10:
            break
        id = batch['clip_ids'][0]
        if id in id_set:
            i += 1
        else:
            id_set.add(id)
            i = 0

        out_A, out_B, T = generate_sample(batch, model, device=device)
        # print("Original Out A Shape:", out_A.shape)
        out_A = out_A[:,T:, :].permute(0, 2, 1)
        # print("New Out A Shape After Permuting:", out_A.shape)
        out_B = out_B[:,T:, :].permute(0, 2, 1)
        wav_A = vocoder.decode(out_A).detach().cpu()
        wav_B = vocoder.decode(out_B).detach().cpu()

        import torch.nn.functional as F
        max_len = max(wav_A.shape[-1], wav_B.shape[-1])
        A_pad = F.pad(wav_A, (0, max_len - wav_A.shape[-1]))
        B_pad = F.pad(wav_B, (0, max_len - wav_B.shape[-1]))

        mono = A_pad + B_pad
        out_wav_path = os.path.join(out_dir, f"{batch['clip_ids'][0]}_{i}_generated.wav")
        torchaudio.save(out_wav_path, mono, 24000)

    print("Processed All Data")



