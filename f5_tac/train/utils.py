import torch
from safetensors.torch import load_file
from peft import get_peft_model

def load_state(local_pretrain_path) -> dict:
    print("Loading pretrained weights...")
    if local_pretrain_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt = load_file(local_pretrain_path, device="cpu")
    else:
        ckpt = torch.load(local_pretrain_path, map_location="cpu")
    state = (
        ckpt.get("model_state_dict", 
        ckpt.get("ema_model_state_dict", ckpt))
    )
    state = {k.replace("ema_model.", ""): v for k, v in state.items()}
    return state

def update_embedding_weights(state, vocab_size) -> dict:
    old_emb = state["transformer.text_embed.text_embed.weight"]  # [V, D]
    if old_emb.shape[0] < vocab_size + 1:
        rand_row = torch.randn(1, old_emb.shape[1])
        state["transformer.text_embed.text_embed.weight"] = torch.cat(
            [old_emb, rand_row], dim=0
        )
        print(f"Vocab Embedding Matrix Updated to Size = {vocab_size + 1}")
    return state

def get_lora_model(model, lora_config, unfrozen_modules):
    model = get_peft_model(model, lora_config)

    from f5_tac.configs.model_kwargs import unfrozen_modules
    trainable = 0
    all = 0
    for name, param in model.named_parameters():
        all += 1
        for name_in in unfrozen_modules:
            if name_in in name:
                param.requires_grad = True
                trainable += 1
    print(f"Unfrozen Module Params = {trainable} | All Named Params = {all} | Proportion = {trainable/all}")        
    model.print_trainable_parameters()
    return model
