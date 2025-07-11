
# TODO: work in progress to make the code more modular...
def get_model():
    print("Instantiating F5-TAC models...")
    transformer_backbone = DiTWithTAC(
        **dit_cfg,
        num_speakers=2, # Critical for TAC blocks
        text_num_embeds=vocab_size,
        mel_dim=mel_spec_kwargs["n_mel_channels"]
    )
    
    model = CFMWithTAC(
        transformer=transformer_backbone,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    print("Loading pretrained weights...")
    # 1) Load the raw checkpoint
    if local_pretrain_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt = load_file(local_pretrain_path, device="cpu")
    else:
        ckpt = torch.load(local_pretrain_path, map_location="cpu")

    # 2) Unwrap any nesting
    state = (
        ckpt.get("model_state_dict", 
        ckpt.get("ema_model_state_dict", ckpt))
    )

    # 3) Strip an ‘ema_model.’ prefix if it sneaked in
    state = {k.replace("ema_model.", ""): v for k, v in state.items()}


    # 5) Finally load with strict=False to pick up whatever lines up
    incompatible = model.load_state_dict(state, strict=False)

    model = get_peft_model(model, lora_configv2)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if "tac" in name:
            param.requires_grad = True