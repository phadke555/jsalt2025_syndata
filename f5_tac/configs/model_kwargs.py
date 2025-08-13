mel_spec_kwargs = dict(
    n_fft=1024, hop_length=256, win_length=1024,
    n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos",
)

dit_cfg = dict(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
)

from peft import LoraConfig, PeftModel, LoraModel, get_peft_model
rank = 64
lora_configv1 = LoraConfig(
    r=rank,
    lora_alpha=128,  # (rank)**0.5,
    # target_modules=["query", "value", "key", "mlp.0", "mlp.2"],
    target_modules=["to_k", "to_q", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"],
    lora_dropout=0.1,
    bias="none",
)
lora_configv2 = LoraConfig(
    r=rank,
    lora_alpha=128,  # (rank)**0.5,
    # target_modules=["query", "value", "key", "mlp.0", "mlp.2"],
    target_modules=["pwconv1", "pwconv2", "to_k", "to_q", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"],
    lora_dropout=0.1,
    bias="none",
)

lora_configv3 = LoraConfig(
    r=rank,
    lora_alpha=128,  # (rank)**0.5,
    # target_modules=["query", "value", "key", "mlp.0", "mlp.2"],
    target_modules=["pwconv1", "pwconv2", "input_embed.conv_pos_embed.conv1d.0", "input_embed.conv_pos_embed.conv1d.2", "input_embed.proj", "to_k", "to_q", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"],
    lora_dropout=0.1,
    bias="none",
)

unfrozen_modules = ["tac", "text_embed.text_embed", "input_embed"]
