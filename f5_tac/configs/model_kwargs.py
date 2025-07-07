mel_spec_kwargs = dict(
    n_fft=1024, hop_length=256, win_length=1024,
    n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos",
)

dit_cfg = dict(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
)