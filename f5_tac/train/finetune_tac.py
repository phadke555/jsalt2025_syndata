import argparse
import os
import shutil
from importlib.resources import files
from functools import partial

import torch
from cached_path import cached_path

import yaml

# --- MODIFICATION: Import your new modules ---
from f5_tac.model.cfm import CFMWithTAC
from f5_tac.model.reccfm import CFMWithTACRecon
from f5_tac.model.backbones.dittac import DiTWithTAC
from f5_tac.model.trainer import Trainer
from f5_tac.model.dataset import load_conversation_dataset, conversation_collate_fn
from f5_tac.configs.model_kwargs import mel_spec_kwargs, dit_cfg, lora_configv3
from f5_tts.model.utils import get_tokenizer

# --- Argument Parsing (adapted for finetuning TAC model) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune F5-TAC Model on two-speaker dataset.")

    # --------------------- Experiment Configuration --------------------- #
    parser.add_argument("--config", "-c", type=str, default=None, help="Optional YAML config file for finetuning.")

    parser.add_argument("--data_root", type=str, default=None, help="Base directory to datasets, overrides defaults.")
    parser.add_argument("--dataset_name", type=str, default="fisher_conversations", help="Name of dataset folder.")

    parser.add_argument("--exp_name", type=str, default="f5_tac_finetune", help="Experiment name for logs/checkpoints.")
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune from a pretrained checkpoint.")

    # --------------------- Checkpoints and Tokenizer --------------------- #
    parser.add_argument("--pretrain", type=str, required=True, help="Path or URL to pretrained F5 checkpoint.")
    parser.add_argument(
        "--tokenizer", type=str, default="custom", choices=["pinyin", "char", "custom"], help="Tokenizer type."
    )
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer vocab file.")

    # --------------------- Training Hyperparameters --------------------- #
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200, help="Batch size in frames per GPU.")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size mode."
    )
    parser.add_argument("--max_samples", type=int, default=16, help="Max sequences per batch.")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--num_warmup_updates", type=int, default=20000, help="Warmup updates for LR scheduler.")

    # --------------------- Logging and Saving --------------------- #
    parser.add_argument("--save_per_updates", type=int, default=5000, help="Save checkpoint every N updates.")
    parser.add_argument("--last_per_updates", type=int, default=5000, help="Save last checkpoint every N updates.")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int, default=-1,
        help="-1 to keep all, 0 to skip, or N to keep most recent N checkpoints."
    )
    parser.add_argument("--logger", type=str, default="wandb", choices=[None, "wandb", "tensorboard"], help="Logger.")
    parser.add_argument("--log_samples", action="store_true", help="Log audio samples during training.")
    parser.add_argument("--bnb_optimizer", action="store_true", help="Use bitsandbytes 8-bit optimizer.")

    return parser.parse_args()


def main():
    args = parse_args()

    # If a YAML config was provided, load it and override any args:
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
            else:
                raise ValueError(f"Unknown config key: {key}")
    
    dataset_path = os.path.join(args.data_root, args.dataset_name) if args.data_root else args.dataset_name
    checkpoint_path = os.path.join(args.data_root, "ckpts", args.exp_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    local_pretrain_path = args.pretrain
    print(f"Using pretrained model from: {args.pretrain}")
    
    # --- 2. Setup Tokenizer ---
    vocab_char_map, vocab_size = get_tokenizer(args.tokenizer_path,args.tokenizer)
    print(f"Loaded tokenizer with vocab size: {vocab_size}")

    # --- 3. Define Model Architecture and Mel Spectrogram settings ---
    # These should match the architecture of the pretrained model you are loading.

    # --- 4. Instantiate Your New Models ---
    print("Instantiating F5-TAC models...")
    transformer_backbone = DiTWithTAC(
        **dit_cfg,
        num_speakers=2, # Critical for TAC blocks
        text_num_embeds=vocab_size,
        mel_dim=mel_spec_kwargs["n_mel_channels"]
    )
    
    model = CFMWithTACRecon(
        transformer=transformer_backbone,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    


    # --- 5. CRITICAL: Load Pretrained Weights into the New Architecture ---
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

    old_embed_weight = state["transformer.text_embed.text_embed.weight"]
    random_spkchg_init = torch.randn(1, old_embed_weight.shape[1]) * 0.01
    print("\n == Spk Chg Token Init ==")
    print(random_spkchg_init)
    with torch.no_grad():
        new_embed_weight = torch.cat([
            old_embed_weight,                               # copy existing rows
            random_spkchg_init  # random init new row
        ], dim=0)
    
    state["transformer.text_embed.text_embed.weight"] = new_embed_weight

    # 5) Finally load with strict=False to pick up whatever lines up
    incompatible = model.load_state_dict(state, strict=False)

    print("✔ loaded partial state:")
    print("  • missing   (should only be tac module keys)   :", incompatible.missing_keys[:5], "…")
    print("  • unexpected  :", incompatible.unexpected_keys[:5], "…")

    # # Freeze initial transformer layers
    # freeze_layers = 8  # freeze first 12 layers
    # for i, block in enumerate(transformer_backbone.transformer_blocks):
    #     if i < freeze_layers:
    #         for name, param in block.named_parameters():
    #             if "norm" not in name and "tac" not in name:  # keep LayerNorm trainable
    #                 param.requires_grad = False
    # print(f"✔️ Frozen first {freeze_layers} layers of DiTWithTAC")

    # ----------------------------------------------------------
    # LoRA Experiment
    from peft import LoraConfig, PeftModel, LoraModel, get_peft_model

    model = get_peft_model(model, lora_configv3)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if "tac" in name or "text_embed.text_embed" in name or "3.dwconv" in name:
            param.requires_grad = True

    # ----------------------------------------------------------


    # --- 6. Instantiate Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=float(args.learning_rate),
        checkpoint_path=checkpoint_path,
        save_per_updates=int(args.save_per_updates),
        batch_size_per_gpu=int(args.batch_size_per_gpu),
        batch_size_type=args.batch_size_type,
        max_samples=int(args.max_samples),
        grad_accumulation_steps=int(args.grad_accumulation_steps),
        max_grad_norm=1.0, # max_grad_norm is not in args
        mix_loss_lambda=1.0,
        logger=args.logger,
        recon_loss = True,
        wandb_project=f"finetune_f5_2speaker",
        wandb_run_name=args.exp_name,
        log_samples=args.log_samples,
        bnb_optimizer=True,
        accelerate_kwargs={"mixed_precision": "bf16"}
    )
    
    # --- 7. Load Dataset and Start Training ---
    print("Loading dataset...")
    train_dataset = load_conversation_dataset(
        dataset_path=dataset_path,
        mel_spec_kwargs=mel_spec_kwargs
    )
    print("Train dataset length:", len(train_dataset))
    
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        collate_fn=conversation_collate_fn, # Pass your custom collate fn
        num_workers=18, # Adjust as needed
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()

