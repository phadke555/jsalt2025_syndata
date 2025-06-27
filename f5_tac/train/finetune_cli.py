import argparse
import os
import shutil
from importlib.resources import files
from functools import partial

import torch
from cached_path import cached_path

# --- MODIFICATION: Import your new modules ---
from f5_tac.model.cfm import CFMWithTAC
from f5_tac.model.backbones.dittac import DiTWithTAC
from f5_tac.model.trainer import Trainer
from f5_tac.model.dataset import load_conversation_dataset, conversation_collate_fn
from f5_tts.model.utils import get_tokenizer

# --- Argument Parsing (adapted for finetuning TAC model) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune F5-TAC Model on a two-speaker dataset.")

    # --- Key Arguments ---
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the prepared dataset directory (containing raw.arrow, etc.).")
    parser.add_argument("--pretrain_ckpt_path", type=str, required=True, help="Path or URL to the pretrained single-speaker F5 model checkpoint (.pt or .safetensors).")
    parser.add_argument("--exp_name", type=str, default="f5_tac_finetune", help="Experiment name for checkpoints and logs.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for finetuning.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200, help="Batch size in frames per GPU.")
    parser.add_argument("--max_samples", type=int, default=16, help="Max sequences per batch.")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    
    # --- Tokenizer ---
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to custom tokenizer vocab file (e.g., the vocab.txt in your dataset dir).")
    
    # --- Checkpointing and Logging ---
    parser.add_argument("--save_per_updates", type=int, default=5000, help="Save checkpoint every N updates.")
    parser.add_argument("--logger", type=str, default="wandb", choices=[None, "wandb", "tensorboard"], help="Logger to use.")
    parser.add_argument("--log_samples", action="store_true", help="Log audio samples during training.")

    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. Setup Checkpoint Directory and Pretrained Model ---
    checkpoint_path = f"./ckpts/{args.exp_name}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Cache the pretrained model locally if it's a URL
    local_pretrain_path = cached_path(args.pretrain_ckpt_path)
    print(f"Using pretrained model from: {local_pretrain_path}")
    
    # --- 2. Setup Tokenizer ---
    tokenizer, vocab_size = get_tokenizer(
        tokenizer_path=args.tokenizer_path,
        tokenizer_type="custom",
    )
    print(f"Loaded tokenizer with vocab size: {vocab_size}")

    # --- 3. Define Model Architecture and Mel Spectrogram settings ---
    # These should match the architecture of the pretrained model you are loading.
    mel_spec_kwargs = dict(
        n_fft=1024, hop_length=256, win_length=1024,
        n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos",
    )
    
    dit_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
    )

    # --- 4. Instantiate Your New Models ---
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
        vocab_char_map=tokenizer.char_map,
    )
    
    # --- 5. CRITICAL: Load Pretrained Weights into the New Architecture ---
    print("Loading pretrained weights...")
    if local_pretrain_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        pretrained_state_dict = load_file(local_pretrain_path, device="cpu")
    else:
        pretrained_state_dict = torch.load(local_pretrain_path, map_location="cpu")
    
    # Handle checkpoints that might be nested (e.g., inside 'model_state_dict' or 'ema_model_state_dict')
    if "model_state_dict" in pretrained_state_dict:
        pretrained_state_dict = pretrained_state_dict["model_state_dict"]
    elif "ema_model_state_dict" in pretrained_state_dict:
        pretrained_state_dict = pretrained_state_dict["ema_model_state_dict"]
        # Remove 'ema_model.' prefix if present
        pretrained_state_dict = {k.replace("ema_model.", ""): v for k, v in pretrained_state_dict.items()}

    # --- MODIFICATION: Use strict=False for cleaner loading ---
    # This will load all matching keys and ignore the new TAC layers.
    incompatible_keys = model.load_state_dict(pretrained_state_dict, strict=False)
    
    print(f"Loaded pretrained model with status: {incompatible_keys}")
    print(f"Missing keys (expected: TAC layers): {incompatible_keys.missing_keys[:5]}...")
    if incompatible_keys.unexpected_keys:
        print(f"WARNING: Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys[:5]}...")


    # --- 6. Instantiate Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_path=checkpoint_path,
        save_per_updates=args.save_per_updates,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type="frame",
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=1.0, # max_grad_norm is not in args
        logger=args.logger,
        wandb_project=f"finetune_{args.exp_name}",
        wandb_run_name=args.exp_name,
        log_samples=args.log_samples,
    )
    
    # --- 7. Load Dataset and Start Training ---
    print("Loading dataset...")
    train_dataset = load_conversation_dataset(
        dataset_path=args.dataset_path,
        mel_spec_kwargs=mel_spec_kwargs
    )
    
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        collate_fn=conversation_collate_fn, # Pass your custom collate fn
        num_workers=8, # Adjust as needed
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()

