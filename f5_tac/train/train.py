import os
from importlib.resources import files
from functools import partial

import hydra
from omegaconf import OmegaConf

# --- MODIFICATION: Import your new modules ---
from f5_tac.model.cfm import CFMWithTAC
from f5_tac.model.trainer import Trainer # Assuming you save the new Trainer here
from f5_tac.model.dataset import load_conversation_dataset, conversation_collate_fn
from f5_tts.model.utils import get_tokenizer # Re-using the tokenizer utility

# --- MODIFICATION: Change working directory for f5_tac ---
try:
    os.chdir(str(files("f5_tac").joinpath("../..")))
except Exception:
    print("Could not change working directory. Assuming running from project root.")

@hydra.main(version_base="1.3", config_path="f5_tac/configs", config_name="train_fisher")
def main(cfg):
    print("--- F5-TAC Training Script ---")
    print(OmegaConf.to_yaml(cfg))

    # --- MODIFICATION: Get backbone class from config ---
    model_cls = hydra.utils.get_class(cfg.model.backbone)

    # --- Set text tokenizer ---
    tokenizer, vocab_size = get_tokenizer(
        tokenizer_path=cfg.model.tokenizer_path,
        tokenizer_type=cfg.model.tokenizer,
    )

    # --- MODIFICATION: Instantiate your new models ---
    transformer_backbone = model_cls(
        **cfg.model.arch,
        text_num_embeds=vocab_size,
        mel_dim=cfg.model.mel_spec.n_mel_channels
    )
    
    model = CFMWithTAC(
        transformer=transformer_backbone,
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=tokenizer.char_map, # Pass the char_map for inference
    )

    # --- Init trainer (the new one) ---
    trainer = Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=cfg.ckpts.save_dir,
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project="CFM-TAC-TTS", # New project name
        wandb_run_name=f"{cfg.model.name}",
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )
    
    # --- MODIFICATION: Load your new dataset ---
    train_dataset = load_conversation_dataset(
        dataset_path=cfg.datasets.name,
        mel_spec_kwargs=cfg.model.mel_spec
    )
    
    # --- MODIFICATION: Pass the correct collate function to the trainer ---
    # We don't need to pass the tokenizer here because the collate_fn doesn't use it,
    # but the CFM model's forward pass does, which is correct.
    trainer.train(
        train_dataset=train_dataset,
        collate_fn=conversation_collate_fn,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,
    )

if __name__ == "__main__":
    main()
