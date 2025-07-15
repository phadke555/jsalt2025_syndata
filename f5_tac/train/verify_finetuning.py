import torch
import argparse
from safetensors.torch import load_file
from f5_tts.model.utils import list_str_to_idx, get_tokenizer

def inspect_lora_weights(sd):
    # find all LoRA params in the after-finetuned checkpoint
    lora_items = {k: v for k,v in sd.items() if ".lora_" in k and isinstance(v, torch.Tensor)}
    if not lora_items:
        print("No LoRA weights found in this checkpoint.")
        return

    print("LoRA adapter parameter stats:\n")
    total_norm2 = 0.0
    total_elements = 0
    for name, tensor in sorted(lora_items.items()):
        t = tensor.cpu()
        mean_abs = t.abs().mean().item()
        max_abs  = t.abs().max().item()
        norm2    = t.norm().item()
        total_norm2 += norm2**2
        total_elements += t.numel()
        print(f"{name:60s}  shape={tuple(t.shape)}  mean|·|={mean_abs:.3e} max|·|={max_abs:.3e} ‖·‖₂={norm2:.3e}")

    overall_fro = total_norm2**0.5
    avg_norm    = overall_fro / (total_elements**0.5)
    print("\nSummary:")
    print(f"  Total LoRA parameters : {total_elements:,}")
    print(f"  Combined Frobenius ‖Δ‖₂ : {overall_fro:.3e}")
    print(f"  Avg per-param scale    : {avg_norm:.3e}")
    print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifying finetuning of F5-TAC Model on two-speaker dataset.")
    parser.add_argument("--checkpoint", type=str, default=None, help="New Checkpoint Path.")
    args = parser.parse_args()

    # Path to your checkpoints
    before_ckpt = "/work/users/r/p/rphadke/JSALT/ckpts/pretrained_model_1250000.safetensors"
    tok_path = "/work/users/r/p/rphadke/JSALT/fisher_chunks_0.1K_v2/vocab.txt"
    vocab_char_map, vocab_size = get_tokenizer(tok_path, "custom")

    # ID of your new speaker token (e.g. tokenizer.convert_tokens_to_ids("<spk>"))
    spk_chg_token = "-"

        # 1) Load state dicts
    sd_before = load_file(before_ckpt, device="cpu")
    print("Prior Checkpoint")
    print(set(sd_before.keys()))
    sd_after  = torch.load(args.checkpoint,  map_location="cpu")["ema_model_state_dict"]
    sd_after = {k.replace("ema_model.", ""): v for k, v in sd_after.items()}
    print("New Checkpoint")
    print(set(sd_after.keys()))

    # 2) Extract embedding weights
    #    adjust the key if your checkpoint nests it differently
    Wb = sd_before["ema_model.transformer.text_embed.text_embed.weight"]
    Wa = sd_after["base_model.model.transformer.text_embed.text_embed.weight"]

    ib = sd_before["ema_model.transformer.input_embed.proj.weight"]
    ia = sd_after["base_model.model.transformer.input_embed.proj.weight"]

    # 3) Compute absolute difference
    delta = (Wa - Wb).abs()

    idelta = (ib - ia).abs()

    # 4) Metrics
    speaker_chg_token_idx = list_str_to_idx([spk_chg_token], vocab_char_map=vocab_char_map)

    max_overall   = delta.max().item()
    max_speaker   = delta[speaker_chg_token_idx].max().item()

    print("\n== Text Embedding Drift ==")
    print(f"→ Max text embedding matrix change |Δ| overall:                  {max_overall:.6f}")
    print(f"→ Max |Δ| on token {speaker_chg_token_idx}: {max_speaker:.6f}")

    print("\n== Input Embedding Drift ==")
    print(f"→ Max input embedding weight change |Δ| overall:                  {idelta.max().item():.6f}")

    print("== LoRA Weight Scales ==")
    inspect_lora_weights(sd_after)

    # compare_text_embeddings(before_ckpt, args.checkpoint, spk_chg_token)
