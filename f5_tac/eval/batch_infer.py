# batch_infer.py
import argparse
import torch
from f5_tac.infer.utils import load_model_and_vocoder, process_all
import os
import multiprocessing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    print(f"Detected {len(devices)} GPUs: {devices}")

    # Load model & vocoder
    model, vocoder = load_model_and_vocoder(args.ckpt_path, args.vocab_file, args.lora)

    # Process all metadata rows
    metadata_path = os.path.join(args.data_root, "metadata.csv")
    print("Processing Eval Metadata and Generating Conversations")
    process_all(metadata_path, args.output_path, model, vocoder, devices)

if __name__ == "__main__":
    main()