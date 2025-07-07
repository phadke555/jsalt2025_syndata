import argparse
import torch
from f5_tac.infer.utils import load_model_and_vocoder, process_all
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & vocoder
    model, vocoder = load_model_and_vocoder(args.ckpt_path, args.vocab_file, device)

    # Process all metadata rows
    metadata_path = os.path.join(args.data_root, "metadata.csv")
    print("Processing Eval Metadata and Generating Conversations")
    process_all(metadata_path, args.output_path, model, vocoder, device)

if __name__ == "__main__":
    main()