from f5_tac.infer.utils import process_directory
import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine subpart generated clips into full recordings.")
    parser.add_argument("--generations_input_dir", type=str, required=True, help="Directory with subpart _generated.wav files")
    parser.add_argument("--generations_output_dir", type=str, required=True, help="Directory to save combined files")
    # parser.add_argument("--real_input_dir", type=str, required=True, help="Directory with subpart _generated.wav files")
    # parser.add_argument("--real_output_dir", type=str, required=True, help="Directory to save combined files")
    args = parser.parse_args()

    print("🔄 Combining generated clips...")
    process_directory(args.generations_input_dir, args.generations_output_dir, mode="generated")

    # print("\n🔄 Combining original chunks...")
    # process_directory(args.real_input_dir, args.real_output_dir, mode="original")

    print("\n✅ Done combining all audio!")


if __name__ == "__main__":
    main()
