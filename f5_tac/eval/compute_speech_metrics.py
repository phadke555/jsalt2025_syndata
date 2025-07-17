import json
from collections import defaultdict
from copy import deepcopy
import os
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns



def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def compute_turn_taking_stats(data):
    data.sort(key=lambda x: float(x['start_time']))  # Ensure chronological order

    speakers = defaultdict(list)
    for entry in data:
        speakers[entry['speaker']].append(entry)

    stats = {
        'IPU': [],
        'Turn': [],
        'Pause': [],
        'Gap': [],
        'Interruption': [],
        'Backchannel': []
    }

    last_speaker = None
    last_end_time = None
    last_speaker_end = {}
    last_speaker_start = {}
    last_ipu_start = None


    for i, entry in enumerate(data):
        speaker = entry['speaker']
        start, end = float(entry['start_time']), float(entry['end_time'])

        if i == 0:
            last_ipu_start = start

        # Compute IPU (Interpausal Unit) for each speaker
        ipu_duration = end - start
        stats['IPU'].append(ipu_duration)

        # Compute Turn for each speaker
        if last_speaker != speaker or i == 0:
            stats['Turn'].append(end - last_ipu_start)
            last_ipu_start = start

        # Compute within-speaker Pause
        if speaker in last_speaker_end.keys() and last_speaker == speaker:
            pause = start - last_speaker_end[speaker]
            if pause > 0:
                stats['Pause'].append(pause)

        # Compute Gap and Overlap
        if last_end_time is not None and last_speaker != speaker:
            silence = start - last_end_time
            if silence > 0:
                stats['Gap'].append(silence)

        # fetch
        other_spk = [x for x in last_speaker_end.keys() if x != speaker]
        is_backchannel = False
        if len(other_spk) > 0:
            other_spk = other_spk[0]
            if start < last_speaker_end[other_spk]:
                # contained inside
                overlap = abs(start - last_speaker_end[other_spk])
                if end < last_speaker_end[other_spk]:
                    stats['Backchannel'].append(end - start)
                    #stats['Overlap'].append(end - start)
                    is_backchannel = True
                else:
                    if overlap > 0:
                        stats['Interruption'].append(overlap)

        if not is_backchannel:
            last_speaker = speaker

        last_speaker_start[speaker] = start
        last_speaker_end[speaker] = end
        last_end_time = max([v for k, v in last_speaker_end.items()])

    return stats


def print_stats(stats):
    for key, value in stats.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(
                    f"{key} ({sub_key}): Count = {len(sub_value)}, Avg Duration = {sum(sub_value) / len(sub_value) if sub_value else 0:.2f}")
        else:
            print(f"{key}: Count = {len(value)}, Avg Duration = {sum(value) / len(value) if value else 0:.2f}")


def process_directory(json_dir, label):
    """Process all JSON files in a directory and return DataFrames."""
    all_durations, all_occurrences, all_means = [], [], []

    json_files = sorted(Path(json_dir).glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {json_dir}")

    for json_file in tqdm(json_files, desc=f"Processing {label}"):
        data = load_json(json_file)
        stats = compute_turn_taking_stats(data)

        session_name = json_file.stem
        durations = {k: np.sum(v) / 60 for k, v in stats.items()}
        occurrences = {k: len(v) for k, v in stats.items()}
        means = {k: np.mean(v) if v else 0 for k, v in stats.items()}

        durations.update({"Session": session_name, "Type": label})
        occurrences.update({"Session": session_name, "Type": label})
        means.update({"Session": session_name, "Type": label})

        all_durations.append(durations)
        all_occurrences.append(occurrences)
        all_means.append(means)

    dur_df = pd.DataFrame(all_durations)
    occ_df = pd.DataFrame(all_occurrences)
    mean_df = pd.DataFrame(all_means)

    return dur_df, occ_df, mean_df

def create_plots(mean_df, total_df, occ_df, output_dir):
    metrics = ["IPU", "Turn", "Pause", "Gap", "Interruption", "Backchannel"]
    os.makedirs(output_dir, exist_ok=True)

    # Bar plots: Mean durations
    mean_grouped = mean_df.groupby("Type")[metrics].mean()
    mean_grouped.T.plot(kind="bar", figsize=(10, 6))
    plt.title("Mean Duration of Metrics (Generated vs Real)")
    plt.ylabel("Duration (seconds)")
    plt.xlabel("Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_duration_bar.png"))
    plt.close()

    # Box plots: Distributions of mean durations
    mean_melted = mean_df.melt(id_vars=["Type"], value_vars=metrics, var_name="Metric", value_name="Mean Duration")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=mean_melted, x="Metric", y="Mean Duration", hue="Type")
    plt.title("Distribution of Mean Durations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_duration_boxplot.png"))
    plt.close()

    # === Occurrences Boxplot ===
    occ_melted = occ_df.melt(id_vars=["Type"], value_vars=metrics,
                            var_name="Metric", value_name="Count")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=occ_melted, x="Metric", y="Count", hue="Type")
    plt.title("Distribution of Metric Occurrences per Session")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occurrences_boxplot.png"))
    plt.close()

    # Bar plots: Total durations
    total_grouped = total_df.groupby("Type")[metrics].sum()
    total_grouped.T.plot(kind="bar", figsize=(10, 6))
    plt.title("Total Duration of Metrics (Generated vs Real)")
    plt.ylabel("Total Duration (minutes)")
    plt.xlabel("Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_duration_bar.png"))
    plt.close()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    # Paths
    base_output = args.data_root
    generated_json_dir = os.path.join(base_output, "generated")
    # real_json_dir = os.path.join(base_output, "real")

    # Process both sets
    dur_gen, occ_gen, mean_gen = process_directory(generated_json_dir, "Ablation-Layered")
    # dur_real, occ_real, mean_real = process_directory(real_json_dir, "Real")

    # Combine
    dur_df = dur_gen
    occ_df = occ_gen
    mean_df = mean_gen

    # Save combined results
    combined_output_dir = os.path.join(base_output, "metrics")
    os.makedirs(combined_output_dir, exist_ok=True)



    dur_df.to_csv(os.path.join(combined_output_dir, "total_duration.csv"), index=False)
    occ_df.to_csv(os.path.join(combined_output_dir, "occurrences.csv"), index=False)
    mean_df.to_csv(os.path.join(combined_output_dir, "mean_duration.csv"), index=False)
    print(f"Metrics saved to {combined_output_dir}")

    create_plots(mean_df, dur_df, occ_df, combined_output_dir)
    print(f"Plots saved to {combined_output_dir}")