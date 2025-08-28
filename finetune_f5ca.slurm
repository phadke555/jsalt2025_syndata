#!/bin/bash
# finetune.slurm

#SBATCH --job-name=ChannelAttentionTestRun2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=80G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --account=a100acct
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rphadke@unc.edu

export WANDB_API_KEY=db06664f93f231ec28f9f074bf8a253862ceb578
export WANDB_ENTITY=rphadke
export HF_HOME="/export/fs06/rphadke1/.cache/huggingface"

source ~/.bashrc

conda activate /home/rphadke1/jsalt2025_syndata/envs/conda_env

cd "/home/rphadke1/jsalt2025_syndata"

export PYTHONPATH=$(pwd)

python f5_tac/train/finetune_ca.py \
--config f5_tac/configs/jhutest.yaml \
--tokenizer_path /export/fs06/rphadke1/data/vocab_files/vocab_v2.1.txt

conda deactivate