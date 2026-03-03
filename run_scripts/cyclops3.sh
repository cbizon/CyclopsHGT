#!/bin/bash -l
#
#SBATCH --job-name=cyclops3
#SBATCH --output=logs/output.cyclops3
#SBATCH --error=logs/output_err.cyclops3
#SBATCH --nodelist=gpu-4-1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -t 24:00:00


module load cuda/latest cudnn/latest
cd /projects/sequence_analysis/vol3/bizon/CyclopsHGT

uv run python src/main.py \
  --graph_dir input_graphs/mf1 \
  --output_dir output/cyclops3 \
  --hidden_dim 32 \
  --num_layers 1 \
  --num_heads 4 \
  --dropout 0.2 \
  --lr 0.001 \
  --epochs 40 \
  --eval_every 5 \
  --device cuda
