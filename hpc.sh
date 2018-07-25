#!/bin/bash

#SBATCH --job-name=hmap-heatonly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:p100:1

echo "START"

module purge

module load pillow/python3.5/intel/4.2.1
module swap python3/intel  python3/intel/3.6.3
#module load python3/intel/3.6.3
module load cudnn/9.0v7.0.5
module load cuda/9.0.176
module load tensorflow/python3.6/1.5.0
module list
echo "Loaded modules"
nvidia-smi

python heat_train.py --name v1 --dataset train --last_epoch 0 --epochs 5
echo "Ended"
