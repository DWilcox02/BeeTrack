#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=dgw22 # required to send email notifcations
export PATH=/vol/bitbucket/dgw22/BeeTracker/venv/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
source activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

python src/tapir_bulk.py