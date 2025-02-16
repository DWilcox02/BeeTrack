#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB        # Request total memory instead of per-CPU
#SBATCH --cpus-per-task=4 # Explicitly specify CPU cores
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgw22
export PATH=/vol/bitbucket/dgw22/BeeTrack/venv/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
source activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

# Display initial memory state
echo "Initial memory status:"
free -h

# Start memory monitoring in background (updates every 5 seconds)
(while true; do
    echo "$(date '+%H:%M:%S') - Memory Status:"
    free -h | grep "Mem:"
    sleep 15
done) &
MEMORY_PID=$!

# Run your Python script
python src/tapir_bulk.py

# Kill the memory monitoring process
kill $MEMORY_PID

# Display final memory state
echo "Final memory status:"
free -h