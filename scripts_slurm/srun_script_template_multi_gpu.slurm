#!/bin/bash -l

# make sure logs dir exists
# add absolute path on your system

# fields with '...' should be filled in based on your system

###############################################
#SBATCH --job-name=<JOB-NAME>   # Job name
#SBATCH --output=./logs/<CONF-DIR>/<JOB-NAME>.out
#SBATCH --error=./logs/<CONF-DIR>/<JOB-NAME>.err
###############################################
#SBATCH --ntasks=<NUM-GPUS>     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=<NUM-GPUS>     # Tasks per node
#SBATCH --cpus-per-task=1     # Threads per task
###############################################
#SBATCH --time=...   # walltime
#SBATCH --mem=<RAM>   # memory per NODE
#SBATCH --partition=...    # Partition
###############################################
#SBATCH --gres=gpu:<NUM-GPUS>
#SBATCH --account=...    # Replace with your system project
###############################################

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

## LOAD MODULES ##
module purge            # clean up loaded modules 

# load necessary modules
# .....
# .....


# activate virtual env
source venv/bin/activate

base_script=<BASE-SCRIPT>

echo "Base script ${base_script}"

# Set up common environment variables
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$(shuf -i 10000-60000 -n 1)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export SLURM_GPUS_ON_NODE=$WORLD_SIZE 


# Run the training script
srun --exclusive bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID

  echo "########################"
  echo "Starting PROC $SLURM_PROCID"
  echo "########################"

  echo "Debug: RANK=$RANK, SLURM_PROCID=$SLURM_PROCID, LOCAL_RANK=$LOCAL_RANK"
  echo "WORLD_SIZE=$WORLD_SIZE"
  echo "MASTER_ADDR=$MASTER_ADDR"
  echo "MASTER_PORT=$MASTER_PORT"

  ## RUN YOUR PROGRAM ##
  python -m baby_lm.train \
    --training_config ./configs/train/<CONF-DIR>/'"$base_script"'\
    --experiment_config ./configs/train/<CONF-DIR>/<JOB-NAME>.yaml
'
