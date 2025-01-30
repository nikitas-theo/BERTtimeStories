#!/bin/bash -l

# make sure logs dir exists
# add absolute path on your system

# fields with '...' should be filled in based on your system

###############################################
#SBATCH --job-name=<JOB-NAME>   # Job name
#SBATCH --output=./logs/<CONF-DIR>/<JOB-NAME>.out
#SBATCH --error=./logs/<CONF-DIR>/<JOB-NAME>.err
###############################################
#SBATCH --ntasks=1     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=1     # Threads per task
###############################################
#SBATCH --time=...   # walltime
#SBATCH --mem=...   # memory per NODE
#SBATCH --partition=...    # Partition
###############################################
#SBATCH --gres=gpu:1
#SBATCH --account=... 
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

## RUN YOUR PROGRAM ##
srun python -m baby_lm.train \
  --training_config ./configs/train/<CONF-DIR>/${base_script}\
  --experiment_config ./configs/train/<CONF-DIR>/<JOB-NAME>.yaml
