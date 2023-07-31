#!/bin/bash
#SBATCH --gres=gpu:p100:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-10:00     # DD-HH:MM:SS
#SBATCH --account=def-jhoey     # DD-HH:MM:SS

nvidia-smi

#module load python/3.6 cuda cudnn

SOURCEDIR=~/VGAF

# Prepare virtualenv
#source ~/vgaf_env/bin/activate
# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.

# Prepare data
#mkdir $SLURM_TMPDIR/data
#tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt

# Start training
python $SOURCEDIR/vgaf_ViTR_AR_16F_train_Tset.py