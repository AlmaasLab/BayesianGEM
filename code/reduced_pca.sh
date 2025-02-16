#!/bin/bash
#SBATCH -J reduced_pca               # sensible name for the job
#SBATCH --output=reduced_pca.out
#SBATCH --nodes=1                    
#SBATCH -c 4
#SBATCH --mem=100G
#SBATCH -t 10:00:00             # Upper time limit for the job
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jakob.p.pettersen@ntnu.no
#SBATCH -p CPUQ
#SBATCH --account=nv-ibt
#SBATCH --export=NONE

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
module purge
module load Anaconda3/2020.07
source ~/.bash_profile
conda activate etcFBA
python reduced_pca.py &> "../results/reduced_smcabc_res/reduced_pca.log"
