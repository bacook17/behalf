#!/bin/bash
#SBATCH -p holyseasgpu
#SBATCH -J merer_576-12_10000_5000_50 # Job Name
#SBATCH -n 576 # Number of MPI tasks
#SBATCH -N 12 # Fix number of nodes
#SBATCH --gres=gpu:2 #Number of GPUs requested per node
#SBATCH --constraint=cuda-7.5 #require CUDA
#SBATCH -t 2-00:00 # runtime in D-HH:MM
#SBATCH --mem-per-cpu 1536 # memory per MPI task
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err
#SBATCH --mail-type=BEGIN,END,FAIL #alert when done
#SBATCH --mail-user=bcook@cfa.harvard.edu # Email to send to

mpiexec -n $SLURM_NTASKS run_merger.py --run-name $SLURM_JOB_NAME --clobber --N-parts 10000 --N-steps 5000 --dt 0.020 --save-every 10
RESULT=${PIPESTATUS[0]}
sacct -j $SLURM_JOB_ID --format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State
exit $RESULT
