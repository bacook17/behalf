#!/bin/bash
#SBATCH -p holyseasgpu
#SBATCH -J gpuc_4-2_16000_3_100 # Job Name
#SBATCH -n 4 # Number of MPI tasks
#SBATCH -N 2 # Fix number of nodes
#SBATCH --gres=gpu:2 #Number of GPUs requested per node
#SBATCH --constraint=cuda-7.5 #require CUDA
#SBATCH -t 0-03:00 # runtime in D-HH:MM
#SBATCH --mem-per-cpu 1536 # memory per MPI task
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err
#SBATCH --mail-type=BEGIN,END,FAIL #alert when done
#SBATCH --mail-user=bcook@cfa.harvard.edu # Email to send to

mpiexec -n $SLURM_NTASKS run_behalf.py --run-name $SLURM_JOB_NAME --clobber --N-parts 16000 --N-steps 3 --dt 0.010 --save-every 1
RESULT=${PIPESTATUS[0]}
sacct -j $SLURM_JOB_ID ----format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State
exit $RESULT
