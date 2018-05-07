#!/bin/bash
#SBATCH -p holyseasgpu
#SBATCH -J gpu_2_1000_100_100 # Job Name
#SBATCH -n 2 # Number of MPI tasks
#SBATCH -N 2 # Ensure on two nodes
#SBATCH --gres=gpu:1 #Number of GPUs requested per node
#SBATCH --constraint=cuda-7.5 #require CUDA
#SBATCH -t 0-06:00 #runtime in D-HH:MM
#SBATCH --mem-per-cpu 1000 #memory per MPI task
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err
#SBATCH --mail-type=BEGIN,END,FAIL #alert when done
#SBATCH --mail-user=bcook@cfa.harvard.edu #Email to send to

mpiexec -n $SLURM_NTASKS run_behalf.py --run-name $SLURM_JOB_NAME --clobber --N-parts 1000 --N-steps 100 --dt 0.01
RESULT=${PIPESTATUS[0]}
sacct -j "${SLURM_JOB_ID}".batch --format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State 
exit $RESULT
