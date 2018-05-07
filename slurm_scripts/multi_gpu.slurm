#!/bin/bash
#SBATCH -p gpu
#SBATCH -J multi_gpu # Job Name
#SBATCH -n 4 # Number of MPI tasks
#SBATCH -N 1 #ensure that all cores are NOT on one machine
#SBATCH --gres=gpu:4 #Number of GPUs requested
#SBATCH --constraint=cuda-7.5 #require CUDA
#SBATCH -t 0-06:00 #runtime in D-HH:MM
#SBATCH --mem-per-cpu 1000 #memory per MPI task
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err
#SBATCH --mail-type=BEGIN,END,FAIL #alert when done
#SBATCH --mail-user=bcook@cfa.harvard.edu #Email to send to

mpiexec -n $SLURM_NTASKS run_behalf.py --run-name $SLURM_JOB_NAME --clobber --N-parts 100 --N-steps 1000 --dt 0.01
RESULT=${PIPESTATUS[0]}
sacct -j "${SLURM_JOB_ID}".batch --format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State 
exit $RESULT
