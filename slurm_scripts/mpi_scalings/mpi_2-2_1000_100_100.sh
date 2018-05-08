#!/bin/bash
#SBATCH -p hernquist
#SBATCH -J mpi_2-2_1000_100_100 # Job Name
#SBATCH -n 2 # Number of MPI tasks
#SBATCH -N 2 #ensure that all cores are on one machine
#SBATCH -t 3-00:00 #runtime in D-HH:MM
#SBATCH --mem-per-cpu 4000 #memory per MPI task
#SBATCH -o ../../logs/%x.out #logs/%x.out
#SBATCH -e ../../logs/%x.err #logs/%x.err
#SBATCH --mail-type=BEGIN,END,FAIL #alert when done
#SBATCH --mail-user=ana-roxana.pop@cfa.harvard.edu #Email to send to

mpiexec -n $SLURM_NTASKS ../../bin/run_behalf.py --run-name $SLURM_JOB_NAME --clobber --verbose --N-parts 1000 --N-steps 100 --dt 0.01
RESULT=${PIPESTATUS[0]}
sacct -j "${SLURM_JOB_ID}".batch --format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State 
exit $RESULT
