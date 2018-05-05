import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # after 100 iterations, Npart 1000, dt 0.1
    Npart = 1000
    Nsteps = 1000
    dt = 10

    NNsteps = 100

    NN = 5
    serial_time = 140.*np.ones(NN)
    parallel_time = np.array([ 140., 102. , 60., 46. ])
    
    ### TO-DO: read from .out file, finding the line with "NNsteps" and reading the computation time up to there

    ratio = serial_time/parallel_time
    k_array = [1, 2, 4, 8, 16]

    plt.figure()
    plt.plot(k_array, ratio, '-ob')
    plt.xlabel('Number of cores (same node)')
    plt.ylabel('Speedup', fontsize=8)

    plt.title('Speedup versus number of cores')
    plt.savefig('mpi_speedup_vs_ncores_Npart%d_Nsteps%d_dt%d.png' %(Npart, Nsteps, dt), dpi=600)
    plt.show()