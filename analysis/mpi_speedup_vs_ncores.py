import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # after 25 iterations, Npart 1000, dt 0.1
    Npart = 1000
    Nsteps = 1000
    dt = 10

    NNsteps = 500

    n_cores = [1, 2, 4, 8, 16,32, 64]
    parallel_time = []
    # Read serial time
    outputFile = '../logs/serial_%d_%d_%d.out' %( Npart, Nsteps, dt)
    with open(outputFile) as fopen:
        lines = fopen.readlines()
        lines = [x.strip() for x in lines]
        temp = lines[NNsteps + 1].split("complete. ",1)
        temp = temp[1].split(" seconds")
        parallel_time.append(float(temp[0]))
    # Read parallel time for ncores
    for ncores in n_cores[1:]:
        outputFile = '../logs/mpi_%d_%d_%d_%d.out' %(ncores, Npart, Nsteps, dt)
        with open(outputFile) as fopen:
            lines = fopen.readlines()
        lines = [x.strip() for x in lines]
        temp = lines[NNsteps + 1].split("complete. ",1)
        temp = temp[1].split(" seconds")
        parallel_time.append(float(temp[0]))

    print 'Running time after %d steps for ncores:' %NNsteps
    NN = np.size(n_cores)

    print 'Serial time: %.2f seconds' %(parallel_time[0])
    for i in range(1,NN):
        print '%d cores: %.2f seconds' %(n_cores[i], parallel_time[i])

    serial_time = parallel_time[0]*np.ones(NN)

    ratio = serial_time/parallel_time
    
    

    plt.figure()
    plt.plot(n_cores, ratio, '-ob')
    plt.xlabel('Number of cores (same node)')
    plt.ylabel('Speedup',fontsize=8)
    plt.xscale('log')
    plt.title('Speedup versus number of cores')
    plt.savefig('mpi_speedup_vs_ncores_Npart%d_Nsteps%d_dt%d_64.png' %(Npart, NNsteps, dt), dpi=600)
    plt.show()