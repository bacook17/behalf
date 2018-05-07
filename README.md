BEHALF
========

BarnEs-Hut ALgorithm For CS205

Please visit out [project website](https://anaroxanapop.github.io/behalf/) to learn more.

### Installation
To install `behalf`, first clone and navigate into the repository. 

From there, it can be installed (with all dependencies, including GPU support) using `make` via
```
make gpu
```

If this fails due to an error with installing PyCUDA, make sure that you have CUDA installed and an NVIDIA device is available. (If on Odyssey, see below). To install only CPU support, use:
```
make cpu
```
or simply (no dependencies installed):
```
make manual
```

### Odyssey GPU installation
First, enable the CUDA module:
```
module load cuda/7.5-fasrc01
```
Then enter an interactive session on a GPU-enabled machine:
```
srun --pty --mem 4096 -n 1 --gres=gpu:1 --constraint=cuda-7.5 -p gpu -t 0-2:00 /bin/bash
```
From there, you should be able to install the GPU support.

### Running BEHALF
```
usage: run_behalf.py [-h] --run-name RUN_NAME --N-parts N_PARTS
                     [--total-mass TOTAL_MASS] [--radius RADIUS]
                     [--N-steps N_STEPS] [--dt DT] [--softening SOFTENING]
                     [--save_every SAVE_EVERY] [--THETA THETA] 
		     [--rand-seed RAND_SEED][--clobber] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --run-name RUN_NAME   REQUIRED - Name of the run (default: None)
  --N-parts N_PARTS     REQUIRED - Number of particles (default: None)
  --total-mass TOTAL_MASS
                        Total mass of the system (in GMsun) (default:
                        100000.0)
  --radius RADIUS       Scale Radius (in kpc) (default: 10.0)
  --N-steps N_STEPS     Number of time steps (default: 1000)
  --dt DT               Size of time step (in Myr) (default: 0.01)
  --softening SOFTENING
                        Softening length (in kpc) (default: 0.01)
  --save_every SAVE_EVERY
                        How often to save output results (default: 10)
  --rand-seed RAND_SEED
                        Random seed to initialize (default: 1234)
  --clobber             Should previous results be overwritten? (default:
                        False)
  --verbose             Should diagnostics be printed? (default: False)
```
