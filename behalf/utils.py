from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from behalf import octree
from builtins import range
from builtins import open
from builtins import int
from builtins import str
from future import standard_library
from builtins import object
import numpy as np
from time import time
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime, timedelta

__CYTHON_AVAIL = False
try:
    from behalf.force import accel_cython
    __CYTHON_AVAIL = True
except: 
    print("Cython module not loaded")

standard_library.install_aliases()

sys.setrecursionlimit(5000)

def construct_tree(pos, mass):
    sim_box = octree.bbox(np.array([np.min(pos, axis=0),
                                    np.max(pos, axis=0)]).T)
    return octree.octree(pos, mass, sim_box)


def compute_accel(tree, part_ids, theta, G, eps=0.1, cython=True):
    if type(part_ids) == int:
        if(cython and __CYTHON_AVAIL):
            return accel_cython(tree, theta, part_ids, G, eps=eps)
        else:
            return tree.accel(theta, part_ids, G, eps=eps)
    else:
        if(cython and __CYTHON_AVAIL):
            return np.array([accel_cython(tree, theta, p_id, G, eps=eps) for p_id in part_ids])
        else:
            return np.array([tree.accel(theta, p_id, G, eps=eps) for p_id in part_ids])


class TimerCollection(object):
    def __init__(self):
        self.start_times = {}
        self.completed_times = {}

    def start(self, watch_name):
        self.start_times[watch_name] = time()
        if watch_name not in self.completed_times:
            self.completed_times[watch_name] = []

    def stop(self, watch_name):
        try:
            dt = time() - self.start_times.pop(watch_name)
            self.completed_times[watch_name].append(dt)
        except KeyError:
            raise KeyError('No such timer started')

    def iter_medians(self):
        for k in sorted(self.completed_times.keys()):
            yield k, np.median(self.completed_times[k])

    def clear(self):
        self.__init__()

def compute_energy(pos, vel, mass=None, G=4.483e-3):
    """
    Returns the total energy of the system defined by
    input positions (pos), velocities (vel), and masses (defaults
    to 1.0 for all particles), using units defined by choice of G.

    Total energy is the sum of potential and kinetic energy
    (see: compute_potential_energy and compute_kinetic_energy)

    Input:
       pos - positions (N x d)
       vel - velocities (N x d)
       mass - masses (optonal. Default: np.ones(N))
       G - Newton's Constant (optional. Default=1.)

    Output:
       E - total energy (float)
    """
    return compute_potential_energy(pos, mass=mass, G=G) +\
        compute_kinetic_energy(vel, mass=mass)


def compute_potential_energy(pos, mass=None, G=4.483e-3):
    """
    Returns the gravitational potential energy of the system defined by input
    positions (pos) and masses (defaults to 1.0 for all particles),
    using units defined by choice of G.
    
    Potential energy is defined as:
    U = - sum_i ( sum_[j > i] ( G * mass[i] * mass[j] / r_ij))
    r_ij = || pos[i] - pos[j] ||

    Input:
       pos - positions (N x d)
       mass - masses (optonal. Default: np.ones(N))
       G - Newton's Constant (optional. Default=1.)

    Output:
       U - Gravitational potential energy (float)
    """
    pos = np.array(pos).astype(float)
    N_part = pos.shape[0]
    if mass is None:
        mass = np.ones(N_part)
    elif type(mass) is float:
        mass = np.ones(N_part) * mass
    else:
        mass = np.array(mass).astype(float)
    assert mass.shape == (N_part,), ("input masses must match length of "
                                     "input positions")
    U = 0.
    for i in range(N_part):
        m_i = mass[i]
        m_j = mass[i+1:]
        dr = np.linalg.norm(pos[i] - pos[i+1:], axis=1)
        U -= np.sum(G * m_i * m_j / dr)
    return U


def compute_kinetic_energy(vel, mass=None):
    """
    Returns the kinetic of the system defined by input
    velocities and mass (defaults to 1.0 for all particles).
    
    Kinetic energy is defined as:
    K = 1/2 sum_i (mass[i] * ||vel[i]||**2)

    Input:
       vel - velocities (N x 3)
       mass - masses (optonal. Default: np.ones(N))

    Output:
       K - kinetic energy (float)
    """
    vel = np.array(vel).astype(float)
    N_part = vel.shape[0]
    if mass is None:
        mass = np.ones(N_part)
    elif type(mass) is float:
        mass = np.ones(N_part) * mass
    else:
        mass = np.array(mass).astype(float)
    N_part = vel.shape[0]
    assert mass.shape == (N_part,), ("input masses must match length of "
                                     "input velocities")
    return np.sum(vel.T**2. * mass) * 0.5


def save_results(out_file, pos, vel, mass, t_start, iter_num, iter_total,
                 num_cores, G=4.483e-3, timers=None):
    """
    Saves the current state of the simulation to "out_file".
    
    Input:
       out_file - filename to save results to
       pos - array of particle positions (N x d)
       vel - array of particle velocities (N x d)
       mass - array of particle masses (N)
       t_start - start time (in seconds) of the simulation
       iter_num - current time step of the simulation
       iter_total - total number of iterations the simulation will run for
       num_cores - number of cores used for computation
       timers - TimerCollection of custom timers to save
    """
    header = ""
    header += 'Iterations: {:d} of {:d}\n'.format(iter_num+1, iter_total)
    K = compute_kinetic_energy(vel, mass=mass)
    U = compute_potential_energy(pos, mass=mass, G=G)
    E_total = K+U
    header += 'Total Energy: {:.3e}\n'.format(E_total)
    header += '   Kinetic Energy: {:.3e}\n'.format(K)
    header += '   Potential Energy: {:.3e}\n'.format(U)
    header += 'Current Time: {:s}\n'.format(str(datetime.now()))
    dt = time()-t_start
    header += 'Elapsed Time: {:s}\n'.format(str(timedelta(seconds=dt)))
    if timers is not None:
        header += '\nMed. Times for Sections\n'
        for name, avg in timers.iter_medians():
            header += '   {:s}:\t\t{:.6f}\n'.format(name, avg)
            header += '       ['
            for t in timers.completed_times[name]:
                header += '{:.6f}, '.format(t)
            header += ']\n'
    header += '\n'
    header += 'x\ty\tz\tvx\tvy\tvz\n'
    np.savetxt(out_file, np.append(pos, vel, axis=-1), header=header,
               fmt='%+8.4f', delimiter='\t')


def summarize_run(out_file, run_name, N_cores, N_parts, M_total, a, theta, dt,
                  N_steps, softening, seed):
    with open(out_file, 'w') as f:
        f.write('# Run Name: {:s}\n'.format(run_name))
        f.write('# Number of Cores: {:d}\n'.format(N_cores))
        f.write('# Num Particles: {:d}\n'.format(N_parts))
        f.write('# Total Mass: {:.2e} M_sun\n'.format(M_total * 1e9))
        f.write('# Scale Radius: {:.2f} kpc\n'.format(a))
        f.write('# Theta: {:.2f}\n'.format(theta))
        f.write('# Time Step: {:.2g} Myr\n'.format(dt))
        f.write('# Number of Steps: {:d}\n'.format(N_steps))
        f.write('# Force Softening: {:.2f} kpc\n'.format(softening))
        f.write('# Random Seed: {:d}\n'.format(seed))

        
def split_size(N_parts, N_chunks, i):
    """
    Returns number of particles (out of N_parts) distributed to
    chunk i of N_chunks

    Input:
       N_parts - number of particles (int)
       N_chunks - number of chunks to distribute to (int)
       i - which chunk to compute number of particles (int, 0-indexed)

    Example: splitting 1000 particles across 10 chunks
    >>> split_size(1000, 11, 0)
    91
    >>> split_size(1000, 11, 10)
    90
    """
    return (N_parts // N_chunks) + int((N_parts % N_chunks) > i)


def parse_name(run_name):
    """
    Parses the run name into number of cores, nodes, gpus, particles,
    steps, and dt

    Format:
       AAA_B-C_D_E_F
    
       AAA - Run Type (gpu, mpi, etc)
       B - Number of cores
       C - Number of nodes
       D - Number of particles
       E - Number of time steps
       F - Time Precision (1 / dt_Myr)
    """
    run_type, cores, parts, steps, time = run_name.split('_')
    if len(cores.split('-')) == 2:
        cores, nodes = cores.split('-')
    else:
        nodes = '1'
    dt = float('{:.3f}'.format(1./float(time)))  # round to 3 sig figs
    cores = int(cores)
    nodes = int(nodes)
    if 'gpu' not in run_name:
        ngpu = 0
    elif cores > nodes:
        ngpu = 2 * nodes
    else:
        ngpu = 1 * nodes
    parts = int(parts)
    steps = int(steps)
    return cores, nodes, ngpu, parts, steps, dt


class RunResults:
    """
    Class for loading and analyzing results from behalf runs
    """
    def __init__(self, run_name):
        self.run_name = run_name
        path = '../results/' + run_name + '/'
        # Load basic properties from run_name
        self.Ncores, self.Nnodes, self.Ngpu, self.Nparts, self.Nsteps, self.dt = parse_name(run_name)
        # Load data from overivew
        with open(path + 'overview.txt', 'r') as f:
            lines = f.readlines()
            lines = [l.rstrip('\n').split(': ')[-1].split(' ')[0] for l in lines]
        self.Mass = float(lines[3])
        self.a = float(lines[4])
        self.theta = float(lines[5])
        self.softening = float(lines[8])
        self.rand_seed = int(lines[9])
        # check properties are correct
        assert(self.Ncores == int(lines[1]))
        assert(self.Nparts == int(lines[2]))
        assert(self.Nsteps == int(lines[7]))
        assert(self.dt == float(lines[6]))
        
        steps = []
        # figure out what time steps were saved
        for infile in glob( os.path.join(path, '*.dat') ):
            steps.append(int(infile.split('/')[-1].lstrip('step_').strip('.dat')))
        self.steps = sorted(steps)
        self.time_elapsed = np.array(self.steps) * self.dt
        K, U, E = [], [], []
        times = {'Force': [], 'Gather': [], 'Overall': [], 'Scatter': [],
                 'Integ.': [], 'Broadcast': [], 'Tree': [], 'Comm.': []}
        # Load in data from all results files
        for i in self.steps:
            infile = path + 'step_{:d}.dat'.format(i)
            with open(infile, 'r') as f:
                lines = []
                for _ in range(30):
                    l = f.readline()
                    if l.startswith('# '):
                        lines.append(l.lstrip('# ').rstrip('\n').split(':')[-1].lstrip(' \t'))
                    else:
                        break
                # Load energy data
                K.append(float(lines[2]))
                U.append(float(lines[3]))
                E.append(float(lines[1]))
                # Load timing values
                times['Force'].append(float(lines[8]))
                times['Gather'].append(float(lines[10]))
                times['Overall'].append(float(lines[12]))
                times['Scatter'].append(float(lines[14]))
                times['Integ.'].append(float(lines[16]))
                times['Broadcast'].append(float(lines[18]))
                times['Tree'].append(float(lines[20]))
                times['Comm.'].append(times['Gather'][-1] + times['Scatter'][-1] + times['Broadcast'][-1])
        self.med_times = {}
        for k, v in times.items():
            self.med_times[k] = np.median(v)
        self.K = np.array(K)
        self.U = np.array(U)
        self.E = np.array(E)
        
    def plot_energy(self, ax=None, color='k'):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel('Time Elapsed (Myr)')
            ax.set_ylabel('Energy (arb. units)')
        ax.plot(self.time_elapsed, self.K, ls=':', color=color)
        ax.plot(self.time_elapsed, self.U, ls='--', color=color)
        ax.plot(self.time_elapsed, self.E, ls='-', color=color,
                label=self.run_name)
        ax.legend(loc=0)
        return ax
    
    def plot_speedups(self, other, ax=None, color=None, marker=None):
        labels = ['Overall', 'Tree', 'Force', 'Integ.', 'Comm.']
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.colors = sns.color_palette('Set2', 8)[::-1]
            self.markers = ['8', 'p', 'D', 'h', 's', '*', '^', 'o']
            ax.set_xlabel('Code Portion')
            ax.set_xticks(np.arange(5))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Speed-Up (x)')
            ax.set_yscale('log')
            title = '{:d} Parts, {:d} Cores, {:d} Nodes, {:d} GPUs'.format(self.Nparts, self.Ncores, self.Nnodes, self.Ngpu)
            ax.set_title('Baseline: {:s}'.format(title))
            ax.axhline(y=1, ls=':')
        if color is None:
            try:
                color = self.colors.pop()
            except IndexError:
                color = 'r'
        if marker is None:
            try:
                marker = self.markers.pop()
            except IndexError:
                marker = 'o'
        for i, k in enumerate(labels):
            label = None
            if i == 0:
                label = ''
                if other.Nparts != self.Nparts:
                    label += '{:d} Parts, '.format(other.Nparts)
                if other.Ncores != self.Ncores:
                    label += '{:d} Cores, '.format(other.Ncores)
                if other.Nnodes != self.Nnodes:
                    label += '{:d} Nodes, '.format(other.Nnodes)
                if other.Ngpu != self.Ngpu:
                    label += '{:d} GPUs'.format(other.Ngpu)
            speedup = self.med_times[k] / other.med_times[k]
            ax.plot(i, speedup, ls='', marker=marker, color=color, label=label)
        ax.legend(loc=0, frameon=True, fontsize='x-small')
        return ax
