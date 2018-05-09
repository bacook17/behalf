from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
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
from datetime import datetime, timedelta

standard_library.install_aliases()

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
