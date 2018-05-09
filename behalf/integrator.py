# integrator.py
# Ben Cook (bcook@cfa.harvard.edu)
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from future import standard_library
import numpy as np
import os
from mpi4py import MPI
from warnings import warn
standard_library.install_aliases()

rank = MPI.COMM_WORLD.Get_rank()

__GPU_AVAIL = True
try:
    # try to split up the GPU devices
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    n_devices = len(devices)
    my_device = rank % n_devices
    os.environ['CUDA_DEVICE'] = devices[my_device]
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
except:
    __GPU_AVAIL = False


def cuda_timestep(p, v, a, dt):
    """
    Returns the updated positions (p) and velocities (v) given accelerations
    (a) and a timestep dt, using the leap-frog algorithm.
    Implemented using pycuda. If GPU isn't available or initialized, will
    raise a warning, and use the serial version.

    Input:
       p - positions (N x d)
       v - velocities (N x d)
       a - accelerations (N x d)
       dt - time step
    
    Output:
       p - updated positions (N x d)
       v - updated velocites (N x d)
    """
    if not __GPU_AVAIL:
        warn('GPU not available, switching to serial implementation')
        return serial_timestep(p, v, a, dt)
    # for the correct leapfrog condition, assume self-started
    # i.e. p = p(i)
    #      v = v(i - 1/2)
    #      a = F(p(i))
    p_d = gpuarray.to_gpu(np.array(p).astype(np.float32))
    v_d = gpuarray.to_gpu(np.array(v).astype(np.float32))
    a_d = gpuarray.to_gpu(np.array(a).astype(np.float32))

    # kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
    v_d += a_d * dt
    # drift step: x(i+1) = x(i) + v(i + 1/2) dt
    p_d += v_d * dt
    return p_d.get().astype(np.float64), v_d.get().astype(np.float64)


def serial_timestep(p, v, a, dt):
    """
    Returns the updated positions (p) and velocities (v) given accelerations
    (a) and a timestep dt, using the leap-frog algorithm.

    Input:
       p - positions (N x d)
       v - velocities (N x d)
       a - accelerations (N x d)
       dt - time step
    
    Output:
       p - updated positions (N x d)
       v - updated velocites (N x d)
    """
    # for the correct leapfrog condition, assume self-started
    # i.e. p = p(i)
    #      v = v(i - 1/2)
    #      a = F(p(i))
    # kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
    v1 = v + a * dt
    # drift step: x(i+1) = x(i) + v(i + 1/2) dt
    p1 = p + v1 * dt
    return p1, v1
