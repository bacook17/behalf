# integrator.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np
import pycuda.gpuarray as gpuarray
# os.environ['CUDA_DEVICE'] = '0'  # figure out how this will work on MPI
import pycuda.autoinit


def cuda_timestep(p, v, a, dt):
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
    return p_d.get(), v_d.get()


def serial_timestep(p, v, a, dt):
    # for the correct leapfrog condition, assume self-started
    # i.e. p = p(i)
    #      v = v(i - 1/2)
    #      a = F(p(i))
    # kick step: v(i + 1/2) = v(i - 1/2) + a(i) * dt
    v += a * dt
    # drift step: x(i+1) = x(i) + v(i + 1/2) dt
    p += v * dt
    return p, v


def serial_leapfrog(pos_init, vel_init, mass, dt, max_steps,
                    acc_func, self_start=True, acc_kwargs={}):
    """
    """
    pos = np.array(pos_init).astype(float)
    vel = np.array(vel_init).astype(float)
    mass = np.array(mass).astype(float)
    
    N_part = pos.shape[0]
    d_space = pos.shape[1]
    assert vel.shape == (N_part, d_space), ("input velocities must mach shape "
                                            "of input positions")
    assert mass.shape == (N_part,), ("input masses must match length of "
                                     "input positions")
    assert type(max_steps) == int, ("max_steps must be an integer")
    assert type(dt) in [int, float], ("dt must be a real number")
    dt = float(dt)

    # the initial self-starting step (Eulerian) to offset velocity by 1/2 dt
    if self_start:
        accel = acc_func(pos, mass, **acc_kwargs)
        assert accel.shape == (N_part, d_space), ("the acceleration function must "
                                                  "return dimensions matching "
                                                  "shape of positions ")
        vel += accel * dt / 2.

    for i in range(max_steps):
        # compute acceleration: a(i+1) = a(x[i+1], m)
        accel = acc_func(pos, mass, **acc_kwargs)
        pos, vel = serial_timestep(pos, vel, accel, dt)
        yield np.copy(pos), np.copy(vel)
