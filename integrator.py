# integrator.py
# Ben Cook (bcook@cfa.harvard.edu)

import numpy as np


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
        # drift step: x(i+1) = x(i) + v(i + 1/2) dt
        pos += vel * dt
        # compute acceleration: a(i+1) = a(x[i+1], m)
        accel = acc_func(pos, mass, **acc_kwargs)
        # kick step: v(i + 3/2) = v(i + 1/2) + a(i+1) dt
        vel += accel * dt
        yield np.copy(pos), np.copy(vel)
