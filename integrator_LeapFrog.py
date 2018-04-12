import numpy as np
from calc_dens_acc import *

def integration_LeapFrog(x_ini, v_ini, dt, maxTimeSteps, m, G, h, nu, k, npoly):
    """
    Performs the time integration using the Leap Frog Method
    and computes the new particle positions, velocities, accelerations,
    as well as density and pressure at the location of each particle at each time step.
    input: 
        x_ini        - (Npart x 3) array of initial positions of particles
                       in cartesian coordinates
        v_ini        - (Npart x 3) array of initial velocities of particles
                       in cartesian coordinates
        dt           - time step size
        maxTimeSteps - final number of time steps
        m            - mass of each star
        G            - gravitational constant
        h            - smoothing length
        nu           - damping parameter
        k            - pressure constant
        npoly        - polytropic index
    output:
        pos          - (Npart x 3) array of positions in cartesian coordinates
        vel          - (Npart x 3) array of velocities in cartesian coordinates
        acc          - (Npart x 3) array of accelerations in cartesian coordinates
        density      - (Npart)     array of densities at the location of each particle
        pressure     - (Npart)     array of pressures at the location of each particle
    """
    Npart = x_ini.shape[0]
    # initialize positions, velocities, density, pressure, acceleration
    pos      = np.zeros((maxTimeSteps,Npart,3))
    vel      = np.zeros((maxTimeSteps,Npart,3))
    acc      = np.zeros((maxTimeSteps,Npart,3))
    density  = np.zeros((maxTimeSteps,Npart))
    pressure = np.zeros((maxTimeSteps,Npart))

    # initial positions and velocities
    x       = x_ini
    v_mhalf = v_ini
    
    a   = np.zeros((Npart,3))
    rho = np.zeros((Npart))
    P   = np.zeros((Npart))
    for i in range(maxTimeSteps):
        v_phalf = v_mhalf + a * dt
        x += v_phalf * dt
        v = 0.5 * (v_mhalf + v_phalf)
        v_mhalf = v_phalf
        # update densities, pressures, accelerations
        rho = calc_density(x,m,h)
        P   = k * rho**(1. + 1./npoly)
        a   = calc_acceleration(x, v, m, rho, P, nu, G, h) 

        # save values of interest for post-processing
        pos[i,:,:] = x
        vel[i,:,:] = v
        acc[i,:,:] = a
        density[i,:] = rho
        pressure[i,:] = P
    return pos, vel, acc, density, pressure











