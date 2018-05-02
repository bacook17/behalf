import numpy as np

def plummer(Npart, a, m=1., G=1.):
    """
    Returns the positions and velocities of particles in the Plummer sphere
    input:
        Npart - total number of particles to be initialized
        a - Plummer radius (scale parameter setting the cluster core)
        m - mass of particles (if single value, all particles will be initialized with the same mass; if array, each particles i will have mass m_i)
        G - Gravitational constant; by default, work in units where G=1
    output:
        [pos, vel] - [(Npart x 3), (Npart x 3)] of positions and velocities in cartesian coordinates

    """
    pos = PlummerDist_3d_xyz(Npart, a)
    if np.size(m) == 1:
        M = Npart * m # if all particles have the same mass
    else:
        M = np.sum(m)
    vel = velDist_Plummer(Npart, pos, M, a, G)
    return [pos, vel]

### Spatial Distribution for Plummer Model
def PlummerDist_3d_xyz(Npart,a):
    """
    Initializes particles in 3d with Plummer density profile.
    input: 
        Npart - total number of particles to be initialized
        a - Plummer radius (scale parameter setting the cluster core)
    output:
        pos   - (Npart x 3) array of positions in cartesian coordinates
    """
    r = np.zeros((Npart))
    phi = np.zeros((Npart))
    theta = np.zeros((Npart))
    x = np.zeros((Npart))
    y = np.zeros((Npart))
    z = np.zeros((Npart))
    for i in range(Npart):
        # Let enclosed mass fraction f_mi be random uniform number between 0 and 1 
        f_mi = np.random.uniform(0.,1.)
        r[i] = a/np.sqrt(f_mi**(-2./3.)-1.)
        # Let the angles be chosen uniformly at random
        theta[i] = 2.*np.pi*np.random.uniform(0.,1.)
        phi[i] = np.pi*np.random.uniform(0.,1.)   
        x[i] = r[i]*np.sin(theta[i])*np.cos(phi[i])
        y[i] = r[i]*np.sin(theta[i])*np.sin(phi[i])
        z[i] = r[i]*np.cos(theta[i])
    pos = np.array([x,y,z]).T
    return pos

### Initial velocities for particles in the Plummer model
def velEscape_Plummer(r,M,a,G):
    """
    Returns the escape velocity of a particle in the Plummer model 
    at a radius r from the center of the Plummer sphere
    input: 
        r - radius away from the center of the Plummer sphere
        M - total mass of the Plummer sphere
        a - Plummer radius (scale parameter setting the cluster core)
        G - gravitational constant
    output:
        escape velocity of a particle at radius r inside a Plummer sphere
    """
    r_abs = np.linalg.norm(r)
    pref = np.sqrt(2.*G*M/a)
    return pref*(1.+(r*r)/(a*a))**(-0.25)

def rejTech_velPlummer(r,M,a,G):
    """
    Uses the rejection technique to return a velocity modulus drawn at random 
    from the velocity distribution of particles at radius r in the Plummer model
    input: 
        r - radius away from the center of the Plummer sphere
        M - total mass of the Plummer sphere
        a - Plummer radius (scale parameter setting the cluster core)
        G - gravitational constant
    output:
        absoulte velocity of a particle at radius r inside a Plummer sphere
    """
    x0 = 0. 
    gmax = 0.1 # slightly bigger than g_max = g(\sqrt(2/9)) = 0.092
    g0 = gmax
    while g0 > g_Plummer(x0):
        # 0 <= v <= v_esc or 0 <= x <= 1 where x = v/v_esc
        x0 = np.random.uniform(0.,1.) 
        # 0 <= g <= g_max
        g0 = np.random.uniform(0.,gmax)
    return x0*velEscape_Plummer(r,M,a,G)

def g_Plummer(x):
    return x*x * (1. - x*x)**3.5 

def rand_unit_vector(d):
    """
    Returns d-dimensional random unit vector (norm = 1)
    """
    #r = np.random.randn(d)
    #norm = np.sqrt(np.sum(r**2))
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    theta = np.arccos(costheta)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    vec = np.array([x,y,z])
    return vec

def velDist_Plummer(Npart,r,M,a,G):
    """
    Returns velocities from the velocity distribution of particles 
    at radius r in the Plummer model
    input: 
        Npart - total number of particles to be initialized
        r     - array of radii away from the center of the Plummer sphere, size Npart
        M     - total mass of the Plummer sphere
        a     - Plummer radius (scale parameter setting the cluster core)
        G     - gravitational constant
    output:
        vel   - array of velocities of particles in Plummer model at radii r
    """
    vel = np.zeros((Npart,3))
    for i in range(Npart):
        r_abs = np.linalg.norm(r[i])
        vel_mod = rejTech_velPlummer(r_abs,M,a,G)
        vel[i,:] = rand_unit_vector(3)*vel_mod
    return vel

def circVel_Plummer(r,M,a,G):
    N = r.shape[0]
    vel = np.zeros((N,3))
    for i in range(N):
        vel[i,:] = rand_unit_vector(3)*velEscape_Plummer(np.linalg.norm(r[i]),M,a,G)/2.
    return vel

