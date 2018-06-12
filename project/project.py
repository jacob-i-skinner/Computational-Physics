import numpy as np, time, scipy.optimize
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
'''
A 3-body gravity simulator. Simulates the mutual attraction and resultant
motion of Sol, Earth, Jupiter. Uses Heun's method to perform an initial step,
then steps into a velocity verlet loop until complete.

All internal calculations are in SI units.
'''

cos = np.cos
sin = np.sin
array = np.array

# Define some useful constants.
m_venus = 4.8675e24
m_sun   = 1.98955e30
R = 1.08208e11 # Venus orbital radius.
alpha =  m_venus/(m_venus + m_sun) # Venus' proportion of system mass
AU = 1.496e11 # 1 AU in meters.
v_year = 0.615198 # Venus year in Earth years.
P = 1.941436081e7 # orbital period of venus in seconds
w = 2*np.pi/P

# Initialize Lagrangian points with appropriate velocities.
L1 = array([107200101984.93997, 0, 0, 0])
L2 = array([109222195896.43077, 0, 0, 0])
L3 = array([-108207845529.61938, 0, 0, 0])
L4 = array([5.41040000e10, 9.37108769e10, 0, 0])
L5 = array([5.41040000e10, -9.37108769e10, 0,0])

L1r, L2r, L3r, L4r, L5r = L1, L2, L3, L4, L5

# Calculate velocities at those locations
# L1,2,3 are all completely in the vertical
L1[2:] = w*np.linalg.norm(L1)*array([cos(np.pi/2),sin(np.pi/2)])
L2[2:] = w*np.linalg.norm(L2)*array([cos(np.pi/2),sin(np.pi/2)])
L3[2:] = w*np.linalg.norm(L3)*array([cos(-np.pi/2),sin(-np.pi/2)])

# L4,5 are both rotated 60 degrees from the vertical
L4[2:] = w*np.linalg.norm(L4)*array([cos(5*np.pi/6),sin(5*np.pi/6)])
L5[2:] = w*np.linalg.norm(L5)*array([cos(np.pi/6),sin(np.pi/6)])

def r_venus(t):
    '''
    Calculate the position of venus at time t.

    Parameters
    ----------
    t : float
        Time inside simulation.
    
    Returns
    -------
    r_venus : numpy array (2)
        Position vector of Venus.
    '''

    # position = r(sin(wt), cos(wt))

    r_venus  = 1.08208e11*array([cos(3.236359604e-7*t),
                          sin(3.236359604e-7*t)])
    
    return r_venus
def inertialAccelerator(r, t):
    '''
    Calculate the net acceleration (Sol + Venus) that an
    object at position r feels at time t.

    Parameters
    ----------
    r : numpy array (2)
        Position vector of object.
    
    t : float
        Time inside simulation.
    
    Returns
    -------
    acceleration : numpy array (2)
        Acceleration vector felt by object.
    '''

    norm = lambda r: (r[0]**2 + r[1]**2)**(0.5)
    # Define time dependent venus position.
    Rv   = lambda t: 1.08208e11*array([cos(3.236359604e-7*t), sin(3.236359604e-7*t)])

    # Find the acceleration due to venus
    venus_sep      = Rv(t)-r
    venus_sep_norm = norm(venus_sep)
    venus_accel    = venus_sep*(6.67428e-11*4.8675e24)/(venus_sep_norm**3)

    # Find acceleration due to sun
    sol_sep      = -r
    sol_sep_norm = norm(sol_sep)
    sol_accel    = sol_sep*(6.67428e-11*1.98955e30)/(sol_sep_norm**3)

    return venus_accel + sol_accel
def inertialPathFinder(swarm, T, dt):
    '''
    Perform gravitational simulation of 2-body Sol-Venus
    system with additional 'massless' small bodies.

    Parameters
    ----------
    swarm : numpy array (n_objects, 4)
        Initial positions and velocities of all
        n objects in the swarm.
    
    T : float
        Total simulation time (in Venusian years).
    
    dt : float
        Simulation timestep in seconds.
    
    Returns
    -------
    paths : numpy array (T/dt, n_objects, 2)
        Array storing position coordinates for 
        all n objects at all times within the simulation.
    '''
    start = time.time()
    # Convert T from Venusian years to seconds.
    T = round(T*31556926*0.615198)
    
    # 'paths' stores positions and velocities
    paths = np.zeros((round(T/dt), swarm.shape[0], 2))

    # Calculate the path for each object - in series.
    for i in range(swarm.shape[0]):
        #Store initial values.
        old_vi = swarm[i,2:]
        old_pi = swarm[i,:2]
        paths[0,i,:] = old_pi

        # Use Heun's method to find 1st position
        pi_bar = old_pi + old_vi*dt
        paths[1,i,:] = old_pi + (dt/2)*(2*old_vi + inertialAccelerator(old_pi, 0) + inertialAccelerator(pi_bar, dt))

        t=dt

        # Use Verlet Integration to calculate all following positions.
        for j in range(2, round(T/dt)):
            paths[j,i,:] = 2*paths[j-1,i,:] + inertialAccelerator(paths[j-1,i,:], t)*dt**2 - paths[j-2,i,:]
            
            t += dt
    
    
    end   = time.time()
    elapsed = end-start
    print('Finished in %s seconds.'%(round(elapsed,1)))

    return paths

# Couldn't get these two working
def rotatingAccelerator(r, v):
    '''
    Calculate the net acceleration (Sol + Venus) that an
    object at position r feels at time t.

    Parameters
    ----------
    r : numpy array (2)
        Position vector of object.
    
    t : float
        Time inside simulation.
    
    Returns
    -------
    acceleration : numpy array (2)
        Acceleration vector felt by object.
    '''

    norm = lambda r: (r[0]**2 + r[1]**2)**(0.5)

    # Find the acceleration due to venus.
    venus_sep      = array([1.08208e11, 0])-r
    venus_sep_norm = norm(venus_sep)
    venus_accel    = venus_sep*(6.67428e-11*4.8675e24)/(venus_sep_norm**3)

    # Find acceleration due to sun.
    sol_sep      = -r
    sol_sep_norm = norm(sol_sep)
    sol_accel    = sol_sep*(6.67428e-11*1.98955e30)/(sol_sep_norm**3)

    # Find the centrifugal acceleration. (2pi/T)**2 is precomputed
    cent_accel = r*1.047402348934733e-13

    # Find the coriolis acceleration. -2 omega X velocity
    corio_accel= -2*np.cross(array([0,0,w]), array([v[0],v[1],0]))

    net_accel = venus_accel + sol_accel + cent_accel + corio_accel[:2]
    return net_accel
def rotatingPathFinder(swarm, T, dt):
    '''
    Perform gravitational simulation (IN A ROTATING REFERENCE FRAME)
    of 2-body Sol-Venus system with additional 'massless' small bodies.

    Parameters
    ----------
    swarm : numpy array (n_objects, 2)
        Initial positions and velocities of all
        n objects in the swarm.
    
    T : float
        Total simulation time (in Venusian years).
    
    dt : float
        Simulation timestep in seconds.
    
    Returns
    -------
    paths : numpy array (T/dt, n_objects, 2)
        Array storing position coordinates for 
        all n objects at all times within the simulation.
    '''
    start = time.time()
    # Convert T from Venusian years to seconds.
    T = round(T*31556926*0.615198)
    
    # 'paths' stores positions and velocities
    paths = np.zeros((round(T/dt), swarm.shape[0], 2))

    # For each object in the swarm.
    for i in range(swarm.shape[0]):
        
        #Store initial values.
        vi = swarm[i,2:]
        paths[0,i,:2] = swarm[i,:2]


        # Use Heun's method to find the i+1th value.
        paths[1,i,:2] = paths[0,i,:2] + vi*dt + 0.5*rotatingAccelerator(paths[0,i,:2], vi)*dt**2
        vi = (paths[1,i,:2]-paths[0,i,:2])/dt

        t = dt

        # Use Verlet method to calculate trajectory for i'th object at time 'j'.
        # Coriolis force is velocity dependent -> this algorithm does not conserve energy.
        for j in range(2, round(T/dt)):
            paths[j,i,:2] = 2*paths[j-1,i,:2] + rotatingAccelerator(paths[j-1,i,:2], vi)*dt**2 - paths[j-2,i,:2]
            vi = (paths[j,i,:2]-paths[j-1,i,:2])/dt

            t += dt
    
    
    end   = time.time()
    elapsed = end-start
    print('Finished in %s seconds.'%(round(elapsed,1)))
    
    return paths
def rotating(dt, paths):
    '''
    Transform position coordinates from a stationary
    reference frame into one corotating with Venus.

    Parameters
    ----------
    dt : float
        Simulation timestep.
    
    paths : numpy array (T/dt, n_objects, 2)
        Position coordinates of each object
        in stationary frame.

    Returns
    -------
    points : numpy array (T/dt, n_objects, 2)
        Position coordinates of each object
        in rotating frame.

    '''
    # Define these locally to speed things up.
    tau = 2*np.pi
    P   = 1.941436081e7
    cos = np.cos
    sin = np.sin

    points = np.zeros(paths.shape)
    
    # For each moment in time.
    for i in range(paths.shape[0]):
        # Define the phase angle corresponding to that time.
        theta = tau*(dt*i%P/P)
        
        # For each object in the swarm.
        for j in range(paths.shape[1]):
            # Explicitly perform 'matrix math' of rotating by -theta.
            points[i,j,0] = cos(theta)*paths[i,j,0] + sin(theta)*paths[i,j,1]
            points[i,j,1] =-sin(theta)*paths[i,j,0] + cos(theta)*paths[i,j,1]
    
    return points

# Time in Venusian years!
T = 2
real_T = T*31556926*v_year # Time in seconds
dt = 1000

'''
# Swarm shape: (number of objects, coords)
swarm = array([L4r, L5r])
# Calculate rotating frame trajectories.
paths = rotatingPathFinder(swarm, T, dt)
'''

# Position perturbation amount
delta = 1e8

# Swarm shape: (number of objects, coords)
swarm = array([L1, L2, L3, L4, L5, array([57909050000, 0, 0, 47362])])
#swarm = array([L4+array([-delta,delta,0,0]), L4+array([-delta,-delta,0,0]), L4+array([-delta,0,0,0]), L4+array([0,0,0,0]),
                #L4+array([0,-delta,0,0]), L4+array([-delta/2,delta/2,0,0]), L4+array([-delta/2,-delta,0,0]), L4+np.array([-delta/2,delta,0,0])])
# Calculate inertial frame trajectories and convert to rotating frame.
paths = inertialPathFinder(swarm, T, dt)
paths = rotating(dt,paths)


#dy = paths[-1,0,1]-r_venus(1)[1]-paths[0,0,1]-r_venus(0)[1]
#dx = paths[-1,0,0]-r_venus(1)[1]-paths[0,0,0]-r_venus(0)[0]
#print(dy, dx)
#print(180 + (180/np.pi)*np.arctan(dy/dx))


plt.figure(figsize=(7,7))
plt.axes().set_aspect('equal', 'datalim')
#plt.grid()
# Plot the swarm.
for i in range(5):
    plt.plot(paths[:,i,0]/AU, paths[:,i,1]/AU, label='L%s'%(i+1))
plt.plot(paths[:,-1,0]/AU, paths[:,-1,1]/AU, label='Mercury')
#plt.plot(0,0,'.')
#plt.plot(dx,dy, '.')
plt.ylabel('y (AU)')
plt.xlabel('x (AU)')

# Plot Venus
venus = plt.Circle((R/AU, 0), 6051.8e3/AU)
fig = plt.gcf()
ax = fig.gca()
ax.add_artist(venus)

# Plot Sol
sol = plt.Circle((0, 0), 695.508e6/AU)
fig = plt.gcf()
ax = fig.gca()
ax.add_artist(sol)

plt.legend(fontsize=16, loc='upper left')
plt.savefig('drift.pdf', bbox_inches='tight')
plt.show()