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

# Initialize Lagrangian points with appropriate velocities.
L1 = array([107200101984.93997, 0, 0, 34693.807965750275])
L2 = array([109222195896.43077, 0, 0, 35348.230270906584])
L3 = array([-108207845529.61938, 0, 0, -35019.950015715345])
L4 = array([5.41040000e10, 9.37108769e10, -35020.00000788533*np.sqrt(3)/2, 35020.00000788533/2])
L5 = array([5.41040000e10, -9.37108769e10, 35020.00000788533*np.sqrt(3)/2, 35020.00000788533/2])

# Initialize Lagrangian points (IN ROTATING FRAME) with appropriate velocities.
L1r = array([107200101984.93997, 0, 0, 0])
L2r = array([109222195896.43077, 0, -10, 0])
L3r = array([-108207845529.61938, 0, 10, 10])
L4r = array([5.41040000e10, 9.37108769e10, 5, -5])
L5r = array([5.41040000e10, -9.37108769e10, 0, 0])

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
    venus_accel    = venus_sep*(6.67428e-11*4.8675e+24)/(venus_sep_norm**3)

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
        paths[0,i,:2] = swarm[i,:2]


        # Use Heun's method to find the i+1th value.
        paths[1,i,:2] = paths[0,i,:2] + old_vi*dt + 0.5*inertialAccelerator(paths[0,i,:2], 0)*dt**2

        t = dt

        # Use Verlet method to calculate trajectory for i'th object at time 'j'.
        for j in range(2, round(T/dt)):
            
            paths[j,i,:2] = 2*paths[j-1,i,:2] + inertialAccelerator(paths[j-1,i,:2], t)*dt**2 - paths[j-2,i,:2]

            t += dt
    
    
    end   = time.time()
    elapsed = end-start
    print('Finished in %s seconds.'%(round(elapsed,1)))

    return paths
def rotatingAccelerator(r):
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
    venus_accel    = venus_sep*(6.67428e-11*4.8675e+24)/(venus_sep_norm**3)

    # Find acceleration due to sun.
    sol_sep      = -r
    sol_sep_norm = norm(sol_sep)
    sol_accel    = sol_sep*(6.67428e-11*1.98955e30)/(sol_sep_norm**3)

    # Find the centrifugal acceleration. (2pi/T)**2 is precomputed
    cent_accel = r*1.047402348934733e-13

    net_accel = venus_accel + sol_accel + cent_accel
    return -net_accel
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

    # Calculate the path for each object - in series.
    for i in range(swarm.shape[0]):
        
        #Store initial values.
        old_vi = swarm[i,2:]
        paths[0,i,:2] = swarm[i,:2]


        # Use Heun's method to find the i+1th value.
        paths[1,i,:2] = paths[0,i,:2] + old_vi*dt + 0.5*rotatingAccelerator(paths[0,i,:2], 0)*dt**2

        t = dt

        # Use Verlet method to calculate trajectory for i'th object at time 'j'.
        for j in range(2, round(T/dt)):
            
            paths[j,i,:2] = 2*paths[j-1,i,:2] + rotatingAccelerator(paths[j-1,i,:2], t)*dt**2 - paths[j-2,i,:2]

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
T = 1
real_T = T*31556926*v_year # Time in seconds
dt = 10000

'''
# Swarm shape: (number of objects, coords)
swarm = array([L4r])
# Calculate rotating frame trajectories.
paths = rotatingPathFinder(swarm, T, dt)
'''

# Swarm shape: (number of objects, coords)
swarm = array([L2])
# Calculate inertial frame trajectories and convert to rotating frame.
paths = inertialPathFinder(swarm, T, dt)
paths = rotating(dt,paths)
# As a check, plot Venus' position in the rotating frame (it shouldn't move).
venus = np.zeros((round(real_T/dt), 1, 2))
x = np.linspace(0,real_T,round(real_T/dt))
for i in range(venus.shape[0]):
    venus[i,0,:] = r_venus(x[i])[:]
v_path = rotating(dt, venus)/AU


plt.figure(figsize=(7,7))
plt.axes().set_aspect('equal', 'datalim')
plt.grid()
# Plot the swarm.
for i in range(paths.shape[1]):
    plt.plot(paths[:,i,0]/AU, paths[:,i,1]/AU, label='L%s'%(i+1))
#plt.plot(v_path[:,0,0], v_path[:,0,1])
plt.ylabel('y (AU)')
plt.xlabel('x (AU)')
#plt.legend(fontsize=16)
#plt.savefig('L5 ensemble.png', bbox_inches='tight', dpi=300)
plt.show()