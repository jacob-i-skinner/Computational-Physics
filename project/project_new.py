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

# Define some useful constants.
m_venus = 4.8675e24
m_sun   = 1.98955e30
R = 1.08208e11 # Venus orbital radius.
alpha =  m_venus/(m_venus + m_sun) # Venus' proportion of system mass
AU = 1.496e11 # 1 AU in meters.
v_year = 0.615198 # Venus year in Earth years.
P = 1.941436081e7 # orbital period of venus in seconds

# Calculate Lagrangian point locations.
L1 = np.array([R*(1-(alpha/3)**(1/3)), 0])
L2 = np.array([R*(1+(alpha/3)**(1/3)), 0])
L3 = np.array([-R*(1-(5*alpha/12)**(1/3)), 0])
L4 = np.array([0.5*R*((m_sun-m_venus)/(m_sun+m_venus)), np.sqrt(3)*0.5*R])
L5 = np.array([0.5*R*((m_sun-m_venus)/(m_sun+m_venus)), -np.sqrt(3)*0.5*R])


def rotatingAccelerator(r, t):
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

    # Find the acceleration due to venus.
    venus_sep      = np.array([1.08208e11, 0])-r
    venus_sep_norm = np.linalg.norm(venus_sep)
    venus_accel    = venus_sep*(6.67428e-11*4.8675e+24)/(venus_sep_norm**3)

    # Find acceleration due to sun.
    sol_sep      = -r
    sol_sep_norm = np.linalg.norm(sol_sep)
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
    
    # 'paths' stores positions
    paths = np.zeros((round(T/dt), swarm.shape[0], 2))

    # Calculate the path for each object - in series.
    for i in range(swarm.shape[0]):
        
        #Store initial values.
        paths[0,i,0], paths[0,i,1] = swarm[i,0], swarm[i,1]
        old_pi = swarm[i]
        old_vi = np.array([0,0])
        t = dt

        # Use Euler method to calculate trajectory for i'th object at time 'j'.
        for j in range(1, round(T/dt)):
            
            new_vi = old_vi + accelerator(old_pi, t)*dt
            new_pi = old_pi + new_vi*dt

            paths[j,i,0], paths[j,i,1] = new_pi[0], new_pi[1]

            old_vi = new_vi
            old_pi = new_pi

            t += dt
    
    end   = time.time()
    elapsed = end-start
    print('Finished in %s seconds.'%(round(elapsed,1)))
    
    return paths

# Time in Venusian years!
T = 200
real_T = T*31556926*v_year # Time in seconds
dt = 10000

# Swarm shape: (number of objects, coords)
swarm = np.array([L4])

# Calculate swarm trajectories.
paths = pathFinder(swarm, T, dt)

# Transform paths into a rotating reference frame
#paths = rotating(dt,paths)

plt.figure(figsize=(8,8))
plt.axes().set_aspect('equal', 'datalim')
plt.grid()
# Plot the swarm.
for i in range(paths.shape[1]):
    plt.plot(paths[:,i,0]/AU, paths[:,i,1]/AU, label='L%s'%(i+1))

venus = plt.Circle((R/AU, 0), 6051.8e3/AU)
fig = plt.gcf()
ax = fig.gca()
ax.add_artist(venus)

plt.ylabel('y (AU)')
plt.xlabel('x (AU)')
plt.legend(fontsize=16)
#plt.savefig('L5 ensemble.png', bbox_inches='tight', dpi=300)
plt.show()