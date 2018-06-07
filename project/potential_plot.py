import numpy as np, time, scipy.optimize
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
'''
Just trying to make an effective potential plot
'''


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

    # Find the acceleration due to object 1.
    venus_sep      = np.array([50, 0])-r
    #print(venus_sep+r)
    venus_sep_norm = np.linalg.norm(venus_sep)
    venus_accel    = 1000*middle/(venus_sep_norm)
    vpot = np.linalg.norm(venus_accel)

    # Find acceleration due to object 0.
    sol_sep      = -r
    sol_sep_norm = np.linalg.norm(sol_sep)
    sol_accel    = 1000*middle/(sol_sep_norm)
    spot = np.linalg.norm(sol_accel)

    # Find the centrifugal acceleration.
    cent_accel = (np.linalg.norm(r)**2)/(0.1)

    net_accel = vpot + spot - cent_accel
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

gridsize = 161

middle = gridsize//2

potential = [[rotatingAccelerator(np.array([x-middle,y-middle])) for x in range(gridsize)] for y in range(gridsize)]

plt.figure(figsize=(5,5))
plt.axes().set_aspect('equal', 'datalim')

plt.pcolormesh(potential)
plt.colorbar()

plt.ylabel('y')
plt.xlabel('x')
#plt.savefig('L5 ensemble.png', bbox_inches='tight', dpi=300)
plt.show()