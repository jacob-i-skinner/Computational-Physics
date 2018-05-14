import numpy as np, time, matplotlib.animation as animation, scipy.optimize
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

def L_points():
    # Because the location of venus is a function of time,
    # the lagrangian points are also functions of time,
    # and they orbit at the same angular velocity as Venus.
    # Here they are only used as hubs around which the swarm is initialized.
    # Their positions could be found by rotating the vector to the correct phase.

    # Find L 1,2,3
    

    def radial_a(r):
        # Zeros of this function are points
        # L1, L2, L3.

        G = 6.67428e-11
        R = 1.08208e11
        M1 = 1.989e30
        M2 = 4.867488088236164e+24

        part1 = -G*M1*np.sign(r)/r**2
        part2 = G*M2*np.sign(R-r)/(R-r)**2
        part3 = G*((M1+M2)*r - M2*R)/R**3

        a = part1 + part2 + part3

        return a

    L1 = scipy.optimize.root(radial_a, [1e11]).x[0]
    L2 = scipy.optimize.root(radial_a, [1.1e11]).x[0]
    L3 = scipy.optimize.root(radial_a, [-1e11]).x[0]
    L4 = 1.08208e11*np.array([1, np.sqrt(3)])/2
    L5 = 1.08208e11*np.array([1, -np.sqrt(3)])/2

    return [np.array([L1,0]), np.array([L2,0]), np.array([L3,0]), L4, L5]
#print(L_points())
# Initialize Lagrangian points with appropriate velocities.
L1 = np.array([107200101984.93997, 0, 0, 34693.807965750275])
L2 = np.array([109222195896.43077, 0, 0, 35348.230270906584])
L3 = np.array([-108207845529.61938, 0, 0, -35019.950015715345])
L4 = np.array([5.41040000e10, 9.37108769e10, -35020.00000788533*np.sqrt(3)/2, 35020.00000788533/2])
L5 = np.array([5.41040000e10, -9.37108769e10, 35020.00000788533*np.sqrt(3)/2, 35020.00000788533/2])

def r_venus(t):
    '''
    Because Venus' position is approximated as perfectly circular,
    attracted only by the sun, it moves about the sun at constant
    radius with a constant angular velocity.
    '''

    # position = r(sin(wt), cos(wt))

    r_venus  = 1.08208e11*np.array([np.cos(2*np.pi*t/1.941436081e7),
                          np.sin(2*np.pi*t/1.941436081e7)])
    
    return r_venus

def accelerator(r, t):
    # Calculate the acceleration at r due to Sol and Venus.

    # Find the acceleration due to venus
    venus_sep      = r_venus(t)-r
    venus_sep_norm = np.linalg.norm(venus_sep)
    venus_accel    = venus_sep*(6.67428e-11*4.867488088236164e+24)/(venus_sep_norm**3)

    # Find acceleration due to sun
    sol_sep      = -r
    sol_sep_norm = np.linalg.norm(sol_sep)
    sol_accel    = sol_sep*(6.67428e-11*1.989e30)/(sol_sep_norm**3)

    return venus_accel + sol_accel
#TODO: Make pathfinder loop over each row of a "swarm" array 
def pathFinder(swarm, T, dt):
    '''
    Takes in a run time, delta-t, and info for 3 bodies.
    Returns timestamped coordinates for the specified run time.
    '''
    
    # Convert T from Venusian years to seconds.
    T = round(T*31556926*0.615198)
    
    paths = np.zeros((round(T/dt), swarm.shape[0], 2))

    # Calculate the path for each object - in series.
    for i in range(swarm.shape[0]):
        
        #Store initial values.
        old_pi = np.array([swarm[i,0], swarm[i,1]])
        old_vi = np.array([swarm[i,2], swarm[i,3]])

        paths[0,i,0], paths[0,i,1] = swarm[i,0], swarm[i,1]


        # Use Heun's method to find the i+1th value.
        pi_bar = old_pi + old_vi*dt
        pi = old_pi + (dt/2)*(2*old_vi + accelerator(pi_bar, 0)*dt)

        paths[1,i,0], paths[1,i,1] = pi[0], pi[1]


        t = dt

        # Use Verlet method to calculate trajectory for i'th object.
        for j in range(2, round(T/dt)):
            
            new_pi = 2*pi + accelerator(pi, t)*dt**2 - old_pi

            # Update the position values
            old_pi = pi
            pi = new_pi

            t += dt

            paths[j,i,0], paths[j,i,1] = pi[0], pi[1]

            
    
    return paths

# Time in Venusian years!
T = 200
dt = 100000

# Swarm shape: (number of objects, coords)
swarm = np.array([L1])#, L2, L3, L4, L5])

# Calculate swarm trajectories.
start = time.time()
paths = pathFinder(swarm, T, dt)
end   = time.time()
elapsed = end-start
print('Finished in %s seconds.'%(round(elapsed,1)))
print(paths.shape)


# Plot stuff!

time = np.linspace(0, T*31556926, round(T*31556926/dt))
venus = r_venus(time)


plt.figure(figsize=(8,8))
plt.axes().set_aspect('equal', 'datalim')
plt.grid()
#plt.plot(venus[0],venus[1], 'r')
#plt.scatter(0,0, s=1e3, c='y')
# Plot the swarm.
for i in range(paths.shape[1]):
    plt.plot(paths[::194,i,0], paths[::194,i,1])
#plt.savefig('Orbital Paths.pdf', bbox_inches='tight')
plt.show()
plt.clf()

'''
#plt.plot(paths[1], paths[2], lw=0.1)
plt.plot(paths[3][0::315569], paths[4][0::315569], lw=1)
#plt.plot(paths[5], paths[6], lw=0.1)
plt.savefig('Earth Poincare.pdf', bbox_inches='tight')
plt.show()
'''
'''
# Make an animation
fig, ax = plt.subplots(figsize=(8,8))
plt.xlim(-2e11, 2e11)
plt.ylim(-2e11, 2e11)

def init(): # Clears the frame for the animator.
    return ax.scatter([],[])
def animate(i): # Returns frames to the animator.  
    return ax.scatter(paths[i,0,0], paths[i,0,1], c='k')

ani = animation.FuncAnimation(fig, animate, init_func=init, blit=False, interval = 17)

plt.show()
'''