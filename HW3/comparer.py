import numpy as np, time
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

#import Kevin's numbers for comparison
earth_orbit = np.transpose(np.genfromtxt('Earth_1K.txt', usecols=(1,2)))
sun_orbit = np.transpose(np.genfromtxt('Sun_1K.txt', usecols=(1,2)))
jupiter_orbit = np.transpose(np.genfromtxt('Jupiter_1K.txt', usecols=(1,2)))

def main(mass_factor, sim_time, dt):
    array = np.array

    G = 39.43

    # Masses of the 3 bodies. (kg)
    m_sun, m_earth, m_jupiter = 1, 3.003e-6, 0.0009546*mass_factor

    # Use Kevin's initial positions
    r_sun, r_earth, r_jupiter = array([0.00000e+00, 0.00000e+00]), array([1.00000e+00, 0.00000e+00]), array([0.00000e+00, 5.20000e+00])

    # move the origin from the sun to the center of mass
    #CoM = 0# -(m_earth*r_earth + m_jupiter*r_jupiter)/(m_sun + m_earth + m_jupiter)
    #r_sun, r_earth, r_jupiter = r_sun + CoM, r_earth + CoM, r_jupiter + CoM

    # Use Kevin's initial velocities
    v_sun, v_earth, v_jupiter = array([-2.62971e-03, 1.88678e-05]), array([0.00000e+00, -6.28300e+00]), array([2.75477e+00, 0.00000e+00])
    # Convert from AU/year to m/s
    v_sun, v_earth, v_jupiter = v_sun, v_earth, v_jupiter

    # Adjust the initial velocity of Sol such that the net momentum of the system is 0.
    #v_sun = -(m_earth*v_earth + m_jupiter*v_jupiter)/m_sun

    # Store this info in a single variable (easier to pass to function).
    Sun = [m_sun, r_sun, v_sun]
    Earth = [m_earth, r_earth, v_earth]
    Jupiter = [m_jupiter, r_jupiter, v_jupiter]

    def accelerator(norm, m, r, r0):
        '''
        Calculates the acceleration due to Gravity
        that an object feels from another.

        e.g. m is mass of earth, r0 is your position, r is the position of earth
        '''

        # Find the sum of the squares of the components of the
        # separation vector for the two objects.
        sep = r-r0
        sep_norm = norm(sep)

        # return F = GM/R^2
        return   sep*(39.43*m)/(sep_norm**3)
    def pathFinder(T, dt, b1, b2, b3):
        '''
        Takes in a run time, delta-t, and info for 3 bodies.
        Returns timestamped coordinates for the specified run time.
        '''
        
        # Optimization?
        norm = np.linalg.norm
        
        t = 0

        m1, p1, v1 = b1[0], b1[1], b1[2]
        m2, p2, v2 = b2[0], b2[1], b2[2]
        m3, p3, v3 = b3[0], b3[1], b3[2]

        paths = np.empty((round(T/dt)+3, 6), dtype=float)

        # Yes, I know this is structured in a silly way.
        #           x1     y1     x2     y2     x3     y3
        paths[1] = [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]

        # Use an Euler step to find the pre-initial position
        old_p1 = p1 - v1*dt
        old_p2 = p2 - v2*dt
        old_p3 = p3 - v3*dt

        t += dt

        paths[0] = [old_p1[0], old_p1[1], old_p2[0], old_p2[1], old_p3[0], old_p3[1]]

        # Now use the "velocity verlet" technique to step positions forward
        # from the already determined first two values.
        for i in range(2, round(T/dt)+3):
            new_p1 = 2*p1 + (accelerator(norm, m2, p2, p1) + accelerator(norm, m3, p3, p1))*dt**2 - old_p1
            new_p2 = 2*p2 + (accelerator(norm, m1, p1, p2) + accelerator(norm, m3, p3, p2))*dt**2 - old_p2
            new_p3 = 2*p3 + (accelerator(norm, m1, p1, p3) + accelerator(norm, m2, p2, p3))*dt**2 - old_p3

            # Update the position values
            old_p1 = p1
            p1 = new_p1

            old_p2 = p2
            p2 = new_p2

            old_p3 = p3
            p3 = new_p3

            t += dt

            paths[i] = [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]
        
        return paths

    start = time.time()
    paths = pathFinder(sim_time, dt, Sun, Earth, Jupiter)
    end   = time.time()
    elapsed = end-start
    print('Finished in %s seconds.'%(round(elapsed,1)))
    
    return paths

'''
plt.figure(figsize=(16,4))
t = np.linspace(0,1000, num = 100003)
# Fecth the paths.
#paths = np.transpose(main(1000, 1000, 0.01))
#plt.plot(t, np.arctan(paths[3]/paths[2])-np.arctan(earth_orbit[1]/earth_orbit[0]), label='x')
#paths = np.transpose(main(100, 1000, 0.01))
#plt.plot(t, np.arctan(paths[3]/paths[2])-np.arctan(earth_orbit[1]/earth_orbit[0]), label='x')
paths = np.transpose(main(10, 1000, 0.01))
plt.plot(t, np.arctan(paths[1]/paths[0])-np.arctan(sun_orbit[1]/sun_orbit[0]), label='Mass factor = 10')
paths = np.transpose(main(1, 1000, 0.01))
plt.plot(t, np.arctan(paths[1]/paths[0])-np.arctan(sun_orbit[1]/sun_orbit[0]), label='Mass factor = 1')
plt.xlabel('t (years)')
plt.ylabel('differnce in Sol\'s angular position (radians)')
plt.xlim(0,1000)
plt.legend()
plt.savefig('covey comparison sun.pdf', bbox_inches='tight')
plt.show()
'''
paths = np.transpose(main(1, 1000, 0.01))
plt.figure(figsize=(12,12))
plt.axes().set_aspect('equal', 'datalim')
#plt.grid()
'''
plt.hist2d(paths[2]-earth_orbit[0], paths[3]-earth_orbit[1], bins = 50, cmap='viridis')
plt.xlabel('difference in x position (AU)')
plt.ylabel('difference in y position (AU)')
plt.savefig('covey comparison earth.pdf', bbox_inches='tight')


plt.plot(paths[0]-sun_orbit[0], paths[1]-sun_orbit[1])
plt.xlabel('difference in x position (AU)')
plt.ylabel('difference in y position (AU)')
plt.savefig('covey comparison sun.pdf', bbox_inches='tight')
'''

plt.plot(paths[4]-jupiter_orbit[0], paths[5]-jupiter_orbit[1])
plt.xlabel('difference in x position (AU)')
plt.ylabel('difference in y position (AU)')
plt.savefig('covey comparison jupiter.pdf', bbox_inches='tight')

plt.show()