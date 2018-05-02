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

#import Kevin's numbers for comparison. convert from AU to meters
earth_orbit = np.transpose(1.495978707e11*np.genfromtxt('Earth_1K.txt', usecols=(1,2)))
sun_orbit = np.transpose(1.495978707e11*np.genfromtxt('Sun_1K.txt', usecols=(1,2)))
jupiter_orbit = np.transpose(1.495978707e11*np.genfromtxt('Jupiter_1K.txt', usecols=(1,2)))

def main(mass_factor, sim_time, dt):
    array = np.array

    G = 6.67428e-11

    # Masses of the 3 bodies. (kg)
    m_sun, m_earth, m_jupiter = 1.989e30, 5.972e24, 1.898e27*mass_factor

    # Use Kevin's initial positions
    r_sun, r_earth, r_jupiter = array([sun_orbit[0][1], sun_orbit[1][1]]), array([earth_orbit[0][1], earth_orbit[1][1]]), array([jupiter_orbit[0][1], jupiter_orbit[1][1]])
    
    # move the origin from the sun to the center of mass
    CoM = -(m_earth*r_earth + m_jupiter*r_jupiter)/(m_sun + m_earth + m_jupiter)
    r_sun, r_earth, r_jupiter = r_sun + CoM, r_earth + CoM, r_jupiter + CoM

    # Use Kevin's initial velocities
    v_sun, v_earth, v_jupiter = array([-2.62971e-03, 1.88678e-05]), array([0.00000e+00, -6.28300e+00]), array([2.75477e+00, 0.00000e+00])
    # Convert from AU/year to m/s
    v_sun, v_earth, v_jupiter = 4.74057172e3*v_sun, 4.74057172e3*v_earth, 4.74057172e3*v_jupiter

    # Adjust the initial velocity of Sol such that the net momentum of the system is 0.
    v_sun = -(m_earth*v_earth + m_jupiter*v_jupiter)/m_sun

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
        return   sep*(6.67428e-11*m)/(sep_norm**3)
    def pathFinder(T, dt, b1, b2, b3):
        '''
        Takes in a run time, delta-t, and info for 3 bodies.
        Returns timestamped coordinates for the specified run time.
        '''
        
        # Optimization?
        norm = np.linalg.norm
        
        # Convert T from years to seconds.
        T = T*31556926
        t = 0

        m1, p1, v1 = b1[0], b1[1], b1[2]
        m2, p2, v2 = b2[0], b2[1], b2[2]
        m3, p3, v3 = b3[0], b3[1], b3[2]

        paths = np.empty((round(T/dt)+1, 6), dtype=float)

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
        for i in range(2, round(T/dt)+1):
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

 

# Fecth the paths.
paths = np.transpose(main(1, 1000, 3.1556926e5))

t = np.linspace(0,1000, num = 100001)

#plt.figure(figsize=(16,4))
#plt.plot(t, (paths[4]-jupiter_orbit[0])/149597870700, label='x')
#plt.plot(t, (paths[3]-earth_orbit[1])/149597870700, label='y')
#plt.grid()
plt.plot(paths[0], paths[1])

#plt.plot(paths[2], paths[3])
plt.plot(paths[4], paths[5])
plt.xlabel('t (years)')
plt.ylabel('x position difference (AU)')
#plt.xlim(0,70)
#plt.legend()
#plt.savefig('covey comparison.pdf', bbox_inches='tight')
plt.show()

#np.savetxt('comparison.txt', paths )