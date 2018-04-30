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


def main(mass_factor, sim_time, dt):
    array = np.array

    G = 6.67428e-11

    # Masses of the 3 bodies. (kg)
    m_sun, m_earth, m_jupiter = 1.989e30, 5.972e24, 1.898e27*mass_factor

    # Eccentricities of the 2 planetary orbits. 
    e_earth, e_jupiter = 0.0167086, 0.0489

    # Semimajor axes of the 2 planetary orbits.
    a_earth, a_jupiter = 1.49598023e11, 7.7857e11

    # Initial positions of the 3 bodies. [x, y] (m)
    # Each planet is initialized at its aphelion, on opposite sides of the sun.
    ap_sun, ap_earth, ap_jupiter = array([0, 0]), array([-1.521e11, 0]), array([8.1662e11, 0])

    # Reposition each body such that [0,0] is the center of mass of the system.
    CoM = -(m_earth*ap_earth + m_jupiter*ap_jupiter)/(m_sun + m_earth + m_jupiter)
    r_sun, r_earth, r_jupiter = ap_sun + CoM, ap_earth + CoM, ap_jupiter + CoM

    # Calculate initial velocities for the two planets at aphelion using v_min (4.11) from Giordano.
    ev_min = ((G*m_sun)*((1-e_earth)/(a_earth*(1+e_earth)))*(1+m_earth/m_sun))**(1/2)
    jv_min = ((G*m_sun)*((1-e_jupiter)/(a_jupiter*(1+e_jupiter)))*(1+(m_jupiter/mass_factor/m_sun)))**(1/2)
    
    # Initial velocities of the 3 bodies. [x, y] (m/s)
    v_sun, v_earth, v_jupiter = array([0, 0]), array([0, -ev_min]), array([0, jv_min])

    #print('Distance to foci (CoM)', r_jupiter[0])
    #print('Semi-major axis', r_jupiter[0]*(1+e_jupiter)/(1-e_jupiter**2))


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

        m1, old_p1, old_v1 = b1[0], b1[1], b1[2]
        m2, old_p2, old_v2 = b2[0], b2[1], b2[2]
        m3, old_p3, old_v3 = b3[0], b3[1], b3[2]

        paths = np.empty((round(T/dt), 6))

        # Yes, I know this is structured in a silly way.
        #        time  x1         y1         x2         y2         x3         y3
        paths[0] = [old_p1[0], old_p1[1], old_p2[0], old_p2[1], old_p3[0], old_p3[1]]

        # Use Heun's method to update positions of each body.
        p1_bar = old_p1 + old_v1*dt
        p1 = old_p1 + (dt/2)*(2*old_v1+(accelerator(norm, m2, old_p2, p1_bar) + accelerator(norm, m3, old_p3, p1_bar))*dt)

        p2_bar = old_p2 + old_v2*dt
        p2 = old_p2 + (dt/2)*(2*old_v2+(accelerator(norm, m1, old_p1, p2_bar) + accelerator(norm, m3, old_p3, p2_bar))*dt)

        p3_bar = old_p3 + old_v3*dt
        p3 = old_p3 + (dt/2)*(2*old_v3+(accelerator(norm, m1, old_p1, p3_bar) + accelerator(norm, m2, old_p2, p3_bar))*dt)

        t += dt

        paths[1] = [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]

        # Now use the "velocity verlet" technique to step positions forward
        # from the already determined first two values.
        for i in range(2, round(T/dt)):
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
    paths = np.transpose(pathFinder(sim_time, dt, Sun, Earth, Jupiter))
    end   = time.time()
    elapsed = end-start
    print('Finished in %s seconds.'%(round(elapsed,1)))
    
    return paths

plt.figure(figsize=(8,8))
plt.axes().set_aspect('equal', 'datalim')
plt.grid()
# Fecth the paths, but in AU.
paths = main(1000, 500, 100000)/149597870700

plt.plot(paths[0], paths[1], lw=0.1)
plt.plot(paths[2], paths[3], lw=0.1)
plt.plot(paths[4], paths[5], lw=0.1)
plt.title('3-Body Simulation of Earth, Sol, and Jupiter.\nMass of Jupiter = $1000M_J$')
plt.ylabel('y (AU)')
plt.xlabel('x (AU)')
plt.savefig('Orbital Paths1000.pdf', bbox_inches='tight')
plt.show()

'''
#plt.plot(paths[1], paths[2], lw=0.1)
plt.plot(paths[3][0::315569], paths[4][0::315569], lw=1)
#plt.plot(paths[5], paths[6], lw=0.1)
plt.savefig('Earth Poincare.pdf', bbox_inches='tight')
plt.show()
'''