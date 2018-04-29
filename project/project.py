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


array = np.array

# Masses of the 2 bodies. (kg)
m_sun, m_venus = 1.989e30, 4.8675e24

# Use the reduced mass to simplify the 2-body calculations.
m_venus = m_sun*m_venus/(m_sun + m_venus)

# Eccentricity of Venus
e_venus = 0.006772

# Semimajor axis of Venus
a_venus = 1.49598023e11


# Initial positions of the Venus.
# Venus is initialized at its aphelion, the sun is stationary at the origin.
r_venus  = array([1.08939e11, 0])

v_venus = array([0, ((6.67428e-11*m_sun)*((1-e_venus)/(a_venus*(1+e_venus)))*(1+m_earth/m_sun))**(1/2)])

# Store this info in a single variable (easier to pass to function).
venus = [m_venus, r_venus, v_venus]

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
def pathFinder(T, dt, venus, swarm):
    '''
    Takes in a run time, delta-t, and info for 3 bodies.
    Returns timestamped coordinates for the specified run time.
    '''
    
    # Optimization?
    norm = np.linalg.norm
    
    # Convert T from years to seconds.
    T = T*31556926
    t = 0

    m_venus, r_venus, v_venus = venus[0], venus[1], venus[2]
    m_sun,   r_sun,   v_sun   = 1.989e30, array([0, 0]), array([0, 0])
    
    paths = np.empty((round(T/dt), 7))

    # Yes, I know this is structured in a silly way.
    #        time  x1         y1         x2         y2         x3         y3
    paths[0] = [t, old_p1[0], old_p1[1], old_p2[0], old_p2[1], old_p3[0], old_p3[1]]

    # Use Heun's method to update positions of each body.
    p1_bar = old_p1 + old_v1*dt
    p1 = old_p1 + (dt/2)*(2*old_v1+(accelerator(norm, m2, old_p2, p1_bar) + accelerator(norm, m3, old_p3, p1_bar))*dt)

    p2_bar = old_p2 + old_v2*dt
    p2 = old_p2 + (dt/2)*(2*old_v2+(accelerator(norm, m1, old_p1, p2_bar) + accelerator(norm, m3, old_p3, p2_bar))*dt)

    p3_bar = old_p3 + old_v3*dt
    p3 = old_p3 + (dt/2)*(2*old_v3+(accelerator(norm, m1, old_p1, p3_bar) + accelerator(norm, m2, old_p2, p3_bar))*dt)

    t += dt

    paths[1] = [t, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]

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

        paths[i] = [t, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]
    
    return paths

start = time.time()
paths = np.transpose(pathFinder(sim_time, dt, venus, swarm))
end   = time.time()
elapsed = end-start
print('Finished in %s seconds.'%(round(elapsed,1)))

return paths

plt.figure(figsize=(8,8))
plt.axes().set_aspect('equal', 'datalim')
plt.grid()
plt.plot(paths[1], paths[2], lw=1)
plt.plot(paths[3], paths[4], lw=1)
plt.plot(paths[5], paths[6], lw=1)
plt.savefig('Orbital Paths.pdf', bbox_inches='tight')
plt.show()

'''
#plt.plot(paths[1], paths[2], lw=0.1)
plt.plot(paths[3][0::315569], paths[4][0::315569], lw=1)
#plt.plot(paths[5], paths[6], lw=0.1)
plt.savefig('Earth Poincare.pdf', bbox_inches='tight')
plt.show()
'''