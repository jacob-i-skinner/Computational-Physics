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
    venus_sep      = r-r_venus(t)
    venus_sep_norm = np.norm(venus_sep)
    
    # A = GM/R^2.
    venus_accel  = sep*(6.67428e-11*4.867488088236164e+24)/(venus_sep_norm**3)

    sol_sep      = -r
    sol_sep_norm = np.norm(sol_sep)
    sol_accel    = sep*(6.67428e-11*1.989e30)/(sol_sep_norm**3)

    return venus_accel + sol_accel
#TODO: Make pathfinder loop over each row of a "swarm" array 
def pathFinder(T, dt, swarm):
    '''
    Takes in a run time, delta-t, and info for 3 bodies.
    Returns timestamped coordinates for the specified run time.
    '''
    
    # Convert T from years to seconds.
    T = T*31556926
    t = 0
    
    paths = np.empty((round(T/dt), 7))

    # Yes, I know this is structured in a silly way.
    #        time  x1         y1         x2         y2         x3         y3
    paths[0] = [t, old_p1[0], old_p1[1], old_p2[0], old_p2[1], old_p3[0], old_p3[1]]

    # Use Heun's method to update positions of each body.
    p1_bar = old_p1 + old_v1*dt
    p1 = old_p1 + (dt/2)*(2*old_v1+accelerator(m3, old_p3, p1_bar)*dt)

    p2_bar = old_p2 + old_v2*dt
    p2 = old_p2 + (dt/2)*(2*old_v2+accelerator(norm, m3, old_p3, p2_bar)*dt)

    p3_bar = old_p3 + old_v3*dt
    p3 = old_p3 + (dt/2)*(2*old_v3+accelerator(norm, m2, old_p2, p3_bar)*dt)

    t += dt

    paths[1] = [t, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]

    # Now use the "velocity verlet" technique to step positions forward
    # from the already determined first two values.
    for i in range(2, round(T/dt)):
        
        phase = 2*np.pi*t/p_venus

        r_venus = np.array([1.08208e11*np.cos(phase), 1.08208e11*np.sin(phase)])

        new_p1 = 2*p1 + accelerator(norm, m3, p3, p1)*dt**2 - old_p1
        new_p2 = 2*p2 + accelerator(norm, m3, p3, p2)*dt**2 - old_p2
        new_p3 = 2*p3 + accelerator(norm, m2, p2, p3)*dt**2 - old_p3

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
paths = np.transpose(pathFinder(sim_time, dt, swarm))
end   = time.time()
elapsed = end-start
print('Finished in %s seconds.'%(round(elapsed,1)))

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