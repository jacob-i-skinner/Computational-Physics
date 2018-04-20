import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'


# Initial velocity of objects 1 & 2.
v0 = np.array([0,0]) #m/s
v1 = np.array([10,0])

# Initial position of the objects
h0 = 3.16557766e3
p = np.array([0,h0])

# These functions perform actual calculations.
def pathFinder(position, velocity, feather_factor, dt):
    '''
    Simulate the path of an object with a given initial position and veloctiy.
    Run until object collides with the ground. Return the time at which the object
    collides with the ground, and a list of timestamped coordinates, giving
    the path the object took through the air. With drag calculations.
    '''
    
    
    norm = np.linalg.norm
    
    # This is a perhaps ill-wrought attempt at optimization
    array = np.array
    g = array([0,-9.81]) #m/s^2
    append = np.append
    
    old_position = position
    old_velocity = velocity
    t=0
    path = array([[t, old_position[0], old_position[1]]])

    while 1:
        
        # Compute new position.
        new_position = old_position + old_velocity*dt
        
        # If new position is below ground (unless dt is already very small),
        # then shrink dt and compute new position again until new
        # position is NOT below ground or dt is too small.
        while new_position[1] < 0 and dt > 1e-9:
            dt = dt/2
            new_position = old_position + old_velocity*dt
        
        # If dt has been shrunk to very small and the new position
        # is STILL below ground, call it quits and break.
        if new_position[1] < 0 and dt <= 1e-9:
            break
        # If the current velocity is zero, do not calculate any drag.
        if old_velocity[0] == 0 and old_velocity[1] == 0:
            new_velocity = old_velocity + g*dt
        else:
        #                             air density * c/m            v**2                     -v hat
            new_velocity = old_velocity + (61.25e-3*(feather_factor)*(norm(old_velocity)**2)*(-old_velocity/norm(old_velocity)) + g)*dt


        old_position = new_position
        old_velocity = new_velocity

        t += dt
        path = append(path, [[t, old_position[0], old_position[1]]], axis=0)

    return path
def noDrag(position, velocity, dt):
    '''
    Same as pathFinder but with no drag.
    '''

    g = np.array([0,-9.81]) #m/s^2

    old_position = position
    old_velocity = velocity
    t=0
    path = np.array([[t, old_position[0], old_position[1]]])
    while 1:
        new_position = old_position + old_velocity*dt
        # This loop shrinks dt just before collision
        # with the ground, unless dt is already very small.
        while new_position[1] < 0 and dt > 1e-9:
            dt = dt/2
            new_position = old_position + old_velocity*dt
        if new_position[1] < 0 and dt <= 1e-9:
            break
        
        new_velocity = old_velocity +  g*dt

        old_position = new_position
        old_velocity = new_velocity

        t += dt
        path = np.append(path, [[t, old_position[0], old_position[1]]], axis=0)

    return path
def diff_with_h():
    # This function finds the difference in landing
    # times as a function of height.
    array = np.array
    transpose = np.transpose
    t_diff = np.empty(1000)
    for i in range(0,1000):
        path2 = transpose(pathFinder(np.array([0,i]), array([10,0]), 0.01, 0.01))
        path1 = transpose(pathFinder(np.array([0,i]), array([0,0]), 0.01, 0.01))
        t_diff[i] = path2[0][-1] - path1[0][-1]
    return t_diff
#t_diff = diff_with_h()
def diff_with_v():
    # This function finds the difference in landing
    # times as a function of launch velocity.
    array = np.array
    p = array([0,3.16557766e3])
    transpose = np.transpose
    t_diff = np.empty(1000)
    for i in range(0,1000):
        path2 = transpose(pathFinder(p, array([i,0]), 0.01, 0.1))
        path1 = transpose(pathFinder(p, array([0,0]), 0.01, 0.1))
        t_diff[i] = path2[0][-1] - path1[0][-1]
    return t_diff
t_diff = diff_with_v()
def diff_with_ff():
    # This function finds the difference in landing
    # times as a function of the "feather factor" C/m.
    array = np.array
    t_diff = np.empty(1000)
    transpose = np.transpose
    for i in range(0,1000):
        path2 = transpose(pathFinder(array([0,3.16557766e3]), array([10,0]), i/1000, 0.1))
        path1 = transpose(pathFinder(array([0,3.16557766e3]), array([0,0]), i/1000, 0.1))
        t_diff[i] = path2[0][-1] - path1[0][-1]
    return t_diff
#t_diff = diff_with_ff()
def deliverables():
    '''
    Create the deliverables!
    '''

    # Save the path.
    path = pathFinder(p, v1, 0.01, 0.01)
    t, x, y = np.transpose(path)[0], np.transpose(path)[1], np.transpose(path)[2]
    
    # Write the path to a file, with nice formatting.
    data = open('Skinner-PHYS486-HW2.tbl', 'w')
    print(' t', '\t\t', 'x', '\t\t', 'y', file = data)
    for i in range(len(t)):
        if len('{:.4f}'.format(t[i])) == 6:
            print('', '{:.4f}'.format(t[i]), '\t', '{:.4f}'.format(x[i]), '\t', '{:.4f}'.format(y[i]), file = data)
        else:
            print('{:.4f}'.format(t[i]), '\t', '{:.4f}'.format(x[i]), '\t', '{:.4f}'.format(y[i]), file = data)
    data.close()
    return
#deliverables()


# Assorted plot code (very boring)
'''
# Show how the outcomes change as dt shrinks.
path1 = np.transpose(pathFinder(p, v1, 0.01, 0.1))
path2 = np.transpose(pathFinder(p, v1, 0.01, 0.01))
path3 = np.transpose(pathFinder(p, v1, 0.01, 0.001))
path4 = np.transpose(pathFinder(p, v1, 0.01, 0.0001))
plt.semilogy(path1[0], path1[2], label='$\Delta t=0.1$')
plt.semilogy(path2[0], path2[2], label='$\Delta t=0.01$')
plt.semilogy(path3[0], path3[2], label='$\Delta t=0.001$')
plt.semilogy(path4[0], path4[2], label='$\Delta t=0.0001$')
plt.ylim(0.001, 5)
plt.xlim(33.91,33.94)
plt.ylabel('Altitude (m)', fontsize=20)
plt.xlabel('Time (s)', fontsize=20)
plt.legend(fontsize=14)
plt.savefig('differing dt.pdf', bbox_inches='tight')
plt.show()

'''
plt.plot(np.linspace(0,999,num=1000), t_diff)
plt.xlim(0,1000)
plt.ylim(0,t_diff[-1])
plt.ylabel('Difference in Landing Time (s)', fontsize=20)
plt.xlabel('Initial Velocity (m/s)', fontsize=20)
plt.savefig('time and v.pdf', bbox_inches='tight')
plt.show()
'''
path0 = np.transpose(pathFinder(p, np.array([0,0]), 0.001, 0.01))
path1 = np.transpose(pathFinder(p, np.array([10,0]), 0.001, 0.01))
#path2 = np.transpose(pathFinder(p, np.array([0.00001,0]), 5e-3, 0.01))
plt.plot(path0[0], path0[2])
plt.plot(path1[0], path1[2])
#plt.plot(path2[0], path2[2])
#plt.xlim(25,26)
#plt.ylim(0,50)
plt.show()
plt.clf()

plt.plot(path1[0], path1[2])
plt.plot(path2[0], path2[2])
#plt.plot(path2[1], path2[2])
plt.show()
'''