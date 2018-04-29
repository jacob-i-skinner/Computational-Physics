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

# Masses of the 3 bodies. (kg)
m_sun, m_earth, m_jupiter = 1.989e30, 5.972e24, 1.898e27

# Eccentricities of the 2 planetary orbits. 
e_earth, e_jupiter = 0.0167086, 0.0489

# Semimajor axes of the 2 planetary orbits.
a_earth, a_jupiter = 1.49598023e11, 7.7857e11

# Initial positions of the 3 bodies. [x, y] (m)
# Each planet is initialized at its aphelion, on opposite sides of the sun.
r_sun, r_earth, r_jupiter = array([0, 0]), array([-1.521e11, 0]), array([8.1662e11, 0])

# Reposition each body such that [0,0] is the center of mass of the system.
CoM = -(m_earth*r_earth + m_jupiter*r_jupiter)/(m_sun + m_earth + m_jupiter)
r_sun, r_earth, r_jupiter = r_sun + CoM, r_earth + CoM, r_jupiter + CoM

x = np.linspace(1,1000,num=10000)

def CoM(f):
    return ((m_earth*r_earth + m_jupiter*f*r_jupiter)/(m_sun + m_earth + f*m_jupiter))[0]

y = [CoM(f) for f in x]

#plt.plot(x, y)
plt.xlim(1,1000)
plt.show()
plt.clf()

def moment(f):
    r_sun, r_earth, r_jupiter = array([0, 0]), array([-1.521e11, 0]), array([8.1662e11, 0])
    
    # Reposition each body such that [0,0] is the center of mass of the system.
    CoM = -(m_earth*r_earth + m_jupiter*f*r_jupiter)/(m_sun + m_earth + f*m_jupiter)
    r_sun, r_earth, r_jupiter = r_sun + CoM, r_earth + CoM, r_jupiter + CoM
    jv_min = ((6.67428e-11*m_sun)*((1-e_jupiter)/(a_jupiter*(1+e_jupiter)))*(1+f*m_jupiter/m_sun))**(1/2)
    p = m_jupiter*f*jv_min
    return p

y= [moment(f) for f in x]

#plt.plot(x, y)
plt.xlim(1,1000)
plt.show()

# Calculate initial velocities for the two planets at aphelion using v_min (4.11) from Giordano.
ev_min = ((6.67428e-11*m_sun)*((1-e_earth)/(a_earth*(1+e_earth)))*(1+m_earth/m_sun))**(1/2)
jv_min = ((6.67428e-11*m_sun)*((1-e_jupiter)/(a_jupiter*(1+e_jupiter)))*(1+m_jupiter/m_sun))**(1/2)

# Initial velocities of the 3 bodies. [x, y] (m/s)
v_sun, v_earth, v_jupiter = array([0, 0]), array([0, -ev_min]), array([0, jv_min])

# Adjust the initial velocity of Sol such that the net momentum of the system is 0.
v_sun = -(m_earth*v_earth + m_jupiter*v_jupiter)/m_sun

print(m_jupiter*r_jupiter[0]*v_jupiter[1])