import numpy as np
import scipy.optimize as sp
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# returns NA, NB, NB/NA.
def NBNA(T, gamma):
    if gamma == 1:
        return np.exp(-T), T*np.exp(-T), T
    return np.exp(-T), (np.exp(-T)-np.exp(-gamma*T))/(gamma-1), (np.exp(-T)-np.exp(-gamma*T))/((gamma-1)*np.exp(-T))

#Minimize me!
def difference(T, ratio, gamma):
    return np.abs(ratio-NBNA(T, gamma)[2])

# This finds the time (T) at which the the an anlytically calculated
# sample ratio matches the measured one. It works by minimizing the
# time function of the difference between them (they should be 0 at the actual age.)
def analyticSolver(ratio, gamma):
    if ratio >= gamma:
        print('WARNING: Unphysical parameters!')
        return
    if gamma ==1:
        return ratio
    return sp.minimize(difference, ratio, args=(ratio, gamma)).x[0]

def numericSolver(ratio, dt, tau_A, tau_B):
    if ratio >= tau_A/tau_B:
        print('WARNING: Unphysical parameters!')
        return

    # Initialize the starting values.
    NA_old = 1
    NB_old = 0
    t = 0
    # Run until the correct ratio is reached.
    while NB_old/NA_old <= ratio:
        # Save the result
        NA_new = -(NA_old/tau_A)*dt + NA_old
        NB_new =  (NA_old/tau_A - NB_old/tau_B)*dt + NB_old
        NA_old = NA_new
        NB_old = NB_new
        t += dt
    return t


# This generates the deliverable necessary to complete the assignment.

def fileBuilder():
    table = open('Skinner-PHYS486-HW1.tbl', 'w')
    print('t/tau_A\tNB/NA', file=table)

    # Initialize the starting values.
    NA_old = 1
    NB_old = 0
    dt = 0.0001
    t = 0
    # Run until the correct ratio is reached.

    while t <= 1.001:
        # Print the result
        print('%s\t%s'%('{0:.3f}'.format(round(t, 3)), '{0:.3f}'.format(round(NB_old/NA_old, 3))), file=table)
        
        NA_new = -(NA_old)*dt + NA_old
        NB_new =  (NA_old - NB_old)*dt + NB_old
        NA_old = NA_new
        NB_old = NB_new
        t += dt
    table.close()
    return
fileBuilder()


#print(numericSolver(2, 0.0001, 1.1, 1)[1])
#print(analyticSolver(2, 1.1))


# Plot the difference between the two as a function of the ratio
analytic_values = np.asarray([analyticSolver(r, 0.7) for r in np.linspace(0,2, num=200)])
numeric_values  = np.asarray([numericSolver(r, 0.0001, 0.7, 1) for r in np.linspace(0,2, num=200)])
r = np.linspace(0,2, num=200)
plt.plot(r, analytic_values-numeric_values, label='$\gamma=0.7$')

analytic_values = np.asarray([analyticSolver(r, 1) for r in np.linspace(0,2, num=200)])
numeric_values  = np.asarray([numericSolver(r, 0.0001, 1, 1) for r in np.linspace(0,2, num=200)])

plt.plot(r, analytic_values-numeric_values, label='$\gamma=1$')

analytic_values = np.asarray([analyticSolver(r, 1.3) for r in np.linspace(0,2, num=200)])
numeric_values  = np.asarray([numericSolver(r, 0.0001, 1.3, 1) for r in np.linspace(0,2, num=200)])

plt.plot(r, analytic_values-numeric_values, label='$\gamma=1.3$')


plt.xlabel('$N_B/N_A$', fontsize=20)
plt.ylabel('Analytic - Numeric', fontsize=20)
plt.legend(fontsize=20)
plt.ylim((analytic_values-numeric_values)[-1], (analyticSolver(2,0.7)-numericSolver(2,0.001,0.7,1)))
plt.xlim(0,2)
#plt.title('Difference Between Analytic and Numeric Results', fontsize=14)
plt.savefig('diff(ratio).pdf', bbox_inches='tight')
plt.show()
plt.clf()

# Plot the difference between the two as a function of the gamma
analytic_values = np.asarray([analyticSolver(0.5, g) for g in np.linspace(0.7,1.3, num=300)])
numeric_values  = np.asarray([numericSolver(0.5, 0.0001, g, 1) for g in np.linspace(0.7,1.3, num=300)])
r = np.linspace(0.7,1.3, num=300)
plt.plot(r, analytic_values-numeric_values)
plt.xlabel('$\gamma$ ($\\frac{N_B}{N_A}$ fixed at 0.5)', fontsize=14)
plt.ylabel('$Analytic - Numeric$', fontsize=20)
plt.ylim((analytic_values-numeric_values)[0], (analytic_values-numeric_values)[-1])
plt.xlim(0.7, 1.3)
#plt.title('Difference Between Analytic and Numeric Results')
plt.savefig('diff(gamma).pdf', bbox_inches='tight')
plt.show()
plt.clf()

# Plot the ratios(t) for varying values of gamma
T = np.linspace(0,2,num=200)
plt.plot(T, NBNA(T, 0.7)[2], label='$\gamma = 0.7$')
plt.plot(T, NBNA(T, 1)[2], label='$\gamma = 1$')
plt.plot(T, NBNA(T, 1.3)[2], label='$\gamma = 1.3$')
plt.legend(fontsize=20)
plt.xlabel('$t/\\tau_A$', fontsize=20)
plt.ylabel('$N_B/N_A$', fontsize=20)
plt.ylim(0, NBNA(T[-1], 0.7)[2])
plt.xlim(0,2)
#plt.title('Sample Ratio as a Function of Sample Age', fontsize=14)
plt.savefig('ratio plot.pdf', bbox_inches='tight')
plt.show()