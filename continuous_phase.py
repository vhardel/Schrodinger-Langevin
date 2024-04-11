import matplotlib.pyplot                     as     plt
import numpy                                 as     np
import matplotlib                            as     mpl
import pandas                                as     pd
import math
from   scipy.integrate                       import simpson, trapz


""" Define the plotting parameters """
mpl.rcParams.update({'axes.grid'       :  False , 
                     'grid.color'      : 'black',
                     'grid.linestyle'  : '-.'   ,
                     'grid.linewidth'  :  0.3   ,
                     'text.usetex'     :  True  ,
                     'figure.figsize'  : [10,14],
                     'figure.dpi'      :  100   ,
                     'font.size'       :  28    ,
                     'lines.linewidth' :  4     ,
                     'lines.markersize':  10    ,
                     'xtick.direction' :  'in'  ,
                     'ytick.direction' :  'in' 
                    })
coeffs  =  [0+0j,1+0j,0+0j]#,0+0j]
kmax    =  len(coeffs)
dt      =  0.001
Nt      =  10000
Nx  =  300
x       =  np.linspace(-5,5,Nx)


def harmonic_oscillator_eigenstate(n, x_grid):
    hbar       =  1.0 
    m          =  1.0 
    omega      =  1.0 
    psi_n      =  np.exp(-m * omega * x_grid**2 / (2 * hbar)) * np.polynomial.hermite.hermval(np.sqrt(m * omega / hbar) * x_grid, np.ones(n+1))
    psi_n     /=  np.sqrt(simpson(abs(psi_n)**2, x_grid))
    return psi_n


def phase(x,psi):
    S  =  np.zeros(len(x))
    S[0]  =  0
    for j in range(0,len(x)):
        aux  =  0
        for k in range(1,j+1):
            aux+= np.angle(psi[k]/psi[k-1])
        S[j]  =  S[0] + aux 
    return S

def SS(psi, grid=x):

        s = np.empty(len(grid))
        ds = np.angle(psi[1:]/psi[:-1])
        s[1:] = s[0] + np.cumsum(ds)
        # for i in range(1, grid_size):
        #     s[i] = s[i-1] + np.angle(psi[i]/psi[i-1])

        return s - trapz(x=grid, y=s*(np.abs(psi)**2))

def solve(x,coefs):
    
    sols  =  np.zeros((Nt,kmax),dtype=complex)
    sols[0]  =  coefs
    
    for i in range(1,Nt):
        psi   =  sols[i-1][0]*harmonic_oscillator_eigenstate(0,x) + sols[i-1][1]*harmonic_oscillator_eigenstate(1,x) + sols[i-1][2]*harmonic_oscillator_eigenstate(2,x) #+ sols[i-1][3]*harmonic_oscillator_eigenstate(3,x)
        # for k in range(0,kmax):
        #     psi += sols[i-1][k]*harmonic_oscillator_eigenstate(k,x)

        S  =  phase(x,psi)
        moy_S  =  simpson(S*abs(psi)**2,x)
        s=SS(psi)
        F_n  =  np.zeros(kmax,dtype=complex)
        for n in range(0,kmax):
            F_n[n] = simpson((s)*harmonic_oscillator_eigenstate(n,x)*psi,x)            
            sols[i][n]  = sols[i-1][n] + -1.j*dt*( (n+0.5)*sols[i-1][n] + F_n[n])
        norm  =  0
        for k in range(0,kmax):
            norm += abs(sols[i][k])**2
        if i%100 == 0:
            print("t = %1.3f \t c0: %1.3f \t c1: %1.3f \t c2: %1.3f \t norm = %1.3f"%(i*dt,abs(sols[i][0])**2,abs(sols[i][1])**2, abs(sols[i][2])**2, norm))#, abs(sols[i][3]), norm))
    return sols
    
        

sols  =  solve(x,coeffs)

plt.plot([i*dt for i in range(0,Nt)],abs(sols[:,0])**2)
plt.plot([i*dt for i in range(0,Nt)],abs(sols[:,1])**2)
plt.plot([i*dt for i in range(0,Nt)],abs(sols[:,2])**2)
# plt.plot([i*dt for i in range(0,Nt)],abs(sols[:,3])**2)
plt.show()
