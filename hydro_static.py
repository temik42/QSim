import sys
sys.path.append('Q:\\python\\lib')
import numpy as np
import Sun


def d(x, order = 1):
    if (order == 1):
        return np.gradient(x, edge_order = 1)
    if (order == 2):
        out = np.roll(x,1)+np.roll(x,-1)-2*x
        out[0] = out[1]
        out[-1] = out[-2]
    return out


def diffuse(dx,dt,X):
    nx = X.shape[0]
    D = Sun.kappa*2./7
    X += (D*dt/dx**2*(np.roll(X,-1)+np.roll(X,1)-2*X))

        
def boundary(dx,X):
    nx = X.shape[0]
    
    X[-1] = X[-3]  
    T0 = 2e4
    X[0] = T0**3.5             
 
  
def hydrostep(dx,dt,X,e):
    nghost = 2
    nx = dx.shape[0]

    E = np.sum(e*dx)
    diffuse(dx,dt,X)
    
    T = X**(2./7)
    n = 1e9*1e6/T
    
    lam = 2e-22
    X -= n**2*lam
    X += e*dt              
    boundary(dx,X)