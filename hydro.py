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


def trisol(D,U,V,B):
    _D = np.array(D)
    _U = np.array(U)
    _V = np.array(V)
    _B = np.array(B)
    
    n = D.shape[0]
    
    for i in range(1,n):
        _D[i] -= _U[i-1]*_V[i-1]/_D[i-1]
        _B[i] -= _B[i-1]*_V[i-1]/_D[i-1]
        
    for i in range(n-2,-1,-1):
        _B[i] -= _B[i+1]*_U[i]/_D[i+1]
        
    return _B/_D


def advect(dxi,q,ui,dt,nghost):
    nx = q.shape[0]
    flux = np.where(ui >= 0., ui*np.roll(q,1),ui*q)
    q[nghost:nx-nghost] -= (dt*(np.roll(flux,-1)-flux)/dxi)[nghost:nx-nghost]
    
    
def _diffuse(dx,dxi,q,T,dt,nghost):
    nx = q.shape[0]
    D = Sun.kappa*T**2.5
    D = 0.5*(np.roll(D,1) + D)
    q[nghost:nx-nghost] += (dt/dxi*(np.roll(D,-1)*(np.roll(T,-1)-T)/dx - D*(T-np.roll(T,1))/np.roll(dx,1)))[nghost:nx-nghost]

    
def diffuse(dx,dxi,w,q,z,dt,nghost):
    nx = w.shape[0]
    
    D = -dx**2/dt-2*q
    D = D[nghost:nx-nghost]
    D[0] = 1.
    D[-1] = 1.
    
    
    U = np.roll(q,-1)
    U = U[nghost:nx-nghost-1]
    U[0] = 0.#2*U[0]
    
    V = np.roll(q,1)
    V = V[nghost+1:nx-nghost]
    V[-1] = 0.#2*V[-1]
    
    B = q*w
    B = np.roll(B,-1)+np.roll(B,1)-2*B - (np.roll(z,-1)+np.roll(z,1)-2*z) - dx**2/dt*w
    B = B[nghost:nx-nghost]
    B[0] = w[nghost]
    B[-1] = w[-1-nghost]
    
    w[nghost:nx-nghost] = trisol(D,U,V,B)
    

  
            
def boundary(rho,rhou,rhoe,condition,nghost):
    nx = rho.shape[0]
    
    if (condition == 'periodic'):
        for i in range(0,nghost):
        
            rho[i] = rho[i-2*nghost]
            rho[-i-1] = rho[2*nghost-i-1]

            rhou[i] = rhou[i-2*nghost]
            rhou[-i-1] = rhou[2*nghost-i-1]

            rhoe[i] = rhoe[i-2*nghost]
            rhoe[-i-1] = rhoe[2*nghost-i-1]
        
    if (condition == 'mirror'):
        for i in range(0,nghost):
            rho[i] = rho[2*nghost-i-1]
            rho[-i-1] = rho[-2*nghost+i]
            
            rhou[i] = -rhou[2*nghost-i-1]
            rhou[-i-1] = -rhou[-2*nghost+i]
            
            rhoe[i] = rhoe[2*nghost-i-1]
            rhoe[-i-1] = rhoe[-2*nghost+i]
 
            
    
def hydrostep(dx,dxi,d2x,rho,rhou,rhoe,gamma,dt,condition):
    nghost = 2
    nx = dx.shape[0]
    boundary(rho,rhou,rhoe,condition,nghost)
    
    u = rhou/rho    
    ui = 0.5*(u + np.roll(u,1))
    
    advect(dxi,rho,ui,dt,nghost)
    advect(dxi,rhou,ui,dt,nghost)
    advect(dxi,rhoe,ui,dt,nghost)
    
    boundary(rho,rhou,rhoe,condition,nghost)
    
    m = Sun.mu_c*Sun.m_p
    u = rhou/rho
    etot = rhoe/rho
    ekin = u**2/2
    eth = etot - ekin
    p = (gamma-1.)*rho*eth
    T = (eth*m/(3*Sun.k_b))
    
    #T[:nghost] = 2e4
    #T[-nghost:] = 2e4
    q = Sun.kappa*T**2.5*m/(3*Sun.k_b)/rho
    z = 2./7*Sun.kappa*T**3.5
    
    rhou[nghost:nx-nghost] -= (dt*d(p)/dx)[nghost:nx-nghost]
    rhoe[nghost:nx-nghost] -= (dt*d(p*u)/dx)[nghost:nx-nghost]
    rhoe -= (dt*(rho/m)**2*2e-22)
    rhoe[nx/2] += dt*1e-1
    #rhoe += dt*1e-4
    
    diffuse(dx,dxi,rhoe,q,z,dt,nghost)
    #_diffuse(dx,dxi,rhoe,T,dt,nghost)