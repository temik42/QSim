import sys
from pathfile import *
import threading
import numpy as np
import Sun
from radloss import *


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


class hydro(threading.Thread):
    def __init__(self,dt,X,rho,rhou,rhoe,hrate,gamma=5./3,maxiter=1e3,each=0):
        threading.Thread.__init__(self)
        self.X = X
        Xi = np.zeros_like(self.X)
        
        for i in range(0,2):
            Xi[:,i] = 0.5*(X[:,i] + np.roll(X[:,i],1))
         
        self.dx = np.sqrt(np.sum([d(self.X[:,i])**2 for i in range(0,2)],0))
        self._dx = np.roll(self.dx,-1)
        self.ddx = self.dx*self._dx*(self.dx+self._dx)
        self.dxi = np.sqrt(np.sum([(np.roll(Xi[:,i],-1)-Xi[:,i])**2 for i in range(0,2)],0))           
        #self.d2x = np.sum([d(self.X[:,i])*d(self.X[:,i],2) for i in range(0,2)],0)/self.dx
        self.s = np.cumsum(self.dx)
        
        self.dt = dt
        self.rho = rho
        self.rhou = rhou
        self.rhoe = rhoe
        self.hrate = hrate
        self.gamma = gamma
        self.maxiter = np.int(maxiter)
        
        self.nx = self.dx.shape[0]
        self.m = Sun.mu_c*Sun.m_p
        self.T0 = 2e4
        
        self.time = 0
        self.status = 'initialized'
        self.each = each

        
    def d_dx(self,q,order = 1):
        if (order == 1):
            out = (self.dx**2*np.roll(q,-1)+(self._dx**2-self.dx**2)*q-self._dx**2*np.roll(q,1))/self.ddx
            out[0] = (-(self.dx[1]**2+2*self.dx[0]*self.dx[1])*q[0]+(self.dx[0]+self.dx[1])**2*q[1]-self.dx[0]**2*q[2])/self.ddx[0]
            out[-1] = (self.dx[-2]**2*q[-3]-(self.dx[-3]+self.dx[-2])**2*q[-2] + 
                       (self.dx[-3]**2+2*self.dx[-2]*self.dx[-3])*q[-1])/self.ddx[-2]
            return out
        if (order == 2):
            out = 2*(self._dx*np.roll(q,1)-(self.dx+self._dx)*q+self.dx*np.roll(q,-1))/self.ddx
            out[0] = out[1]
            out[-1] = out[-2]
            return out
            

    def advect(self,q):
        flux = np.where(self.ui >= 0., self.ui*np.roll(q,1),self.ui*q)
        q -= (self.dt*(np.roll(flux,-1)-flux)/self.dxi)
    
    def _diffuse(self):
        dd = Sun.kappa*self.T**2.5*self.m/(3*Sun.k_b)/self.rho
        z = 2./7*Sun.kappa*self.T**3.5

        D = -self.dx**2/self.dt-2*dd
        D[0] = 1.
        D[-1] = 1.

        U = np.roll(dd,-1)
        U = U[:self.nx-1]
        U[0] = -1.

        V = np.roll(dd,1)
        V = V[1:]
        V[-1] = -1.

        B = dd*self.rhoe
        B = d(B,2) - d(z,2) - self.dx**2/self.dt*self.rhoe
        
        B[0] = 0.
        B[-1] = 0.
        
        self.rhoe = trisol(D,U,V,B)

    def diffuse(self):
        
        dd = Sun.kappa*self.T**2.5*self.m/(3*Sun.k_b)/self.rho
        z = 2./7*Sun.kappa*self.T**3.5

        D = -1/self.dt-2*dd/(self.dx*self._dx)
        D[0] = 1.
        D[-1] = 1.

        U = 2*np.roll(dd,-1)/(self._dx*(self.dx+self._dx))
        U = U[:self.nx-1]
        U[0] = -1.

        V = 2*np.roll(dd,1)/(self.dx*(self.dx+self._dx))
        V = V[1:]
        V[-1] = -1.

        B = dd*self.rhoe
        B = self.d_dx(B,2) - self.d_dx(z,2) - self.rhoe/self.dt
        
        B[0] = 0.
        B[-1] = 0.
        
        self.rhoe = trisol(D,U,V,B)
        
            
    def boundary(self):
        #self.T[0] = self.T0
        #self.T[-1] = self.T0

        self.rho[0] = self.rhoe[1]/(3*Sun.k_b/self.m*self.T0)
        self.rho[-1] = self.rhoe[-2]/(3*Sun.k_b/self.m*self.T0)
        
        #self.rho[0] = self.rho[1]
        #self.rho[-1] = self.rho[-2]
   
        #F0 = 2./7*Sun.kappa*(self.T[1]**3.5-self.T0**3.5)/self.dx[0]
        #_F0 = 2./7*Sun.kappa*(self.T0**3.5-self.T[-2]**3.5)/self.dx[-1]
            
        #self.rhou[0] = 0.2*F0/(Sun.k_b*self.T0)*self.m
        #self.rhou[-1] = 0.2*_F0/(Sun.k_b*self.T0)*self.m
        
        self.rhou[0] = self.rhou[1]
        self.rhou[-1] = self.rhou[-2]
        
        self.rhoe[0] = self.rhoe[1]
        self.rhoe[-1] = self.rhoe[-2]
 
    def getpt(self):
        self.etot = self.rhoe/self.rho
        self.ekin = self.u**2/2
        self.eth = self.etot - self.ekin
        self.p = (self.gamma-1.)*self.rho*self.eth
        self.T = (self.eth*self.m/(3*Sun.k_b))
    
    def hydrostep(self):
        self.u = self.rhou/self.rho    
        self.ui = 0.5*(self.u + np.roll(self.u,1))
        
        self.boundary()

        self.advect(self.rho)
        self.advect(self.rhou)
        self.advect(self.rhoe)
        
        self.boundary()
        self.getpt()
             
        self.rhoe -= self.dt*(self.rho/self.m)**2*radloss(self.T)
        self.rhou -= self.dt*self.d_dx(self.p)
        self.rhoe -= self.dt*self.d_dx(self.p*self.u)
        self.rhoe += self.dt*self.hrate
        
        self.boundary()
        self.getpt()
        
        self.diffuse()
        
        


        
    def run(self):     
        self.status = 'in progress'
        if (self.each != 0):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure()
        
        for i in range(0,self.maxiter):
            self.hydrostep()
            self.time += self.dt
            if (self.each != 0):
                if (i % self.each == 0):
                    plt.subplot(211)
                    plt.title('Density', size = 18)
                    plt.plot(self.s, np.log10(self.rho/self.m))
                    plt.axis([0,np.max(self.s),8,11])
                    plt.subplot(212)
                    plt.title('Temperature', size = 18)
                    plt.plot(self.s, np.log10(self.T))
                    plt.axis([0,np.max(self.s),4,7])
                    
                    fname = (img_dir + str(np.int(i/self.each)).zfill(np.ceil(np.log10(self.maxiter/self.each)).astype(np.int)) + '.png')
                    fig.savefig(fname)
                    fig.clf()

        self.status = 'done'
        