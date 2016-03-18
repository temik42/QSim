import sys
from pathfile import *
import threading
import numpy as np
import Sun
#from radloss import *
import AIA

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
        
        for i in range(0,3):
            Xi[:,i] = 0.5*(X[:,i] + np.roll(X[:,i],1))
         
        self.ds = np.sqrt(np.sum([d(self.X[:,i])**2 for i in range(0,3)],0))
        self.dx = np.sqrt(np.sum([(np.roll(X[:,i],1)-X[:,i])**2 for i in range(0,3)],0))
        self.dx[0] = self.dx[1]
        self._dx = np.roll(self.dx,-1)
        self.ddx = self.dx*self._dx*(self.dx+self._dx)
        self.dxi = np.sqrt(np.sum([(np.roll(Xi[:,i],-1)-Xi[:,i])**2 for i in range(0,3)],0))           
        #self.d2x = np.sum([d(self.X[:,i])*d(self.X[:,i],2) for i in range(0,2)],0)/self.dx
        self.s = np.cumsum(self.ds)
        self.L = self.s[-1]
        
        self.dt = dt
        self.rho = rho
        self.rhou = rhou
        self.rhoe = rhoe
        self.hrate = hrate
        self.g = -d(X[:,2])/self.ds*Sun.g_sun
        self.gamma = gamma
        self.maxiter = np.int(maxiter)
        
        self.nx = self.dx.shape[0]
        self.m = Sun.mu_c*Sun.m_p
        self.T0 = 2e4
        self.rho0 = 1e10*self.m
        self.p0 = 2*self.rho0/self.m*Sun.k_b*self.T0
        
        self.time = 0     
        self.each = each
        
        self.I171 = []
        self.I193 = []
        self.I211 = []
        
        self.load_rl()
        self.status = 'initialized'
        
    def load_rl(self):
        rl = np.load(path + '\\radloss.npz')
        self.rlRate = 10**rl['rlRate']
        self.rlTemperature = 10**rl['temperature']   
    
    def radloss(self, T):
        return np.interp(T, self.rlTemperature, self.rlRate, left = 0, right = 0)
    
    def d_dx(self,q,order = 1):
        if (order == 1):
            out = (self.dx**2*np.roll(q,-1)+(self._dx**2-self.dx**2)*q-self._dx**2*np.roll(q,1))/self.ddx
            out[0] = (-(self.dx[2]**2+2*self.dx[1]*self.dx[2])*q[0]+(self.dx[1]+self.dx[2])**2*q[1]-self.dx[1]**2*q[2])/self.ddx[1]
            out[-1] = (self.dx[-1]**2*q[-3]-(self.dx[-2]+self.dx[-1])**2*q[-2] + 
                       (self.dx[-2]**2+2*self.dx[-1]*self.dx[-2])*q[-1])/self.ddx[-2]
            return out
        if (order == 2):
            out = 2*(self._dx*np.roll(q,1)-(self.dx+self._dx)*q+self.dx*np.roll(q,-1))/self.ddx
            out[0] = out[1]
            out[-1] = out[-2]
            return out
            

    def advect(self,q):
        #dq = q - np.roll(q,1)
        #r = np.where(self.ui >= 0., (np.roll(q,1)-np.roll(q,2))/dq, (np.roll(q,-1)-q)/dq)
        #a = np.min([[np.ones(self.nx)],[2*r]], axis = 0)
        #b = np.min([[2*np.ones(self.nx)],[r]], axis = 0)
        #phi = np.max([[np.zeros(self.nx)],[a],[b]], axis = 0)
        
        flux = np.where(self.ui >= 0., self.ui*np.roll(q,1),self.ui*q)
        #flux = flux + 0.5*np.abs(self.ui)*(1-np.abs(self.ui*self.dt/self.dx))*phi*dq
        
        q -= (self.dt*(np.roll(flux,-1)-flux)/self.dxi)
        

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
        
            
    def _boundary(self):
        self.T[:2] = self.T0
        self.T[-2:] = self.T0
        
        self.p[0] = self.p[2]
        self.p[1] = self.p[3]
        self.p[-1] = self.p[-3]
        self.p[-2] = self.p[-4]
        
        self.rhou[0] = self.rhou[2]
        self.rhou[1] = self.rhou[3]
        self.rhou[-1] = self.rhou[-3]
        self.rhou[-2] = self.rhou[-4]
        
        self.rhoe[:2] = self.p[:2]/(self.gamma-1.)
        self.rhoe[-2:] = self.p[-2:]/(self.gamma-1.)
        
        self.rho[:2] = self.p[:2]/(2*Sun.k_b/self.m*self.T[:2])
        self.rho[-2:] = self.p[-2:]/(2*Sun.k_b/self.m*self.T[-2:])
        
        
    def boundary(self):
        self.T[0] = self.T0
        self.T[-1] = self.T0
        
        self.p[0] = self.p[1]
        self.p[-1] = self.p[-2]
        
        #self.rho[0] = self.rho[1]
        #self.rho[-1] = self.rho[-2]
        
        self.rho[0] = self.p[0]/(2*Sun.k_b/self.m*self.T[0])
        self.rho[-1] = self.p[-1]/(2*Sun.k_b/self.m*self.T[-1])
        
        self.rhoe[0] = self.p[0]/(self.gamma-1.)
        self.rhoe[-1] = self.p[-1]/(self.gamma-1.)
        
        self.rhou[0] = self.rho[0]*(4./7*(0.5*Sun.kappa*(self.T[1]**2.5+self.T[0]**2.5)*(self.T[1]-self.T[0])/self.dx[1]-
                                     1e-31*(self.p[0]/2/Sun.k_b)**2*self.dx[1])/self.p[0]-
                                    3./7*self.rhou[1]/self.rho[1])
        self.rhou[-1] = self.rho[-1]*(4./7*(0.5*Sun.kappa*(self.T[-2]**2.5+self.T[-1]**2.5)*(self.T[-1]-self.T[-2])/self.dx[-1]+
                                     1e-31*(self.p[-1]/2/Sun.k_b)**2*self.dx[-1])/self.p[-1]-
                                    3./7*self.rhou[-2]/self.rho[-2])

    def getptu(self):
        self.u = self.rhou/self.rho
        #self.ui = 0.5*(self.u + np.roll(self.u,1))
        #self.ui[0] = 0.
        
        self.etot = self.rhoe/self.rho
        self.ekin = self.u**2/2
        self.eth = self.etot - self.ekin
        self.p = (self.gamma-1.)*self.rho*self.eth
        #self.p = (self.gamma-1.)*self.rhoe
        self.T = (self.eth*self.m/(3*Sun.k_b))
        #self.T = (self.p/self.rho*self.m/(2*Sun.k_b))
        
    def getui(self):
        self.ui = 0.5*(self.u + np.roll(self.u,1))
        self.ui[0] = 0.
    
    def hydrostep(self):      
        self.getptu()
        self.boundary()
        self.getui()

        self.advect(self.rho)
        self.advect(self.rhou)
        self.advect(self.rhoe)
          
        self.getptu()
        self.boundary()
        self.getui()
        
        self.diffuse()
        
        self.getptu()
        self.boundary()
        self.getui()

        q = np.where(self.ui >= 0., self.ui*np.roll(self.p,1),self.ui*self.p)
        self.rhoe -= self.dt*(np.roll(q,-1)-q)/self.dxi
        #self.rhoe -= self.dt*self.d_dx(self.p*self.u)
        #self.rhou -= self.dt*self.d_dx(self.p)
        
        #self.rhoe -= self.dt*(np.roll(self.p*self.u,-1) - np.roll(self.p*self.u,1))/2/self.dxi
        self.rhou -= self.dt*(np.roll(self.p,-1) - np.roll(self.p,1))/2/self.dxi
        
        self.rhoe -= self.dt*(self.rho/self.m)**2*self.radloss(self.T)
        self.rhoe += self.dt*self.rhou*self.g
        self.rhoe += self.dt*self.hrate(self.time)
             
        self.rhou += self.dt*self.rho*self.g    



        
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
                    I1 = (self.rho/self.m)**2*AIA.t171(self.T)*1e8
                    I2 = (self.rho/self.m)**2*AIA.t193(self.T)*1e8
                    I3 = (self.rho/self.m)**2*AIA.t211(self.T)*1e8
                    self.I171 += [np.sum((I1*self.dx)[np.where(np.abs(self.s-self.L*0.5) < self.L*0.5-self.L*0.15)])/self.L]
                    self.I193 += [np.sum((I2*self.dx)[np.where(np.abs(self.s-self.L*0.5) < self.L*0.5-self.L*0.15)])/self.L]
                    self.I211 += [np.sum((I3*self.dx)[np.where(np.abs(self.s-self.L*0.5) < self.L*0.5-self.L*0.15)])/self.L]
                    
                    plt.subplot(211)
                    plt.title('Density 10^4 / Temperature', size = 18)
                    plt.plot(self.s, np.log10(self.rho/self.m)-4)
                    plt.plot(self.s, np.log10(self.T))
                    plt.axis([0,np.max(self.s),4,7])
                    #plt.axis([0,1e9,4,7])
                    plt.subplot(212)
                    #plt.plot(self.s, self.rhou)
                    #plt.axis([0,1e9,-1e-8,1e-8])
                    plt.title('AIA 171/193/211 intensity', size = 18)
                    plt.plot(self.s,np.log10(I1))
                    plt.plot(self.s,np.log10(I2))
                    plt.plot(self.s,np.log10(I3))
                    plt.axis([0,np.max(self.s),-2,4])
                    
                    fname = (img_dir + str(np.int(i/self.each)).zfill(np.ceil(np.log10(self.maxiter/self.each)).astype(np.int)) + '.png')
                    fig.savefig(fname)
                    fig.clf()

        self.status = 'done'
        