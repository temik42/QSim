# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 14:37:16 2015

@author: artem
"""

import numpy as np
#from scipy.ndimage.filters import gaussian_filter
from numpy import fft
from numpy import linalg
#from scipy import interpolate
#from scipy import optimize
from sklearn.cluster import DBSCAN
#from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import time


def FField(data, z=0):

    dim = np.shape(data)

    temp = np.pad(data,((0,dim[0]),(0,dim[1])),'constant')
    
    xi = np.roll(np.arange(-dim[0],dim[0]),dim[0])
    yi = np.roll(np.arange(-dim[1],dim[1]),dim[1])
    
    xi = np.resize(xi,(2*dim[0],2*dim[1]))/(2.*dim[0])
    yi = (np.resize(yi,(2*dim[1],2*dim[0]))/(2.*dim[1])).T
        
    q = np.sqrt(xi**2 + yi**2)
    fdata = fft.fft2(temp)*np.exp(-2*np.pi*q*z) 
    
    b = np.real(fft.ifft2(fdata))[0:dim[0],0:dim[1]]    
    
    bx = np.real(fft.ifft2(fdata*1j*xi*2*np.pi))[0:dim[0],0:dim[1]]
    by = np.real(fft.ifft2(fdata*1j*yi*2*np.pi))[0:dim[0],0:dim[1]]
    
    bxx = np.real(fft.ifft2(-fdata*xi**2*(2*np.pi)**2))[0:dim[0],0:dim[1]]
    byy = np.real(fft.ifft2(-fdata*yi**2*(2*np.pi)**2))[0:dim[0],0:dim[1]]
    bxy = np.real(fft.ifft2(-fdata*xi*yi*(2*np.pi)**2))[0:dim[0],0:dim[1]]
    

    return [b, bx, by, bxx, bxy, byy]


class fLine:
    def __init__(self, field, verts, ends, color=(0,0,0), type='standard'):
        self.field = field
        self.verts = verts
        self.ends = ends
        self.color = color
        self.type = type
        
        if self.type == 'fan':
            self.linestyle = '--'
        else:
            self.linestyle = '-'
    #!!!!!!!!!!!!!!!!! FIX ME!!!!!!!!!!!!!    
    #def colored(self, a=0.8):
    #    if self.ends[0] >= 0 and self.ends[1] >= 0:
    #        color = str(float((np.argwhere(self.field.posi == self.ends[1])*self.field.negi.shape[0]+
    #                           np.argwhere(self.field.negi == self.ends[0])))/(self.field.posi.shape[0]*self.field.negi.shape[0])+0.0001)  
    #        r,g,b = float(color[2])/10.*a, float(color[3])/10.*a, float(color[4])/10.*a
    #    else:
    #        r,g,b = .4,.4,.4
    #    self.color = (r,g,b)
    #    return self
"""
class fPlane:
    def __init__(self, field, data):
        print 'Interpolating field...'
        time0 = time.time()
        
        self.field = field
        self.data = FouField(data, 1)[0]
        self.dim = np.shape(self.data)

        self.skeleton = skeletonize(np.abs(self.data) > self.field.threshold)
        self.iData = interpolate.interp2d(np.arange(self.dim[0]),np.arange(self.dim[1]),self.data)
        print 'Done! Elapsed time: '+str(time.time()-time0)        
"""
        
class fNull:
    def __init__(self, field, x, index=0):
        self.field = field
        self.x = x
        self.eigen = linalg.eig(self.field.jacobian(self.x))
        self.det = self.eigen[0][0]*self.eigen[0][1]*self.eigen[0][2]   
        
        if np.sign(self.det) < 0:
            self.sign = 'pos'
            self.marker = 'v'
            self.name = 'B' + str(index)
        else:
            self.sign = 'neg'
            self.marker = '^'
            self.name = 'A' + str(index)
        
        if np.sign(self.det)*np.sign(self.eigen[0][2]) < 0:
            self.type = 'prone'
        else:
            self.type = 'upright'
            self.name = self.name + 'u'
        
        dx = self.eigen[1]
        self.flines = []
        
        for j in range(0,3):
            for i in range(0,2):
                mul = np.sign(self.eigen[0][j])
                if np.sign(self.det)*mul < 0:
                    type = 'fan'
                else:
                    type = 'spine'
                f = self.field.rk2(self.x + (-1)**i*dx[:,j], mul = mul)
                fline = fLine(self.field, np.column_stack((self.x,f[0])), [self,f[1]], type = type)
                self.flines.extend([fline])
        

class fCharge:
    def __init__(self, field, q, x):
        self.field = field
        self.q = q
        self.x = x
        
        if self.q > 0:
            self.marker = '+'
            self.markersize = 10
            self.sign = 'pos'
            self.name = 'P'+str(field.n_pos+1)
            self.field.n_pos += 1
        else:
            self.marker = 'x'
            self.markersize = 8
            self.sign = 'neg'
            self.name = 'N'+str(field.n_neg+1)
            self.field.n_neg += 1

class fInfinity:
    def __init__(self, field, q):
        self.field = field
        self.q = q
        self.x = np.inf

class Field:
    def __init__(self, data, z = 1, nz = 64):
        self.data = FField(data, z)
        self.dim = np.append(np.array(self.data[0].shape), nz)

    def X(self, sign = 'all'):
        return np.array([charge.x for charge in self.Charges if charge.sign == sign or sign == 'all'])
    
    def Q(self, sign = 'all'):
        return np.array([charge.q for charge in self.Charges if charge.sign == sign or sign == 'all'])
    
    def get(self, x):
        dx = (x - self.X())
        dr = np.sqrt(np.sum(dx**2, axis = 1)).clip(min = 1e-10)
        self._minr = np.min(dr)
        self._argminr = np.argmin(dr)
        return np.dot(self.Q()/dr**3, dx)    
    
    def jacobian(self, x):
        dx = (x - self.X())
        dr = np.sqrt(np.sum(dx**2, axis = 1)).clip(min = 1e-10)
        return np.sum(self.Q()/dr**3)*np.identity(3)-3*np.dot(self.Q()/dr**5*dx.T, dx)
    
    def set_charges(self, q, x): 
        self.n_charges = q.shape[0]
        self.n_pos = 0
        self.n_neg = 0
        
        self.Charges = []
        for i in range(0,self.n_charges):
            self.Charges.extend([fCharge(self, q[i], x[:,i])])
            
        self.Infinity = fInfinity(self, -np.sum(q))
        return self

    def search_charges(self, threshold = 30):
        print 'Searching charges...'
        time0 = time.time()        
        
        det = self.data[3]*self.data[5]-self.data[4]**2

        dx = -(self.data[1]*self.data[5]-self.data[2]*self.data[4])/det
        dy = -(self.data[2]*self.data[3]-self.data[1]*self.data[4])/det
        
        datamax = self.data[0]+self.data[1]*dx+self.data[2]*dy+self.data[3]*dx**2/2+self.data[4]*dx*dy+self.data[5]*dy**2/2        
        t = np.where((np.abs(dx) < 1)*(np.abs(dy) < 1)*(np.abs(datamax) > threshold)*(det > 0))        
        
        x = np.array([t[1]+dx[t], t[0]+dy[t]]).T
        
        db = DBSCAN(min_samples = 1, eps = 1)
        db.fit_predict(x)
        
        n_charges = np.max(db.labels_)+1
        qi = np.zeros(n_charges)
        xi = np.zeros((3,n_charges))
        
        for i in range(0, n_charges):
            xi[0:2,i] = np.mean(x[db.labels_ == i,:], axis=0)
            qi[i] = np.mean(datamax[t][db.labels_ == i])
        
        
        self.set_charges(qi,xi)
        print 'Done! Elapsed time: '+str(time.time()-time0)
        return self
        
    def search_nulls(self, maxi = 10, h = 0.1, d = 0.5):
        
        print 'Searching nulls...'
        time0 = time.time()                
        
        xx, yy = np.mgrid[0:self.dim[0]:d, 0:self.dim[1]:d]
        zz = np.zeros(self.dim[0]*self.dim[1]/d**2)

        xx = xx.flatten()
        yy = yy.flatten()
        
        #for i in range(0, maxi):
        #    self.Update(np.array([xx, yy, zz])).Get().Jacobian()
        #    p2 = np.sum(self._p**2, axis = 0)
        #    gr = np.sum(self._jacobian*self._p, axis = 1)
        #    f = p2/gr[0:2,:]
        #    xx = xx - f[0,:]*0.5
        #    yy = yy - f[1,:]*0.5      
        
        for i in range(0, maxi):
            
            self.Update(np.array([xx, yy, zz])).Get().Jacobian()
            gr = np.sum(self._jacobian*self._p, axis = 1)
        
            self.Update(np.array([xx+h, yy, zz])).Get().Jacobian()
            hesx = np.sum(self._jacobian*self._p, axis = 1)
        
            self.Update(np.array([xx-h, yy, zz])).Get().Jacobian()
            hesx = (hesx - np.sum(self._jacobian*self._p, axis = 1))/(2*h)
    
            self.Update(np.array([xx, yy+h, zz])).Get().Jacobian()
            hesy = np.sum(self._jacobian*self._p, axis = 1)
        
            self.Update(np.array([xx, yy-h, zz])).Get().Jacobian()
            hesy = (hesy - np.sum(self._jacobian*self._p, axis = 1))/(2*h)
        
            det = hesx[0,:]*hesy[1,:] - hesx[1,:]*hesy[0,:]
            
            dx = (gr[0,:]*hesy[1,:] - gr[1,:]*hesx[1,:])/det
            dy = (gr[1,:]*hesx[0,:] - gr[0,:]*hesy[0,:])/det
        
            t = np.where((np.abs(dx) < d)*(np.abs(dy) < d))
            
            xx = xx[t] - dx[t]
            yy = yy[t] - dy[t]
            zz = zz[t]

        self.Update(np.array([xx, yy, zz])).Get()
        p2 = np.sum(self._p**2, axis = 0)
            
        t = np.where(p2 < 1e-8)
        xx = xx[t]
        yy = yy[t]
        zz = zz[t]        
        
        xn = np.array([xx,yy,zz]).T
        
        db = DBSCAN(min_samples = 1, eps = 0.5)
        db.fit_predict(xn)

        self.n_nulls = np.max(db.labels_)+1
        self.Nulls = []
        
        for i in range(0, self.n_nulls):
            xi = np.mean(xn[db.labels_ == i,:], axis=0)
            self.Nulls.extend([fNull(self, xi, i+1)])
        
        print 'Done! Elapsed time: '+str(time.time()-time0)    
        return self

    def update(self, x):  
        self._x = x
        self._dx = (self._x - self.X())
        self._dr = np.sqrt(np.sum(self._dx**2, axis = 1))
        _f = self.Q()/self._dr**3
        
        self._p = np.dot(_f, self._dx)
        self._jacobian = np.sum(_f)*np.identity(3)-3*np.dot(_f/self._dr**2*self._dx.T, self._dx)                
        self._minr = np.min(self._dr)
        self._argminr = np.argmin(self._dr)
        
    def rk2(self, x, mul = 1, maxi = 200, step = 0.5):
        self.update(x)
        stack = self._x
        
        i = 0      
        while i < maxi:       
            dx = self._p/linalg.norm(self._p)*step*mul
            
            if self._minr < step:
                stack = np.column_stack((stack,self.X()[self._argminr,:]))
                return (stack, self.Charges[self._argminr])
            else:                    
                self._p = self._p + 0.5*np.dot(self._jacobian, dx)
                dx = self._p/linalg.norm(self._p)*step*mul
                self.update(self._x + dx)
                stack = np.column_stack((stack,self._x))
            i += 1
        return (stack, self.Infinity)
    
    def Get(self):
        self._p = np.array([np.sum(self._f*dx, axis = 1) for dx in self._dx])
        return self
    
    def Jacobian(self):
        n = self._x.shape[1]
        F = np.sum(self._f, axis = 1)
        fr = self._f/self._dr
        self._jacobian = (np.array([[F if i==j else np.zeros(n) for i in range(0,3)] for j in range(0,3)])+
                    np.array([[-3.*np.sum(fr*dxi*dxj, axis = 1) for dxi in self._dx] for dxj in self._dx]))
        return self
           
    def Update(self, x):
        self._x = x
        self._dx = [np.resize(self._x[i,:], (self.n_charges,self._x.shape[1])).T - self.X()[:,i] for i in range(0,3)]
        self._dr = sum([self._dx[i]**2. for i in range(0,3)]).clip(min = 1e-10) 
        self._f = self.Q()/self._dr**1.5        
        self._argminr = np.argmin(self._dr, axis = 1)
        self._minr = np.min(self._dr, axis = 1)    
        return self
    
    def Rk2(self, x, mul = 1, maxi = 200, step = 0.5):
        self.Update(x).Get().Jacobian()
        n = x.shape[1]
        
        i = 0
        
        result = - np.ones(n)
        ids = np.where(np.ones(n))[0]      
        
        while i < maxi:                   
            
            dx = self._p/np.sqrt(np.sum(self._p**2., axis=0))*step*mul
            self._p = self._p + 0.5*np.sum(self._jacobian*dx, axis = 1)
            dx = self._p/np.sqrt(np.sum(self._p**2., axis=0))*step*mul
            
            self.Update(self._x + dx).Get().Jacobian()

            t = np.where(self._minr > 0.25*step**2)[0]
            _t = np.where(self._minr < 0.25*step**2)[0]
            result[ids[_t]] = self._argminr[_t]
            ids = ids[t]
            
            self._p = self._p[:,t]
            self._x = self._x[:,t]  
            self._jacobian = self._jacobian[:,:,t]            
 
            i += 1
    
        return result.astype(np.int16)        
    
    def connectivity(self):
        xx, yy = np.meshgrid(np.arange(0,self.dim[0]), np.arange(0,self.dim[1]))
        zz = np.zeros((self.dim[0],self.dim[1]))#+1
        xi = np.array([xx.flatten(), yy.flatten(), zz.flatten()])

        g1 = self.Rk2(xi)
        g2 = self.Rk2(xi, -1)
        
        nch = self.n_charges
        
        t1 = np.where((g1 == -1) + (g1 == g2))        
        t2 = np.where((g2 == -1) + (g1 == g2))        
        
        g1[t1] = nch
        g2[t2] = nch
        
        self.g = g1*(nch+1) + g2        
        self._connectivity = np.bincount(self.g, minlength = (nch+1)**2)
        self.g.shape = (self.dim[0],self.dim[1])
        self._connectivity.shape = (nch+1, nch+1)

        m = np.where(np.sum(self._connectivity, axis=0))[0]
        n = np.where(np.sum(self._connectivity, axis=1))[0]

        self._connectivity = self._connectivity[n][:,m]
        return self._connectivity    
        
    
    def fline(self, x):
        f1 = self.rk2(x)
        f2 = self.rk2(x, mul = -1)
        if f1[1] != f2[1]:
            verts = np.column_stack((np.fliplr(f1[0]), f2[0]))
            ends = [f1[1],f2[1]]
            return fLine(self, verts, ends)
        
    def check_ranges(self,x):
        return np.all(np.abs(x[0:2] - self.dim[0:2]/2) < self.dim[0:2]/2)

    def draw_footprint(self, figsize = (10,10)):
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        lim = np.max(np.abs(self.Q()))
        plt.imshow(self.data[0], vmin = -lim, vmax = lim)

        for i in range(0, self.n_charges):
            charge = self.Charges[i]
            plt.plot(charge.x[0], charge.x[1], charge.marker, color = 'green', markersize = charge.markersize, mew = 1.5)
            ax.annotate(charge.name, (charge.x[0], charge.x[1]), size = 12)

        for i in range(0, self.n_nulls):
            null = self.Nulls[i]
            plt.plot(null.x[0],null.x[1], null.marker, color = 'green', markersize = 8)
            ax.annotate(null.name, (null.x[0], null.x[1]), size = 12)
            a = null.flines
            for j in range(0,4):
                plt.plot(a[j].verts[0,:], a[j].verts[1,:], color = 'green', linestyle = a[j].linestyle)

        ax.invert_yaxis()
        plt.axis([0,self.dim[0]-1,0,self.dim[1]-1])
        plt.show()
        return self

        