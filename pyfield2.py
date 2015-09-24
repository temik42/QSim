# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 14:37:16 2015

@author: artem
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from numpy import fft
from numpy import linalg
from scipy import interpolate
from scipy import optimize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import time


def FouField(data):

    dim = np.shape(data)

    temp = np.pad(data,((0,dim[0]),(0,dim[1])),'constant')
    
    xi = np.roll(np.arange(-dim[0],dim[0]),dim[0])
    yi = np.roll(np.arange(-dim[1],dim[1]),dim[1])
    
    xi = np.resize(xi,(2*dim[0],2*dim[1]))/(2.*dim[0])
    yi = (np.resize(yi,(2*dim[1],2*dim[0]))/(2.*dim[1])).T
        
    q = np.sqrt(xi**2 + yi**2)
    fdata = fft.fft2(temp)
    
    gx = xi*(-1j)/q.clip(min = (1./(dim[0]*dim[1])))   
    gy = yi*(-1j)/q.clip(min = (1./(dim[0]*dim[1])))   
    
    bx = np.real(fft.ifft2(fdata*gx))[0:dim[0],0:dim[1]]
    by = np.real(fft.ifft2(fdata*gy))[0:dim[0],0:dim[1]]

    return [bx, by]


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

class fPlane:
    def __init__(self, field, data):
        print 'Interpolating field...'
        time0 = time.time()
        
        self.field = field
        self.data = gaussian_filter(data, 1)
        self.dim = np.shape(self.data)

        self.skeleton = skeletonize(np.abs(self.data) > self.field.threshold)
        self.iData = interpolate.interp2d(np.arange(self.dim[0]),np.arange(self.dim[1]),self.data)
        print 'Done! Elapsed time: '+str(time.time()-time0)        
        
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
    def __init__(self, data, nz = 64, threshold = 50):
        self.threshold = threshold
        self.fplane = fPlane(self, data)
        self.dim = np.append(self.fplane.dim, nz)

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
    
    def Get(self, xi):
        nch = self.n_charges
        ni = xi.shape[1]
    
        X = self.X()
        Q = self.Q()
    
        dx = [np.resize(xi[i,:], (nch,ni)).T - X[:,i] for i in range(0,3)]
        dr = np.sqrt(sum([dx[i]**2. for i in range(0,3)])).clip(min = 1e-10)
    
        self._argminr = np.argmin(dr, axis = 1)
        self._minr = np.min(dr, axis = 1)
        
        return np.array([np.dot(dx[i]/dr**3., Q) for i in range(0,3)])    
    
    def jacobian(self, x):
        dx = (x - self.X())
        dr = np.sqrt(np.sum(dx**2, axis = 1)).clip(min = 1e-10)
        return np.sum(self.Q()/dr**3)*np.identity(3)-3*np.dot(self.Q()/dr**5*dx.T, dx)
        
    def Jacobian(self, x):
        nch = self.n_charges
        n = x.shape[1]
    
        X = self.X()
        Q = self.Q()
        
        dx = [np.resize(x[i,:], (nch,n)).T - X[:,i] for i in range(0,3)]
        dr = np.sqrt(sum([dx[i]**2. for i in range(0,3)])).clip(min = 1e-10)
        
        f = Q/dr**3.
        return (np.array([[np.sum(f, axis = 1) if i==j else np.zeros(n) for i in range(0,3)] for j in range(0,3)])+
                    np.array([[-3.*np.sum(f/dr**2.*dxi*dxj, axis = 1) for dxi in dx] for dxj in dx]))
         
    
    def set_charges(self, q, x): 
        self.n_charges = q.shape[0]
        self.n_pos = 0
        self.n_neg = 0
        
        self.Charges = []
        for i in range(0,self.n_charges):
            self.Charges.extend([fCharge(self, q[i], x[:,i])])
            
        self.Infinity = fInfinity(self, -np.sum(q))
        return self

    def search_charges(self, maxi = 100):
        print 'Searching charges...'
        time0 = time.time()        
        
        x = []
        
        idx = np.where(self.fplane.skeleton)

        for i in range(0, idx[0].shape[0]):
            x0 = (idx[1][i],idx[0][i])
            
            opt = optimize.fmin(lambda y: -np.abs(self.fplane.iData(y[0],y[1])[0]), 
                                 x0, disp = False, maxiter = maxi, full_output = True, ftol = 1e-2)
            
            xmin = opt[0]
            n_iter = opt[2]
            
            if (self.check_ranges(xmin) and n_iter < maxi and
                np.abs(self.fplane.iData(xmin[0],xmin[1])[0]) > self.threshold):
                x.extend([xmin])
        
        x = np.array(x)
        
        db = DBSCAN(min_samples = 1, eps = 2)
        db.fit_predict(x)
        
        n_charges = np.max(db.labels_)+1
        qi = np.zeros(n_charges)
        xi = np.zeros((3,n_charges))
        
        for i in range(0, n_charges):
            xi[0:2,i] = np.mean(x[db.labels_ == i,:], axis=0)
            qi[i] = self.fplane.iData(xi[0,i],xi[1,i])
        
        qi /= np.max(np.abs(qi))
        
        self.set_charges(qi,xi)
        print 'Done! Elapsed time: '+str(time.time()-time0)
        return self
    
    def search_nulls(self, maxi = 10):
        def optima(x, x0):
            dx = np.zeros(3)
            i = 0
            n = 0
            while n < 3:
                opt = optimize.fmin_bfgs(lambda y: np.sum(self.get(y)**2), x0+dx, 
                            lambda y: 2*np.dot(self.jacobian(y), self.get(y)),
                            disp = False, full_output = True, gtol = 1e-9, maxiter = 50)                    
                if not opt[6] and opt[1] < 1e-10:
                    x.extend([opt[0]])
                    n += 1
                    dx = np.append(np.random.normal(0, 2, size=2),0)
                if i == maxi:
                    n += 1
                    dx = np.append(np.random.normal(0, 2, size=2),0)
                    i = 0
                    
                dx = dx + np.append(np.random.normal(0, 2, size=2),0)
                i += 1
        
        def q_metric(x1,x2):
            return np.sum((x1-x2)**2)/(self.fplane.iData(x1[0],x1[1])*self.fplane.iData(x2[0],x2[1]))
        
        def nei_search(x, sign = 'all'):
            nei = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric = q_metric).fit(self.X(sign)[:,0:2])
            dist, idx = nei.kneighbors(self.X(sign)[:,0:2])
            idx = np.sort(idx, axis = 1)
            m = max(idx[:,1])+1
            idx = idx[:,0]*m+idx[:,1]
            idx = np.unique(idx)                     
            idx = np.array(divmod(idx,m)).T         
            
            for i in range(0, idx.shape[0]):
                id1 = idx[i,0]
                id2 = idx[i,1]                
                
                q1 = np.sqrt(np.abs(self.Q(sign)[id1]))
                q2 = np.sqrt(np.abs(self.Q(sign)[id2]))
                faq = q2/(q1+q2)
                x0 = self.X(sign)[id1,:]*faq+self.X(sign)[id2,:]*(1-faq)
                optima(x, x0)
        
        def random_search(x, m = 8):
            for i in range(0,m):
                xi = np.array([np.random.uniform(0, self.dim[0]),np.random.uniform(0, self.dim[1]),0])
                optima(x, xi)
                
        
        print 'Searching nulls...'
        time0 = time.time()                
        
        x = []


        n_uprights = 0
        n_prones = 0
        n_a = 0
        n_b = 0
        
        while n_prones != self.n_charges + n_uprights - 2:# or n_b - n_a != self.n_pos - self.n_neg :
            n_uprights = 0
            n_prones = 0
            n_a = 0
            n_b = 0
            
            nei_search(x, 'pos')
            nei_search(x, 'neg')
            #random_search(x)    
            xn = np.array(x)
        
            db = DBSCAN(min_samples = 1, eps = 0.5)
            db.fit_predict(xn)
        
            n_nulls = np.max(db.labels_)+1
            nulls = []
        
            for i in range(0, n_nulls):
                xi = np.mean(xn[db.labels_ == i,:], axis=0)
                null = fNull(self, xi, i+1)
                nulls.extend([null])                
                
                if null.type == 'prone':
                    n_prones += 1
                else:
                    n_uprights += 1
                if null.sign == 'pos':
                    n_b += 1
                else:
                    n_a += 1
                   
        #print n_b - n_a, self.n_pos - self.n_neg
        self.Nulls = nulls        
        self.n_nulls = n_nulls
        self.n_prones = n_prones
        self.n_uprights = n_uprights
        
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
        
    def Update(self, x):
        nch = self.n_charges
        
        self._x = x
        n = self._x.shape[1]
    
        X = self.X()
        Q = self.Q()
        
        self._dx = [np.resize(self._x[i,:], (nch,n)).T - X[:,i] for i in range(0,3)]
        self._dr = np.sqrt(sum([self._dx[i]**2. for i in range(0,3)])).clip(min = 1e-10)
        
        _f = Q/self._dr**3.
        self._p = np.array([np.sum(Q/self._dr**3.*dx, axis = 1) for dx in self._dx])
        self._jacobian = (np.array([[np.sum(_f, axis = 1) if i==j else np.zeros(n) for i in range(0,3)] for j in range(0,3)])+
                    np.array([[-3.*np.sum(_f/self._dr**2.*dxi*dxj, axis = 1) for dxi in self._dx] for dxj in self._dx]))
        
        self._argminr = np.argmin(self._dr, axis = 1)
        self._minr = np.min(self._dr, axis = 1)    
    
    def Rk2(self, x, mul = 1, maxi = 200, step = 0.5):
        self.Update(x)
        n = x.shape[1]
        
        i = 0
        
        result = - np.ones(n)
        ids = np.where(np.ones(n))[0]      
        
        while i < maxi:                   
                      
            
            dx = self._p/np.sqrt(np.sum(self._p**2., axis=0))*step*mul
            self._p = self._p + 0.5*np.sum(self._jacobian*dx, axis = 1)
            dx = self._p/np.sqrt(np.sum(self._p**2., axis=0))*step*mul
            
            self.Update(self._x + dx)
            
            t = np.where(self._minr > 0.5*step)[0]
            _t = np.where(self._minr < 0.5*step)[0]
            result[ids[_t]] = self._argminr[_t]
            ids = ids[t]
            
            self._p = self._p[:,t]
            self._x = self._x[:,t]  
            self._jacobian = self._jacobian[:,:,t]            
            
            
            i += 1
    
        return result.astype(np.int16)        

    
    """
    def rk2(self, x, mul = 1, maxi = 200, step = 0.5):
        _x = x
        stack = x
        
        i = 0      
        while i < maxi:       
            p = self.get(_x)
            dx = p/linalg.norm(p)*step*mul
            
            if self._minr < step:
                _x = self.X()[self._argminr,:]
                stack = np.column_stack((stack,_x))
                return (stack, self.Charges[self._argminr])
            else:                    
                p = p + 0.5*np.dot(self.jacobian(_x), dx)
                dx = p/linalg.norm(p)*step*mul
                _x = _x + dx
                #if self.check_ranges(_x):
                stack = np.column_stack((stack,_x))
                #else:
                #    break
            i += 1
        return (stack, self.Infinity)
        

    def _rk2(self, xi, mul = 1, maxi = 200, step = 0.5):
        _xi = xi
        ni = xi.shape[1]
        
        i = 0
        
        result = - np.ones(ni)
        ids = np.where(np.ones(ni))[0]      
        
        while i < maxi:       
            pi = self._get(_xi)
            
            t = np.where(self._minr > step)[0]
            _t = np.where(self._minr < step)[0]
            result[ids[_t]] = self._argminr[_t]
            ids = ids[t]
            
            pi = pi[:,t]
            _xi = _xi[:,t]             
            
            dxi = pi/np.sqrt(np.sum(pi**2, axis=0))*step*mul        
            pi = 0.5*(pi+self._get(_xi+dxi))            
            dxi = pi/np.sqrt(np.sum(pi**2, axis=0))*step*mul
            
            _xi = _xi + dxi
            i += 1
    
        return result.astype(np.int16)
    """


    
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
        plt.imshow(self.fplane.data, vmin = -200, vmax = 200)

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
        plt.axis([0,self.dim[0],0,self.dim[1]])
        plt.show()
        return self

        