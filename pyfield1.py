# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 14:37:16 2015

@author: artem
"""

import numpy as np
from numpy import fft
from numpy import linalg
from scipy import interpolate
from scipy import optimize
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import time


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
    def colored(self, a=0.8):
        if self.ends[0] >= 0 and self.ends[1] >= 0:
            color = str(float((np.argwhere(self.field.posi == self.ends[1])*self.field.negi.shape[0]+
                               np.argwhere(self.field.negi == self.ends[0])))/(self.field.posi.shape[0]*self.field.negi.shape[0])+0.0001)  
            r,g,b = float(color[2])/10.*a, float(color[3])/10.*a, float(color[4])/10.*a
        else:
            r,g,b = .4,.4,.4
        self.color = (r,g,b)
        return self

class fPlane:
    def __init__(self, field, data, threshold = 50):
        print 'Interpolating field...'
        time0 = time.time()
        
        self.field = field
        self.data = data
        self.dim = np.shape(data)

        temp = np.pad(data,((0,self.dim[0]),(0,self.dim[1])),'constant')
    
        xi = np.roll(np.arange(-self.dim[0],self.dim[0]),self.dim[0])
        yi = np.roll(np.arange(-self.dim[1],self.dim[1]),self.dim[1])
    
        xi = np.resize(xi,(2*self.dim[0],2*self.dim[1]))/(2.*self.dim[0])
        yi = (np.resize(yi,(2*self.dim[1],2*self.dim[0]))/(2.*self.dim[1])).T
        
        q = np.sqrt(xi**2 + yi**2)
        fdata = fft.fft2(temp)*np.exp(-2*np.pi*q)
    
        gx = xi*(-1j)/q.clip(min = (1./(self.dim[0]*self.dim[1])))   
        gy = yi*(-1j)/q.clip(min = (1./(self.dim[0]*self.dim[1])))   
    
        bx = np.real(fft.ifft2(fdata*gx))[0:self.dim[0],0:self.dim[1]]
        by = np.real(fft.ifft2(fdata*gy))[0:self.dim[0],0:self.dim[1]]
        bz = np.real(fft.ifft2(fdata))[0:self.dim[0],0:self.dim[1]]
        
        bx_interp = interpolate.interp2d(np.arange(self.dim[0]),np.arange(self.dim[1]),bx)
        by_interp = interpolate.interp2d(np.arange(self.dim[0]),np.arange(self.dim[1]),by)
        bz_interp = interpolate.interp2d(np.arange(self.dim[0]),np.arange(self.dim[1]),bz)
                
        self.b = (bx, by, bz)
        self.mask = skeletonize(#(np.sqrt(self.b[0]**2+self.b[1]**2)/np.abs(self.b[2]).clip(min = 1) < 0.3)*
                     (np.abs(self.data) > threshold))
        self.iField = (bx_interp, by_interp, bz_interp)
        self.iData = interpolate.interp2d(np.arange(self.dim[0]),np.arange(self.dim[1]),data)
        print 'Done! Elapsed time: '+str(time.time()-time0)        
        
class fNull:
    def __init__(self, field, x, index=0):
        self.field = field
        self.x = x
        self.eigen = linalg.eig(self.field.jacobian(self.x))
        self.det = self.eigen[0][0]*self.eigen[0][1]*self.eigen[0][2]   
        self.sign = -np.sign(self.det)
        
        if self.sign > 0:
            self.marker = 'v'
            self.name = 'B' + str(index)
        else:
            self.marker = '^'
            self.name = 'A' + str(index)
        
        if self.sign == np.sign(self.eigen[0][2]):
            self.type = 'prone'
        else:
            self.type = 'upright'
            self.name = self.name + 'u'
        
        dx = self.eigen[1]

        
        self.flines = []
        
        for j in range(0,3):
            for i in range(0,2):
                mul = np.sign(self.eigen[0][j])
                if mul == self.sign:
                    type = 'fan'
                else:
                    type = 'spine'
                f = self.field.rk2(self.x + (-1)**i*dx[:,j], mul = mul)
                fline = fLine(self.field, np.column_stack((self.x,f[0])), [self,self.field.Charges[f[1]]], type = type)
                self.flines.extend([fline])
        

class fCharge:
    def __init__(self, field, q, x, index = 0):
        self.field = field
        self.q = q
        self.x = x
        
        if self.q > 0:
            self.marker = '+'
            self.markersize = 10
            self.name = 'P'+str(index)
        else:
            self.marker = 'x'
            self.markersize = 8
            self.name = 'N'+str(index)

class Field:
    def __init__(self, data, nz = 64):
        self.fplane = fPlane(self, data, 50)
        self.dim = np.append(self.fplane.dim, nz)
    
    def get(self, x):
        dx = (x - self.X)
        dr = np.sqrt(np.sum(dx**2, axis = 1))
        self._minr = np.min(dr)
        self._argminr = np.argmin(dr)
        return np.dot(self.Q/dr**3, dx)
        
    def jacobian(self, x):
        dx = (x - self.X)
        dr = np.sqrt(np.sum(dx**2, axis = 1))
        return np.sum(self.Q/dr**3)*np.identity(3)-3*np.dot(self.Q/dr**5*dx.T, dx)
    
    def set_charges(self, q, x):
        
        self.Q = np.append(q[q>0], q[q<0])
        self.X = np.column_stack((x[:,q>0],x[:,q<0])).T
        self.n_charges = self.Q.shape[0]
        
        self.posi = np.where(self.Q > 0)[0]
        self.negi = np.where(self.Q < 0)[0]
        
        self.n_pos = self.posi.shape[0]
        self.n_neg = self.negi.shape[0]
        
        self.Charges = []
        for i in range(0,self.n_charges):
            self.Charges.extend([fCharge(self, self.Q[i], self.X[i,:], i+1)])
            
        self.Charges.extend([fCharge(self, -np.sum(self.Q), np.inf)])
        
        return self

    def search_charges(self, threshold = 50, maxi = 50):
        print 'Searching charges...'
        time0 = time.time()        
        
        x = []
        
        idx = np.where(self.fplane.mask)

        for i in range(0, idx[0].shape[0]):
            x0 = (idx[1][i],idx[0][i])
            
            opt = optimize.fmin(lambda y: -np.abs(self.fplane.iField[2](y[0],y[1])[0]), 
                                 x0, disp = False, maxiter = maxi, full_output = True)
            
            xmin = opt[0]
            n_iter = opt[2]
            
            if (self.check_ranges(xmin) and n_iter < maxi and
                np.abs(self.fplane.iData(xmin[0],xmin[1])[0]) > threshold):
                x.extend([xmin])
        
        x = np.array(x)
        
        db = DBSCAN(min_samples = 1, eps = 2.)
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
            #i = 0
            n = 0
            while n < 3:
                opt = optimize.fmin_bfgs(lambda y: np.sum(self.get(y)**2), x0+dx, 
                            lambda y: 2*np.dot(self.jacobian(y), self.get(y)),
                            disp = False, full_output = True, gtol = 1e-9, maxiter = 50)                    
                if (self.check_ranges(opt[0]) and not opt[6]):
                    x.extend([opt[0]])
                    n += 1
                    #dx = np.append(np.random.normal(0, 2, size=2),0)
                dx = np.append(np.random.normal(0, 2, size=2),0)
                #i += 1
        
        def q_metric(x1,x2):
            return np.sum((x1-x2)**2)/(self.fplane.iData(x1[0],x1[1])*self.fplane.iData(x2[0],x2[1]))
        
        def tri_search(x, ids):
            tri = Delaunay(self.X[ids,0:2])
            for i in range(0, tri.nsimplex):
                x0 = np.mean(self.X[ids[tri.simplices[i,:]],:], axis=0)
                optima(x, x0)
                idx = ids[tri.simplices[i,:]]
                for j in range(-1,2):
                    x0 = 0.5*(self.X[idx[j],:]+self.X[idx[j+1],:])                
                    optima(x, x0)
        
        def nei_search(x, ids):
            nei = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric = q_metric).fit(self.X[ids,0:2])
            dist, idx = nei.kneighbors(self.X[ids,0:2])
            idx = np.sort(idx, axis = 1)
            m = max(idx[:,1])+1
            idx = idx[:,0]*m+idx[:,1]
            idx = np.unique(idx)                     
            idx = np.array(divmod(idx,m)).T         
            
            for i in range(0, idx.shape[0]):
                id1 = ids[idx[i,0]]
                id2 = ids[idx[i,1]]                
                
                q1 = np.sqrt(np.abs(self.Q[id1]))
                q2 = np.sqrt(np.abs(self.Q[id2]))
                faq = q2/(q1+q2)
                x0 = self.X[id1,:]*faq+self.X[id2,:]*(1-faq)
                optima(x, x0)
                
        print 'Searching nulls...'
        time0 = time.time()                
        
        x = []
        nei_search(x, self.posi)
        nei_search(x, self.negi)
                  
        x = np.array(x)
        
        db = DBSCAN(min_samples = 1, eps = 0.5)
        db.fit_predict(x)
        
        self.n_nulls = np.max(db.labels_)+1
        self.Nulls = []
        
        for i in range(0, self.n_nulls):
            xi = np.mean(x[db.labels_ == i,:], axis=0)
            self.Nulls.extend([fNull(self, xi, i+1)])
        
        print 'Done! Elapsed time: '+str(time.time()-time0)
        return self
    
    def rk2(self, x, mul = 1, maxi = 200, step = 0.5):
        _x = x
        stack = x
        
        i = 0      
        while i < maxi:       
            p = self.get(_x)
            dx = p/linalg.norm(p)*step*mul
            
            if self._minr < step:
                _x = self.X[self._argminr,:]
                stack = np.column_stack((stack,_x))
                return (stack, self._argminr)
            else:
                #p = 0.5*(p+self.get(_x+dx))
                p = p + 0.5*np.dot(self.jacobian(_x), dx)
                dx = p/linalg.norm(p)*step*mul
                if self._minr < step:
                    _x = self.X[self._argminr,:]
                    stack = np.column_stack((stack,_x))
                    return (stack, self._argminr)
                else:
                    _x = _x + dx
                    if self.check_ranges(_x):
                        stack = np.column_stack((stack,_x))
                    else:
                        break
            i += 1
        return (stack, -1)
    
    def fline(self, x):
        f1 = self.rk2(x)
        f2 = self.rk2(x, mul = -1)
        if f1[1] != f2[1]:
            verts = np.column_stack((np.fliplr(f1[0]), f2[0]))
            ends = [self.Charges[f1[1]],self.Charges[f2[1]]]
            return fLine(self, verts, ends)
        
    def check_ranges(self,x):
        return np.all(np.abs(x[0:2] - self.dim[0:2]/2) < self.dim[0:2]/2)

    def draw_footprint(self, figsize = (10,10)):
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        plt.imshow(self.fplane.b[2], vmin = -200, vmax = 200)

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
        plt.show()
        return self

        