# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:34:10 2015

@author: artem
"""

import numpy as np
#import Sun as S
#import AIA

lambda0 = 8.87e-17
beta = 1/3.
l = 1e9

def simulate(H, H0=5e5, P0=0.01, T0=5e5, dt=12):
    
    N = np.size(H)
    A = H0/P0

    P=P0   
    I = np.zeros(N)  

    for i in range (0,N):        

        dP = 2./3*(H[i] - A*P)/l*dt
        P = P + dP

        I[i] = 1e-18*A*P/(6*lambda0)

    return (I)
