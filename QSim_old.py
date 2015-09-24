# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:34:10 2015

@author: artem
"""

import numpy as np
import Sun as S
import AIA


lambda0 = 8.87e-17
beta = 1/3.#(S.gamma+1)/S.gamma
T_tr = 1e5  ## [K]

q=1e5

def simulate(H, H0=5e5, P0=0.01, T0=5e5, dt=12):
    N = np.size(H)
    A = H0/P0#**2
    l = S.h_t(T0)
    #l=1e9
    
    P=P0
    
    I1 = np.zeros(N)  
    #I1[0] = 2*AIA.cresp(T0,'A171')*H0/lambda0*np.sqrt(T_tr)
    I1[0] = AIA.cresp(T0,'A171')*H0/(6*lambda0)#*q+AIA.tresp(T0,'A171')*(P0/(2*S.k_b*T0))**2*l*(1-q)
    
    #I2 = np.zeros(N)  
    #I2[0] = 2*AIA.cresp(T0,'A193')*H0/lambda0*np.sqrt(T_tr)
    #I2[0] = AIA.tresp(T0,'A193')*1e5*H0/(6*lambda0)#*q+AIA.tresp(T0,'A193')*(P0/(2*S.k_b*T0))**2*l*(1-q)   
    
    #I3 = np.zeros(N)      
    #I3[0] = 3*2.5e-20*H0/(6*lambda0)#*q+3*AIA.tresp(T0,'A131')*(P0/(2*S.k_b*T0))**2*l*(1-q)
    #I3[0] = 0#AIA.tresp(T0,'A193')*1e5*H0/(6*lambda0)    
    
    
    for i in range (1,N):        
        #dP = 2./3*(H[i] - A*P**2)/l*dt
        dP = 2./3*(H[i] - A*P)/l*dt
        P = P + dP
        #T = T0*(1+(P-P0)/P0*beta)
        T = T0*(P/P0)**beta        
        l = S.h_t(T)
        #I1[i] = 2*AIA.cresp(T, 'A171')*A*P**2/lambda0*np.sqrt(T_tr)
        #I2[i] = 2*AIA.cresp(T, 'A193')*A*P**2/lambda0*np.sqrt(T_tr)
        I1[i] = AIA.cresp(T,'A171')*A*P/(6*lambda0)
        #I2[i] = AIA.tresp(T,'A193')*1e5*A*P/(6*lambda0)
        #I3[i] = 3*2.5e-20*A*P/(6*lambda0)
        #I3[i] = 0#AIA.tresp(T,'A193')*1e5*A*P/(6*lambda0)
    return (I1)
