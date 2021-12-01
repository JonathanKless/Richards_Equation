"""
Richard's equation explicit numerical scheme
"""

import numpy as np
import matplotlib.pyplot as plt

#Define dictionary with input parameters (soil characteristics)
VGP = dict()    
VGP['theta_s'] = 0.43
VGP['theta_r'] = 0.078
VGP['alpha'] = 0.016  
VGP['n'] = 2.16
VGP['Ksat'] = 0.001 
VGP['lambda_k'] = 0.5 

#Define geometry and time information
depth = -100              
dz = 1 
nz = int(abs(depth-dz)/dz)              
z = np.linspace(0,depth, nz)  
# dz=abs(z[2]-z[1])  

tFinal = 3*60*60 
dt = 1 
nt = int((tFinal-dt)/dt)
t = np.linspace(0, tFinal, nt)
# dt=t[2]-t[1]


#Finding approximate solution for C(h) equation, needs to be copy-pasted :-(
# def derive_Ch():
#     import sympy as sym
#     theta_s = sym.symbols("theta_s")
#     theta_r = sym.symbols("theta_r")
#     alpha = sym.symbols("alpha")
#     n = sym.symbols("n")
#     h = sym.symbols("h")
#     m = 1-1/n
#     theta = (theta_s - theta_r)/(1 ++(alpha*abs(h))**n)**m + theta_r
#     C = sym.differentiate_finite(theta, h)
#     return C


#Using Van-Genuchten-Mualem model to get theta, K, C 
def VGE(h, VGP):
    theta_r = VGP['theta_r']
    theta_s = VGP['theta_s']
    alpha = VGP['alpha']
    n = VGP['n']
    Ksat = VGP['Ksat']
    lambda_k = VGP['lambda_k']
    m = 1-1/n
    
    theta = (theta_s - theta_r)/(1 + (alpha*abs(h))**n)**m +theta_r
    Theta = (theta - theta_r)/(theta_s - theta_r)
    K = Ksat*Theta**(lambda_k)*(1 - (1 - Theta**(1/m))**m)**2 
    C = -(-theta_r + theta_s)*((alpha*abs(h - 1/2))**n + 1)**(-1 + 1/n)\
        + (-theta_r + theta_s)*((alpha*abs(h + 1/2))**n + 1)**(-1 + 1/n)
    
    return theta, K, C

#Skeleton matrices 
h = np.zeros((nz,nt))
theta = np.zeros((nz,nt))

#Define initial conditions of the system (t=0)
hBot_t0 = -200                 # h = matric potential
HBot_t0 = hBot_t0+z[-1]        # H = water potential  
HTop_t0 = HBot_t0                 
hTop_t0 = HTop_t0-z[0]             
h0 = np.linspace(hTop_t0,hBot_t0, nz)
theta0,K0,C0 = VGE(h0,VGP) 

#Storing initial conditions in matrices
h[:,0] = h0
theta[:,0] = theta0

#Known boundary conditions   
h0[0] = -5
qBot = 0 

#Function to compute Kplus at (i+0.5) and Kminus at (i-0.5) //geometric mean method
def K_mean(K):
    Kplus = np.sqrt(K[1:]*K[:nz-1])
    Kplus = np.append(Kplus,[K[-1]], axis=0)           
    Kminus = np.sqrt(K[:-1]*K[1:])
    Kminus = np.append([K[0]],Kminus, axis=0)          
    return Kplus, Kminus
    
#Filling the matrices using richard's equation, numerical, linearised implementation
for j in range(0, nt-1):
    i = 0                                                       #boundary condition (z=0)
    h[i,j+1] = -5                                               
                                                                    
    theta_tem, K, C = VGE(h[:,j],VGP)
    Kplus, Kminus = K_mean(K)
    
    for i in range (1, nz-1): 
        h[i,j+1]=(dt/(C[i]*dz))*(Kminus[i]*(h[i-1,j]-h[i,j])/dz\
        -Kplus[i]*(h[i,j]-h[i+1,j])/dz\
        +(Kminus[i]- Kplus[i])) + h[i,j]
             
    i = -1                                                      #boundary condition (z=-100)
    h[i,j+1] = (dt/(C[i]*dz))*(Kminus[i]*(h[i-1,j]-h[i,j])/dz\
                + qBot\
                + (Kminus[i]- 0))\
                + h[i,j]
    
    theta[:, j+1] = theta_tem
    
    
#Plotting results!
plt.figure(1)
plt.subplot(1,2,1)
plt.plot( h[:,0],z)
plt.plot(h[:,-1],z)
plt.ylabel("Depth [cm]")
plt.xlabel("Soil Matric Potential [cm]")
plt.legend(['t='+str(t[0]) + 's', 't='+str(t[-1]/60/60) + 'h'], frameon=False)
plt.show()

plt.subplot(1,2,2)
plt.plot(theta[:,0],z)
plt.plot(theta[:,-1],z)
plt.ylabel("Depth [cm]")
plt.xlabel("Soil Water Content [$cm^{3} cm^{-3}]$")
plt.legend(['t='+str(t[0]) + 's', 't='+str(t[-1]/60/60) + 'h'], frameon=False)
plt.show()
plt.tight_layout()

plt.figure(2,figsize=(9,5))
plt.subplot(1,2,1)
plt.imshow(np.abs(h), cmap='jet', extent=[t[0],t[-1]/60/60,z[-1],z[0]], aspect='auto')
plt.colorbar(label='Soil Water Potential [cm]')
plt.xlabel("Time [h]")
plt.ylabel("Depth [cm]")

plt.subplot(1,2,2)
plt.imshow(theta, cmap='jet', extent=[t[0],t[-1]/60/60,z[-1],z[0]], aspect='auto')
plt.colorbar(label='Soil Water Content [$cm^{3} cm^{-3}]$')
plt.xlabel("Time [h]")
plt.ylabel("Depth [cm]")

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.35, wspace=0.4)
plt.tight_layout()










    
        