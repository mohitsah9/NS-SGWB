
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy
import time
from scipy.optimize import fsolve 
from scipy import interpolate
import warnings
import ray




tol=1.49e-12

G=6.674*10**(-11)
M0=1.989*10**(30)
H=70*3.24044*10**(-20)
om=0.30

c=3*10**(8)
yrtosec=3600*24*365
Gpctometer=3.086*10**(25)
hundmyrtosec=10**(8)*yrtosec


ray.shutdown()
ray.init()



tol=epsrel=1.49e-06

rho=3*H**(2)/(8*np.pi*G)



# Frequency dependence during the inspiral, merger, and ringdown phases of the gravitational wave
def G1(z,f,m1,m2):
    k=(M0*m1)*(M0*m2)/((M0*m1)+(M0*m2))**(2)
    fmerger= c**(3)*(2.9740*10**(-1)* k**(2) +  4.4810*10**(-2)* k + 9.5560*10**(-2))/(np.pi * G*(M0*m1+M0*m2))
    fring= c**(3)*(5.9411*10**(-1)* k**(2) +  8.9794*10**(-2)* k +  19.111*10**(-2))/(np.pi * G*(M0*m1+M0*m2))
    fcut= c**(3)*(8.484510**(-1)* k**(2) +  12.848*10**(-2)* k + 27.299*10**(-2))/(np.pi * G*(M0*m1+M0*m2))
    fw= c**(3)*(5.0801*10**(-1)* k**(2) +  7.7515*10**(-2)* k +  2.2369*10**(-2))/(np.pi * G*(M0*m1+M0*m2))

    if (1+z)*f<(fmerger):
        return ((1+z)*f)**(-1/3)

    if (fmerger)<=(1+z)*f<(fring):
        return ((1+z)*f)**(2/3)/(fmerger)

    if (fring)<=(1+z)*f<(fcut):
        return (1/((fmerger)*(fring)**(4/3)))*((1+z)*f/(1+((((1+z)*f-(fring))/((fw)/2))**2)))**2
    else:
        return 0



#  Energy emission per frequency bin in the source frame (dE/dfr)
def Ef(z,f,m1,m2):
    return (G*np.pi)**(2/3) * (m1*M0*m2*M0)/((M0*m1+M0*m2)**(1/3))*G1(z,f,m1,m2)/3



K1=[];K2=[];K3=[];K4=[];K5=[]

F=np.linspace(20,100,161)



zmin=0.6

Ro=20
R0=Ro/(yrtosec*Gpctometer**(3)); sig=0.5; Mc=30; alpha=1.5;Mmax=Mc*10+50




def RPBH(z):

    return R0*(1+z)**(alpha)


N,_=integrate.quad(lambda m: (1/((2*np.pi)**(1/2)*sig*m)) * np.exp(-(np.log(m/Mc))**2/(2*sig**2)),0.01,Mmax)

def PMF2(m):

    return (1/((2*np.pi)**(1/2)*sig*m)) * np.exp(-(np.log(m/Mc))**2/(2*sig**2))/N


@ray.remote
def Bac1(f):
    rho=3*H**(2)/(8*np.pi*G)
    g=lambda z,m1,m2:10**(-5)*(f*(1+z)/(rho*c**(2))) *  (1/(H*(1+z)*(om*(1+z)**(3)+(1-om))**(1/2))) * ((RPBH(z)*PMF2(m1)*PMF2(m2))/(1+z)) *  Ef(z,f,m1,m2)
    y1,_= integrate.tplquad(g,0.01,Mmax,0.01,Mmax,zmin,10,epsabs=tol,epsrel=tol)

    return y1*10**(5)

