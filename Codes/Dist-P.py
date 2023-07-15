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

G=6.674*10**(-11)
M0=1.989*10**(30)
H=70*3.24044*10**(-20)    #H=67.8*3.24044*10**(-20)
om=0.30                   #om=0.307
c=3*10**(8)
yrtosec=3600*24*365
Gpctometer=3.086*10**(25)
hundmyrtosec=10**(8)*yrtosec
rho=3*H**(2)/(8*np.pi*G)


ray.shutdown()
ray.init(num_cpus=16)

tol=epsrel=1.49e-03






from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

def dm(z):
    return ((cosmo.comoving_distance(z)/u.Mpc)*3.086e+22)

def dl(z):
    return ((cosmo.luminosity_distance(z)/u.Mpc)*3.086e+22)


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
    return (G*np.pi)**(2/3) *  (m1*M0*m2*M0)/((M0*m1+M0*m2)**(1/3))*G1(z,f,m1,m2)/3




Z1=np.linspace(0.9,10,80)

npix= 10000
obst= 10000


F1=[20,40,60,80,100,120]


Ro=20
Mc=60;alpha=1.5

R0=Ro/(yrtosec*Gpctometer**(3)); sig=0.5; Mmax=Mc*10+50



def RPBH(z):

    return R0*(1+z)**(alpha)



N,_=integrate.quad(lambda m: (1/((2*np.pi)**(1/2)*sig*m)) * np.exp(-(np.log(m/Mc))**2/(2*sig**2)),0.01,Mmax)

def PMF2(m):

    return (1/((2*np.pi)**(1/2)*sig*m)) * np.exp(-(np.log(m/Mc))**2/(2*sig**2))/N




def Sample2(Nsample):
    def Pf2(m1):
        y,_=integrate.quad(lambda m: PMF2(m),0.01,m1)
        return y

    Samp2=[]
    for i in range(Nsample):
        y=np.random.uniform(low=0,high=1)
        a, b = 0.01, Mmax
        res_x = scipy.optimize.bisect(lambda m: Pf2(m) - y,a, b)
        Samp2.append(res_x)
    
    return Samp2





def gw1(j):                       
    lam,_=integrate.quad(lambda z:(obst)* RPBH(z)*4*np.pi*c*(1/(H*(1+z)*(om*(1+z)**(3)+(1-om))**(1/2)))*dm(z)**(2),Z1[j],Z1[j+1])   
 
    ev=np.random.poisson(lam=lam, size=None)
    A=np.zeros(len(F1))
    if ev==0:
        warnings.filterwarnings("ignore")
        return A
    
    else:      
        for i in range(ev):
            Ms=Sample2(2)
            	
            for l in range(len(F1)):
                x1 = ((F1[l]*(1+(Z1[j]+Z1[j+1])/2))/(rho*c**(2))) *  (1/(4*np.pi*c*(1+(Z1[j]+Z1[j+1])/2)*dm((Z1[j]+Z1[j+1])/2)**2)) * Ef((Z1[j]+Z1[j+1])/2,F1[l],Ms[1],Ms[0])
                A[l]=A[l]+x1
	    	
            
        warnings.filterwarnings("ignore") 
        return A

    
@ray.remote
def GW1(i):
    S=[]
    for j in range(len(Z1)-1):
        S.append(gw1(j))
       
    S=np.array(S)   

    Bg=[]
    for l in range(len(F1)):
        Bg.append(np.sum(S[:,l])/obst)

    return Bg




s=time.time()

BG=[]

for i in range(npix):
    BG.append(GW1.remote(i))

BG=ray.get(BG)
BG=np.array(BG)

print(np.shape(BG))

g=open('Dist-'+str(Z1[0])+'-PBH'+str(obst)+'Samples='+str(npix)+'RP='+str(Ro)+'Mc='+str(Mc)+'alpha='+str(alpha)+'sigma='+str(sig)+'.txt','w')


for i in range(npix):
    for l in range(len(F1)): 
        g.write(str(BG[i,l]))
        g.write('\t')
    g.write('\n')

g.close()

print('Time',time.time()-s)





