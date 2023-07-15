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
ray.init(num_cpus=20)

tol=epsrel=1.49e-03

npix= 10000
obst= 10000



F1=[20,40,60,80,100,120]

Z1=np.linspace(0.135,10,80)


td=1; Mp=80
td1=td*hundmyrtosec; k=1;fr=0; a=1.5*M0; b=-1; RA0=30; Mmax=Mp+60


# Star Formation Rate
def Rsfr(z):
    return ((1+z)**2.7)/(1+(((1+z)/2.9)**5.6))
    
# Merger rates at z=0
def Rabh0(fr):
    return (RA0)/((1+fr)*yrtosec*Gpctometer**(3))


def R2a0(fr):
    return ((RA0)*fr)/((1+fr)*yrtosec*Gpctometer**(3))


def PA1(z,m,a,b,Mp):
    sig=5; l=0.03
    mp=(Mp -( a*b* z)/M0)
    return (1-l) * ((m)**(-2.3))* (1.3/(5**(-1.3)- mp**(-1.3))) +  l*2* (1/((2*np.pi)**(1/2)*sig))*np.exp(-((m-mp)**(2))/(2*sig**(2)))
    
def PA2(z,m,a,b,Mp):
    mp=(Mp -( a*b* z)/M0)
    return  ((m)**(-2.3))* (1.3/(5**(-1.3)- mp**(-1.3))) 
    

# Time dalay distribution

def t(zm):
    y,err=integrate.quad(lambda z:-1/(H*(1+z)*(om*(1+z)**(3)+(1-om))**(1/2)),0, zm) 
    return y # + 4.35*10**(17)



def Ptd1(z,zm,td,k):
    if (t(zm)-t(z))>td:
        return (t(zm)-t(z))**(-k)
    
    else:
        return 0


def Ptd2(z,zm,td2,k):
    if (t(zm)-t(z))>td2:
        return (t(zm)-t(z))**(-k)
    
    else:
        return 0




# Window Funtion of Black hole formed at a redshift z

def Ws(z,m,a,b,Mp):
    if 5*M0<=(M0*m)<=(Mp*M0 - a*b*z):
        return 1/(Mp-5 - (a*b*z)/M0)
    else:
        return 0




# Window Funtion of Black hole at a redshift z

def W(zm,m,td,k,a,b,Mp):
    zd=fsolve(lambda z: t(zm)-t(z)-td,zm+0.002)
    y1,_ = integrate.quad(lambda z: Ptd1(z,zm,td,k) * Ws(z,m,a,b,Mp) *(1/(H*(1+z)*(om*(1+z)**(3)+(1-om))**(1/2))),zm,28,points=zd,epsabs=tol, epsrel=tol)

    warnings.filterwarnings("ignore")
    return y1




@ray.remote
def N1(zm,td,k,a,b,Mp):
    zd=fsolve(lambda z: t(zm)-t(z)-td,zm+0.002)
    mp=(Mp - a*b*zd/M0)
    
    y1,_ = integrate.quad(lambda m:W(zm,m,td,k,a,b,Mp)*PA1(zd,m,a,b,Mp),5,mp,epsabs=tol, epsrel=tol)
    y2,_ = integrate.quad(lambda m:W(zm,m,td,k,a,b,Mp)*PA1(zd,m,a,b,Mp),mp,Mmax,epsabs=tol, epsrel=tol)
    
    warnings.filterwarnings("ignore")
    return (y1+y2)

@ray.remote
def N2(zm,td,k,a,b,Mp):
    zd=fsolve(lambda z: t(zm)-t(z)-td,zm+0.002)
    mp=(Mp - a*b*zd/M0)
    
    y1,_ = integrate.quad(lambda m:W(zm,m,td,k,a,b,Mp)*PA2(zd,m,a,b,Mp),5,mp,epsabs=tol, epsrel=tol)
    y2,_ = integrate.quad(lambda m:W(zm,m,td,k,a,b,Mp)*PA2(zd,m,a,b,Mp),mp,Mmax,epsabs=tol, epsrel=tol)
    
    warnings.filterwarnings("ignore")
    return (y1+y2)



def N3(td,k):
    y,_ = integrate.quad(lambda z: (1/(H*(1+z)*(om*(1+z)**(3)+(1-om))**(1/2))) * Ptd1(z,0,td,k) * Rsfr(z),0,30,epsabs=tol, epsrel=tol)
    warnings.filterwarnings("ignore")
    return y


M1=np.linspace(5,Mmax,250)
Z3=np.linspace(0,10,100)


A1=[]

for i in range(len(Z3)):
    A1.append(N1.remote(Z3[i],td1,k,a,b,Mp))

N11=ray.get(A1)

B1=[]

for i in range(len(Z3)):
    B1.append(N2.remote(Z3[i],td1,k,a,b,Mp))

N22=ray.get(B1)




@ray.remote
def Pm1(i,zd,z,m,td1,k,a,b,Mp):

    if N11[i]>0:
        
        return W(z,m,td1,k,a,b,Mp)*PA1(zd,m,a,b,Mp)/N11[i]
        
    else:
        return 0

@ray.remote
def Pm2(i,zd,z,m,td1,k,a,b,Mp):

    if N22[i]>0:
        
        return W(z,m,td1,k,a,b,Mp)*PA2(zd,m,a,b,Mp)/N22[i]
        
    else:
        return 0



A2=[]

for i in range(len(Z3)):
    zdd=fsolve(lambda z: t(Z3[i])-t(z)-td1,Z3[i]+0.002)
    zd=zdd[0]
    for j in range(len(M1)):
        A2.append(Pm1.remote(i,zd,Z3[i],M1[j],td1,k,a,b,Mp))

A2=ray.get(A2)
PMF1= interpolate.interp2d(Z3,M1,A2, kind='linear')


B2=[]

for i in range(len(Z3)):
    zdd=fsolve(lambda z: t(Z3[i])-t(z)-td1,Z3[i]+0.002)
    zd=zdd[0]
    for j in range(len(M1)):
        B2.append(Pm2.remote(i,zd,Z3[i],M1[j],td1,k,a,b,Mp))

B2=ray.get(B2)
PMF2= interpolate.interp2d(Z3,M1,B2, kind='linear')



N33=N3(td1,k)*yrtosec*Gpctometer**(3)

@ray.remote
def Rabh(zm,td1,fr,k):
    if N33==0:
        return 0
    else:
        zd=fsolve(lambda z: t(zm)-t(z)-td1,zm+0.002)
        y1,_ = integrate.quad(lambda z:(yrtosec*Gpctometer**(3))  * (1/(H*(1+z)*(om*(1+z)**(3)+(1-om))**(1/2))) * Ptd1(z,zm,td1,k) * Rsfr(z) * Rabh0(fr), zm,30,points=zd,epsabs=tol, epsrel=tol)
        warnings.filterwarnings("ignore")
        return y1/N33


A3=[]
Z2=np.linspace(0,10,100)



for i in range(len(Z2)):
    A3.append((Rabh.remote(Z2[i],td1,fr,k)))

A3=ray.get(A3)

RABH= interpolate.interp1d(Z2,A3, kind='linear')



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
    return (G*np.pi)**(2/3) * (m1*M0*m2*M0)/((M0*m1+M0*m2)**(1/3))*G1(z,f,m1,m2)/3






def Sample1(z):
    def Pf1(m1):
        y,_=integrate.quad(lambda m: PMF1(z,m),5,m1)
        return y
        
    n1,_=integrate.quad(lambda m: PMF1(z,m),5,Mmax)

    def Pf2(m1,Mx):
        n2,_=integrate.quad(lambda m: PMF2(z,m),5,Mx)
        y,_=integrate.quad(lambda m: PMF2(z,m)/n2,5,m1)
        return y
        

    Samp1=[]

    y1=np.random.uniform(low=0,high=1)
    a, b = 5, Mmax
    x1 = scipy.optimize.bisect(lambda m: (Pf1(m)/n1) - y1,a, b)
    Samp1.append(x1)

    y2=np.random.uniform(low=0,high=1)
    a,b = 5, x1
    x2 = scipy.optimize.bisect(lambda m: (Pf2(m,x1)) - y2,a, b)
    Samp1.append(x2)


    return Samp1
    



def gw1(j):                       
    lam,_=integrate.quad(lambda z:(obst)* RABH(z)*4*np.pi*c*(1/(H*(1+z)*(om*(1+z)**(3)+(1-om))**(1/2)))*dm(z)**(2),Z1[j],Z1[j+1])   
 
    ev=np.random.poisson(lam=lam, size=None)
    A=np.zeros(len(F1))
    if ev==0:
        warnings.filterwarnings("ignore")
        return A
    
    else:      
        for i in range(ev):
            Ms=Sample1((Z1[j]+Z1[j+1])/2)
            	
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


g=open('Dist-'+str(Z1[0])+'-ABH'+str(obst)+'Samples='+str(npix)+'RA='+str(RA0)+'Mp='+str(Mp)+'td1='+str(td)+'.txt','w')



for i in range(npix):
    for l in range(len(F1)): 
        g.write(str(BG[i,l]))
        g.write('\t')
    g.write('\n')

g.close()

print('Time',time.time()-s)




