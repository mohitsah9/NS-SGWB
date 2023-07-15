import numpy as np
import matplotlib.pyplot as plt


from scipy import interpolate

yrtosec=3600*24*365
H=70*3.24044*10**(-20)



freq1=np.loadtxt('noise.txt',usecols=0)
nos=np.loadtxt('noise.txt',usecols=1)

freq2=np.loadtxt('figure1.txt',usecols=0)
red12=np.loadtxt('figure1.txt',usecols=1)
red13=np.loadtxt('figure1.txt',usecols=2)
red23=np.loadtxt('figure1.txt',usecols=3)



Nos= interpolate.interp1d(freq1,nos, kind='linear')
Red12= interpolate.interp1d(freq2,red12, kind='linear')
Red13= interpolate.interp1d(freq2,red13, kind='linear')
Red23= interpolate.interp1d(freq2,red23, kind='linear')


F=np.linspace(20,100,161)



Nos_pow12=[];Nos_pow13=[];Nos_pow23=[]

T=2*10**(2); delf=(F[1]-F[0])


for j in range(len(F)):
    Nos_pow12.append((5*2**(1/2)*np.pi**(2)*F[j]**(3) * Nos(F[j])**(2))/(3*H**(2)*(T*delf)**(1/2)*abs(Red12(F[j]))))
    Nos_pow13.append((5*2**(1/2)*np.pi**(2)*F[j]**(3) * Nos(F[j])**(2))/(3*H**(2)*(T*delf)**(1/2)*abs(Red13(F[j]))))
    Nos_pow23.append((5*2**(1/2)*np.pi**(2)*F[j]**(3) * Nos(F[j])**(2))/(3*H**(2)*(T*delf)**(1/2)*abs(Red23(F[j]))))

Nos_pow12=np.array(Nos_pow12)
Nos_pow13=np.array(Nos_pow13)
Nos_pow23=np.array(Nos_pow23)








F=np.linspace(20,100,161)


import numpy as np

Mc=30; alpha=1.5; dM=4; dalp=0.2     

            
AA1=np.loadtxt('1-0.6-PBH200Samples=300000RP=20Mc=28.5alpha=1.5sigma=0.5.txt')
AA2=np.loadtxt('1-0.6-PBH200Samples=300000RP=20Mc=32alpha=1.5sigma=0.5.txt')
AA3=np.loadtxt('1-0.6-PBH200Samples=300000RP=20Mc=30alpha=1.4sigma=0.5.txt')
AA4=np.loadtxt('1-0.6-PBH200Samples=300000RP=20Mc=30alpha=1.6sigma=0.5.txt')
AA5=np.loadtxt('1-0.6-PBH200Samples=300000RP=20Mc=30alpha=1.5sigma=0.5.txt')


A1 = AA1
A2 = AA2
A3 = AA3
A4 = AA4
A5 = AA5



Nobst=300000


A1avg=np.mean(A1,axis=0)
A2avg=np.mean(A2,axis=0)
A3avg=np.mean(A3,axis=0)
A4avg=np.mean(A4,axis=0)
A5avg=np.mean(A5,axis=0)






Cs1= np.cov(A1, rowvar=False)
Cs2= np.cov(A2, rowvar=False)
Cs3= np.cov(A3, rowvar=False)
Cs4= np.cov(A4, rowvar=False)
Cs5= np.cov(A5, rowvar=False)




CCa= (Cs2-Cs1)/dM
CCb= (Cs4-Cs3)/dalp



Cn=np.zeros((len(F),len(F)),float)

for i in range(len(F)):
    Cn[i][i]=1/((1/(Nos_pow12[i])**(2))+(1/(Nos_pow13[i])**(2))+(1/(Nos_pow23[i])**(2)))

        
C4=np.zeros((len(F),len(F)),float)



C4=Cs5+Cn


Dab=np.zeros((len(F),len(F)),float)
Dba=np.zeros((len(F),len(F)),float)
Daa=np.zeros((len(F),len(F)),float)
Dbb=np.zeros((len(F),len(F)),float)



for i in range(len(F)):
    for j in range(len(F)):
        Daa[i][j]= ((A2avg[i]-A1avg[i])/dM) * ((A2avg[j]-A1avg[j])/dM) + ((A2avg[j]-A1avg[j])/dM) * ((A2avg[i]-A1avg[i])/dM)


for i in range(len(F)):
    for j in range(len(F)):
        Dab[i][j]= ((A2avg[i]-A1avg[i])/dM) * ((A4avg[j]-A3avg[j])/dalp) + ((A2avg[j]-A1avg[j])/dM) * ((A4avg[i]-A3avg[i])/dalp)


for i in range(len(F)):
    for j in range(len(F)):
        Dba[i][j]= ((A4avg[i]-A3avg[i])/dalp) * ((A2avg[j]-A1avg[j])/dM) + ((A4avg[j]-A3avg[j])/dalp) * ((A2avg[i]-A1avg[i])/dM)


for i in range(len(F)):
    for j in range(len(F)):
        Dbb[i][j]= ((A4avg[i]-A3avg[i])/dalp) * ((A4avg[j]-A3avg[j])/dalp) + ((A4avg[j]-A3avg[j])/dalp) * ((A4avg[i]-A3avg[i])/dalp)








C4inv= np.linalg.inv(C4)
Csinv= np.linalg.inv(Cs4)
Cninv= np.linalg.inv(Cn)

Ea=np.dot(C4inv,CCa)
Eb=np.dot(C4inv,CCb)



Fis=np.zeros((2,2),float)

Fis[0][0]= (1/2)* np.trace(Nobst*np.dot(Ea,Ea) + Nobst* np.dot(C4inv,Daa))
Fis[0][1]= (1/2)* np.trace(Nobst*np.dot(Ea,Eb) + Nobst* np.dot(C4inv,Dab))
Fis[1][0]= (1/2)* np.trace(Nobst*np.dot(Eb,Ea) + Nobst* np.dot(C4inv,Dba))
Fis[1][1]= (1/2)* np.trace(Nobst*np.dot(Eb,Eb) + Nobst* np.dot(C4inv,Dbb))

Fis1=np.zeros((2,2),float)
Fis2=np.zeros((2,2),float)
Fis3=np.zeros((2,2),float)

Fis1[0][0]= (1/2)* np.trace(Nobst*np.dot(Ea,Ea))
Fis1[0][1]= (1/2)* np.trace(Nobst*np.dot(Ea,Eb))
Fis1[1][0]= (1/2)* np.trace(Nobst*np.dot(Eb,Ea))
Fis1[1][1]= (1/2)* np.trace(Nobst*np.dot(Eb,Eb))

Fis2[0][0]= (1/2)* np.trace(Nobst* np.dot(Cninv,Daa))
Fis2[0][1]= (1/2)* np.trace(Nobst* np.dot(Cninv,Dab))
Fis2[1][0]= (1/2)* np.trace(Nobst* np.dot(Cninv,Dba))
Fis2[1][1]= (1/2)* np.trace(Nobst* np.dot(Cninv,Dbb))


Fis3[0][0]= (1/2)* np.trace(Nobst* np.dot(C4inv,Daa))
Fis3[0][1]= (1/2)* np.trace(Nobst* np.dot(C4inv,Dab))
Fis3[1][0]= (1/2)* np.trace(Nobst* np.dot(C4inv,Dba))
Fis3[1][1]= (1/2)* np.trace(Nobst* np.dot(C4inv,Dbb))



Fis1_inv=np.linalg.inv(Fis1)
Fis2_inv=np.linalg.inv(Fis2)
Fis_inv=np.linalg.inv(Fis)

print(Fis1_inv**(1/2))
print(Fis2_inv**(1/2))
print(Fis_inv**(1/2))

Fis1_inv=Fis1_inv.tolist()
Fis2_inv=Fis2_inv.tolist()
Fis_inv=Fis_inv.tolist()

print(100*((Fis2_inv[0][0])**(1/2)-(Fis_inv[0][0])**(1/2))/(Fis2_inv[0][0])**(1/2))

print('Fis1_inv=',Fis1_inv)
print('Fis2_inv=',Fis2_inv)
print('Fis_inv=',Fis_inv)

Fis1=Fis1.tolist()
Fis2=Fis2.tolist()
Fis=Fis.tolist()

print('Fis1=',Fis1)
print('Fis2=',Fis2)
print('Fis=',Fis)
