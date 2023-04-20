# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 08:56:38 2023

@author: jgall
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoide (v):
    '''función logística'''
    return (1./(1.+np.exp(-v)))


u=0
k=1.6222
tao=0.3699
delta=0.1 # Muestreo 

yr = 0.05 # Entrada del sistema
y = np.array([0])
yla = np.array([0])
ylc = np.array([0])
ey=np.array([0,0])
spp=[]

w_1=0.5*np.random.random(size=(3,3)) # Pesos iniciales de W_ji
w_2=-2.5*np.random.random(size=(1,3)) # Pesos iniciales de V_j

u_t=np.array([0]) # Inicializo la salida de la RNA en 1
n_etha=1
n=n_etha
alpha=0.1
alf = alpha
epocas = 15000

nombre='Tanque_cerrado_sin_retardo'   

with open(f"{nombre}_n_=_{n}_alpha_=_{alf}.csv", 'w', newline='') as csvfile:
    fieldnames = ['n', 'alfa','set-point','salida controlada','ley de control','error']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'n':n_etha,'alfa':alpha,'set-point':yr,'salida controlada':y[-1],'ley de control':u_t[-1],'error':ey[-1]})
    for t in range (epocas):
        #Lazo cerrado con control
    
        if t > 1500:
            yr=0.2
        if t > 3000:
            yr=0.4
        if t > 4500:
            yr=0.6
        if t > 6000:
            yr=0.5
        if t > 7500:
            yr=0.3
        if t > 9000:
            yr=0.1
        if t > 11500:
            yr=0
        if t > 13000:
            yr=0.5
        if t > 14500:
            yr=0.9
        if t > 14000:
            yr=0
        
        ey=np.append(ey,yr-y[-1]) # Error del sistema
        error=np.array([ey[-1],ey[-2],ey[-3]]) # Arreglo de Errores atrasados
        aux=np.dot(w_1,error) # Producto punto entre entradas y pesos W_1
        aux_2=np.reshape(sigmoide(aux),(1,3)) # Evaluación con la funcion sigmoide
        aux_4=sigmoide(np.dot(aux_2,np.reshape(w_2,(3,1))))
        u_t=np.append(u_t,aux_4)
    
        delta_1= ey[-1]*u_t[-1]*(1-u_t[-1]) # Generacion de delta 1
        delta_2= delta_1*w_2*aux_2*(1-aux_2) # Generacion de delta 2
        aux_3=alpha*np.abs(ey[-1])
        n_etha=n_etha+aux_3 # Generacion del etha variable
    
        y=np.append(y,(((u_t[-1]*k)-y[-1])*delta)/(tao) + y[-1]) # Ecuacion en diferencias de orden uno
        w_2=w_2+(n_etha*delta_1*aux_2) # Actualizacion de pesos de V_j
        w_1=w_1+(n_etha*delta_2*(error)) # Actualizacion de pesos de W_ji
        spp.append(yr)
        writer.writerow({'n':n_etha,'alfa':alpha,'set-point':yr,'salida controlada':y[-1],'ley de control':u_t[-1],'error':ey[-1]})

plt.figure(figsize=(13,7))
plt.plot(y,c='blue',label='Salida Controlada',linestyle='-')
plt.plot(spp,c='red',label='Set Point',linestyle='-')
plt.plot(u_t,c='green',label='Ley de control',linestyle='--')
plt.title(f"{nombre} n = {n} alpha = {alf}",c='black',size=20)
plt.legend()
plt.grid()
plt.savefig(f"{nombre}_n_=_{n}_alpha_=_{alf}.jpg")
plt.show()