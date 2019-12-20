# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:21:15 2019

@author: shirley
"""

import numpy as np
import pandas as pd

def action(U):
    initial = 0
    actio = {1:{1:initial, 2:initial, 3:initial}, 2:{1:initial, 3:initial}, 3:{1:initial, 2:initial, 3:initial},4:{1:initial}}
    for j in [0,1,2,3,4]:
        U[0][j] = U[1][j]
        U[5][j] = U[4][j]

    for i in [0,1,2,3,4,5]:
        U[i][0] = U[i][1]
        U[i][4] = U[i][3]
        
    actio[1][2] = {'u': 0.8*U[1][3] + 0.2*U[1][2],
                   'l': 0.8*U[1][2] + 0.1*U[1][1] + 0.1*U[1][3],
                   'd': 0.8*U[1][1] + 0.2*U[1][2],
                   'r': 0.8*U[1][2] + 0.1*U[1][1] + 0.1*U[1][3]}

    actio[2][1] = {'u': 0.8*U[2][1] + 0.1*U[1][1] + 0.1*U[1][3],
                   'l': 0.8*U[1][1] + 0.2*U[2][1],
                   'd': 0.8*U[2][1] + 0.1*U[1][1] + 0.1*U[1][3],
                   'r': 0.8*U[1][3] + 0.2*U[2][1]}
    
    actio[3][2] = {'u': 0.8*U[3][3] + 0.1*U[3][2] + 0.1*U[4][2],
                   'l': 0.8*U[3][2] + 0.1*U[3][1] + 0.1*U[3][3],
                   'd': 0.8*U[3][1] + 0.1*U[3][2] + 0.1*U[4][2],
                   'r': 0.8*U[4][2] + 0.1*U[3][1] + 0.1*U[3][3]}

    actio[2][3] = {'u': 0.8*U[2][3] + 0.1*U[3][3] + 0.1*U[1][3],
                   'l': 0.8*U[1][3] + 0.2*U[2][3],
                   'd': 0.8*U[2][3] + 0.1*U[3][3] + 0.1*U[1][3],
                   'r': 0.8*U[3][3] + 0.2*U[2][3]}
    
    actio[4][1] = {'u': 0.8*U[4][2] + 0.1*U[3][1] + 0.1*U[4][1],
                   'l': 0.8*U[3][1] + 0.1*U[4][1] + 0.1*U[4][2],
                   'd': 0.9*U[4][1] + 0.1*U[3][1],
                   'r': 0.9*U[4][1] + 0.2*U[4][2]}
    
   
    for i in [1,3]:
        for j in [1,3]:
              actio[i][j] = {'u': 0.8*U[i][j+1] + 0.1*U[i+1][j] + 0.1*U[i-1][j],
                             'l': 0.8*U[i-1][j] + 0.1*U[i][j-1] + 0.1*U[i][j+1],
                             'd': 0.8*U[i][j-1] + 0.1*U[i-1][j] + 0.1*U[i+1][j],
                             'r': 0.8*U[i+1][j] + 0.1*U[i][j-1] + 0.1*U[i][j+1]}
  
    return actio

def iteration(ac):
    
    u = 0
    U1 = {0:{0:u, 1:u, 2:u, 3:u, 4:u}, 1:{0:u, 1:u, 2:u, 3:u, 4:u}, 2:{0:u, 1:u, 3:u, 4:u}, 3:{0:u, 1:u, 2:u, 3:u, 4:u}, 4:{0:u, 1:u, 2:-1, 3:1, 4:u}, 5:{0:u, 1:u, 2:u, 3:u, 4:u}}

    for i in [1,2,3,4]:
        U1[i][1] = R[i][1] + discount * ac[i][1][max(ac[i][1],key=ac[i][1].get)]
    
    for i in [1,3]:
        U1[i][2] = R[i][2] + discount * ac[i][2][max(ac[i][2],key=ac[i][2].get)]
 
    for i in [1,2,3]:
        U1[i][3] = R[i][3] + discount * ac[i][3][max(ac[i][3],key=ac[i][3].get)]
    
    return U1

def different(U1, U):
    diff = []
    for i in [1,2,3,4]:
        diff.append(U1[i][1]-U[i][1])
    
    for i in [1,3]:
        diff.append(U1[i][2]-U[i][2])
    
    for i in [1,2,3]:
        diff.append(U1[i][3]-U[i][3])
    
    return diff

#here I use the utility initialized as 0, which also can be random like below
u = 0
U = {0:{0:u, 1:u, 2:u, 3:u, 4:u}, 1:{0:u, 1:u, 2:u, 3:u, 4:u}, 2:{0:u, 1:u, 3:u, 4:u}, 3:{0:u, 1:u, 2:u, 3:u, 4:u}, 4:{0:u, 1:u, 2:-1, 3:1, 4:u}, 5:{0:u, 1:u, 2:u, 3:u, 4:u}}
#u = np.random.rand(29)
#U = {0:{0:u[0], 1:u[1], 2:u[2], 3:u[3], 4:u[4]}, 1:{0:u[5], 1:u[6], 2:u[7], 3:u[8], 4:u[9]}, 2:{0:u[10], 1:u[11], 3:u[12], 4:u[13]}, 3:{0:u[14], 1:u[15], 2:u[16], 3:u[17], 4:u[18]}, 4:{0:u[19], 1:u[20], 2:u[21], 3:u[22], 4:u[23]}, 5:{0:u[24], 1:u[25], 2:u[26], 3:u[27], 4:u[28]}}


#which can be replaced by -0.2, -0.01
r = -2
R = {1:{1:r, 2:r, 3:r}, 2:{1:r, 3:r}, 3:{1:r, 2:r, 3:r}, 4:{1:r, 2:-1, 3:1}}

#discount 
discount = 0.99

def convergen(delta, U):
    diff = np.ones(9)
    U_old = U
    while max(np.abs(diff))>delta:# I want to break when the difference of this two very small
        
         #renew the old_one as the new one
        ac_old = action(U_old)
        #get a futher new one
        U_new = iteration(ac_old) 
        #compute the difference between two
        diff = different(U_new,U_old)
        U_old = U_new
      
    return U_new

U_result = convergen(0.001, U)
ac= action(U_result)
for i in [1,2,3,4]:
    print(1,i,max(ac[i][1],key=ac[i][1].get))
    
for i in [1,3]:
    print(2,i,max(ac[i][1],key=ac[i][2].get))

for i in [1,2,3]:
    print(3,i,max(ac[i][1],key=ac[i][3].get))