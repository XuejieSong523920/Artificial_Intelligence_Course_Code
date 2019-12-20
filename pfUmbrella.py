# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:56:59 2019

@author: Xuejie Song
"""
import numpy as np
import pandas as pd

Rain = (1, 1)

numSamples = int(input())
numSteps = int(input())

observations_Umbrella=[]
for i in range(numSteps):
    observations_Umbrella.append(int(input()))


start_probability = {'T': 0.5, 'F': 0.5}

transition_probability = {
    1 : {1: 0.7, 0: 0.3 },
    0 : {1: 0.3, 0: 0.7 },
}

sensor_probability = {
    1 : {1: 0.9, 0: 0.1},
    0 : {1: 0.2, 0: 0.8}
}

def transition_model(r1):
    
    rd = np.random.uniform(0,1)
    if r1 == 1:
        if rd < 0.7:
            r2 = 1
        else:
            r2 = 0
            
    if r1 == 0:
        if rd < 0.3:
            r2 = 1
        else:
            r2 = 0
        
    return r2

particle = np.zeros(numSamples)
for i in range(numSamples):
    rd = np.random.uniform(0,1)
    if rd < 0.5:
        particle[i] = 1
    else:
        particle[i] = 0
        
particle = pd.DataFrame(particle)

def particle_filtering(NumSamples, NumSteps, sp, u):
    particle1 = particle
    weight = np.zeros((NumSteps, NumSamples))
    for i in range(NumSteps):
        for j in range(NumSamples):
            particle1.iloc[j,0] = transition_model(particle1.iloc[j,0])
            weight[i, j]=sp[particle1.iloc[j, 0]][u[i]]
        particle1 = particle1.sample(n=NumSamples, frac=None, replace=True, weights=weight[i, :], random_state=None, axis=0)
        
        
    return particle1.sum()/len(particle1)


print(particle_filtering(numSamples, numSteps, sensor_probability, observations_Umbrella))
