# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:12:03 2019

@author: shirley
"""
import numpy as np

states = (1, 0)

#observations  = (0,1,0,1,1,1,1,0,0,0)
#observations  = (0,1,0,1,0,1,0,1,0,1)
numSteps = int(input())

observations_Umbrella=[]
for i in range(numSteps):
    observations_Umbrella.append(int(input()))
    
    
start_probability = {1: 0.5, 0: 0.5}

transition_probability = {
    1 : {1: 0.7, 0: 0.3 },
    0 : {1: 0.4, 0: 0.6 },
}

sensor_probability = {
    1 : {1: 0.9, 0: 0.1},
    0 : {1: 0.3, 0: 0.7}
}

def viterbi(obs, states, start_p, trans_p, sensor_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0) t is coresponding to obervations
    for y in states:
        V[0][y] = start_p[y] * sensor_p[y][obs[0]]#for different stat T and F
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * sensor_p[y][obs[t]], y0) for y0 in states])
            #(prob, state) = np.dot(max([(V[t-1][y0] * trans_p[y0][y], y0) for y0 in states]), sensor_p[y][obs[t]]) 
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    #print_dptable(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (path[state])


s = viterbi(observations_Umbrella,states,start_probability,transition_probability,sensor_probability)

print(s)