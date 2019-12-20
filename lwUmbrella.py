# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 07:42:44 2019

@author: shirley
"""

import numpy as np

numSamples = int(input())
numSteps = int(input())

evidence = []
for i in range(numSteps):
    evidence.append(int(input()))


def transit_model(num):
    """
    :param num: int
    :return: next_num : int
    """
    r_num = np.random.uniform()
    if num == 1:
        if r_num < 0.7:
            next_num = 1
        else:
            next_num = 0
    else:
        if r_num < 0.3:
            next_num = 1
        else:
            next_num = 0
    return next_num


def particle_filter(numSamples, numSteps, evi):
    """
    :param numSamples: int
    :param numSteps: int
    :param evi: List[int]
    :return: estimate prob: float
    """

    sample = []
    w_true = 0
    w_overall = 0

    for i in range(numSamples):
        sub_sample = []
        w = 1
        first = np.random.randint(0, 2)

        for j in range(numSteps):
            next_var = transit_model(first)
            sub_sample.append(next_var)
            first = next_var

        for k in range(len(sub_sample)):
            if sub_sample[k] == 1 and evi[k] == 1:
                w *= 0.9
            elif sub_sample[k] == 1 and evi[k] == 0:
                w *= 0.1
            elif sub_sample[k] == 0 and evi[k] == 1:
                w *= 0.2
            elif sub_sample[k] == 0 and evi[k] == 0:
                w *= 0.8
        if sub_sample[numSteps - 1] == 1:
            w_true += w
        w_overall += w
        sample.append(sub_sample)

    prob = round(w_true / w_overall, 5)

    return prob


result = particle_filter(numSamples, numSteps, evidence)

print(result)


