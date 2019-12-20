# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:52:13 2019

@author: shirley
"""

# for thirst step, I want to choose the two attributes have the most large 2 Gani:
import pandas as pd
import numpy as np
data = pd.read_csv("dataset.csv", header = None)
data1 = data.rename(columns = {0:'Alt',1:'Bar',2:'Fri',3:'Hun',4:'Pat',5:'Price',6:'Rain',7:'Res',8:'Type',9:'Est',10:'Target'})
#this is the line I missed
Idx_infer ={'Alt':0,'Bar':1,'Fri':2,'Hun':3,'Pat':4,'Price':5,'Rain':6,'Res':7,'Type':8,'Est':9,'Target':10}
data_name = data1.columns.values.tolist()
data_copy = data.copy()

Alt = data.iloc[:,0]
Bar = data.iloc[:,1]
Fri = data.iloc[:,2]
Hun = data.iloc[:,3]
Pat = data.iloc[:,4]
Price = data.iloc[:,5]
Rain = data.iloc[:,6]
Res = data.iloc[:,7]
Type = data.iloc[:,8]
Est = data.iloc[:,9]
Target = data.iloc[:,10]

from math import log
def B(x):
    if x != 0 and x!= 1:
        return -(x * log(x,2) + (1-x) * log(1-x,2))
    elif x == 0:
        return 0
    elif x == 1:
        return 0
    
def Gani(X,parent_examples):
    
    parent_examples_T = 0
    for i in parent_examples.index.tolist():
        if Target.loc[i] == ' T':
            parent_examples_T += 1
    
    Gani_parents = B(parent_examples_T/len(parent_examples))
            
    value_cnt = {}  
    for value in X:
        value_cnt[value] = value_cnt.get(value, 0) + 1

    keys = []
    for key in value_cnt.keys():
        keys.append(key)
        
    if len(keys) == 1:
        return 0
        
    if len(keys) == 2:
                  
        num_1 = 0#the number of 'T'
        num_2 = 0
        sum_value_1_target_T = 0
        sum_value_2_target_T = 0
        for i in parent_examples.index.tolist():#here is some problem that the idx should be the number of row of parent_examples
            if X[i] == keys[0]:
                num_1 +=1 
                if Target.loc[i] == ' T':
                     sum_value_1_target_T += 1
            else:
                num_2 += 1
                if Target.loc[i] == ' T':
                    sum_value_2_target_T += 1    
        a = num_1/len(parent_examples)
        c = num_2/len(parent_examples)
        
    
        if num_1 != 0:
            b = sum_value_1_target_T/num_1
        else:
             b = 0
            
        if num_2 != 0:
            d = sum_value_2_target_T/num_2
        else:
            d = 0

        return Gani_parents-(a*B(b)+c*B(d))
    
    if len(keys) == 4:
                                      
        num_1 = 0#the number of 'T'
        num_2 = 0
        num_3 = 0#the number of 'T'
        num_4 = 0
        sum_value_1_target_T = 0
        sum_value_2_target_T = 0
        sum_value_3_target_T = 0
        sum_value_4_target_T = 0
        for i in parent_examples.index.tolist():#here is some problem that the idx should be the number of row of parent_examples
            if X[i] == keys[0]:
                num_1 +=1 
                if Target.loc[i] == ' T':
                     sum_value_1_target_T += 1
            elif X[i] == keys[1]:
                num_2 += 1
                if Target.loc[i] == ' T':
                    sum_value_2_target_T += 1  
                
            elif X[i] == keys[2]:
                num_3 += 1
                if Target.loc[i] == ' T':
                    sum_value_3_target_T += 1  
                
            elif X[i] == keys[3]:
                num_4 += 1
                if Target.loc[i] == ' T':
                    sum_value_4_target_T += 1 
                    
        a = num_1/len(parent_examples)           
        c = num_2/len(parent_examples)
        e = num_3/len(parent_examples)
        g = num_4/len(parent_examples)
        
        if num_1 != 0:
            b = sum_value_1_target_T/num_1
        else:
             b = 0
            
        if num_2 != 0:
            d = sum_value_2_target_T/num_2
        else:
            d = 0
            
        if num_3 != 0:
            f = sum_value_3_target_T/num_3
        else:
            f = 0
            
        if num_4 != 0:
            h = sum_value_4_target_T/num_4
        else:
            h = 0
            
        return Gani_parents-(a*B(b)+c*B(d)+e*B(f)+g*B(h))
        
#         return Gani_parents-((num_1/len(parent_examples))*B(sum_value_1_target_T/num_1)+(num_2/len(parent_examples))*B(sum_value_2_target_T/num_2)+(num_3/len(parent_examples))*B(sum_value_3_target_T/num_3)+(num_4/len(parent_examples))*B(sum_value_4_target_T/num_4))

    if len(keys) == 3:
        
        num_1 = 0#the number of 'T'
        num_2 = 0
        num_3 = 0
        sum_value_1_target_T = 0
        sum_value_2_target_T = 0
        sum_value_3_target_T = 0
        for i in parent_examples.index.tolist():#here is some problem that the idx should be the number of row of parent_examples
            if X[i] == keys[0]:
                num_1 +=1 
                if Target.loc[i] == ' T':
                    sum_value_1_target_T += 1

            elif X[i] == keys[1]:
                num_2 += 1
                if Target.loc[i] == ' T':
                    sum_value_2_target_T += 1   

            else:
                num_3 += 1
                if Target.loc[i] == ' T':
                    sum_value_3_target_T += 1
                    
        a = num_1/len(parent_examples)
        c = num_2/len(parent_examples)
        e = num_3/len(parent_examples)
        
        if num_1 != 0:
            b = sum_value_1_target_T/num_1
        else:
             b = 0
            
        if num_2 != 0:
            d = sum_value_2_target_T/num_2
        else:
            d = 0
            
        if num_3 != 0:
            f = sum_value_3_target_T/num_3
        else:
            f = 0
            
        return Gani_parents-(a*B(b)+c*B(d)+e*B(f))
    
    
def importance(examples):
    
    importance = dict()
    
    for i in range(examples.shape[1]-1):
        importance[i] = Gani(examples.iloc[:,i], examples)
        
    return importance
#     return examples.iloc[:,max(importance(examples), key=lambda x: importance(examples)[x])]

def decision_list(data):
    #firstly I want use one test contains one attributes to separate data:
    Im = importance(data)
    #find the most important one's # of column
    L = sorted(Im.items(),key=lambda item:item[1],reverse=True)
    L = L[:2]
    Idx_1 = L[0][0]# the attributes which has the biggest Gani
    Idx_2 = L[1][0]# the attributes which has the biggest Gani
    # Then I will pick these two attributes 
    Attributes_1 = data.iloc[:,Idx_1]
    Attributes_2 = data.iloc[:,Idx_2]
    keys_1 = np.unique(Attributes_1)
    keys_2 = np.unique(Attributes_2)
    from itertools import product
    combin = list(product(keys_1, keys_2))
    combin_T_num = np.zeros(len(combin))
    combin_F_num = np.zeros(len(combin))
    key = np.unique(data.iloc[:,10])
    for i in range(data.shape[0]):
        for j in range(len(combin)):
            if data.iloc[i,Idx_1] == combin[j][0] and data.iloc[i,Idx_2] == combin[j][1] and  data.iloc[i,10] == key[0]:
                combin_T_num[j] += 1
            elif data.iloc[i,Idx_1] == combin[j][0] and data.iloc[i,Idx_2] == combin[j][1] and  data.iloc[i,10] == key[1]:
                combin_F_num[j] += 1

    #get rid of all combinations have different target:
    combin_T_num_copy = combin_T_num.copy()
    combin_F_num_copy = combin_F_num.copy()
    for i in range(len(combin_T_num)):
        if combin_T_num_copy[i] != 0:
            combin_F_num[i] = 0
    for i in range(len(combin_T_num)):
        if combin_F_num_copy[i] != 0:
            combin_T_num[i] = 0

    if max(combin_F_num)>max(combin_T_num):
        location = np.argmax(combin_F_num)
    elif max(combin_F_num)<max(combin_T_num):
        location = np.argmax(combin_T_num)
    else:
        rd = np.random.rand(1)
        if rd<0.5:
            location = np.argmax(combin_T_num)
        else:
            location = np.argmax(combin_F_num)

    for i in range(data.shape[0]):
        if data.iloc[i,Idx_1] == combin[location][0] and data.iloc[i,Idx_2] == combin[location][1]:
            l_target = i

    #build the test :
    #where A_1 is the first attribute, A_2 is the second attribute, V_1 is the value of fisrt attribute we use to decision, the same as V_2, 'decision' is when we see V_1 and V_1 the decision we make, 'next' is the next test!
    test = {'A_1':data_name[Idx_1],'V_1': combin[location][0],'A_2':data_name[Idx_2],'V_2': combin[location][1],'decision' : data.iloc[l_target, 10],'next':{}}

    #drop all examples are tested:
    drop_idx = []
    for i_f in data.index.tolist():#must be the index of rows
        if data.loc[i_f,Idx_1] ==combin[location][0] and data.loc[i_f,Idx_2] ==combin[location][1]:
            
            drop_idx.append(i_f)

    data_drop = data.drop(drop_idx)
    return test,data_drop

test_1, data_1 = decision_list(data)
test_2, data_2 = decision_list(data_1)
test_3, data_3 = decision_list(data_2)
test_4, data_4 = decision_list(data_3)
test_5, data_5 = decision_list(data_4)
test_6, data_6 = decision_list(data_5)
test_7, data_7 = decision_list(data_6)
test_6['next'] = test_7
test_5['next'] = test_6
test_4['next'] = test_5
test_3['next'] = test_4
test_2['next'] = test_3
test_1['next'] = test_2

print('this is the decision_list that I learned')
print(test_1)

test = test_1

#here is the function to predict using learned decision_list
def predict(test, data):
    if data[Idx_infer[test['A_1']]] == test['V_1'] and data[Idx_infer[test['A_2']]] == test['V_2']:
        return test['decision']
    elif data[Idx_infer[test['next']['A_1']]] == test['next']['V_1'] and data[Idx_infer[test['next']['A_2']]] == test['next']['V_2']:
        return test['next']['decision']
    elif data[Idx_infer[test['next']['next']['A_1']]] == test['next']['next']['V_1'] and data[Idx_infer[test['next']['next']['A_2']]] == test['next']['next']['V_2']:
        return test['next']['next']['decision']
    elif data[Idx_infer[test['next']['next']['next']['A_1']]] == test['next']['next']['next']['V_1'] and data[Idx_infer[test['next']['next']['next']['A_2']]] == test['next']['next']['next']['V_2']:
        return test['next']['next']['next']['decision']
    elif data[Idx_infer[test['next']['next']['next']['next']['A_1']]] == test['next']['next']['next']['next']['V_1'] and data[Idx_infer[test['next']['next']['next']['next']['A_2']]] == test['next']['next']['next']['next']['V_2']:
        return test['next']['next']['next']['next']['decision']
    elif data[Idx_infer[test['next']['next']['next']['next']['next']['A_1']]] == test['next']['next']['next']['next']['next']['V_1'] and data[Idx_infer[test['next']['next']['next']['next']['next']['A_2']]] == test['next']['next']['next']['next']['next']['V_2']:
        return test['next']['next']['next']['next']['next']['decision']
    elif data[Idx_infer[test['next']['next']['next']['next']['next']['next']['A_1']]] == test['next']['next']['next']['next']['next']['next']['V_1'] and data[Idx_infer[test['next']['next']['next']['next']['next']['next']['A_2']]] == test['next']['next']['next']['next']['next']['next']['V_2']:
        return test['next']['next']['next']['next']['next']['next']['decision']

print('here is the predict result')
for i in range(data.shape[0]):
    print(predict(test, data.iloc[i,:]))

def error_rate(data, test):
    pre = []
    for i in range(data.shape[0]):
        pre.append(predict(test, data.iloc[i,:]))
    
    
    for i in range(len(pre)):
        if pre[i] == ' T':
            pre[i] = 1
        else:
            pre[i] = 0
    
    orig_target = []
    for i in range(len(data.iloc[:,10])):
        if data.iloc[:,10][i] == ' T':
            orig_target.append(1)
        else:
            orig_target.append(0)
    
    num_wrong = 0
    for i in range(len(pre)):
        if pre[i] != orig_target[i]:
            num_wrong+=1
    
    return num_wrong/len(pre)

#here is the  classiﬁcation error rate on the training set of the learned decision tree
print('here is the error rate of training data')
print(error_rate(data, test))

#below is the code used for LOOCV error rate
train = []
for i in [1,2,3,4,5,6,7,8,9,10,11,12]:

    frontPart = data.iloc[:i-1, :]
    rearPart = data.iloc[i:, :]
    tarinData = np.concatenate([np.array(frontPart), np.array(rearPart)], axis=0)
    traindata = pd.DataFrame(tarinData)
    train.append(traindata)

def finally_dlist_for_6(data):
    
    test_1, data_1 = decision_list(data)
    test_2, data_2 = decision_list(data_1)
    test_3, data_3 = decision_list(data_2)
    test_4, data_4 = decision_list(data_3)
    test_5, data_5 = decision_list(data_4)
    test_6, data_6 = decision_list(data_5)
    
    
    test_5['next'] = test_6
    test_4['next'] = test_5
    test_3['next'] = test_4
    test_2['next'] = test_3
    test_1['next'] = test_2

    
    return test_1

def predict_for_6(test, data):
    if data[Idx_infer[test['attribute_1']]] == test['value_1'] and data[Idx_infer[test['attribute_2']]] == test['value_2']:
        return test['result']
    elif data[Idx_infer[test['next']['attribute_1']]] == test['next']['value_1'] and data[Idx_infer[test['next']['attribute_2']]] == test['next']['value_2']:
        return test['next']['result']
    elif data[Idx_infer[test['next']['next']['attribute_1']]] == test['next']['next']['value_1'] and data[Idx_infer[test['next']['next']['attribute_2']]] == test['next']['next']['value_2']:
        return test['next']['next']['result']
    elif data[Idx_infer[test['next']['next']['next']['attribute_1']]] == test['next']['next']['next']['value_1'] and data[Idx_infer[test['next']['next']['next']['attribute_2']]] == test['next']['next']['next']['value_2']:
        return test['next']['next']['next']['result']
    elif data[Idx_infer[test['next']['next']['next']['next']['attribute_1']]] == test['next']['next']['next']['next']['value_1'] and data[Idx_infer[test['next']['next']['next']['next']['attribute_2']]] == test['next']['next']['next']['next']['value_2']:
        return test['next']['next']['next']['next']['result']
    elif data[Idx_infer[test['next']['next']['next']['next']['next']['attribute_1']]] == test['next']['next']['next']['next']['next']['value_1'] and data[Idx_infer[test['next']['next']['next']['next']['next']['attribute_2']]] == test['next']['next']['next']['next']['next']['value_2']:
        return test['next']['next']['next']['next']['next']['result']
    
'''  
test = finally_dlist(data.iloc[1:,:])
print(predict(test, data.iloc[0,:]))
for i in [1,2,3,4,5,6,7,8,9,10,11]:
    test = finally_dlist_for_6(train[i])
    print(predict_for_6(test, data.iloc[i,:]))
'''
