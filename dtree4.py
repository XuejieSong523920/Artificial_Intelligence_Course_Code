# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:08:06 2019

@author: shirley
"""

import pandas as pd
import numpy as np
data = pd.read_csv("dataset.csv", header = None)
data1 = data.rename(columns = {0:'Alt',1:'Bar',2:'Fri',3:'Hun',4:'Pat',5:'Price',6:'Rain',7:'Res',8:'Type',9:'Est',10:'Target'})
Idx_infer ={'Alt':0,'Bar':1,'Fri':2,'Hun':3,'Pat':4,'Price':5,'Rain':6,'Res':7,'Type':8,'Est':9,'Target':10}
data_name = data1.columns.values.tolist()
data_copy = data.copy()
# data_name.pop(2)
data_name
data1['Type']
data_copy

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

def leaf_decision(examples):
    
    if len(examples) == 0:
        return 'without example'
    
    else:
        num_T = 0
        num_F = 0
        n = examples.shape[1]-1
        for idx, value in examples.iloc[:,n]:
            if value == 'T':
                num_T += 1

            if value == 'F':
                num_F += 1

    #     print(num_T)
    #     print(num_F)
        if num_T == len(examples):
            return 'T'

        elif num_F == len(examples):
            return 'F'

        else:
            return examples
        

def Plurality_Value(examples):
    num_T = 0
    num_F = 0
    T = Target[0]
    F = Target[1]
    n = examples.shape[1]-1
    for idx, value in examples.iloc[:,n]:
        if value == 'T':
            num_T += 1
            
        elif value == 'F':
            num_F += 1
            
    if num_T > num_F:
        return 'T'
    
    elif num_T < num_F:
        return 'F'
    
    elif num_T == num_F:
        rd = np.random.rand(1)
        if rd<0.5:
            return 'T'
        else:
            return 'F'


def get_sub_value_example(X, parent_examples):
    # for Alt, Bar, Fri, Hun, Rain, Res
    
    for i in range(data_copy.shape[1]):
        for key in np.unique(X).tolist():
            if data_copy.iloc[1,i] == key:
                k = i
                break
            
    keys = np.unique(data_copy.iloc[:,k]).tolist()
            
    
    if len(keys) == 2:
        idx_1 = []
        idx_2 = []
        for i in parent_examples.index.tolist():#here is some problem that the idx should be the number of row of parent_examples
            if X[i] == keys[0]:
                idx_1.append(i)
            else:
                idx_2.append(i)

        examples_1 = parent_examples.loc[idx_1,:]
        examples_2 = parent_examples.loc[idx_2,:]
        return examples_1,examples_2
    
    if len(keys) == 3:
        idx_1 = []
        idx_2 = []
        idx_3 = []
        for i in parent_examples.index.tolist():#here is some problem that the idx should be the number of row of parent_examples
            if X[i] == keys[0]:
                idx_1.append(i)
            elif X[i] == keys[1]:
                idx_2.append(i)
            else:
                idx_3.append(i)

        examples_1 = parent_examples.loc[idx_1,:]
        examples_2 = parent_examples.loc[idx_2,:]
        examples_3 = parent_examples.loc[idx_3,:]
        return examples_1,examples_2,examples_3
    
    if len(keys) == 4:
        idx_1 = []
        idx_2 = []
        idx_3 = []
        idx_4 = []
        for i in parent_examples.index.tolist():#here is some problem that the idx should be the number of row of parent_examples
            if X[i] == keys[0]:
                idx_1.append(i)
            elif X[i] == keys[1]:
                idx_2.append(i)
            elif X[i] == keys[2]:
                idx_3.append(i)
            else:
                idx_4.append(i)

        examples_1 = parent_examples.loc[idx_1,:]
        examples_2 = parent_examples.loc[idx_2,:]
        examples_3 = parent_examples.loc[idx_3,:]
        examples_4 = parent_examples.loc[idx_4,:]
        return examples_1,examples_2,examples_3,examples_4      
    
tree1 = {'root':{},'leaf':{}, 'node':{}}
tree2 = {'root':{},'leaf':{}, 'node':{}}
tree3 = {'root':{},'leaf':{}, 'node':{}}
tree4 = {'root':{},'leaf':{}, 'node':{}}

def build_subtree(data, tree):
    tree = {'root':{},'leaf':{}, 'node':{}}
    importance(data)
    #find the most important one's # of column
    Id_most_important = max(importance(data), key=lambda x: importance(data)[x])
    #get this attribute
    attribute_name = data_name[Id_most_important]
    #delete this attribute when finish use it
    data_name.pop(Id_most_important)
#   attribute_name = data1.loc[Id_most_important].columns.values
    tree['root'] = attribute_name
    Attri_most_imortant = data.iloc[:,Id_most_important]
    examples_new = 0
   #Then according to its values separate it
   #after I run the code, I know there are 3 parts
    
#     value_cnt = {}  
#     for value in Attri_most_imortant:
#         value_cnt[value] = value_cnt.get(value, 0) + 1

#     keys = []
#     for key in value_cnt.keys():
#         keys.append(key)
    for i in range(data_copy.shape[1]):
        for key in np.unique(Attri_most_imortant).tolist():
            if data_copy.iloc[1,i] == key:
                kk = i
                break
            
    keys = np.unique(data_copy.iloc[:,kk]).tolist()
    
    if len(keys) == 3:
        
        a1, a2, a3 = get_sub_value_example(Attri_most_imortant, data)
        
       #for every part run the function
        for a in [a1, a2, a3]:
            colu_idx = a.columns.values.tolist()[Id_most_important]
            
            if len(a) == 0:
                tree['leaf']['no example'] = Plurality_Value(data)
            
            elif len(leaf_decision(a)) == 1:
                Idx = a.index.tolist()[0]
                leaf = a.loc[Idx, colu_idx]
                tree['leaf'][leaf] = leaf_decision(a)
                
            else:
                Idx1 = a.index.tolist()[0]
                node = a.loc[Idx1, colu_idx]
                tree['node'][node] = leaf_decision(a)
                examples = tree['node'][node]
                examples_new = examples.drop(Id_most_important, axis=1)
                
    if len(keys) == 2:
        
        a1, a2 = get_sub_value_example(Attri_most_imortant, data)
       #for every part run the function
    
        for a in [a1, a2]:
            colu_idx = a.columns.values.tolist()[Id_most_important]
            
            if len(a) == 0:
                tree['leaf']['no example'] = Plurality_Value(data)
            
            elif len(leaf_decision(a)) == 1:
                Idx = a.index.tolist()[0]
                leaf = a.loc[Idx, colu_idx]
                tree['leaf'][leaf] = leaf_decision(a)
                    
            else:
                Idx1 = a.index.tolist()[0]
                node = a.loc[Idx1, colu_idx]
                tree['node'][node] = leaf_decision(a)
                examples = tree['node'][node]
                examples_new = examples.drop(Id_most_important, axis=1)
                
    if len(keys) == 4:
        
        
        a1, a2, a3, a4 = get_sub_value_example(Attri_most_imortant, data)
        
       #for every part run the function
            
        for a in [a1, a2, a3, a4]:
            colu_idx = a.columns.values.tolist()[Id_most_important]
            
            if len(a) == 0:
                tree['leaf']['no example'] = Plurality_Value(data)
            
            elif len(leaf_decision(a)) == 1:
                Idx = a.index.tolist()[0]
                leaf = a.loc[Idx, colu_idx]
                tree['leaf'][leaf] = leaf_decision(a)
                
            else:
                Idx1 = a.index.tolist()[0]
                node = a.loc[Idx1, colu_idx]
                tree['node'][node] = leaf_decision(a)
                examples = tree['node'][node]
                examples_new = examples.drop(Id_most_important, axis=1)
#     examples = examples.drop('column_name', Id_most_important)
        
            
    #therefore we can get the first subtree
        
#     print(delt_row)
#     for i,d in enumerate(delt_row):
#         for e,f in enumerate(d):
#             examples_new = examples.drop([d], inplace = False)
    return tree, examples_new


data = pd.read_csv("dataset.csv", header = None)
#data1 = data.iloc[0:8,:]
data1 = data.rename(columns = {0:'Alt',1:'Bar',2:'Fri',3:'Hun',4:'Pat',5:'Price',6:'Rain',7:'Res',8:'Type',9:'Est',10:'Target'})
data_name = data1.columns.values.tolist()

def tree(data):
    
    new_tree, new_example= build_subtree(data, tree1)
    new_new_tree, new_new_example = build_subtree(new_example, tree2)
    new_new_new_tree, new_new_new_example = build_subtree(new_new_example, tree3)
    new_new_new__new_tree, new_new_new_new_example = build_subtree(new_new_new_example, tree4)
    tree_1 = new_tree
    tree_2 = new_new_tree
    tree_3 = new_new_new_tree
    tree_4 = new_new_new__new_tree
    tree_3['node'] = tree_4
    tree_2['node'] = tree_3
    tree_1['node'] = tree_2
    
    return tree_1

#define a function to predict the target from data using decision tree
def predict(tree_new, data):
    
    if data[Idx_infer[tree_new['root']]] in tree_new['leaf']:
        for key in tree_new['leaf']:
            if data[Idx_infer[tree_new['root']]] == key:
                return tree_new['leaf'][key]
            
    elif data[Idx_infer[tree_new['node']['root']]] in tree_new['node']['leaf']:
        for key in tree_new['node']['leaf']: 
            if data[Idx_infer[tree_new['node']['root']]] == key:
                return tree_new['node']['leaf'][key]
            
    elif data[Idx_infer[tree_new['node']['node']['root']]] in tree_new['node']['node']['leaf']:
        for key in tree_new['node']['node']['leaf']: 
            if data[Idx_infer[tree_new['node']['node']['root']]] == key:
                return tree_new['node']['node']['leaf'][key]
            
    elif data[Idx_infer[tree_new['node']['node']['node']['root']]] in tree_new['node']['node']['node']['leaf']:
        for key in tree_new['node']['node']['node']['leaf']: 
            if data[Idx_infer[tree_new['node']['node']['node']['root']]] == key:
                return tree_new['node']['node']['node']['leaf'][key]

def error_rate(data, tree):
    pre = []
    for i in range(data.shape[0]):
        pre.append(predict(tree, data.iloc[i,:]))
    
    for i in range(len(pre)):
        if pre[i] == 'T':
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


tree_train = tree(data)
print('This is the decision tree I get.')
print(tree_train)

print('Here is the predict result of trainning data.')
for i in range(data.shape[0]):
        print(predict(tree_train, data.iloc[i,:]))

print('This the error rate on the training set of the learned decision tree:')
print(error_rate(data, tree_train))


#below is the code to get LOOCV:
def tree_for_2(data):
    tree1 = {'root':{},'leaf':{}, 'node':{}}
    tree2 = {'root':{},'leaf':{}, 'node':{}}
    tree3 = {'root':{},'leaf':{}, 'node':{}}
    tree4 = {'root':{},'leaf':{}, 'node':{}}
    new_tree, new_example= build_subtree(data, tree1)
    new_new_tree, new_new_example = build_subtree(new_example, tree2)
    tree_1 = new_tree
    tree_2 = new_new_tree
    
    tree_1['node'] = tree_2
    
    return tree_1

def predict_for_2(tree_new, data):
    
    if data[Idx_infer[tree_new['root']]] in tree_new['leaf']:
        for key in tree_new['leaf']:
            if data[Idx_infer[tree_new['root']]] == key:
                return tree_new['leaf'][key]
            
    elif data[Idx_infer[tree_new['node']['root']]] in tree_new['node']['leaf']:
        for key in tree_new['node']['leaf']: 
            if data[Idx_infer[tree_new['node']['root']]] == key:
                return tree_new['node']['leaf'][key]
            
   
train = []
data1 = data.rename(columns = {0:'Alt',1:'Bar',2:'Fri',3:'Hun',4:'Pat',5:'Price',6:'Rain',7:'Res',8:'Type',9:'Est',10:'Target'})
Idx_infer ={'Alt':0,'Bar':1,'Fri':2,'Hun':3,'Pat':4,'Price':5,'Rain':6,'Res':7,'Type':8,'Est':9,'Target':10}
data_name = data1.columns.values.tolist()
for i in [1,2,3,4,5,6,7,8,9,10,11,12]:

    frontPart = data.iloc[:i-1, :]
    rearPart = data.iloc[i:, :]
    tarinData = np.concatenate([np.array(frontPart), np.array(rearPart)], axis=0)
    traindata = pd.DataFrame(tarinData)
    train.append(traindata)
    
'''    
k=11
data_name = data1.columns.values.tolist()
tree_o = tree(train[k]) or tree_for_2(train[k])
print(predict(tree_o, data.iloc[k,:])) or use predict_for_2()
'''