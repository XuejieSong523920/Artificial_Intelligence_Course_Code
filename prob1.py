# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:05:06 2019

@author: Xuejie Song
"""

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

breat_X,breast_y = load_breast_cancer(return_X_y=True)
breastcancer = np.column_stack((breat_X,breast_y))
breastcancer = shuffle(breastcancer)

def split_folds(data):
    """
    data : array
    output : every factor in folds contains a train set and a validation set
    """
    # split the data into five folds
    nFolds = 5
    folds = []
    
    numSamples = data.shape[0]
    numLeaveOutPerFold = numSamples // nFolds

    for i in range(nFolds):
        startInd = i * numLeaveOutPerFold
        endInd = min((i + 1) * numLeaveOutPerFold, numSamples)

        frontPart = data[:startInd, :]
        midPart = data[startInd : endInd, :]
        rearPart = data[endInd:, :]


        foldData = np.concatenate([frontPart, rearPart], axis=0)
        foldInfo = {
            'train_x' : foldData[:, :-1],
            'train_y' : foldData[:, -1],
            'valid_x' : midPart[:, :-1],
            'valid_y' : midPart[:, -1]
        }

        folds.append(foldInfo)

    return folds

breastcancer_split_folds = split_folds(breastcancer)

def error_rate_for_logist(data, c):
    
    # error rate in every cross validation
    error_rate = []
    for i in range(5):
        
        X_train = data[i]['train_x']
        y_train = data[i]['train_y']
        X_test = data[i]['valid_x']
        y_test = data[i]['valid_y']

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        clf = LogisticRegression(random_state=0, C=c, solver='liblinear').fit(X_train_std,y_train)
        y_pred = clf.predict(X_test_std)
        error = sum(y_test != y_pred)/len(y_test)
        error_rate.append(error)
#     return error_rate
    return np.mean(error_rate),np.std(error_rate)

#plot mean classiﬁcation error rate in breast cancer data set
alter_C = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
log_alter_C =[math.log(i,10) for i in alter_C]
result = [error_rate_for_logist(breastcancer_split_folds,i) for i in alter_C]
result = pd.DataFrame(result)
ero = result.iloc[:,0]
std = result.iloc[:,1]
plt.errorbar(log_alter_C,ero,std,color='blue')
plt.title('breast cancer data set', fontsize=20)
plt.xlabel('log(C) in logistic regression')
plt.ylabel('error rate')
plt.show()

def error_rate_for_perceptron(data, a):
    
    # error rate in every cross validation
    error_rate = []
    for i in range(5):
        
        X_train = data[i]['train_x']
        y_train = data[i]['train_y']
        X_test = data[i]['valid_x']
        y_test = data[i]['valid_y']

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        ppn = Perceptron(penalty = 'l2',alpha = a, eta0=0.1,random_state = 0)
        ppn.fit(X_train_std,y_train)
        y_pred = ppn.predict(X_test_std)
        error = sum(y_test != y_pred)/len(y_test)
        error_rate.append(error)
    
    return np.mean(error_rate),np.std(error_rate)

#as the plot above is kind of weird and we can not see the error rate when alpha is very samll,
#so here I will plot error rate vs log(alpha)
alter_a = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
log_alter_a = [math.log(i,10) for i in alter_a]
result_pe = [error_rate_for_perceptron(breastcancer_split_folds,i) for i in alter_a]
result_pe = pd.DataFrame(result_pe)
ero_pe = result_pe.iloc[:,0]
std_pe = result_pe.iloc[:,1]
plt.errorbar(log_alter_a,ero_pe,std_pe,color='red')
plt.title('breast cancer data set', fontsize=20)
plt.xlabel('log(alpha) in perceptron')
plt.ylabel('error rate')
plt.show()

def error_rate_for_linear_SVM(data, c):
    
    # error rate in every cross validation
    error_rate = []
    for i in range(5):
        
        X_train = data[i]['train_x']
        y_train = data[i]['train_y']
        X_test = data[i]['valid_x']
        y_test = data[i]['valid_y']

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        clf = LinearSVC(C=c, random_state=0, tol=1e-5,max_iter=1000000)
        clf.fit(X_train_std,y_train)  
        y_pred = clf.predict(X_test_std)
        error = sum(y_test != y_pred)/len(y_test)
        error_rate.append(error)
    
    return np.mean(error_rate),np.std(error_rate)

alter_C_SVM = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
log_alter_C_SVM = [math.log(i,10) for i in alter_C_SVM]
result_SVM = [error_rate_for_linear_SVM(breastcancer_split_folds,i) for i in alter_C_SVM ]

result_SVM = pd.DataFrame(result_SVM)
ero_SVM = result_SVM.iloc[:,0]
std_SVM = result_SVM.iloc[:,1]
plt.errorbar(log_alter_C_SVM,ero_SVM,std_SVM,color='green')
plt.title('breast cancer data set', fontsize=20)
plt.xlabel('log(C) in linear SVM')
plt.ylabel('error rate')
plt.show()

def error_rate_for_KNN(data, k):
    
    # error rate in every cross validation
    error_rate = []
    for i in range(5):
        
        X_train = data[i]['train_x']
        y_train = data[i]['train_y']
        X_test = data[i]['valid_x']
        y_test = data[i]['valid_y']
        
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train_std,y_train)  
        y_pred = neigh.predict(X_test_std)
        error = sum(y_test != y_pred)/len(y_test)
        error_rate.append(error)
    
    return np.mean(error_rate),np.std(error_rate)

#plot mean classiﬁcation error rate in breast cancer data set for k-nearest neighbor(KNN)
alter_k = np.zeros(21)
for i in range(21):
    alter_k[i] = 6*i+1
    
result_KNN = [error_rate_for_KNN(breastcancer_split_folds,int(i)) for i in alter_k ]
result_KNN = pd.DataFrame(result_KNN)
ero_KNN = result_KNN.iloc[:,0]
std_KNN = result_KNN.iloc[:,1]
plt.errorbar(alter_k ,ero_KNN,std_KNN,color= 'skyblue')
plt.title('breast cancer data set', fontsize=20)
plt.xlabel('k in KNN')
plt.ylabel('error rate')
plt.show()

# Then deal with the iris data set:
iris_X,iris_y = load_iris(return_X_y=True)
iris = np.column_stack((iris_X,iris_y))
iris = shuffle(iris)
iris_split_folds = split_folds(iris)

def error_rate_for_logist_for_iris(data, c):
    
    # error rate in every cross validation
    error_rate = []
    for i in range(5):
        
        X_train = data[i]['train_x']
        y_train = data[i]['train_y']
        X_test = data[i]['valid_x']
        y_test = data[i]['valid_y']

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        clf = LogisticRegression(random_state=0, C=c,  solver='lbfgs',multi_class='multinomial').fit(X_train_std,y_train)
        y_pred = clf.predict(X_test_std)
        error = sum(y_test != y_pred)/len(y_test)
        error_rate.append(error)
#     return error_rate
    return np.mean(error_rate),np.std(error_rate)

result_iris = [error_rate_for_logist_for_iris(iris_split_folds,i) for i in alter_C]
result_iris = pd.DataFrame(result_iris)
ero_iris = result_iris.iloc[:,0]
std_iris = result_iris.iloc[:,1]
plt.errorbar(log_alter_C,ero_iris,std_iris,color='blue')
plt.title('iris data set', fontsize=20)
plt.xlabel('log(C) in logistic regression')
plt.ylabel('error rate')
plt.show()

result_pe_iris = [error_rate_for_perceptron(iris_split_folds,i) for i in alter_a]
result_pe_iris = pd.DataFrame(result_pe_iris)
ero_pe_iris = result_pe_iris.iloc[:,0]
std_pe_iris = result_pe_iris.iloc[:,1]
plt.errorbar(log_alter_a,ero_pe_iris,std_pe_iris,color='red')
plt.title('iris data set', fontsize=20)
plt.xlabel('log(alpha) in perceptron')
plt.ylabel('error rate')
plt.show()

result_SVM = [error_rate_for_linear_SVM(iris_split_folds,i) for i in alter_C_SVM ]
result_SVM = pd.DataFrame(result_SVM)
ero_SVM = result_SVM.iloc[:,0]
std_SVM = result_SVM.iloc[:,1]
plt.errorbar(log_alter_C_SVM,ero_SVM,std_SVM,color='green')
plt.title('iris data set', fontsize=20)
plt.xlabel('log(C) in linear SVM')
plt.ylabel('error rate')
plt.show()


alter_k = np.zeros(4)
for i in range(4):
    alter_k[i] = 6*i+1
result_KNN = [error_rate_for_KNN(iris_split_folds,int(i)) for i in alter_k ]
result_KNN = pd.DataFrame(result_KNN)
ero_KNN = result_KNN.iloc[:,0]
std_KNN = result_KNN.iloc[:,1]
plt.errorbar(alter_k ,ero_KNN,std_KNN,color= 'skyblue')
plt.title('iris data set', fontsize=20)
plt.xlabel('k in KNN')
plt.ylabel('error rate')
plt.show()


#Then deal with digits 
digits_X,digits_y = load_digits(return_X_y=True)
digits= np.column_stack((digits_X,digits_y))
digits = shuffle(digits)
digits_split_folds = split_folds(digits)

def error_rate_for_logist_for_digits(data, c):
    
    # error rate in every cross validation
    error_rate = []
    for i in range(5):
        
        X_train = data[i]['train_x']
        y_train = data[i]['train_y']
        X_test = data[i]['valid_x']
        y_test = data[i]['valid_y']

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        clf = LogisticRegression(random_state=0, C=c,max_iter=100,  solver='saga',multi_class='multinomial').fit(X_train_std,y_train)
        y_pred = clf.predict(X_test_std)
        error = sum(y_test != y_pred)/len(y_test)
        error_rate.append(error)
#     return error_rate
    return np.mean(error_rate),np.std(error_rate)

result_digits = [error_rate_for_logist_for_digits(digits_split_folds,i) for i in alter_C]
result_digits = pd.DataFrame(result_digits)
ero_digits = result_digits.iloc[:,0]
std_digits = result_digits.iloc[:,1]
plt.errorbar(log_alter_C,ero_digits,std_digits,color='blue')
plt.title('digits data set', fontsize=20)
plt.xlabel('log(C) in logistic regression')
plt.ylabel('error rate')
plt.show()

result_pe_digits = [error_rate_for_perceptron(digits_split_folds,i) for i in alter_a]
result_pe_digits = pd.DataFrame(result_pe_digits)
ero_pe_digits = result_pe_digits.iloc[:,0]
std_pe_digits = result_pe_digits.iloc[:,1]
plt.errorbar(log_alter_a,ero_pe_digits,std_pe_digits,color='red')
plt.title('digits data set', fontsize=20)
plt.xlabel('log(alpha) in perceptron')
plt.ylabel('error rate')
plt.show()

result_SVM = [error_rate_for_linear_SVM(digits_split_folds,i) for i in alter_C_SVM ]
result_SVM = pd.DataFrame(result_SVM)
ero_SVM = result_SVM.iloc[:,0]
std_SVM = result_SVM.iloc[:,1]
plt.errorbar(log_alter_C_SVM,ero_SVM,std_SVM,color='green')
plt.title('digits data set', fontsize=20)
plt.xlabel('log(C) in linear SVM')
plt.ylabel('error rate')
plt.show()


alter_k = np.zeros(21)
for i in range(21):
    alter_k[i] = 6*i+1  
result_KNN = [error_rate_for_KNN(digits_split_folds,int(i)) for i in alter_k ]
result_KNN = pd.DataFrame(result_KNN)
ero_KNN = result_KNN.iloc[:,0]
std_KNN = result_KNN.iloc[:,1]
plt.errorbar(alter_k ,ero_KNN,std_KNN,color= 'skyblue')
plt.title('digits data set', fontsize=20)
plt.xlabel('K in KNN')
plt.ylabel('error rate')
plt.show()



# Then deal with wine 
wine_X,wine_y = load_wine(return_X_y=True)
wine= np.column_stack((wine_X,wine_y))
wine = shuffle(wine)
wine_split_folds = split_folds(wine)

def error_rate_for_logist_for_wine(data, c):
    
    # error rate in every cross validation
    error_rate = []
    for i in range(5):
        
        X_train = data[i]['train_x']
        y_train = data[i]['train_y']
        X_test = data[i]['valid_x']
        y_test = data[i]['valid_y']

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        clf = LogisticRegression(random_state=0, C=c,max_iter=100,  solver='saga',multi_class='multinomial').fit(X_train_std,y_train)
        y_pred = clf.predict(X_test_std)
        error = sum(y_test != y_pred)/len(y_test)
        error_rate.append(error)
#     return error_rate
    return np.mean(error_rate),np.std(error_rate)

result_wine = [error_rate_for_logist_for_wine(wine_split_folds,i) for i in alter_C]
result_wine = pd.DataFrame(result_wine)
ero_wine = result_wine.iloc[:,0]
std_wine = result_wine.iloc[:,1]
plt.errorbar(log_alter_C,ero_wine,std_wine,color='blue')
plt.title('wine data set', fontsize=20)
plt.xlabel('log(C) in logistic regression')
plt.ylabel('error rate')
plt.show()

result_pe_wine = [error_rate_for_perceptron(wine_split_folds,i) for i in alter_a]
result_pe_wine = pd.DataFrame(result_pe_wine)
ero_pe_wine = result_pe_wine.iloc[:,0]
std_pe_wine = result_pe_wine.iloc[:,1]
plt.errorbar(log_alter_a,ero_pe_wine,std_pe_wine,color='red')
plt.title('wine data set', fontsize=20)
plt.xlabel('log(alpha) in perceptron')
plt.ylabel('error rate')
plt.show()

result_SVM = [error_rate_for_linear_SVM(wine_split_folds,i) for i in alter_C_SVM ]
result_SVM = pd.DataFrame(result_SVM)
ero_SVM = result_SVM.iloc[:,0]
std_SVM = result_SVM.iloc[:,1]
plt.errorbar(log_alter_C_SVM,ero_SVM,std_SVM,color='green')
plt.title('wine data set', fontsize=20)
plt.xlabel('log(C) in linear SVM')
plt.ylabel('error rate')
plt.show()

    
result_KNN = [error_rate_for_KNN(wine_split_folds,int(i)) for i in alter_k ]
result_KNN = pd.DataFrame(result_KNN)
ero_KNN = result_KNN.iloc[:,0]
std_KNN = result_KNN.iloc[:,1]
plt.errorbar(alter_k ,ero_KNN,std_KNN,color= 'skyblue')
plt.title('wine data set', fontsize=20)
plt.xlabel('k in KNN')
plt.ylabel('error rate')
plt.show()

