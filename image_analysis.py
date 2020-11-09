# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:40:53 2020

@author: gmche
"""
import numpy as np 
import dim_red as dr
import matplotlib.pyplot as plt 
#from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as nb 
basepath = 'PACS/photo/' 

#extract data from directories by instantiating the class object
data = dr.dim_red(basepath)

#DATA RE-SHUFFLE 
images, lab = data.get_shuffle()

#GET IMAGE DATA 
n_samples = 400
images, lab = data.get_data(n_samples)

#DATA NORMALIZE
norm_images = data.rescale(np.mean(images, axis = 0 ), np.std(images, axis = 0))

######################################## DIMENSIONAL REDUCTION ############################################
#APPLYING PCA: CHOOSING NUMBER OF COMPONENTS
comp = 360
proj_im = data.pca(comp)



#########################################  IMAGES PLOT   #################################
#DATA DENORMALIZE

###################################  NAIVE-BAYES CLASSIFICATION  ###########################################

from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as nb
from sklearn import metrics 

def validation(X,Y,test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    gnb = nb.GaussianNB()
    train = gnb.fit(X_train,y_train)
    y_pred = train.predict(X_test)
    #accuracy = np.sum(y_pred == y_test)
    return metrics.accuracy_score(y_test, y_pred)

def mean_acc(matrix,n_val, n_comp,l):
    mean_acc = []
    for i in range(1,n_comp):
        acc_data = list(map(lambda j: validation(matrix[:,:i],l,0.3), range(0,n_val)))
        mean_acc.append(np.mean(acc_data))
    return mean_acc

ag_data = [images, norm_images, proj_im]

n_val = 40

mean_acc_data = [mean_acc(x,40,comp, lab) for x in ag_data]

plt.figure(figsize = (14,9))
plt.title('7-classes Naive Bayes classifier Expected Accuracy', fontsize = 15)
plt.plot(mean_acc_data[2])
plt.xlabel('Number of Principal Compoenents', fontsize = 15)
plt.ylabel('Expected Accuracy', fontsize = 15)
plt.grid()
plt.show()
    
    
    
    
    
    
    
    
    
    





