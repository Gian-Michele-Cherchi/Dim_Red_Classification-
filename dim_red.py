# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:07:37 2020

@author: gmche
"""
#This class extracts images from pre-categorized directories and creates the data structure
#suited for dimensional reduction. (__init__)
#It also performs The PCA image reconstruction, as well data normalization and denormalization 

import os
from PIL import Image
#import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.decomposition import PCA

class dim_red:
    def __init__(self, path):
        self.__path = path 
        self.__im_data = []
        self.__labels = []
        self.__direct = os.listdir(self.__path)
        for lab in self.__direct:
            self.__im_names = os.listdir(path+lab+'/')[:181]
            self.__images = []
            for name in self.__im_names:
                self.__im = np.asarray(Image.open(self.__path+lab+'/'+name))
                self.__images.append(self.__im.ravel()) #flattening the image data structure(RGB)
            self.__labels.append(np.array(len(self.__images)*[lab], dtype = object))
            self.__im_data.append(np.array(self.__images))
        self.__im_data = np.concatenate(self.__im_data)
        self.__labels = np.concatenate(self.__labels)
        
        
        
    
    def get_data(self, n_samples):
        self.__samp = n_samples
        self.__im_data = self.__im_data[:self.__samp,:]
        self.__labels = self.__labels[:self.__samp]
        return self.__im_data, self.__labels 
        
    def get_shuffle(self):
         self.__perm = list(np.random.permutation(self.__im_data.shape[0]))
         self.__labels = self.__labels[self.__perm]
         self.__im_data = self.__im_data[self.__perm,:]
         return self.__im_data, self.__labels 
     
    def rescale(self, means, std_dev):
        self.__means, self.__std_dev = means, std_dev
        self.__im_data = (self.__im_data-self.__means)/self.__std_dev
        return self.__im_data
    
    def org_scale(self,proj):
        self.__proj = proj
        return np.array([self.__proj[:,i]*self.__std_dev[i]+self.__means[i] for i in range(len(self.__proj[0,:]))]).transpose()
    
    def pca(self, comp_list):
        self.__list_comp = comp_list
        try:
            self.__pca = [PCA(c) for c in self.__list_comp]
            self.__pca_eig = [self.__pca[i].fit_transform(self.__im_data) for i in range(len(self.__list_comp))]
            #self.__im_proj = [self.__pca[i].inverse_transform(self.__pca_eig[i]) for i in range(len(self.__list_comp))]
        except:
            self.__pca = PCA(self.__list_comp)
            self.__pca_eig = self.__pca.fit_transform(self.__im_data) 
            #self.__im_proj = self.__pca.inverse_transform(self.__pca_eig) 
        
        return self.__pca_eig
    
    
    
    
    
    
    
    
    