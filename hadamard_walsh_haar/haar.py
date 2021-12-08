# -*- coding: utf-8 -*-
"""
Created on Tue May 18 08:17:56 2021

@author: matheus
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import log10,ceil,pow,sqrt
from datetime import datetime

def turn_gray(img):
    r = 0
    g = 1
    b = 2
    img_gray = np.empty((len(img),len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pixel = img[i][j]
            if (len(pixel)) > 1:
                img_gray[i][j] = round(255*((pixel[r] + pixel[g] + pixel[b])/3))
    return img_gray


def create_haar_matrix(n):
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = create_haar_matrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # parte de cima
    h_n = np.kron(h, [1, 1])
    # parte de baixo
    h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])    # combine parts
    h = np.vstack((h_n, h_i))
    return h

def haar_transformation(img,inverse = False):
    if(img.shape[0] != img.shape[1]):
        print("erro, precisa ser uma img quadrada")
        return
    size = img.shape[0]
    haar = create_haar_matrix(size)/sqrt(size)
    if inverse:
        output = np.dot(np.dot(haar.T,img),haar.T)
    else:
        output = np.dot(np.dot(haar,img),haar)
    return output



def pad_power_2(img):
    if(img.shape[0] != img.shape[1]):
        print("erro, precisa ser uma img quadrada")
        return
    size = img.shape[0]
    if ((log10(size)/log10(2)).is_integer()): 
        return img
    else:
        next_value = pow(2, ceil(log10(size)/log10(2)));
        diference = int(next_value-img.shape[0])
        return np.pad(img, ((diference,0),(diference,0)), 'constant')        


h4 = create_haar_matrix(4)
h8 = create_haar_matrix(8)
h16 = create_haar_matrix(16)

teste = np.array([[1,1,1,1],[1,2,2,1],[1,2,2,1],[1,1,1,1]])
transformada = haar_transformation(teste)
inversa = haar_transformation(transformada,inverse=True)


img_original = mpimg.imread('lenna.png')

img_gray = turn_gray(img_original)

imgplot = plt.imshow(img_gray,cmap='gray')
plt.title("imagem cinza")
plt.show()

img_paded = pad_power_2(img_gray)

haar_img = haar_transformation(img_paded)
imgplot = plt.imshow(20*np.log10(abs(haar_img)),cmap='gray')
plt.title("imagem haar")
plt.show()

devolta = haar_transformation(haar_img,inverse=True)
imgplot = plt.imshow(abs(devolta),cmap='gray')
plt.title("imagem haar inversa")
plt.show()


#cortando
for i in range(haar_img.shape[0]):
    for j in range(haar_img.shape[0]):
        if i < 100 and j < 512:
            haar_img[i][j]=0
            
imgplot = plt.imshow(20*np.log10(abs(haar_img)),cmap='gray')
plt.title("imagem haar cortada")
plt.show()

devolta = haar_transformation(haar_img,inverse=True)
imgplot = plt.imshow(abs(devolta),cmap='gray')
plt.title("imagem haar inversa cortada")
plt.show()