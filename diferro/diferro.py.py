# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:14:06 2021

@author: matheus
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

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

def diferro(img_a):
    img_a = img_a/255
    img_b = np.empty_like(img_a)
    for l in range(len(img_a)):
        for c in range(len(img_a[0])):
            erro = 0
            if (img_a[l][c]>0.5):
                img_b[l][c] = 1.0
            else:
                img_b[l][c] = 0.0
    
            erro = (img_b[l][c]-img_a[l][c])
     
            if ((l+1)<len(img_a)):
                img_a[l+1][c]=img_a[l+1][c]-(erro*3/8)
            if (((c+1)<len(img_a[0])) and  ((l+1)<len(img_a))):
                img_a[l+1][c+1]=img_a[l+1][c+1]-(erro/4)
            if ((c+1)<len(img_a[0])):
                img_a[l][c+1]=img_a[l][c+1]-(erro*3/8);
    return img_b*255

def fixed(img_a):
    img_a = img_a/255
    img_b = img_a
    for l in range(len(img_a)):
        for c in range(len(img_a[0])):
            if (img_a[l][c]>0.5):
                img_b[l][c] = 1
            else:
                img_b[l][c] = 0
            
    return img_b*255



img_original = mpimg.imread('lenna.png')

imgplot = plt.imshow(img_original)
plt.title("imagem original")
plt.show()

img_gray = turn_gray(img_original)

imgplot = plt.imshow(img_gray,cmap='gray')
plt.title("imagem cinza")
plt.show()


img_fixed = fixed(img_gray)
imgplot = plt.imshow(img_fixed,cmap='gray')
plt.title("imagem fixed")
plt.show()


img_dif = diferro(img_gray)
imgplot = plt.imshow(img_dif,cmap='gray')
plt.title("imagem dif")
plt.show()