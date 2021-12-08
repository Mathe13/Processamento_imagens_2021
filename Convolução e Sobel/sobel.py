# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:36:31 2021

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

def basic_convolution(m1,m2):
    #numero de linhas e colunas
    m1cols = m1.shape[1]
    m1rows = m1.shape[0]
    m2cols = m2.shape[1]
    m2rows = m2.shape[0]
    
    #newRows = m1rows + m2rows - 1
    #newColumns = m1cols + m2cols - 1

    # cria matriz saida
    y = np.zeros((m1rows,m1cols))
    
    # go over input locations
    for m in range(m1rows):
        for n in range(m1cols):
     
             for i in range(m2rows):
                 for j in range(m2cols):
                      if (m-i >= 0) and (m-i < m1rows ) and (n-j >= 0) and (n-j < m1cols):
  
                          y[m,n] = y[m,n] + m2[i,j]*m1[m-i,n-j]
    
    return y

def adjust(img,distance=0):
    img = np.where(img < 0, 0, img)
    img = np.where(img > 255, 255, img)

    return img.astype(int)

img_original = mpimg.imread('lenna.png')

img_gray = turn_gray(img_original)

imgplot = plt.imshow(img_gray,cmap='gray')
plt.title("imagem cinza")
plt.show()

sobel_x = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
sobel_y = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])

test_img = basic_convolution(img_gray,sobel_x)
test_adjust = adjust(test_img)
imgplot = plt.imshow(adjust(test_img),cmap='gray')
plt.title("imagem sobel x")
plt.show()

test_img2 = basic_convolution(img_gray,sobel_y)
imgplot = plt.imshow(adjust(test_img2),cmap='gray')
plt.title("imagem sobel y")
plt.show()

test_img3 = basic_convolution(test_img,sobel_y)
imgplot = plt.imshow(adjust(test_img3),cmap='gray')
plt.title("imagem sobel x e y")
plt.show()

test_img3 = basic_convolution(test_img2,sobel_x)
imgplot = plt.imshow(adjust(test_img3),cmap='gray')
plt.title("imagem sobel y e x")
plt.show()

#fazer o de angulo