# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 20:34:08 2021

@author: matheus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import log10,ceil,pow,sqrt,isinf
from datetime import datetime

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


def dilation(m1,struct_element):
    #numero de linhas e colunas
  
    y = basic_convolution(m1, struct_element)
    
    return np.where(y >= 1,1,0)

def erosion(m1,struct_element):
    #numero de linhas e colunas
    limiar = 0
    for i in range(struct_element.shape[0]):
        for j in range(struct_element.shape[0]):
            if (struct_element[i][j] == 1):
                limiar = limiar + 1
    
    y = basic_convolution(m1, struct_element)
    
    return np.where(y >= limiar,1,0)


img_teste =  np.array([[0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,1,1,1,1,1,1,0,0,0],
                       [0,0,1,1,1,1,1,1,0,0,0],
                       [0,0,1,1,1,1,1,1,0,0,0],
                       [0,0,0,1,0,0,1,1,0,0,0],
                       [0,0,0,1,1,0,1,1,0,0,0],
                       [0,0,0,0,0,1,1,1,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0],])


imgplot = plt.imshow(img_teste,cmap='gray')
plt.title("imagem teste")
plt.show()

imgplot = plt.imshow(np.ones((3,3)),cmap='gray')
plt.title("struct teste")
plt.show()

img_dilation = dilation(img_teste,np.ones((3,3)))

imgplot = plt.imshow(img_dilation,cmap='gray')
plt.title("dilatação")
plt.show()


img_erosion = erosion(img_dilation,np.ones((3,3)))

imgplot = plt.imshow(img_erosion,cmap='gray')
plt.title("erosão")
plt.show()


img_teste2 = np.array([[0,0,0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,0,1,0,0],
                       [0,1,0,0,0,0,0,1,0,0],
                       [1,0,0,0,0,0,0,1,0,0],
                       [0,0,0,1,0,0,0,1,0,1],
                       [0,0,0,0,1,0,0,1,1,0],
                       [0,0,0,0,0,1,0,1,0,0],
                       [0,0,0,0,0,0,1,1,0,0],
                       [0,0,0,0,0,1,0,1,0,0],
                       [0,0,0,0,1,0,0,1,1,0],
                       [0,0,0,1,0,0,0,1,0,1],
                       [0,0,1,0,0,0,0,1,0,0],
                       ])

imgplot = plt.imshow(img_teste2,cmap='gray')
plt.title("imagem teste2")
plt.show()

struc_element1 = np.array([[0,0,1],
                          [0,1,0],
                          [1,0,0]])

imgplot = plt.imshow(struc_element1,cmap='gray')
plt.title("struct teste")
plt.show()

img_erosion2 = erosion(img_teste2,struc_element1)

imgplot = plt.imshow(img_erosion2,cmap='gray')
plt.title("erosão2")
plt.show()

img_dilation2 = dilation(img_erosion2,struc_element1)

imgplot = plt.imshow(img_dilation2,cmap='gray')
plt.title("dilatação2")
plt.show()

img_dilation3 = dilation(img_erosion2,np.ones((3,3)))

imgplot = plt.imshow(img_dilation3,cmap='gray')
plt.title("dilatação2")
plt.show()



