# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:46:52 2021

@author: matheus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import log10,ceil,pow,sqrt,isinf
from datetime import datetime
from mpl_toolkits.axes_grid.axislines import SubplotZero


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
                     if m>m2.shape[0] and n>m2.shape[1]:
                          if (m-i >= 0) and (m-i < m1rows ) and (n-j >= 0) and (n-j < m1cols):
  
                              y[m,n] = y[m,n] + m2[i,j]*m1[m-i,n-j]
    
    return y

def adjust(img,distance=0):
    img = np.where(img < 0, 0, img)
    img = np.where(img > 255, 255, img)

    return img.astype(int)

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

sobel_x = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
sobel_y = np.array([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])

img_original = mpimg.imread('teste3.png')

img_gray = turn_gray(img_original)

imgplot = plt.imshow(img_gray,cmap='gray')
plt.title("imagem cinza")
plt.show()

img_x = adjust(basic_convolution(img_gray,sobel_x))
img_y = adjust(basic_convolution(img_gray,sobel_y))


sobel2 = fixed(np.sqrt(np.add(img_x**2,img_y**2)))

imgplot = plt.imshow(sobel2,cmap='gray')
plt.title("imagem sobel x e y")
plt.show()

img = fixed(sobel2)

#hough
thetas = np.deg2rad(np.arange(0.0, 180.0))
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)
num_thetas = len(thetas)
phos = np.zeros((img.shape[0]*img.shape[0],len(thetas)))
y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

for i in range(len(x_idxs)):
  x = x_idxs[i]
  y = y_idxs[i]

  for t_idx in range(num_thetas):
    phos[i][t_idx] = round(x * cos_t[t_idx] + y * sin_t[t_idx],2)

plt.imshow(phos, cmap='jet',
           extent=[np.deg2rad(0), np.rad2deg(180), np.min(phos), np.max(phos)])
plt.show()

uniques,counts = np.unique(phos,return_counts = True)
acumulador = np.zeros((len(uniques),num_thetas))
print(np.max(counts))
print(np.mean(counts))

limiar = 140

a = b = []
for count_idx in range(len(counts)):
    if counts[count_idx]>=limiar:
        pho = uniques[count_idx]
        pho_idx,theta_idx = np.where(phos == pho)
        theta = thetas[theta_idx[0]]
        a_aux = np.cos(theta)/np.sin(theta)
        b_aux = -pho/np.sin(theta)
        if not isinf(a_aux) and not isinf(b_aux):
            a.append(a_aux)
            b.append(b_aux)
    
x = np.arange(0,img.shape[1],1)

for i in range(len(a)):
    y = ((a[i])*x) + (b[i])
    plt.plot(x, y)


for i in range(len(uniques)):
    pho = uniques[i]
    pho_idx,theta_idx = np.where(phos == pho)
    for j in range(len(theta_idx)):
        acumulador[i][theta_idx] +=1
            

plt.imshow(sobel2,cmap='gray')

plt.show()

