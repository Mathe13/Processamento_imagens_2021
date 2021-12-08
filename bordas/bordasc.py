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

def get_histogram(img):
    if (img.max()) <= 1:
        img = img*255
    
    cont = np.zeros((256))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pixel = round(img[i][j])
            cont[pixel] = cont[pixel] + 1
            
    return cont
    
def get_distp(img,y_hist):
    y_distp = y_hist/(len(img)*len(img[0]))
    return y_distp

def equalize(img,distp):
    cont = 0
    for i in range(len(distp)):
        cont = cont + distp[i]
        img = np.where(img == i,-cont*255,img)
    img = np.where(img < 0,-img,img)
    return img

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

def fixed(img_a,limiar=0.5):
    img_a = img_a/255
    img_b = img_a
    for l in range(len(img_a)):
        for c in range(len(img_a[0])):
            if (img_a[l][c]>limiar):
                img_b[l][c] = 1
            else:
                img_b[l][c] = 0
            
    return img_b*255


def get_circle(xc,yc,radius,img):
    width, height = img.shape
    xx, yy = np.mgrid[:width, :height]
    circle = ((xx - xc) ** 2 + (yy - yc) ** 2)
    return ((circle < (radius+1)**2) & (circle > (radius**2))).astype(int)

def hough_circle(img,max_radius,min_radius=0):
    acumulator_frames = []
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges
    for r in range(min_radius,max_radius):
        aux_frame = np.zeros_like(img)
        for i in range(len(y_idxs)):
            circle = get_circle(x_idxs[i],y_idxs[i],r+1,img)
            aux_frame = np.add(aux_frame,circle)
        acumulator_frames.append(aux_frame)
    return acumulator_frames


sobel_x = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
sobel_y = np.array([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])

img_original = mpimg.imread('imagec.png')

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


img = sobel2

acumulator_frames = hough_circle(img, 12)
mean_value = round(np.mean(acumulator_frames))
max_value = round(np.max(acumulator_frames))
global_limiar = round(mean_value + (max_value-mean_value)*0.5)


result = np.zeros_like(img)
for i in range(len(acumulator_frames)):
    r = i+1
    mean_on_frame = round(np.mean(acumulator_frames[i]))
    max_on_frame  = round(np.max(acumulator_frames[i]))
    if max_on_frame > global_limiar:
        limiar = round(mean_on_frame + (max_on_frame-mean_on_frame)*0.7)
        y_idxs, x_idxs = np.where(acumulator_frames[i] > limiar)
        for j in range(len(y_idxs)):
            result = np.add(result,get_circle(x_idxs[j],y_idxs[j],r,result))

result[0][0] = 255
result_distp = get_distp(result,get_histogram(result))    
plt.imshow(img_original)
plt.imshow(equalize(result,result_distp),cmap='gray')
plt.title("resultado")
plt.show()        


