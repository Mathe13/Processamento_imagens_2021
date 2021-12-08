# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:54:50 2021

@author: matheus
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def dft_2d(image):
    #seguindo a formula apresentada em aula
    M,N = image.shape
    dft2d = np.zeros((M,N),dtype=complex)
    
    #percorre toda a imagem
    for x in range(M):
        for y in range(N):
            sum_matrix = 0.0
            #somatorios
            for u in range(M):
                for v in range(N):
                    expoent = np.exp((-2j * np.pi)*( ((u*x)/M) + ((v*y)/N) ))
                    sum_matrix += image[u,v]*expoent
                    
            dft2d[x,y] = sum_matrix
    return dft2d*(1/(M*N))
    
def dft_2d_inv(image):
    M,N = image.shape
    inv = np.zeros((M,N),dtype=complex)
    
    #percorre toda a imagem
    for x in range(M):
        for y in range(N):
            sum_matrix = 0.0
            #somatorios
            for u in range(M):
                for v in range(N):
                    expoent = np.exp((2j * np.pi)*( ((u*x)/M) + ((v*y)/N) ))
                    sum_matrix += image[u,v]*expoent
                    
            inv[x,y] = sum_matrix
    return inv

#https://images.slideplayer.com/26/8685943/slides/slide_3.jpg
def dft(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    temp = np.dot(M, x)*(1/N)

    N = temp.shape[1]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return (np.dot(M, temp.T)).T  

def dft_inv(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    temp = np.dot(M, x)

    N = temp.shape[1]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return (np.dot(M, temp.T)*(N)).T  
    

def representation(data):
    datatiled = data.reshape((16,)*4).transpose(0,2,1,3).reshape((256,256))
    plt.subplot(121)
    plt.imshow(np.real(datatiled), cmap='gray', interpolation='nearest')
    plt.subplot(122)
    plt.imshow(np.imag(datatiled), cmap='gray', interpolation='nearest')
    plt.show()

              

img_original = mpimg.imread('lenna.png')

img_gray = turn_gray(img_original)

imgplot = plt.imshow(img_gray,cmap='gray')
plt.title("imagem cinza")
plt.show()

print("transformada inicio ", datetime.now())
dft2d_img = dft(img_gray)
inv = dft_inv(dft2d_img)
print("transformada fim ", datetime.now())

a = np.copy(dft2d_img)
x,y = dft2d_img.shape
for i in range(x):
    for j in range(y):
        if i>150 and i < 250:
            a[i][j]=0
        if j>150 and j < 250:
            a[i][j]=0

a_inv = dft_inv(a)
imgplot = plt.imshow(20*np.log10(abs(np.real(a))),cmap='gray')
plt.title("imagem teste")
plt.show()
imgplot = plt.imshow((np.real(a_inv).astype(int)),cmap='gray')
plt.title("imagem inversa")
plt.show()



imgplot = plt.imshow(20*np.log10(abs(np.real(dft2d_img))),cmap='gray')
plt.title("imagem teste")
plt.show()

imgplot = plt.imshow((np.real(inv).astype(int)),cmap='gray')
plt.title("imagem inversa")
plt.show()






