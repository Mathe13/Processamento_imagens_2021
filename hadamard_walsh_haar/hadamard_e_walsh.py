# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import log10,ceil,pow
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


def create_hadamard_matrix(size=8):
    power2 = log10(size)/log10(2)
    if (not power2.is_integer()): 
        print("size precisa ser uma potenciaa de 2")
        return 
    repeat_times = int(size/2)
    base_hadamard = np.array([[1,1],[1,-1]]) #h2
    
    if (repeat_times == 1):
        return base_hadamard
    
    for k in range(int(power2-1)):
        #tem q repetir pra cada potencia de 2 até a pedida
        a = np.tile(base_hadamard,(2,2))
        for i in range(int(a.shape[0]/2),a.shape[0]):
            for j in range(int(a.shape[1]/2),a.shape[1]):
                a[i][j] = a[i][j] * -1
        base_hadamard = a
    return a

def create_walsh_matrix(size):
    hadamard = create_hadamard_matrix(size)
    aux = np.zeros_like(hadamard)
    for i in range(hadamard.shape[0]):
        cont = 0
        for j in range(hadamard.shape[1]):
            if (j == 0):
                ant = hadamard[i][j]
            if ( ((ant ^ hadamard[i][j]) >> 31) ):
                cont = cont + 1
            ant = hadamard[i][j]
        aux[cont] = hadamard[i]
    return aux
    

def hadamard_transformation(img):
    if(img.shape[0] != img.shape[1]):
        print("erro, precisa ser uma img quadrada")
        return
    size = img.shape[0]
    hadamard = create_hadamard_matrix(size)
    output = np.dot(np.dot(hadamard,img),hadamard) * (1/(img.shape[0]**2)) 
    return output

def walsh_transformation(img):
    if(img.shape[0] != img.shape[1]):
        print("erro, precisa ser uma img quadrada")
        return
    size = img.shape[0]
    walsh = create_walsh_matrix(size)
    output = np.dot(np.dot(walsh,img),walsh) * (1/(img.shape[0]**2)) 
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


teste = np.array([[1,1,1,1],[1,2,2,1],[1,2,2,1],[1,1,1,1]])
transformada = hadamard_transformation(teste)

teste2 = np.array([[1,1,1,1,1],[1,2,2,1,1],[1,2,2,1,1],[1,1,1,1,1],[1,1,1,1,1]])
a = pad_power_2(teste2)

b = create_walsh_matrix(8)




img_original = mpimg.imread('lenna.png')

img_gray = turn_gray(img_original)

imgplot = plt.imshow(img_gray,cmap='gray')
plt.title("imagem cinza")
plt.show()

img_paded = pad_power_2(img_gray)

hadamard_img = hadamard_transformation(img_paded)
imgplot = plt.imshow(20*np.log10(abs(hadamard_img)),cmap='gray')
plt.title("imagem hadamard")
plt.show()

walsh_img = walsh_transformation(img_paded)
imgplot = plt.imshow(20*np.log10(abs(walsh_img)),cmap='gray')
plt.title("imagem walsh")
plt.show()

for i in range(hadamard_img.shape[0]):
    for j in range(hadamard_img.shape[1]):
        if i < 100 and j < 100:
            hadamard_img[i][j] = 0


imgplot = plt.imshow(20*np.log10(abs(hadamard_img)),cmap='gray')
plt.title("imagem hadamard")
plt.show()

volta = hadamard_transformation(hadamard_img)
imgplot = plt.imshow(volta,cmap='gray')
plt.title("imagem hadamard")
plt.show()
