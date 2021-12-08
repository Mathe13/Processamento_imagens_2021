# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:33:17 2021

@author: matheus
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import cmath


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

#inicio
img_original = mpimg.imread('texto.png')

#imagem cinza
img_gray = turn_gray(img_original)
imgplot = plt.imshow(img_gray,cmap='gray')
plt.title("imagem cinza")
plt.show()

#histograma
histograma = get_histogram(img_gray)
hist_plot = plt.bar(range(0,256),histograma)
plt.title("histograma")
plt.show()
#distp
distp = get_distp(img_gray, histograma)
hist_plot = plt.bar(range(0,256),distp)
plt.title("distp")
plt.show()

#inicialização do laço
zmax = np.max(img_gray).astype(int)
i = 1
T_i = zmax - i
Erro = 10**6
vetor_erro = []
vetor_t = []



while(i<zmax):
#calcular p1 e p2
    p1 = 0
    p2 = 0
    for p1_idx in range(0,T_i,1):
        p1 = p1 + distp[p1_idx]
        
    for p2_idx in range(T_i,zmax+1,1):
        p2 = p2 + distp[p2_idx]

#determinar media_1
    #media1 = np.where(distp[:T_i] == np.max(distp[:T_i]))[0][0]
    media1 = (len(distp[:T_i])//2)
    if media1 > 255:
        media1 =255

#determinar media_2
    #media2 = T_i + np.where(distp[T_i:] == np.max(distp[T_i:]))[0][0]
    media2 = T_i + (len(distp[T_i:])//2)
    if media2 > 255:
        media2 =255
        
#determinar variancia_1
    z = media1
    S = distp[z]
    while ((S <= (p1 * 0.34)) and z<255):
        z = z+1
        S = S + distp[z]
    variancia1 = abs(media1 - z)

#determinar variancia_2
    z = media2
    S = distp[z]
    while ((S <= (p2 * 0.34)) and z > 0):
        z = z-1
        S = S + distp[z]
    variancia2 = abs(z - media2)


#calcular A
    A = variancia1**2 - variancia2**2

#calcular B
    B = 2*(media1*variancia2**2 - media2*variancia1**2)

#calcular C
    
    if (variancia1*p2 != 0) and ((variancia2*p1)/(variancia1*p2)) > 0:
        C = ((media2**2)*(variancia1**2)) - ((media1**2)*(variancia2**2)) + 2*(variancia1**2)*(variancia2**2)*math.log((variancia2*p1)/(variancia1*p2))

#calcular T
        d = (B**2) - (4*A*C)  
        t1 = (-B-cmath.sqrt(d))/(2*A)  
        t2 = (-B+cmath.sqrt(d))/(2*A) 
        vetor_t.append(t1)
        vetor_t.append(t2)

        if (t1.imag == 0 and t1.real > 0):
            E_i = t1.real - T_i
            if (E_i < Erro):
                Erro = E_i
                vetor_erro.append(Erro)
                Limiar = T_i
        if (t2.imag == 0 and t2.real > 0):
            E_i = t2.real - T_i
            if (E_i < Erro):
                Erro = E_i
                vetor_erro.append(Erro)
                Limiar = T_i

    i = i + 1
    T_i = zmax - i

result = np.copy(img_gray)
result = np.where(img_gray>Limiar,255,0)

imgplot = plt.imshow(result,cmap='gray')
plt.title("resultado")
plt.show()


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

subject_opening = np.zeros_like(result)

for i in range(subject_opening.shape[0]):
    for j in range(subject_opening.shape[1]):
        if result[i][j] == 0:
           subject_opening[i][j] = 1 
           
imgplot = plt.imshow(subject_opening,cmap='gray')
plt.title("invertido")
plt.show()

struct_element = np.array([[0,1,0],
                           [1,1,1],
                           [0,1,0]])

imgplot = plt.imshow(erosion(dilation(subject_opening,struct_element),struct_element),cmap='gray')
plt.title("teste opening")
plt.show()