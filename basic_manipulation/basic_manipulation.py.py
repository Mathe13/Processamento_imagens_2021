
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

#aceita imagens cinzas
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
  
def change_brightness(img,coefficient):
    if (img.max()) <= 1:
        print("must be a 255 encoded img")
    new_img = np.zeros_like(img)
    temp = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            temp = img[i][j] + coefficient
            if (temp > 255):
                temp = 255
            elif(temp < 0):
                temp = 0
            new_img[i][j] = temp
    return new_img
            
def adjust_contrast(img,distance=0):
    if (img.max()) <= 1:
        print("must be a 255 encoded img")
    uniques = np.sort(np.unique(img))
    max_d = 255/len(uniques)
    if(distance > max_d or distance == 0):
        print("this distance is invalid, using max distance:",max_d)
        distance = max_d
    for i in range(len(uniques)):
        img = np.where(img == uniques[i], i*distance, img)

    return img

def equalize(img,distp):
    cont = 0
    for i in range(len(distp)):
        cont = cont + distp[i]
        img = np.where(img == i,-cont*255,img)
    img = np.where(img < 0,-img,img)
    return img
        
def get_contrast(img):
    if (img.max()) <= 1:
        print("must be a 255 encoded img")
    print("contras is :",((img.max()-img.min())/255))

def get_brightness(img):
    if (img.max()) <= 1:
        print("must be a 255 encoded img")
    print("brightness is :",(img.max()/255))

if __name__ == "__main__":
    img_original = mpimg.imread('lenna.png')
    
    imgplot = plt.imshow(img_original)
    plt.title("imagem original")
    plt.show()
    
    img_gray = turn_gray(img_original)
    get_brightness(img_gray)
    get_contrast(img_gray)
    
    imgplot = plt.imshow(img_gray,cmap='gray')
    plt.title("imagem cinza")
    plt.show()
    
    
    y_hist = get_histogram(img_gray)
    x_hist = range(0,256)
    hist_plot = plt.bar(x_hist,y_hist)
    plt.title("histograma")
    plt.show()
    
    y_distp = get_distp(img_gray,y_hist)
    x_distp = range(0,256)
    distp_plot = plt.bar(x_distp,y_distp)
    plt.title("distp")
    plt.show()
    
    equalized_img = equalize(img_gray,y_distp)
    imgplot_equalized = plt.imshow(equalized_img,cmap='gray')
    plt.title("equalizada")
    plt.show()
    
    y_hist_equalized = get_histogram(equalized_img)
    x_hist_equalized = range(0,256)
    hist_plot_equalized = plt.bar(x_hist_equalized,y_hist_equalized)
    plt.title("histograma equalized")
    plt.show()
    
    y_distp_equalized = get_distp(equalized_img,y_hist_equalized)
    x_distp_equalized = range(0,256)
    distp_plot_equalized = plt.bar(x_distp_equalized,y_distp_equalized)
    plt.title("distp equalized")
    plt.show()
    
    
    img_gray_brilho = change_brightness(img_gray,50)
    imgplot_brilho = plt.imshow(img_gray_brilho,cmap='gray')
    plt.title("mudando o brilho")
    plt.show()
    
    y_hist_brilho = get_histogram(img_gray_brilho)
    x_hist_brilho = range(0,256)
    hist_plot_brilho = plt.bar(x_hist_brilho,y_hist_brilho)
    plt.title("histograma brilho")
    plt.show()
    
    y_distp_brilho = get_distp(img_gray_brilho,y_hist)
    x_distp_brilho = range(0,256)
    distp_plot_brilho = plt.bar(x_distp_brilho,y_distp_brilho)
    plt.title("distp brilho")
    plt.show()
    
    img_gray_contraste = adjust_contrast(img_gray)
    imgplot_contraste = plt.imshow(img_gray_contraste,cmap='gray')
    plt.title("mudando contraste")
    plt.show()
    
    y_hist_contraste = get_histogram(img_gray_contraste)
    x_hist_contraste = range(0,256)
    hist_plot = plt.bar(x_hist_contraste,y_hist_contraste)
    plt.title("histograma contraste")
    plt.show()
    
    y_distp_contraste = get_distp(img_gray_contraste,y_hist_contraste)
    x_distp_contraste = range(0,256)
    distp_plot = plt.bar(x_distp_contraste,y_distp_contraste)
    plt.title("distp contraste")
    plt.show()
    
    
    
    
