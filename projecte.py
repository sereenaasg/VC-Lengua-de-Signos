''' PRàCTICA 3 '''

import os
import skimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage import data, io, filters, feature, segmentation, measure
from skimage.io import imread
import cv2
#from moviepy.editor import ImageSequenceClip
from scipy.signal import convolve2d, correlate2d, fftconvolve
from scipy import signal, ndimage
from PIL import Image, ImageOps
from scipy.ndimage import shift
import time
from mpl_toolkits import mplot3d
import math
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

def carregarImatges(path):
    imatges = []
    etiquetes = []

    for etiqueta in os.listdir(path): ## per cada subcarpeta de la carpeta frames
        path_total = path+etiqueta+'/' # construim el path de les imatges d'aquesta etiqueta
        for imatge in os.listdir(path_total): # per cada imatge de la carpeta
            path_imatge = path_total+imatge
            imatge = Image.open(path_imatge)
            imatges.append(np.array(imatge))
            etiquetes.append(etiqueta)

    return np.array(imatges), etiquetes

def carregarVideo(name):
    capture = cv2.VideoCapture(name+'.mp4')
    cont = 0
    path = 'frames/'+name+'/'

    while (capture.isOpened()):
        ret, frame = capture.read()
        if (ret == True):
            if(cont%5==0):
                cv2.imwrite(path + 'IMG_%04d.jpg' % cont, frame)
            cont += 1
            if (cv2.waitKey(1) == ord('s')):
                break
        else:
            break

    capture.release()


def partirDataset(path):
    imagenes, etiquetas = carregarImatges(path)
    x_train, x_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def aplanarImagenes(train, test):
    new_train = []
    new_test = []
    for img in train:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # aplanar la matriz numpy en un vector
        img_vector = img_gray.ravel()
        new_train.append(img_vector)

    for img in test:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # aplanar la matriz numpy en un vector
        img_vector = img_gray.ravel()
        new_test.append(img_vector)

    return new_train, new_test

def guardarImatge(name, img):
    skimage.io.imsave(name+'.jpg', img)


if __name__ == "__main__":
	names = ['subnormal', 'positivo', 'negativo']
	path = 'frames/'
	os.mkdir(path)
	for i in names:
		os.mkdir(path+i)
		carregarVideo(i)

    #x_train, x_test, y_train, y_test = partirDataset(path)

    #x_train, x_test = aplanarImagenes(x_train, x_test)

    # Cargar la imagen
    # img = x_train[0]

    # # Convertir la imagen a espacio de color HSV
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # # Definir los valores de color de la piel en HSV
    # lower_skin = np.array([20, 20, 70], dtype=np.uint8)
    # upper_skin = np.array([140, 255, 255], dtype=np.uint8)

    # # Segmentar la imagen utilizando el umbral de color
    # mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # # Aplicar operaciones morfol�gicas para eliminar peque�as �reas
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # # Detectar los contornos de la regi�n segmentada
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Dibujar los contornos en la imagen original
    # plt.contour(mask, colors='r', levels=[0], alpha=0.5)

    # # Mostrar la imagen resultante
    # plt.imshow(img)

    # model = KNeighborsClassifier(5)
    # model.fit(x_train, y_train)
    # predicciones = model.predict(x_test)
    # acc = model.score(x_test, y_test)
    # print(acc)




