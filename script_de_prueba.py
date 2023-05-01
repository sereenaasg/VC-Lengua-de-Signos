# -*- coding: utf-8 -*-
"""
Created on Mon May  1 10:34:14 2023

@author: javie
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy import signal


# Devuelve la imagen en escala de grises
def llegir_imatge(path):
	return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


# Imatge és un numpy array d'una imatge en escala de grisos uint8
def contorn(imatge):
	# Derivada parcial en x
	dx = cv2.filter2D(imatge, -1, np.array([[-1, 0, 1]]))
	dx = np.float16(dx)
	# Derivada parcial en y
	dy = cv2.filter2D(imatge, -1, np.array([[1],[0],[-1]]))
	dy = np.float16(dy)
	# Retornem el modul del vector gradient (sqrt(dx^2 + dy^2))
	return np.sqrt((dx**2) + (dy**2))




path = "./frames/positivo/IMG_0015.jpg"

# Imagen de Serena
imagen = llegir_imatge(path)
plt.imshow(imagen, cmap='gray')
plt.show()
# Imagen contorno de Serena
imagen_contorn = contorn(imagen)
plt.imshow(imagen_contorn, cmap='gray')
plt.show()

imagen_contorn = np.uint8(imagen_contorn)



path2 = "./mans/ma_si.png"
kernel = llegir_imatge(path2)
ma = contorn(kernel)
ma = np.uint8(ma)
#corr = cv2.filter2D(imagen_contorn, -1, np.array([[-1, 0, 1]]))
corr = cv2.filter2D(imagen_contorn, -1, ma)


