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


path = "./mans/ma_subnormal.png"

"""
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
"""


imagen = cv2.imread(path)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)



# Imatge és un numpy array d'una imatge en escala de grisos
def contorn(imatge):
	# Derivada parcial en x
	dx = cv2.filter2D(imatge, -1, np.array([[-1, 0, 1]]))
	dx = np.float16(dx)
	# Derivada parcial en y
	dy = cv2.filter2D(imatge, -1, np.array([[1],[0],[-1]]))
	dy = np.float16(dy)
	# Retornem el modul del vector gradient (sqrt(dx^2 + dy^2))
	return np.sqrt((dx**2) + (dy**2))


a = contorn(imagen_gris)
plt.imshow(a, cmap='gray')