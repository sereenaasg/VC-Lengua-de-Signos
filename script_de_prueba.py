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
import scipy



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

def flood_fill(field, x ,y, old, new):
    # we need the x and y of the start position, the old value,
    # and the new value
    # the flood fill has 4 parts
    # firstly, make sure the x and y are inbounds
    if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
        return
    # secondly, check if the current position equals the old value
    if field[y][x] != old:
        return

    # thirdly, set the current position to the new value
    field[y][x] = new
    # fourthly, attempt to fill the neighboring positions
    flood_fill(field, x+1, y, old, new)
    flood_fill(field, x-1, y, old, new)
    flood_fill(field, x, y+1, old, new)
    flood_fill(field, x, y-1, old, new)

def flood_fill2(field, x ,y, old, new):
	if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
		return
	stack = []
	stack.append([x,y])
	sz = field.shape
	visits = np.zeros(sz, dtype=int)
	while (stack): # while stack not void
		i, j = stack.pop()
		visits[i][j] = 1
		# check if the current position equals the old value
		if field[i][j] != old:
			continue
	    # set the current position to the new value
		field[i][j] = new
	    # attempt to fill the neighboring positions
		if i+1 < sz[0]:
			if visits[i+1][j] == 0:
				stack.append([i+1,j])
		if i-1 > 0:
			if visits[i-1][j] == 0:
				stack.append([i-1,j])
		if j+1 < sz[1]:
			if visits[i][j+1] == 0:
				stack.append([i,j+1])
		if j-1 > 0:
			if visits[i][j-1] == 0:
				stack.append([i,j-1])


path = "./frames/positivo/IMG_0015.jpg"

# Imagen de Serena
imagen = llegir_imatge(path)
plt.imshow(imagen, cmap='gray')
plt.show()




thr = 65
img_thr = np.uint8(imagen < thr)

plt.imshow(img_thr, cmap='gray')
plt.show()

histograma, bins = np.histogram(imagen, bins=255)
bins = bins[:-1].astype(np.uint8)
plt.plot(bins, histograma)
plt.show()


kernel = np.ones((11,11),np.uint8)
erosion = cv2.erode(img_thr,kernel)
dilate = cv2.dilate(erosion, kernel)




plt.imshow(erosion, cmap='gray')
plt.show()
plt.imshow(dilate, cmap='gray')
plt.show()
'''



sh = imagen.shape
pos = np.where(dilate == 1)

y_min = np.min(pos[1])
y_max = np.max(pos[1])
im = dilate[:470, y_min:y_max]
plt.imshow(im, cmap='gray')
plt.show()

im = im[::1,::1]
plt.imshow(im, cmap='gray')
plt.show()
'''
flood_fill2(dilate, 0 ,0, 0, 1)


plt.imshow(dilate, cmap='gray')
plt.show()

"""
detector = cv2.SimpleBlobDetector()
kp = detector.detect(im)






# Imagen contorno de Serena
imagen_contorn = contorn(imagen)
plt.imshow(imagen_contorn, cmap='gray')
plt.show()

imagen_contorn = np.uint8(imagen_contorn)


# Imagen de la mano
path2 = "./mans/ma_si.png"
kernel = llegir_imatge(path2)
ma = contorn(kernel)
ma = np.uint8(ma)
plt.imshow(ma, cmap='gray')
plt.show()
#corr = cv2.filter2D(imagen_contorn, -1, np.array([[-1, 0, 1]]))
corr = cv2.filter2D(imagen_contorn, -1, ma)
"""









