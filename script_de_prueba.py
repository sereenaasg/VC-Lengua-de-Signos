# -*- coding: utf-8 -*-
"""
Created on Mon May  1 10:34:14 2023

@author: javie
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops, regionprops_table
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import math


# Devuelve la imagen en escala de grises
def llegir_imatge(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


# Imatge és un numpy array d'una imatge en escala de grisos uint8
def contorn(imatge):
    # Derivada parcial en x
    dx = cv2.filter2D(imatge, -1, np.array([[-1, 0, 1]]))
    dx = np.float16(dx)
    # Derivada parcial en y
    dy = cv2.filter2D(imatge, -1, np.array([[1], [0], [-1]]))
    dy = np.float16(dy)
    # Retornem el modul del vector gradient (sqrt(dx^2 + dy^2))
    return np.sqrt((dx**2) + (dy**2))


def flood_fill(field, x, y, old, new):
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
    flood_fill(field, x + 1, y, old, new)
    flood_fill(field, x - 1, y, old, new)
    flood_fill(field, x, y + 1, old, new)
    flood_fill(field, x, y - 1, old, new)


def flood_fill2(field, x, y, old, new):
    if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
        return
    stack = []
    stack.append([x, y])
    sz = field.shape
    visits = np.zeros(sz, dtype=int)
    while stack:  # while stack not void
        i, j = stack.pop()
        visits[i][j] = 1
        # check if the current position equals the old value
        if field[i][j] != old:
            continue
            # set the current position to the new value
        field[i][j] = new
        # attempt to fill the neighboring positions
        if i + 1 < sz[0]:
            if visits[i + 1][j] == 0:
                stack.append([i + 1, j])
        if i - 1 > 0:
            if visits[i - 1][j] == 0:
                stack.append([i - 1, j])
        if j + 1 < sz[1]:
            if visits[i][j + 1] == 0:
                stack.append([i, j + 1])
        if j - 1 > 0:
            if visits[i][j - 1] == 0:
                stack.append([i, j - 1])


def preprocessament(path):
    # Imagen de Serena
    imagen = llegir_imatge(path)
    # plt.imshow(imagen, cmap="gray")
    # plt.show()

    thr = 65
    img_thr = np.uint8(imagen < thr)

    # plt.imshow(img_thr, cmap="gray")
    # plt.show()

    histograma, bins = np.histogram(imagen, bins=255)
    bins = bins[:-1].astype(np.uint8)
    # plt.plot(bins, histograma)
    # plt.show()

    kernel = np.ones((11, 11), np.uint8)
    erosion = cv2.erode(img_thr, kernel)
    dilate = cv2.dilate(erosion, kernel)

    # plt.imshow(erosion, cmap="gray")
    # plt.show()
    # plt.imshow(dilate, cmap="gray")
    # plt.show()

    flood_fill2(dilate, 0, 0, 0, 1)

    # plt.imshow(dilate, cmap="gray")
    # plt.show()

    image = np.where(dilate == 1, 0, 255).astype("uint8")
    # plt.imshow(image, cmap="gray")
    # plt.show()
    return image


def get_characterics(image):
    label_img = label(image)
    regions = regionprops(label_img)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    index_max_area = 0
    max_area = 0
    for index, props in enumerate(regions):
        max_area = max(max_area, props.area)
        if max_area == props.area:
            index_max_area = index
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), "-r", linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), "-r", linewidth=2.5)
        ax.plot(x0, y0, ".g", markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, "-b", linewidth=2.5)

    ax.axis((0, 600, 600, 0))
    # plt.show()

    ratio = (regions[index_max_area].bbox[2] - regions[index_max_area].bbox[0]) / (
        regions[index_max_area].bbox[3] - regions[index_max_area].bbox[1]
    )
    img_carac = [regions[index_max_area].area, regions[index_max_area].area_bbox, ratio]
    return img_carac


if __name__ == "__main__":
    data_path = "frames"
    labels = os.listdir(data_path)
    x = []
    y = []
    for dirname in labels:
        filepath = os.path.join(data_path, dirname)
        for file in os.listdir(filepath):
            filename = os.path.join(filepath, file)
            image = preprocessament(filename)
            charac = get_characterics(image)
            x.append(charac)
            y.append(dirname)
