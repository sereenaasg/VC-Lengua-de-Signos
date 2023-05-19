"""
PRUEBA 1: DETECCIÓN DE REGIONES Y CALCULO DE CARACTERÍSTICAS EN BASE A LAS MEDIDAS
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops, regionprops_table
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import math
from sklearn.model_selection import train_test_split


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

def flood_fill2(imagen):
    # Obtener las dimensiones de la imagen
    filas, columnas = imagen.shape
    
    # Encontrar el límite izquierdo y derecho de cada fila
    limite_izquierdo = np.zeros(filas, dtype=int)
    limite_derecho = np.zeros(filas, dtype=int)
    
    # Encontrar el límite izquierdo y derecho en cada fila
    for i in range(filas):
        for j in range(columnas):
            if imagen[i, j] == 0:
                break
            else:
                imagen[i,j] = 0
        
        for j in range(columnas-1, -1, -1):
            if imagen[i, j] == 0:
                break
            else:
                imagen[i, j] = 0
    
    return imagen
                
def crop_black_region(imagen):
    # Obtener las dimensiones de la imagen
    filas, columnas = imagen.shape
    
    # Encontrar los límites superior, inferior, izquierdo y derecho
    limite_superior = None
    limite_inferior = None
    limite_izquierdo = None
    limite_derecho = None
    
    # Encontrar el límite superior
    for i in range(filas):
        if limite_superior is None and np.any(imagen[i] == 0):
            limite_superior = i
            break
    
    # Encontrar el límite inferior
    for i in range(filas-1, -1, -1):
        if limite_inferior is None and np.any(imagen[i] == 0):
            limite_inferior = i
            break
    
    # Encontrar el límite izquierdo
    for j in range(columnas):
        if limite_izquierdo is None and np.any(imagen[:, j] == 0):
            limite_izquierdo = j
            break
    
    # Encontrar el límite derecho
    for j in range(columnas-1, -1, -1):
        if limite_derecho is None and np.any(imagen[:, j] == 0):
            limite_derecho = j
            break
    
    # Recortar la región negra de la imagen original
    imagen_recortada = imagen[limite_superior:limite_inferior+1, limite_izquierdo:limite_derecho+1]
    
    return imagen_recortada

def preprocessament(path):

    imagen = llegir_imatge(path)

    blur = cv2.GaussianBlur(imagen,(5,5),0)
    ret1,th1 = cv2.threshold(blur,65,255,cv2.THRESH_BINARY)

    th1 = np.where(th1 == 255, 0, 255).astype("uint8")
    
    
    kernel = np.ones((11, 11), np.uint8)
    erosion = cv2.erode(th1, kernel)
    dilate = cv2.dilate(erosion, kernel)
    image = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    image = image/255.0
    
    image = np.where(image == 1, 0, 255).astype("uint8")
    image = crop_black_region(image)
    

    image = flood_fill2(image)
    
    image = np.where(image == 255, 0, 255).astype("uint8")
    kernel = np.ones((15, 15), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    image = np.where(image == 255, 0, 255).astype("uint8")
    plt.imshow(image, cmap='gray')
    plt.show()
    
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
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
        #         x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        #         y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        #         x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        #         y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length


        ax.plot((x0, x1), (y0, y1), "-r", linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), "-r", linewidth=2.5)
        ax.plot(x0, y0, ".g", markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, "-b", linewidth=2.5)
    h, w = image.shape
    ax.axis((0, w, h, 0))
    plt.show()
    
    ratio = (regions[index_max_area].bbox[2] - regions[index_max_area].bbox[0]) / (
        regions[index_max_area].bbox[3] - regions[index_max_area].bbox[1]
    )
    minr, minc, maxr, maxc = regions[index_max_area].bbox
    area_bbox = (maxr - minr) * (maxc - minc)
    
    #     ratio = (regions[index_max_area].bbox[2] - regions[index_max_area].bbox[0]) / (
    #         regions[index_max_area].bbox[3] - regions[index_max_area].bbox[1]
    #     )
    #     img_carac = [regions[index_max_area].area, regions[index_max_area].area_bbox, ratio]

    img_carac = [regions[index_max_area].area, area_bbox, ratio]
    return img_carac


def modelKNN(x, y):
    x_train, x_test, y_train, y_test = train_test_split( 
        x, y, test_size=0.5, random_state=42, stratify= y
    )
    
    model = KNeighborsClassifier(5)
    model.fit(x_train, y_train)
    predicciones = model.predict(x_test)
    print(predicciones)
    print(y_test)
    acc = model.score(x_test, y_test)
    f1 = f1_score(y_test, predicciones, average="micro")
    print(acc, f1) 

   
if __name__ == "__main__":
    data_path = "../data/frames"
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
            
    modelKNN(x,y)
    
''' FUNCIONES ANTIGUAS    
# def flood_fill(field, x, y, old, new):
#     # we need the x and y of the start position, the old value,
#     # and the new value
#     # the flood fill has 4 parts
#     # firstly, make sure the x and y are inbounds
#     if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
#         return
#     # secondly, check if the current position equals the old value
#     if field[y][x] != old:
#         return

#     # thirdly, set the current position to the new value
#     field[y][x] = new
#     # fourthly, attempt to fill the neighboring positions
#     flood_fill(field, x + 1, y, old, new)
#     flood_fill(field, x - 1, y, old, new)
#     flood_fill(field, x, y + 1, old, new)
#     flood_fill(field, x, y - 1, old, new)

# def preprocessament(path):
#     # Imagen de Serena
#     imagen = llegir_imatge(path)
#     # plt.imshow(imagen, cmap="gray")
#     # plt.show()

#     thr = 65
#     img_thr = np.uint8(imagen < thr)

#     # plt.imshow(img_thr, cmap="gray")
#     # plt.show()

#     histograma, bins = np.histogram(imagen, bins=255)
#     bins = bins[:-1].astype(np.uint8)
#     # plt.plot(bins, histograma)
#     # plt.show()

#     kernel = np.ones((11, 11), np.uint8)
#     erosion = cv2.erode(img_thr, kernel)
#     dilate = cv2.dilate(erosion, kernel)

#     # plt.imshow(erosion, cmap="gray")
#     # plt.show()
#     # plt.imshow(dilate, cmap="gray")
#     # plt.show()

#     flood_fill2(dilate, 0, 0, 0, 1)

#     # plt.imshow(dilate, cmap="gray")
#     # plt.show()

#     image = np.where(dilate == 1, 0, 255).astype("uint8")
#     # plt.imshow(image, cmap="gray")
#     # plt.show()
#     return image '''
