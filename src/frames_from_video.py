""" ALGUNAS FUNCIONES UTILES Y CARGA DEL DATASET """

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image


def carregarImatges(path):
    imatges = []
    etiquetes = []

    for etiqueta in os.listdir(path):  ## per cada subcarpeta de la carpeta frames
        path_total = (
            path + etiqueta + "/"
        )  # construim el path de les imatges d'aquesta etiqueta
        for imatge in os.listdir(path_total):  # per cada imatge de la carpeta
            path_imatge = path_total + imatge
            imatge = Image.open(path_imatge)
            imatges.append(np.array(imatge))
            etiquetes.append(etiqueta)

    return np.array(imatges), etiquetes


def carregarVideo(name):
    capture = cv2.VideoCapture("../data/videos/" + name + ".mp4")
    cont = 0
    path = "../data/frames/" + name + "/"

    while capture.isOpened():
        ret, frame = capture.read()
        if ret == True:
            # Guardar un de cada 5 frames
            if cont % 5 == 0:
                cv2.imwrite(path + "IMG_%04d.jpg" % cont, frame)
            cont += 1
            if cv2.waitKey(1) == ord("s"):
                break
        else:
            break

    capture.release()


def partirDataset(path):
    imagenes, etiquetas = carregarImatges(path)
    x_train, x_test, y_train, y_test = train_test_split(
        imagenes, etiquetas, test_size=0.2, random_state=42
    )

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


if __name__ == "__main__":
    names = ["tijeras", "positivo", "negativo"]
    path = "../data/frames/"
    os.mkdir(path)
    for i in names:
        os.mkdir(path+i)
        carregarVideo(i)

