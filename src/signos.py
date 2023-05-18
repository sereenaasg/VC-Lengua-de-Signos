''' PRUEBA 2: DETECCIÓN DE CONTORNOS/USO DEL HOG '''

import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from skimage.feature import hog

def carregarTrain(data_path):
    x_hog = []
    labels = os.listdir(data_path)
    etiquetas = []
    x_image = []

    for dirname in labels:
        filepath = os.path.join(data_path, dirname)
        for file in os.listdir(filepath):
            filename = os.path.join(filepath, file)

            image = cv2.imread(filename)
            image = cv2.resize(image, (300, 300))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(image, 100, 200)
            x_image.append(edges.ravel())
            etiquetas.append(dirname)

            # Find HOG features
            fd, hog_image = hog(
                edges,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
            )
            x_hog.append(fd)
    
    return x_hog, etiquetas

def carregarTest(etiquetes_path):
    labels = os.listdir(etiquetes_path)
    y_test = []
    x_test = []
    x_image = []
    
    for dirname in labels:
        filepath = os.path.join(etiquetes_path, dirname)
        for file in os.listdir(filepath):
            filename = os.path.join(filepath, file)

            image = cv2.imread(filename)
            image = cv2.resize(image, (300, 300))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(image, 100, 200)
            x_image.append(edges.ravel())
            y_test.append(dirname)

            # Find HOG features
            fd, hog_image = hog(
                edges,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
            )
            x_test.append(fd)
    return x_test, y_test
    

def modelKNN(x_train, y_train, x_test, y_test):
    model = KNeighborsClassifier(2)
    model.fit(x_train, y_train)
    predicciones = model.predict(x_test)
    acc = model.score(x_test, y_test)
    f1 = f1_score(y_test, predicciones, average="micro")
    print("KNN accuracy: " + str(acc) + " f1-score: " + str(f1))

def modelSVC(x_train, y_train, x_test, y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    f1 = f1_score(y_test, preds, average="micro")
    acc = clf.score(x_test, y_test)
    print("SVM accuracy:" + str(acc) + " f1-score: " + str(f1))
    
if __name__ == "__main__":
    data_path = "data/mans"
    etiquetes_path = "data/mans_test"
    
    x_train, y_train = carregarTrain(data_path)
    x_test, y_test = carregarTest(etiquetes_path)
    
    modelKNN(x_train, y_train, x_test, y_test)
    modelSVC(x_train, y_train, x_test, y_test)
    
