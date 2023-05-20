''' PRUEBA 2: DETECCIÓN DE CONTORNOS/USO DEL HOG '''

import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from skimage.feature import hog
import pickle
import matplotlib.pyplot as plt
from skimage import io, exposure

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
            
            image = cv2.GaussianBlur(image,(5,5),0)
            image = exposure.rescale_intensity(image)

            edges = cv2.Canny(image, 100, 200)
            x_image.append(edges.ravel())
            etiquetas.append(dirname)
            # plt.imshow(edges, cmap='gray')
            # plt.axis('off')
            # plt.show()

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
            
            image = cv2.GaussianBlur(image,(5,5),0)
            image = exposure.rescale_intensity(image)

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
    # save the model to disk
    filename = 'hog_knn.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    predicciones = model.predict(x_test)
    acc = model.score(x_test, y_test)
    f1 = f1_score(y_test, predicciones, average="micro")
    print("KNN accuracy: " + str(acc) + " f1-score: " + str(f1))

def modelSVC(x_train, y_train, x_test, y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    # save the model to disk
    filename = 'hog_svm.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
    preds = clf.predict(x_test)
    f1 = f1_score(y_test, preds, average="micro")
    acc = clf.score(x_test, y_test)
    print("SVM accuracy:" + str(acc) + " f1-score: " + str(f1))
    
if __name__ == "__main__":
    
    binaries = 0
    
    if binaries == 1:
        data_path = "../data/augmented_bin_train"
        etiquetes_path = "../data/augmented_bin_test"
        
        x_train, y_train = carregarTrain(data_path)
        x_test, y_test = carregarTest(etiquetes_path) 
    else:
        # una persona pel test i la resta pel train
        data_path = "../data/augmented_data_train"
        etiquetes_path = "../data/augmented_data_test"
        
        # train i test aleatori
        data_path = "../data/augmented_data_train2"
        etiquetes_path = "../data/augmented_data_test2"
        
        x_train, y_train = carregarTrain(data_path)
        x_test, y_test = carregarTest(etiquetes_path)
    
    modelKNN(x_train, y_train, x_test, y_test)
    modelSVC(x_train, y_train, x_test, y_test)
    
    
    
    # labels = os.listdir(etiquetes_path)

    
    # for dirname in labels:
    #     filepath = os.path.join(etiquetes_path, dirname)
    #     for file in os.listdir(filepath):
    #         filename = os.path.join(filepath, file)

    #         image = cv2.imread(filename)
    #         image = cv2.resize(image, (300, 300))
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         image = cv2.GaussianBlur(image,(5,5),0)
            
            
    #         imagen1 = cv2.equalizeHist(image)
    #         plt.imshow(imagen1, cmap='gray')
    #         plt.axis('off')
    #         plt.show()
            
            
    #         imagen2 = exposure.rescale_intensity(image)
    #         plt.imshow(imagen2, cmap='gray')
    #         plt.axis('off')
    #         plt.show()
