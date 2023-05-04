import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from skimage.feature import hog

x_image = []
x_hog = []
etiquetas = []
data_path = 'mans'
labels = os.listdir(data_path)

for dirname in labels:
  filepath = os.path.join(data_path, dirname)
  for file in os.listdir(filepath):
    filename = os.path.join(filepath, file)

    image = cv2.imread(filename)
    image = cv2.resize(image, (300,300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image,100,200)
    x_image.append(edges.ravel())
    etiquetas.append(dirname)

    # Find HOG features
    fd, hog_image = hog(edges, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True)
    x_hog.append(fd)

# KNN I SVM AMB LES IMATGES DE CONTORNS
x_train, x_test, y_train, y_test = train_test_split(x_hog, etiquetas, test_size=0.2, random_state=42)
model = KNeighborsClassifier(2)
model.fit(x_train, y_train)
predicciones = model.predict(x_test)
acc = model.score(x_test, y_test)
f1 = f1_score(y_test, predicciones, average='micro')
print('Knn accuracy: '+str(acc)+ ' f1: ' + str(f1))

clf = svm.SVC()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
f1 = f1_score(y_test, preds, average='micro')
acc = clf.score(x_test, y_test)
print('SVM acc:' +str(acc)+' f1-score: ' + str(f1))

