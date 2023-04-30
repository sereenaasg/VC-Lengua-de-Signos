import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

x_image = []
y_label = ['no', 'si', 'subnormal']

for file in os.listdir('mans'):
    filename = os.path.join('mans', file)
    print(filename)
    img = cv2.imread(filename, 0)

    edges = cv2.Canny(img,100,200)

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(edges,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(edges, kp)
    x_image.append(des)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

x_image_features = np.vstack(x_image)
print(x_image_features.shape)

scaler = MinMaxScaler(feature_range=(0,1))
x_image_scaled = scaler.fit_transform(x_image_features)

# Using KMeans to compute centroids to build bag of visual words,n_clusters = 3, 
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_image_scaled)
y_kmeans = kmeans.predict(x_image_scaled)

plt.scatter(x_image_scaled[:, 0], x_image_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
# Como que no queda muy bien no?

