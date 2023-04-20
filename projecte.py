''' PRàCTICA 3 '''

import os
import skimage 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage import data, io, filters, feature, segmentation, measure
from skimage.io import imread
import cv2
from moviepy.editor import ImageSequenceClip
from scipy.signal import convolve2d, correlate2d, fftconvolve
from scipy import signal, ndimage
from PIL import Image, ImageOps
from scipy.ndimage import shift
import time
from mpl_toolkits import mplot3d
import math

def carregarVideo(name):
    capture = cv2.VideoCapture(name)
    cont = 0
    path = 'frames/'
    
    while (capture.isOpened()):
        ret, frame = capture.read()
        if (ret == True):
            cv2.imwrite(path + 'IMG_'+name+'_%04d.jpg' % cont, frame)    
            cont += 1
            if (cv2.waitKey(1) == ord('s')):
                break
        else:
            break

    capture.release()

def partirDataset(datset):
    pass
    

def classificador(train, test):
    pass
    
def guardarImatge(name, img):
    skimage.io.imsave(name+'.jpg', img)
    



if __name__ == "__main__":
    names = ['video1.mp4', 'video2.mp4', 'video3.mp4']
    for i in names:
         carregarVideo(i)
    
    
    
