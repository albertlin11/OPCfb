from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
# import matplotlib.pyplot as plt
import cv2
import sys


def testGenerator(test_path,num_image,target_size = (256, 256)):
    filenames = glob.glob(os.path.join(test_path, "*.png"))
    filenames = sorted(filenames)

    for i in range(num_image):
    
        # verbose
        progress = (i+1) / num_image * 100       
        sys.stdout.write("\rProgress: [{:<50}] {:.2f}%".format("=" * int(progress / 2), progress))
        sys.stdout.flush()

        img = cv2.imread(filenames[i], cv2.IMREAD_GRAYSCALE)
        img = img / 255
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        img = np.reshape(img,img.shape+(1,))

        img[img > 0.2] = 1.0
        img[img <= 0.2] = 0.0

        yield img
    print("\n")
    print("Done loading {} pictures".format(num_image))


def saveResult(path, npyfile, loss_fun = "mse"):
    for i,item in enumerate(npyfile): 
        cv2img = item * 255
        cv2img = np.where(cv2img > 255, 255, cv2img)
        cv2img = np.where(cv2img < 0, 0, cv2img)
        filename = os.path.join(path,"%04d_predict_%s.png"%(i, loss_fun))
        cv2.imwrite(filename, cv2img)

def saveDiff(path, npyfile, loss_fun = "mse"):
    for i,item in enumerate(npyfile):
        filename = os.path.join(path,"%04d_difference_%s.png"%(i, loss_fun))
        cv2.imwrite(filename, cv2img)


def SaveFigureBy0to1(path,npyfile, name="difference", loss_fun = "mse"):
    figure_name = "%04d_" + name + "_" + loss_fun + ".png"
    for i,item in enumerate(npyfile):
        cv2img = item * 255
        cv2.imwrite(os.path.join(path,figure_name%i), cv2img)
        
def SaveFigureBy0to255(path,npyfile, loss_fun = "mse"):
    figure_name = "%04d_" + loss_fun + ".png"
    for i,item in enumerate(npyfile):
        cv2.imwrite(os.path.join(path,figure_name%i), item)