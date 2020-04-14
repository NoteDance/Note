import cv2 as cv
import numpy as np
import glob


def read(image_filename,p='*.jpg',images=True):
    if images==True:
        images_filenames=glob.glob(image_filename+p)
    else:
        images_filenames=image_filename
    if type(images_filenames)==list:
        length=len(images_filenames)
    elif type(images_filenames)==str:
        image_array=cv.imread(images_filenames)
        shape=image_array.shape
        images_arrays=np.zeros(shape=(1,shape[0],shape[1],shape[2]))
        images_arrays=images_arrays+image_array
        return images_arrays
    for i in range(length):
        if i==0:
            image_array=cv.imread(images_filenames[i])
            shape=image_array.shape
            images_arrays=np.zeros(shape=(len(images_filenames),shape[0],shape[1],shape[2]))
            images_arrays[i]=images_arrays[i]+image_array
            if length==1:
                break
        image_array=cv.imread(images_filenames[i])
        images_arrays[i]=images_arrays[i]+image_array
    return images_arrays