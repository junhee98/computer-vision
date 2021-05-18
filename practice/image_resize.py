# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:54:45 2021

@author: taeso
"""

import torchvision.transforms.functional as TF
from PIL import Image
import imageload as il


# define a helper function to resize images
def resize_img_label(image, label=(0., 0.), target_size=(28,28)):
    w_orig, h_orig = image.size
    w_target, h_target = target_size
    cx, cy = label
    image_new = TF.resize(image, target_size)
    label_new = cx/w_orig*w_target, cy/h_orig*h_target
    return image_new, label_new

mypath='./raw-img/cane/'
filenames = il.imageLoad()

for n in range(0, len(filenames)):
    image = Image.open(mypath+filenames[n])
    resize_img, resize_label = resize_img_label(image)
    resize_img.save(mypath+filenames[n])
    

