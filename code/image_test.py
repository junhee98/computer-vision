#image print
from os import listdir
from os.path import isfile, join
import numpy
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
mypath='./Animal_dataset/cane_test/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ] #file name
images = numpy.empty(len(onlyfiles), dtype=object) #file pixel info
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )


for n in range(0, len(onlyfiles)):
    path = mypath + onlyfiles[n]
    image = img.imread(path)
    plt.imshow(image)
    plt.show()
