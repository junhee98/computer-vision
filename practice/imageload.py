#image print
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img

def imageLoad():
    mypath='./raw-img/cane/'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ] #file name
    images = np.empty(len(onlyfiles), dtype=object) #file pixel info
    
    #image 정보 출력
    for n in range(0, len(onlyfiles)):
      images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
      #print(images[n].size)
    
    
    #image 그림 출력
    for i in range(1):
        path = mypath + onlyfiles[0]
        image = img.imread(path)
        plt.imshow(image)
        plt.show()
        
    return onlyfiles
        
        
        
#imageLoad()
