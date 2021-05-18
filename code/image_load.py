# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:01:11 2021

@author: taeso
"""

#사용하는 모듈
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt


#파일 열기
path = './raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg'

image_pil = Image.open(path)
image = np.array(image_pil)



image.shape


#이미지 range 확인
np.min(image), np.max(image)


#이미지 시각화
plt.hist(image.ravel(),256,[0,256])
plt.show()


#이미지 보기
plt.imshow(image)
plt.colorbar()
plt.show()
