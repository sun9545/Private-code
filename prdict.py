import cv2
import numpy as np
from tensorflow import keras
from Unet import unet_predict
from core import locate, erode,dilate

import tensorflow as tf
print(tf.__version__)

UNet = tf.keras.models.load_model('unet.h5')

img_src, img_mask = unet_predict(UNet,'/home/sjq/81.jpg')  #测试图片
# cv2.imshow('mask',img_mask)
img2gray = erode(img_mask)
# cv2.imshow('img2gray',img2gray)
ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow('mask',mask)
cv2.imshow('img_mask',cv2.bitwise_and(img_src,img_src,mask=cv2.bitwise_not(mask)))
cv2.waitKey(0)
gray2img = cv2.cvtColor(img2gray, cv2.COLOR_GRAY2BGR)
img_src_copy = locate(img_src, gray2img)


# cv2.imshow('show', img_mask)

cv2.imshow('plate',img_src_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()