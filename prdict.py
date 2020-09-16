import cv2
import numpy as np
from tensorflow import keras
from Unet import unet_predict
from core import locate

import tensorflow as tf
print(tf.__version__)

UNet = tf.keras.models.load_model('unet.h5')
img_src, img_mask = unet_predict(UNet,'/home/sjq/43.jpg')  #测试图片
img_src_copy = locate(img_src, img_mask)

cv2.imshow('plate',img_src_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()