import numpy as np 
import os
#import skimage.io as io
#import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
from metrics import *
from unet import *
import nibabel as nib
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------------------------------------------#

#导入模型
model=load_model('image_segmentation_model_3.h5',custom_objects={'dice_loss':dice_loss,'DICE':DICE,'Specificity':Specificity,'Precision':Precision,'Recall':Recall})
#--------------------------------------------------------------------------------------------------------------------#
#导入图片
image_input='C:/Users/24710/Desktop/image_processing_project/train_img'
label_input='C:/Users/24710/Desktop/image_processing_project/train_label'
list_f=[os.path.join(image_input,f) for f in os.listdir(image_input)]
slices=[nib.load(f) for f in list_f] #slices中为1-6号的图像（320,320,x）
list_l=[os.path.join(label_input,f) for f in os.listdir(label_input)]
labels=[nib.load(l) for l in list_l]#labels中为1-6号的label（320,320,x）
input_arr=[slices[i].get_fdata() for i in range(18)]
input_label=[labels[i].get_fdata() for i in range(18)]


#--------------------------------------------------------------------------------------------------------------------#
#根据需要做预测
img=input_arr[5][:,:,50]
ground_truth=input_label[5][106:250:,106:250:,50]
ground_truth=ground_truth.reshape((1,)+ground_truth.shape)
img=img*1.0/2171
img=img[106:250:,106:250:]
img=img.reshape((1,)+img.shape)
result=model.predict(img)

#--------------------------------------------------------------------------------------------------------------------#
#画出结果
'''plt.figure(figsize=(12,12))
plt.figure(1)
sub1=plt.subplot(221)
plt.imshow(np.round(result[0,:,:,0]),cmap='gray')
sub2=plt.subplot(222)
plt.imshow(input_arr[5][106:250:,106:250:,50],cmap='gray')
sub3=plt.subplot(223)
plt.imshow(ground_truth[0,:,:],cmap='gray')'''

figure=plt.figure(figsize=(12,12))
sub1=figure.add_subplot(221)
sub1.imshow(np.round(result[0,:,:,0]),cmap='gray')
sub2=figure.add_subplot(222)
sub2.imshow(input_arr[5][106:250:,106:250:,50],cmap='gray')
sub3=figure.add_subplot(223)
sub3.imshow(ground_truth[0,:,:],cmap='gray')
plt.show()