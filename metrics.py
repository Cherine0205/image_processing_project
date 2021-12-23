import numpy as np 
import os
#import skimage.io as io
#import skimage.transform as trans
from keras.models import *
from keras.layers import *
#from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K


def cal_base(y_true, y_pred):
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive
 
    y_positive = K.round(K.clip(y_true, 0, 1))
    y_negative = 1 - y_positive
 
    TP = K.sum(y_positive * y_pred_positive)
    TN = K.sum(y_negative * y_pred_negative)
 
    FP = K.sum(y_negative * y_pred_positive)
    FN = K.sum(y_positive * y_pred_negative)
 
    return TP, TN, FP, FN


def DICE(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score  

def Recall(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP/(TP + FN + K.epsilon())
    return SE

def Specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP/(TP + FP + K.epsilon())
    return PC

def Precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + K.epsilon())
    return SP