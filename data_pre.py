import numpy as np 
import os
#import skimage.io as io
#import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
from metrics import *

def data_prepare(input_path):
