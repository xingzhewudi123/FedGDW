
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import hashlib
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
import time
import dill
import json
import copy
import math
import random
import struct
import binascii
import torch
import numpy as np
from sklearn.utils import shuffle
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.7
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout,Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential

from tensorflow.keras import layers, models, optimizers, losses, Input
from torch.utils.tensorboard import SummaryWriter
import os


def Lenet(input):
    tf.compat.v1.random.set_random_seed(1234)
    lenet = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=120, activation='relu'),
        tf.keras.layers.Dense(units=84, activation='relu'),
        tf.keras.layers.Dense(units=10)
    ])
    return lenet

def Alexnet(input):
    alex_net= keras.Sequential([
    keras.layers.Conv2D(96, 11, 4),  
    keras.layers.ReLU(), 
    keras.layers.MaxPooling2D((3, 3), 2),  
    keras.layers.BatchNormalization(),
    
        ConvWithPadding(kernel=256, filters=5, strides=1, padding=[[0, 0], [2, 2], [2, 2], [0, 0]]),
    keras.layers.Conv2D(256, 5, 1, padding='same'), 
    keras.layers.ReLU(),
    keras.layers.MaxPooling2D((3, 3), 2), 
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(384, 3, 1, padding='same'),  
    keras.layers.ReLU(),
    
    keras.layers.Conv2D(384, 3, 1, padding='same'),  
    keras.layers.ReLU(),
    
    keras.layers.Conv2D(256, 3, 1, padding='same'),  
    keras.layers.ReLU(),
    keras.layers.MaxPooling2D((3, 3), 2),  

    keras.layers.Flatten(),  
    keras.layers.Dense(4096), 
    keras.layers.ReLU(),
    keras.layers.Dropout(0.25), 

    keras.layers.Dense(4096),  
    keras.layers.Dropout(0.25),  

    #keras.layers.Dense(10)  
    keras.layers.Dense(10) 
    ])

    return alex_net


def identity_block(X, f, filters):
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1,1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1,1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    
    X = tf.keras.layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, s=2):
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(F1, (1, 1), strides=(s,s))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = tf.keras.layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet18(input_shape=(32, 32, 3), classes=100):
    X_input = Input(input_shape)

    X = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f=3, filters=[128,128,512], s=2)
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])
    X = identity_block(X, 3, [128,128,512])

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = GlobalAveragePooling2D()(X)
    #X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax')(X)

    resNet18 = Model(inputs=X_input, outputs=X, name='ResNet18')

    return resNet18


