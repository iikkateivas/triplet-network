# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:33:48 2019

@author: Iikka
"""
from keras.layers import Activation, concatenate
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model

output_dim_1 = 50

def get_output_dim_1():
    return output_dim_1

def get_base_cnn_1(input_shape):
    
    model = Sequential()
    
    model.add(Conv2D(6, (3, 3), strides=(1, 1), padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.25))
    
    model.add(Conv2D(12, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
     
    model.add(Dense(output_dim_1))
#    model.add(Activation('relu'))
    return model

output_dim_2 = 1024

def get_output_dim_2():
    return output_dim_2

def get_base_cnn_2(input_shape):
    
    model = Sequential()
    
    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.2))
    
    model.add(Conv2D(12, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Conv2D(12, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Conv2D(12, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.2))
        
    model.add(Conv2D(24, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Conv2D(24, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Conv2D(24, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.2))
    
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
     
    model.add(Dense(output_dim_2))
#    model.add(Activation('relu'))
    return model

output_dim_3 = 1024

def get_output_dim_3():
    return output_dim_3

def get_base_cnn_3(input_shape):
    
    model = Sequential()
    
    model.add(Conv2D(24, (3, 3), strides=(3, 3), padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.25))
    
    model.add(Conv2D(48, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.25))
        
    model.add(Conv2D(96, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.25))
    
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
     
    model.add(Dense(output_dim_3))
#    model.add(Activation('relu'))
    return model


output_dim_4 = 1024

def get_output_dim_4():
    return output_dim_4

def get_base_cnn_4(input_shape):
    
    model = Sequential()
    
    model.add(Conv2D(24, (3, 3), strides=(1, 1), padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.25))
    
    model.add(Conv2D(48, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.25))
        
    model.add(Conv2D(96, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3),padding='same', data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2) ,data_format="channels_first"))
    model.add(Dropout(.25))
    
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
     
    model.add(Dense(output_dim_4))
#    model.add(Activation('relu'))
    return model

#model = build_siamese_network(x_train)
#model.layers[3].layers[-6].output
