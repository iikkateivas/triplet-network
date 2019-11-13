# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:35:01 2019

@author: Iikka

"""
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import numpy as np
import os
from keras.layers import Activation, concatenate
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model

from base_cnn import get_base_vgg16
output_dim = 512

def triplet_loss(y_true, y_pred, alpha = 0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ',y_pred)
    
    anchor = y_pred[:,:output_dim]
    positive = y_pred[:,output_dim:output_dim+output_dim]
    negative = y_pred[:,output_dim+output_dim:]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss


def build_triplet_network(base_model, input_dim, opt):  
    """
    Build triplet network
    Arguments:
    base_model -- keras model for the CNN
    input_dim -- train data input dimensions (1, y, x)
    opt -- fit optimizer
    Returns:
    model -- keras triplet network model
    """
    # Create the 3 inputs
    anchor_in = Input(shape=input_dim)
    pos_in = Input(shape=input_dim)
    neg_in = Input(shape=input_dim)
    
    # Share base network with the 3 inputs
    anchor_out = base_model(anchor_in)
    pos_out = base_model(pos_in)
    neg_out = base_model(neg_in)
    merged_vector = concatenate([anchor_out, pos_out, neg_out], axis=-1)
    
    # Define the trainable model
    model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)
    model.compile(optimizer=opt,loss=triplet_loss)
    return model

'''Load data in size (samples, 3, 1, y, x)'''

#X = np.load('triplet_data_75.npy') # load here
#X = X[:150000]
## suffle the data with train_test_split
#x_train, x_test = train_test_split(X, test_size=.0001)
#del X
#
#x_train = np.true_divide(x_train, 255, dtype=np.float32)

# get input dimensions
in_dims = (200, 200, 3)
# build base model of choosing
base_model = get_base_vgg16()
'''Check for Dense layer input info'''
print(base_model.layers[-6].output)
# build triplet model for training
model = build_triplet_network(base_model, in_dims, Adam())

# create y_dummie for the keras fit function
#y_dummie = np.zeros(len(x_train))
#
##model.layers[3].load_weights('weights/face_triplet_face_1.h5')
## Train the model
#epochs = 100
#model.fit([x_train[:, 0], x_train[:, 1] , x_train[:, 2]], y_dummie, batch_size=512, verbose=1, epochs=epochs)
## create weights folder if not existing
#if not os.path.exists('weights'):
#    os.makedirs('weights')
## save weights
#model.layers[3].save_weights('weights/triplet_75_2.h5')

def generate_batches(files, batch_size):
    while 1:
        for d in datasets:
#            print('Loading data batch...')
            x_part = np.load(d)   
            
            x_train, x_test = train_test_split(x_part, test_size=.0001)
            del x_part
#            print('Normalizing...')
            x_train = np.divide(x_train[:15000], 255, dtype=np.float32)
#            print('Done')
            batches = int(len(x_train) / batch_size)
            
            for i in range(batches):
                x = x_train[i*batch_size:(i+1)*batch_size]
                y_dummie = np.zeros(len(x))
                yield ([x[:, 0], x[:, 1] , x[:, 2]], y_dummie)
       

total_samples = 90000
batch_size = 25
steps = int(total_samples/batch_size)
datasets = ['./train_batch_1.npy', './train_batch_2.npy', './train_batch_3.npy', './train_batch_4.npy', './train_batch_5.npy', './train_batch_6.npy']
model.fit_generator(generate_batches(datasets, batch_size),
        steps_per_epoch=steps, epochs=30)



'''Validating output sanity'''

#intermed_model = Model(inputs=model.layers[3].inputs, outputs=model.layers[3].layers[-1].get_output_at(-1))
#
#x1 = np.expand_dims(x_train[0,0], axis=0)
#out = intermed_model.predict(x1)
#
#unique_vals ,val_counts = np.unique(out, return_counts=True)




