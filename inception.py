# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:04:39 2019

@author: Iikka
"""
import numpy as np
from keras.utils import np_utils, plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate as Concat, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD

'''Prepare a test dataset'''

# Download
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize
X_train = np.divide(X_train, 255, dtype=np.float32)
X_test = np.divide(X_test, 255, dtype=np.float32)

# To categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

'''Create an inception module'''

def inception_3b(input_t, fc1, fc2_in, fc2_out, fc3_in, fc3_out, fc4_out):
    
    layer_1x1 = Conv2D(fc1, (1,1), padding='same', activation='relu')(input_t)
    
    layer_3x3 = Conv2D(fc2_in, (1,1), padding='same', activation='relu')(input_t)
    layer_3x3 = Conv2D(fc2_out, (3,3), padding='same', activation='relu')(layer_3x3)
    
    layer_5x5 = Conv2D(fc3_in, (1,1), padding='same', activation='relu')(input_t)
    layer_5x5 = Conv2D(fc3_out, (5,5), padding='same', activation='relu')(layer_5x5)
    
    layer_pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_t)
    layer_pool = Conv2D(fc4_out, (1,1), padding='same', activation='relu')(layer_pool)
    
    output = Concat([layer_1x1, layer_3x3, layer_5x5, layer_pool], axis = -1)
    return output

'''Create the network'''

# Input shape
input_dims = X_train.shape[1:]

# Input tensor
input_t = Input(shape = (input_dims[0], input_dims[1], input_dims[2]))
# add inception module
layer = inception_3b(input_t, 64, 64, 64, 64, 64, 64)
# add another
layer = inception_3b(layer, 64, 64, 64, 64, 64, 64)

output = Flatten()(layer)
out = Dense(10, activation='softmax')(output)

# create model
model = Model(inputs=input_t, outputs=out)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file='inception_module.png')

'''Test'''

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#61%