
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
import numpy as np
from keras.layers import Activation, concatenate
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model

import matplotlib.pyplot as plt

from base_cnn import get_base_cnn_2, get_output_dim_2
output_dim = get_output_dim_2()

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def build_siamese_network(base_model, input_dim):
    """
    Build siamese network
    Arguments:
    base_model -- keras model for the CNN
    input_dim -- train data input dimensions (1, y, x)
    Returns:
    model -- keras siamese network model
    """
    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)
    
    base_network = base_model
    feat_vecs_a = base_network(img_a)
    feat_vecs_b = base_network(img_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
    model = Model(inputs=[img_a, img_b], outputs=distance)
    return model

'''Simulate'''

test_sets = np.load('val_set_75.npy')
x_test = test_sets[1]
x_test = np.expand_dims(x_test, axis=2)
x_test = np.true_divide(x_test, 255, dtype=np.float32)

in_dims = x_test.shape[2:]
# build base model of choosing
base_model = get_base_cnn_2(in_dims)

model = build_siamese_network(base_model, in_dims)
model.layers[2].load_weights('weights/triplet_75_1.h5')

final = []
#    top = []
for anchor in range(len(x_test)):
    print(anchor)

    a = np.expand_dims(x_test[anchor,0], axis=0)
    
    res = []
    
    for i in range(len(x_test)):
        
        b = np.expand_dims(x_test[i,1], axis=0)
        
        
        # match left ears
        pred = model.predict([a, b])
        
        res.append(pred)
        
    if res.index(min(res)) == anchor:
        final.append(1)
    else:
        final.append(0)
    
#        vv = res[anchor]
#        res.sort()
#        top.append(res.index(vv))
    
int(sum(final)/len(x_test) * 100)

anc = 69
plt.imshow(x_test[anc,0,0])
plt.figure() 
plt.imshow(x_test[anc,1,0])
