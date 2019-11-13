import numpy as np
import cv2
import glob


from keras import backend as K
from keras.layers import Activation, concatenate
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import skimage

from keras.preprocessing.image import ImageDataGenerator
import copy

import matplotlib.pyplot as plt

def generate_images(img, img_number):
    # add the original
    img_collect = []
    img_collect.append(img.astype('uint8'))
    
    # reshape the image for the generator
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        data_format ="channels_first")
    
    datagen.fit(img)
    
    i = 1
    for X_batch in datagen.flow(img):
        img_collect.append(X_batch[0].astype('uint8'))
        if i % img_number == 0:
            break
        i=i+1
    return img_collect

'''Face gen'''
    
samples = 10
gen_count = 10
gray = False
size = (200, 200)

face_list = []
for f in glob.glob('./faces\*'):
    face_list.append(f)

gen_data = []

for face_path in face_list:
    # list all images
    image_list = []
    for f in glob.glob(face_path + '\*.jpg'):
        image_list.append(f)
    
    generated_faces = []
    for i in range(samples):
        
        if (gray):
            image = cv2.imread(image_list[i], 0)
        else:
            image = cv2.imread(image_list[i])
#        temp = 255*skimage.exposure.equalize_adapthist(image)
#        image = temp.astype('uint8')
        #reduce the size
        if image is None:
            continue
        image = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
        gen = generate_images(image, gen_count)
        generated_faces.extend(gen)

    if len(generated_faces) == samples*(gen_count+1):
        gen_data.append(generated_faces)
    
plt.imshow(gen_data[10][55])
#np.save('data_200.npy', gen_data)

gen_data = np.load('data_200.npy')
x_train = gen_data[:300]

def generate_triplet_data(x_train, sample_range, total_sample_size):
    
    dim1 = x_train.shape[2]
    dim2 = x_train.shape[3]
    dim3 = x_train.shape[4]
    # triplet data
    x_triplets = np.zeros([total_sample_size, 3, dim1, dim2, dim3], dtype=np.uint8)
    
    classes = x_train.shape[0]
    samples = x_train.shape[1]
    
    count = 0
    
    for i in range(sample_range[0], sample_range[1]):
        for j in range(int(total_sample_size/classes)):
            ind_a = 0
            ind_p = 0
            
            # read images from same directory (positive pair)
            while ind_a == ind_p:
                ind_a = np.random.randint(samples)
                ind_p = np.random.randint(samples)

                
            while True:
                class_n = np.random.randint(classes)
                ind_n = np.random.randint(samples)
                if i != class_n:
                    break
                           
            img_a = x_train[i, ind_a]
            img_p = x_train[i, ind_p]
            img_n = x_train[class_n, ind_n]
            
            x_triplets[count, 0, :, :, :] = img_a
            x_triplets[count, 1, :, :, :] = img_p
            x_triplets[count, 2, :, :, :] = img_n  
            count += 1
            
    return x_triplets

triplet_data = generate_triplet_data(x_train, (250, 300), 20000)
np.save('train_batch_6.npy', triplet_data)

anc = 1
plt.imshow(triplet_data[anc,0])
plt.figure()
plt.imshow(triplet_data[anc,1])
plt.figure()
plt.imshow(triplet_data[anc,2])

'''Test data'''

gen_data = np.load('data_200.npy')
x_test = gen_data[:300]

def generate_pair_data(x_test):
    
    dim1 = x_test.shape[2]
    dim2 = x_test.shape[3]

    classes = x_test.shape[0]
    samples = x_test.shape[1]
    
    sets = []
    
    test_sets = 10
    for i in range(test_sets):
      
        pair_1 = x_test[:, i]
        pair_1 = np.expand_dims(pair_1, axis=1)
        
        while True:
            ind_pair = np.random.randint(samples)
            if i != ind_pair:
                break
        
        pair_2 = x_test[:, ind_pair]
        pair_2 = np.expand_dims(pair_2, axis=1)
        
        pairs = np.concatenate([pair_1, pair_2], axis=1)
        sets.append(pairs)
            
    return sets

test_sets = generate_pair_data(x_test)
#np.save('val_set_75.npy', test_sets)










