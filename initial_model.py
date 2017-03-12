# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:38:25 2017

@author: zhangzhi
"""

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential,model_from_json, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Input
from keras.utils import np_utils
from keras import backend as K
import os 
import cv2
from glob import glob
model_dir = 'model'
from sklearn.cross_validation import train_test_split
def save_model(model, name = 'zz'):
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)
    out_model_dir = os.path.join(model_dir,name)
    if not os.path.exists(out_model_dir): 
        os.mkdir(out_model_dir)
    json_string = model.to_json()  #等价于 json_string = model.get_config()  
    open(os.path.join(out_model_dir,name+'.json'),'w').write(json_string)    
    model.save_weights(os.path.join(out_model_dir, name+'.h5')) 

def load_model( model_name = 'zz'):
    out_model_dir = os.path.join(model_dir,model_name)
    model = model_from_json(open(os.path.join(out_model_dir,model_name+'.json')).read())
    model.load_weights(os.path.join(out_model_dir,model_name+'.h5'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])      
    return model

batch_size = 32
nb_classes = 2
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 227, 227
# number of convolutional filters to use


# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()


    
#load data

#data_dir = 'D:/zz/defect_detect/data/7953/'
#sub_data_list = glob(data_dir+'*')
#
#ims = []
#labels = []
#for sub_data_dir in sub_data_list:
#    if sub_data_dir.split('\\')[-1]=='bad':
#        label = 1
#    elif sub_data_dir.split('\\')[-1]=='good':
#        label = 0
#    else:
#        label = 0
#    for im_dir in glob(sub_data_dir+'/*.jpg'):
#        print im_dir
#        im = cv2.imread(im_dir)
#        im = cv2.resize(im,(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
##        im = np.transpose(im,axes=(2,0,1))
#        ims.append(im)
#        labels.append(label)
#        
#        
#ims = np.array(ims)
#labels = np.array(labels)
#
#ims = ims.astype('float32')
#ims /= 255
#
#print('the size of images:', ims.shape)
##Ys = np_utils.to_categorical(labels, nb_classes)
#Ys = labels
#
#X_train, X_test, y_train, y_test = train_test_split(ims, Ys, 
#                                                    test_size=0.4, random_state=0)



#    ims = ims.reshape(ims.shape[0], 3, img_rows, img_cols)
#input_shape = (img_rows, img_cols, 3)

#ims = np.transpose(ims,axes=(0,1,3,2))
#def build_model():
#    model = Sequential()
#    
#    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                            border_mode='valid',
#                            input_shape=input_shape))
#    model.add(Activation('relu'))
#    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=pool_size))
#    model.add(Dropout(0.25))
#    
#    model.add(Flatten())
#    model.add(Dense(128))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(nb_classes))
#    model.add(Activation('softmax'))
#    
#    model.compile(loss='categorical_crossentropy',
#                  optimizer='adadelta',
#                  metrics=['accuracy'])
#    return model
#    
    
def alex_model():
    model = Sequential()
    
    model.add(Convolution2D(96, 11, 11, subsample=(4,4), input_shape=input_shape, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())   
    
    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Convolution2D(384, 3, 3, border_mode='same'))
#    model.add(BatchNormalization())
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    model.add(Convolution2D(384, 3, 3, border_mode='same'))
#    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
#    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
#    model.add(BatchNormalization())
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))    
    
    model.add(Dense(nb_classes))
#    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])   
    return model
    
    
def VGG16(include_top=True,
          input_tensor=None):
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''

    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 227, 227)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (227, 227 , 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1, activation='sigmoid', name='predictions')(x)

    # Create model
    model = Model(img_input, x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])   

    return model
    
#model=alex_model()
model = VGG16()
model.summary()
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1,validation_data = (X_test,y_test))
#model_name = 'defect_0.1'
#
#save_model(model,model_name)          

