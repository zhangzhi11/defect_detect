# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:06:04 2017

@author: zhangzhi
"""


import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import model_from_json
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os 
import cv2
from glob import glob
model_dir = 'model'

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



# image size 
img_rows, img_cols = 400, 800

    
#load data

pos_dir = 'D:/zz/defect_detect/output/7953/good'
neg_dir = 'D:/zz/defect_detect/output/7953/bad'

pos_ims = []
neg_ims = []
for im_dir in glob(pos_dir+'/*.jpg'):
    
    im = cv2.imread(im_dir,0)
    im = cv2.resize(im,(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
    pos_ims.append(im)
    
for im_dir in glob(neg_dir+'/*.jpg'):    
    im = cv2.imread(im_dir,0)
    im = cv2.resize(im,(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
    neg_ims.append(im)


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
#        im = np.transpose(im,axes=(2,0,1))
#        ims.append(im)
#        labels.append(label)
        
        
pos_ims = np.array(pos_ims).astype('float32')
neg_ims = np.array(neg_ims).astype('float32')

#ims = ims.astype('float32')
pos_ims /= 255
neg_ims /= 255

print('the size of train images:', pos_ims.shape)

    
if K.image_dim_ordering() == 'th':
    pos_ims = pos_ims.reshape(pos_ims.shape[0], 1, img_rows, img_cols)
    neg_ims = neg_ims.reshape(neg_ims.shape[0], 1, img_rows, img_cols)    
    input_shape = (1, img_rows, img_cols)
else:
    pos_ims = pos_ims.reshape(pos_ims.shape[0], img_rows, img_cols, 1)
    neg_ims = neg_ims.reshape(neg_ims.shape[0], img_rows, img_cols, 1)  
    input_shape = (img_rows, img_cols, 1)

    
    
       
#===================================================================
    
input_img = Input(shape=input_shape)

#x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#encoded = MaxPooling2D((2, 2), border_mode='same')(x)
#
## at this point the representation is (8, 4, 4) i.e. 128-dimensional
#
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Convolution2D(16, 3, 3, activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
#decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)






# Block 1
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(input_img)
x = MaxPooling2D((2, 2), name='block1_pool')(x)

# Block 2
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
x = MaxPooling2D((2, 2), name='block2_pool')(x)

# Block 3
#256
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
x = MaxPooling2D((2, 2), name='block3_pool')(x)

# Block 4
# 512
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
x = MaxPooling2D((2, 2), name='block4_pool')(x)

## Block 5
#x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
#x = MaxPooling2D((2, 2), name='block5_pool')(x)

#===============================================================


#x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 2))(x)

x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)


x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)

x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)

x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)



autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')    
    
print autoencoder.summary()

filepath='auto_encoder-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False)
callbacks_list = [checkpoint]
autoencoder.fit(pos_ims, pos_ims,
                nb_epoch=100,
                batch_size=16 ,
                shuffle=True,
                validation_data=(neg_ims, neg_ims),
                callbacks=callbacks_list)
                
                
thred = 1
losses = autoencoder.predict(neg_ims)



# save model
json_string = autoencoder.to_json()
open('auto_encoder.json','w').write(json_string) 



# a sliding window to detect where the defect is




#model.fit(ims, Ys, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1,validation_split = 0.5)
#model_name = 'defect_0.1'
#
#save_model(model,model_name)          

