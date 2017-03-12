# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:57:06 2017

@author: zhangzhi
"""


import cv2
import numpy as np
from keras.models import model_from_json
import os
from glob import glob
from keras import backend as K

# load model 
model = model_from_json(open(os.path.join('auto_encoder.json')).read())
model.load_weights(os.path.join('auto_encoder-11-0.0376.hdf5'))


pos_dir = 'D:\\zz\\defect_detect\\output\\7953\\bad_seg\\good'
neg_dir = 'D:\\zz\\defect_detect\\output\\7953\\bad_seg\\bad'

# image size 
img_rows, img_cols = 400, 800

pos_ims = []
neg_ims = []
pos_ims_name = []
neg_ims_name = []
for im_dir in glob(pos_dir+'/*.jpg'):
    
    im = cv2.imread(im_dir,0)
    im = cv2.resize(im,(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
    pos_ims.append(im)
    pos_ims_name.append(im_dir.split('\\')[-1])
    
for im_dir in glob(neg_dir+'/*.jpg'):    
    im = cv2.imread(im_dir,0)
    im = cv2.resize(im,(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
    neg_ims.append(im)
    neg_ims_name.append(im_dir.split('\\')[-1])

pos_ims = np.array(pos_ims).astype('float32')
neg_ims = np.array(neg_ims).astype('float32')

#ims = ims.astype('float32')
pos_ims /= 255
neg_ims /= 255

    
if K.image_dim_ordering() == 'th':
    pos_ims = pos_ims.reshape(pos_ims.shape[0], 1, img_rows, img_cols)
    neg_ims = neg_ims.reshape(neg_ims.shape[0], 1, img_rows, img_cols)    
    input_shape = (1, img_rows, img_cols)
else:
    pos_ims = pos_ims.reshape(pos_ims.shape[0], img_rows, img_cols, 1)
    neg_ims = neg_ims.reshape(neg_ims.shape[0], img_rows, img_cols, 1)  
    input_shape = (img_rows, img_cols, 1)
    



# predict
pos_predicted = model.predict(pos_ims)
neg_predicted = model.predict(neg_ims)


out_pos_dir = 'D:\\zz\\defect_detect\\output\\7953\\bad_seg\\predicted_good1\\'
out_neg_dir = 'D:\\zz\\defect_detect\\output\\7953\\bad_seg\\predicted_bad1\\'


out_abs_pos_dir = 'D:\\zz\\defect_detect\\output\\7953\\bad_seg\\predicted_abs_good1\\'
out_abs_neg_dir = 'D:\\zz\\defect_detect\\output\\7953\\bad_seg\\predicted_abs_bad1\\'

out_mse_pos_dir = 'D:\\zz\\defect_detect\\output\\7953\\bad_seg\\predicted_mse_good1\\'
out_mse_neg_dir = 'D:\\zz\\defect_detect\\output\\7953\\bad_seg\\predicted_mse_bad1\\'


# caculate loss
pos_abs_loss = []
pos_mse_loss = []
pos_zhijiejian_loss = []
for i in range(pos_ims.shape[0]):
    pos_im = pos_ims[i,:,:,0]
    predicted = pos_predicted[i,:,:,0]

    
    pos_abs_loss.append(abs(pos_im-predicted))
    pos_mse_loss.append((pos_im-predicted)**2)
    pos_zhijiejian_loss.append((pos_im-predicted))
    

    
    name = pos_ims_name[i]   
    abs_predicted = pos_abs_loss[i] * 255
    abs_predicted = np.array(abs_predicted).astype('int')
    cv2.imwrite(out_abs_pos_dir+name,abs_predicted)
    
    
    name = pos_ims_name[i]  
    mse_predicted = pos_mse_loss[i] * 255
    mse_predicted = np.array(mse_predicted).astype('int')
    cv2.imwrite(out_mse_pos_dir+name,mse_predicted)
    
    predicted = predicted * 255
    predicted = np.array(predicted).astype('int') 
    name = pos_ims_name[i]   
    cv2.imwrite(out_pos_dir+name,predicted)
    
neg_abs_loss = []
neg_mse_loss = []
neg_zhijiejian_loss = []
for i in range(neg_ims.shape[0]):
    neg_im = neg_ims[i,:,:,0]
    predicted = neg_predicted[i,:,:,0]

    
    neg_abs_loss.append(abs(neg_im-predicted))
    neg_mse_loss.append((neg_im-predicted)**2)
    neg_zhijiejian_loss.append(((neg_im-predicted)))
    
    

    
    name = neg_ims_name[i]   
    abs_predicted = neg_abs_loss[i] * 255
    abs_predicted = np.array(abs_predicted).astype('int')
    cv2.imwrite(out_abs_neg_dir+name,abs_predicted)
    
    
    name = neg_ims_name[i]  
    mse_predicted = neg_mse_loss[i] * 255
    mse_predicted = np.array(mse_predicted).astype('int')
    cv2.imwrite(out_mse_neg_dir+name,mse_predicted)
 

    predicted = predicted * 255
    predicted = np.array(predicted).astype('int') 
    name = neg_ims_name[i]  
    cv2.imwrite( out_neg_dir+name,predicted)   
#
##
##kernel_size = (3, 3)
##sigma = 1
##blurred = cv2.GaussianBlur(neg_im, kernel_size, sigma)
#cv2.imshow('im', blurred)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()
#
#np.sum(b - predicted[0,:,:,0])
