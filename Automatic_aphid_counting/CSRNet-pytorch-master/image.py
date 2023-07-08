#!usr/bin/python
# -*- coding: utf-8 -*-

import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    #print('img.resize:', img.size)
    #print('target.resize:', target.shape)

    #resize all of images and labels before train and test
    '''
    img = img.resize((640, 640), Image.ANTIALIAS)  # resize image with high-quality
    target = cv2.resize(target, (640, 640), interpolation=cv2.INTER_CUBIC)
    print('img.resize:', img.size)
    print('target.resize:', target.shape)
    '''


    # only resize the images which they are too large to lead to a CUDA OUT OF MEMORY
    # if use Augumentation, it will crop the image, so may not use this part if no 'CUDA OUT OF MEMORY' appears 
    '''
    if 1700<=np.array(img).shape[0]:
        print('img.size', img.size)
        print('target.shape', target.shape)
        img = img.resize((1700, 1700), Image.ANTIALIAS) #resize image with high-quality
        target = cv2.resize(target, (1700, 1700), interpolation=cv2.INTER_CUBIC)
        print('img.resize:', img.size)
        print('target.resize:', target.shape)
    '''



    # CRSNET-AUGUMENTATION原理:https://blog.csdn.net/weixin_45753850/article/details/109698549
    #if False: #no augumentation,
    #if True:  #augumenation for train and test
    if train:  #augumenation only for train
        #print('Augumentation')
        crop_size = (int(img.size[0]/2),int(img.size[1]/2))
        if random.randint(0,9)<= -1: #no any work
        #if random.randint(0, 9) <= 4.5:  # no any work
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)

        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        

        
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    
    
    target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
    #print('--img.resize:', img.size)
    #print('--target.resize:', target.shape)

    
    
    return img,target
