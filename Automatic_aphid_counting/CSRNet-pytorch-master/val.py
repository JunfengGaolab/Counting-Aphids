#!/usr/bin/env python
# coding: utf-8

# In[42]:


import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from math import sqrt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


from torchvision import datasets, transforms


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])


# In[3]:


root = './Shanghai/'#修改


# In[4]:


#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_val = os.path.join(root,'part_A_final/val_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_test]


# In[5]:


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


# In[6]:


model = CSRNet()


# In[7]:


model = model.cuda()


# In[38]:



#checkpoint = torch.load('./models/model_best.pth')#修改
#model.load_state_dict(checkpoint)
checkpoint = torch.load('0checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

# In[39]:





# In[45]:

###how to calculate the mae,mse,rmse: https://blog.csdn.net/weixin_43894340/article/details/121602065
mae = 0
mse = 0
rmse = 0
for i in range(len(img_paths)):
    '''
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    img[0,:,:]=img[0,:,:]-92.8207477031
    img[1,:,:]=img[1,:,:]-95.2757037428
    img[2,:,:]=img[2,:,:]-104.877445883
    img = img.cuda()
    '''

    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    with torch.no_grad():
        output = model(img.unsqueeze(0))

    mae+=abs(np.sum(groundtruth)-output.detach().cpu().sum().numpy())
    mse+=(np.sum(groundtruth)-output.detach().cpu().sum().numpy())*(np.sum(groundtruth)-output.detach().cpu().sum().numpy())

    print('i,groundtruth_num,predict_num:',img_paths[i],np.sum(groundtruth),output.detach().cpu().sum().numpy())
    print('\n')

print ('mae:',mae/len(img_paths))
print ('mse:',mse/len(img_paths))
print ('rmse:',sqrt(mse/len(img_paths)))

