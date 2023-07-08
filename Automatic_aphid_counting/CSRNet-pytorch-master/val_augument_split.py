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
from torch.autograd import Variable
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error,mean_absolute_error

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
checkpoint = torch.load('0model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred= []
gt = []

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
    img = img.unsqueeze(0)

    h, w = img.shape[2:4]
    h_d = h // 2
    w_d = w // 2
    img_1 = Variable(img[:, :, :h_d, :w_d].cuda())
    img_2 = Variable(img[:, :, :h_d, w_d:].cuda())
    img_3 = Variable(img[:, :, h_d:, :w_d].cuda())
    img_4 = Variable(img[:, :, h_d:, w_d:].cuda())
    with torch.no_grad():
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()

    pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]


    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])

    pred_sum = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
    pred.append(pred_sum)
    gt.append(np.sum(groundtruth))

    mae += abs(np.sum(groundtruth) - pred_sum)
    mse += (np.sum(groundtruth) - pred_sum) * (
                np.sum(groundtruth) - pred_sum)

    print('i,groundtruth_num,predict_num:',img_paths[i],np.sum(groundtruth),pred_sum)
    print('\n')

print ('mae:',mae/len(img_paths))
print ('mse:',mse/len(img_paths))
print ('rmse:',sqrt(mse/len(img_paths)))

#计算平均绝对误差
mae = mean_absolute_error(pred, gt)
#计算根均方误差
rmse = np.sqrt(mean_squared_error(pred, gt))

print('pred:', pred)
print('gt:', gt)
print('MAE: ', mae)
print('RMSE: ', rmse)
