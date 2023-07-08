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
from matplotlib import cm as c
from seg_pinjie import clip_one_picture,merge_picture
import sys
from torchvision import datasets, transforms
from data_loader2 import ImageDataLoader



#root = '/home/comin/CSRNet-pytorch-master/Shanghai/'#修改

#获得某个文件夹下的所有图片并保存到list中
def get_imglist(rootdir):
    # 扫描完一次文件夹，把文件放在列表里
    #print(rootdir)
    imglist = list()
    #T = list([])# 图像的创建时间T
    # root:E:/mcfly/images 输入的根目录
    # dirs:[] 表示root下面无文件夹
    # imgs:['DJI_1.png', 'DJI_2.png', 'DJI_3.png']
    for root, dirs, imgs in os.walk(rootdir):
        for img in imgs:
            if 1: # os.access(img,os.R_OK):
                if img.endswith('png') or img.endswith('jpg'):
                    rootimg = root + '/' + img
                    imglist.append(rootimg)
                    #T.append(os.path.getctime(rootimg))

    #print("imglist:", imglist)
    #print("T0:", T)
    return imglist

#清空文件夹下所有文件函数
def remove_imglist(imagelist):
    for index, name in enumerate(imagelist):
        os.remove(name)


# 重新运行之前删除路径中所有的图像和txt文件
imglist = get_imglist('image/crop/')
remove_imglist(imglist)
imglist = get_imglist('image/output/')
remove_imglist(imglist)


#保存密度图
def save_density_map(density_map, output_dir, fname='results.png'):
    density_map = 255 * density_map / np.max(density_map)

    density_map = density_map[0][0]
    cv2.imwrite(os.path.join(output_dir, fname), density_map)


def main():

    #分割图片为子块
    path_original = 'image/'  # 要裁剪的图片所在的文件夹
    filename = '4rgb_crop.tif'  # 要裁剪的图片名
    cols = 2000  # 小图片的宽度（列数）**需注意太大会大致cuda溢出
    rows = 1125  # 小图片的高度（行数）**需注意太大会大致cuda溢出
    clip_one_picture(path_original, filename, cols, rows)

    path_sets='./image/crop'#裁剪后的子图块的目录
    output_dir = './image/output/' #预测后的密度图输出目录

    img_paths = get_imglist(path_sets)

    model = CSRNet()
    model = model.cuda()
    checkpoint = torch.load('/home/comin/CSRNet-pytorch-master/models/163model_best.pth')#修改
    model.load_state_dict(checkpoint)

    transform=transforms.Compose([
                           transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                       ])

    sum_count=0
    for i in range(len(img_paths)):
        img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
        print(img_paths[i],end=" ")
        output = model(img.unsqueeze(0))
        count = int(output.detach().cpu().sum().numpy())
        if int(output.detach().cpu().sum().numpy())<0:
            count=0
            
        print("Predicted Count : ", count)
        sum_count = sum_count + count

        output = output.data.cpu().numpy()
        #print(img_paths[i].split('/')[3])
        save_density_map(output, output_dir, (img_paths[i].split('/')[3]).strip('.png') + '.png')

    print("sum_count:",sum_count)

    #拼接
    merge_path = output_dir  # 要合并的小图片所在的文件夹
    merge_save_path = "image/"  # 合并后的图片的保存路径
    num_of_cols = 6  # 列数 (根据原图与clip_one_picture的切割大小进行运算而得，最好是整除，（如果不是整除，则拼接时非整除部分会被切掉）)
    num_of_rows = 8  # 行数 (根据原图与clip_one_picture的切割大小进行运算而得，最好是整除，（如果不是整除，则拼接时非整除部分会被切掉）)
    merge_picture(merge_path,merge_save_path,num_of_cols,num_of_rows)

if __name__ == "__main__":
    sys.exit(main())




   

















