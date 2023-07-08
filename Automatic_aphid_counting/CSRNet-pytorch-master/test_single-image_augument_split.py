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
from torch.autograd import Variable
from torchvision import datasets, transforms

#due to qt problem on my linux env, so add these two lines,
envpath = '/home/xumin/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


'''
class torchvision.transforms.ToTensor
把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
class torchvision.transforms.Normalize(mean, std)
给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。即：Normalized_image=(image-mean)/std。
'''
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

model = CSRNet()

#defining the model

model = model.cuda()

#loading the trained weights

#checkpoint = torch.load('0model_best.pth.tar')
checkpoint = torch.load('0checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
#checkpoint = torch.load('models/model_best.pth')#修改
#model.load_state_dict(checkpoint)

#img_path = 'Shanghai/part_A_final/test_data/images/IMG_123_3.jpg'
#h5_path = 'Shanghai/part_A_final/test_data/ground_truth/IMG_123_3.h5'

img_path = '/home/xumin/phd/CSRNet-pytorch-master/Shanghai/part_A_final/other_data/images/2_189.jpg'
h5_path = '/home/xumin/phd/CSRNet-pytorch-master/Shanghai/part_A_final/other_data/ground_truth/2_189.h5'

img = transform(Image.open(img_path).convert('RGB')).cuda()#修改

'''
unsqueeze（arg）是增添第arg个维度为1，以插入的形式填充
相反，squeeze（arg）是删除第arg个维度(如果当前维度不为1，则不会进行删除)
'''
'''
#directly predict
with torch.no_grad():
    output = model(img.unsqueeze(0))
print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
print('output.detach().cpu().shape[2],output.detach().cpu().shape[3]',output.detach().cpu().shape[2],output.detach().cpu().shape[3])
temp[temp<0.001]=0
'''


# split image to 1/4 for predict
with torch.no_grad():
    output_src = model(img.unsqueeze(0)) #not use for predict, just for shape the size of temp
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
output = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
print("Predicted Count : ", int(output))
temp = np.asarray(output_src.detach().cpu().reshape(output_src.detach().cpu().shape[2],output_src.detach().cpu().shape[3]))
print('output.detach().cpu().shape[2],output.detach().cpu().shape[3]',output_src.detach().cpu().shape[2],output_src.detach().cpu().shape[3])
temp[temp<0.001]=0
#print('temp',temp)


# True density map
true_temp = h5py.File(h5_path, 'r')#修改
temp_1 = np.asarray(true_temp['density'])
#print("Original Count : ",round(np.sum(temp_1)) + 1)
print("Original Count : ",round(np.sum(temp_1)))

#density_map = 255*output/np.max(output)
#cv2.imshow('image',density_map)
#cv2.imwrite('result.jpg',density_map)


# visualization seperately
 # show src image
plt.imshow(plt.imread(img_path))#修改
plt.show()

 # true label map show
plt.imshow(temp_1,cmap = c.jet)
#plt.imshow(temp_1, alpha=0.75)
plt.show()

 # prediction show
plt.imshow(temp, cmap = c.jet)
plt.show()



# visualization together
img = plt.imread(img_path)
font_size=20
plt.figure(figsize=(15, 10))
plt.subplot(231), plt.title("Original image",fontsize=font_size), plt.axis('off')
plt.imshow(img, cmap='gray')

plt.subplot(232), plt.title("True_density map",fontsize=font_size), plt.axis('off')
#plt.imshow(temp_1,cmap = c.jet) #the BG is blue
plt.imshow(temp_1,cmap = 'gray') #the BG is gray

plt.subplot(233), plt.title("Pre_density map",fontsize=font_size), plt.axis('off')
#plt.imshow(temp, cmap = c.jet)
plt.imshow(temp, cmap='gray')

plt.tight_layout()
plt.show()
#plt.ion()
#plt.savefig('histogram.jpg')
#plt.pause(3)
#plt.close()


