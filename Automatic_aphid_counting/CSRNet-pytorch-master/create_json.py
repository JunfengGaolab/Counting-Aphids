#!usr/bin/python
# -*- coding: utf-8 -*-

import json
from os.path import join
import glob

#产生json文件，train.json和val.json文件分别是训练和测试文件的路径，里面包含所有图片的路径
if __name__ == '__main__':
    # 包含图像的文件夹路径
    #img_folder = './Shanghai/part_A_final/train_data/images'
    #img_folder = './Shanghai/part_A_final/val_data/images'
    img_folder = './Shanghai/part_A_final/test_data/images'

    # 最终json文件的路径
    #output_json = './train.json'
    #output_json = './val.json'
    output_json = './test.json'

    img_list = []

    # glob.glob()函数：返回所有匹配的文件路径列表
    # append()函数：用于在列表末尾添加新的对象。
    for img_path in glob.glob(join(img_folder, '*.jpg')):
        img_list.append(img_path)

    #将图片路径写入json文件
    with open(output_json, 'w') as f:
        json.dump(img_list, f)

