import numpy as np
import cv2
import os

#参考网址：https://blog.csdn.net/weixin_38517705/article/details/82703252
#https://github.com/guoya1003/data_process/blob/master/clip_merge_picture.py

"""
输入：图片路径(path+filename)，裁剪获得小图片的列数、行数（也即宽、高）
"""
seg_save_path_original='image/crop/'
def clip_one_picture(path,filename,cols,rows):
    img=cv2.imread(path+filename,-1)##读取彩色图像，图像的透明度(alpha通道)被忽略，默认参数;灰度图像;读取原始图像，包括alpha通道;可以用1，0，-1来表示
    sum_rows=img.shape[0]   #高度
    sum_cols=img.shape[1]    #宽度
    #save_path=path+"\\crop{0}_{1}\\".format(cols,rows)  #保存的路径  #保存的路径
    save_path = seg_save_path_original.format(cols, rows)  # 保存的路径  #保存的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("裁剪所得{0}列图片，{1}行图片.".format(int(sum_cols/cols),int(sum_rows/rows)))

    for i in range(int(sum_cols/cols)):
        for j in range(int(sum_rows/rows)):
            #cv2.imwrite(save_path+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1],img[j*rows:(j+1)*rows,i*cols:(i+1)*cols,:])
            cv2.imwrite(save_path + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + '.png',img[j * rows:(j + 1) * rows, i * cols:(i + 1) * cols, :])

            #print(path+"\crop\\"+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1])
    print("裁剪完成，得到{0}张图片.".format(int(sum_cols/cols)*int(sum_rows/rows)))
    print("裁剪所得图片的存放地址为：{0}".format(save_path))


"""调用裁剪函数示例"""
path='image/'   #要裁剪的图片所在的文件夹
filename='370RGB1.tif'    #要裁剪的图片名
cols=2000        #小图片的宽度（列数）
rows=1125        #小图片的高度（行数）
#clip_one_picture(path,filename,cols,rows)




"""
输入：图片路径(path+filename)，裁剪所的图片的列的数量、行的数量
输出：无
"""

def merge_picture(merge_path,merge_save_path,num_of_cols,num_of_rows):
    #filename=file_name(merge_path,".tif")
    filename = file_name(merge_path, ".png") #注意需要拼接的子图的格式(上面clip_one_picture后的子图格式为)
    #shape=cv2.imread(filename[0],-1).shape    #三通道的影像需把-1改成1

    '''
    #彩色图
    shape = cv2.imread(filename[0], 1).shape
    print(shape)
    cols=shape[1]
    rows=shape[0]
    channels=shape[2]
    dst=np.zeros((rows*num_of_rows,cols*num_of_cols,channels),np.uint8)
    
    for i in range(len(filename)):
        img=cv2.imread(filename[i],-1)
        cols_th=int(filename[i].split("_")[-1].split('.')[0])
        rows_th=int(filename[i].split("_")[-2])
        roi=img[0:rows,0:cols,:]
        dst[rows_th*rows:(rows_th+1)*rows,cols_th*cols:(cols_th+1)*cols,:]=roi

    cv2.imwrite(merge_save_path + "merge.tif", dst)
    '''


    # 灰度图
    shape = cv2.imread(filename[0], 0).shape
    #print(shape)
    cols=shape[1]
    rows=shape[0]
    #channels=shape[2]
    dst=np.zeros((rows*num_of_rows,cols*num_of_cols),np.uint8)
    for i in range(len(filename)):
        img=cv2.imread(filename[i],-1)
        cols_th=int(filename[i].split("_")[-1].split('.')[0])
        rows_th=int(filename[i].split("_")[-2])
        roi=img[0:rows,0:cols]
        dst[rows_th*rows:(rows_th+1)*rows,cols_th*cols:(cols_th+1)*cols]=roi
    #cv2.imwrite(merge_path+"merge.tif",dst)
    cv2.imwrite(merge_save_path + "merge.tif", dst)




"""遍历文件夹下某格式图片"""
def file_name(root_path,picturetype):
    filename=[]
    for root,dirs,files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1]==picturetype:
                filename.append(os.path.join(root,file))
    return filename


"""调用合并图片的代码"""
merge_path="image/crop/"   #要合并的小图片所在的文件夹
merge_save_path="image/" #合并后的图片的保存路径
num_of_cols=8   #列数 (根据打印数据填写)
num_of_rows=14    #行数
#merge_picture(merge_path,merge_save_path,num_of_cols,num_of_rows)




