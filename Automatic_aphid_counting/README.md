
# Automatic aphid counting base on improved Yolov5


<img src="https://github.com/JunfengGaolab/Aphid-recognition-and-counting-/blob/main/Automatic%20aphid%20counting%20base%20on%20improved%20Yolov5/demo_data/2(328).jpg" width="240" height="240"><img src="https://github.com/JunfengGaolab/Aphid-recognition-and-counting-/blob/main/Automatic%20aphid%20counting%20base%20on%20improved%20Yolov5/demo_data/2(185).jpg" width="240" height="240"><img src="https://github.com/JunfengGaolab/Aphid-recognition-and-counting-/blob/main/Automatic%20aphid%20counting%20base%20on%20improved%20Yolov5/demo_data/2(189).jpg" width="240" height="240"><img src="https://github.com/JunfengGaolab/Aphid-recognition-and-counting-/blob/main/Automatic%20aphid%20counting%20base%20on%20improved%20Yolov5/demo_data/2(190).jpg" width="240" height="240"/>



## Overview
Aphids can cause direct damage and indirect virus transmission to crops. Timely monitoring the number of aphids and cutting off the transmission source can prevent the large-scale outbreak of aphids. However, the manual counting of aphids which is commonly used at present is inefficient. Therefore, we proposed an automatic aphid counting method based on yellow water pan trap imagery and deep learning. Our proposed method is the improved Yolov5. The improvements mainly include four aspects: a) insert deformable convolution in backbone, b) add small object detection layer, c) split-merge detection strategy, d) soft-NMS post-processing. They mainly focus on the two major difficulties in the aphid counting task, aphids are very small in shape and aphids are sometimes densely distributed. Through the experiments, we found that the mAP of our proposed method on the aphid test set is 64.6%, which is 4.7% higher than that of original Yolov5. So, we conclude that our proposed method is superior to the original Yolov5 in the aphid counting task. Especially for extremely small aphids and densely distributed aphids, the detection and counting effect of our proposed method is obviously better than the original Yolov5.



## Prerequisites

This work is based on https://github.com/ultralytics/yolov5. 



## Installation and Run

**1. Create virtual environment**   
Clone my repository into your workspace and recompile it:  
`conda create --name=yolov5 python=3.8 #choose your machine python version`  
`source activate yolov5`      

**2. Install yolov5 and test**   
`Download this workpackge from github`  

`Enter into dir 'A novel hybrid network based on deep learning for the automatic peach-potato aphid counting in sugar beet fields'`

`pip install -r requirements.txt`

`./data/scripts/download_weights.sh #(download pretrained weights)`  

`python inference.py #(image test)`

`python detect.py --source 0`

**3. Training and test model**

 **3.1 Detection network**

**Prepare dataset**    
Label the aphid and generate the VOC format dataset (#./dataset/voc2011/).

`cd ./dataset/voc2011/`

#`python txt.py #(split the dataset and get the ./voc2011/ImageSets/Main/xxx.txt)` Please don't execute this command if you want to use the pre-split dataset from our

`python voc_label_2011.py #(convert the dataset to yolo format)`

**Modify the configuration from code**    
**(1)** Go to ./data/, creat aphid_voc.yaml according to VOC.yaml. Modify the some configurations:

train: /home/xumin/yolov5/dataset/2011_train.txt

val: /home/xumin/yolov5/dataset/2011_val.txt

test: /home/xumin/yolov5/dataset/2011_test.txt

\# Classes\
nc: 1  # number of classes\
names: ['aphid']  # class names

**(2)** Go to the ./models/, create yolov5s-2-DCN2.yaml according to yolov5s.yaml. Modify configuration and network:

nc: 1  # number of classes

anchors: 3

backbone:

  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2\
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4\
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8\
   [-1, 6, C3, [256]],
   [-1, 1, DCNConv, [512, 3, 2]],  # 5-P4/16 #insert Deformable Convolution\
   [-1, 9, C3, [512]],
   [-1, 1, DCNConv, [1024, 3, 2]],  # 7-P5/32 #insert Deformable Convolution\
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9\
  ]

head:

  [[-1, 1, Conv, [512, 1, 1]],  #20x20\
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], #40x40\
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4  40x40\
   [-1, 3, C3, [512, False]],  # 13     40x40

   [-1, 1, Conv, [256, 1, 1]], #40x40\
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3   80x80\
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)  80x80

   [-1, 1, Conv, [256, 1, 1]], #18  80x80\
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], #19  160x160,  add small object layer\
   [[-1, 2], 1, Concat, [1]], #20 cat backbone p2  160x160, add small object layer\
   [-1, 3, C3, [256, False]], #21 160x160, add small object layer 

   [-1, 1, Conv, [256, 3, 2]],  #22   80x80\
   [[-1, 18], 1, Concat, [1]], #23 80x80\
   [-1, 3, C3, [256, False]], #24 80x80

   [-1, 1, Conv, [256, 3, 2]], #25  40x40\
   [[-1, 14], 1, Concat, [1]],  # 26  cat head P4  40x40\
   [-1, 3, C3, [512, False]],  # 27 (P4/16-medium) 40x40

   [-1, 1, Conv, [512, 3, 2]],  #28  20x20\
   [[-1, 10], 1, Concat, [1]],  #29 cat head P5  #20x20\
   [-1, 3, C3, [1024, False]],  # 30 (P5/32-large)  20x20

   [[21, 24, 27, 30], 1, Detect, [nc, anchors]],  # Detect(p2, P3, P4, P5)\
  ]

**(3)** Modify utils/dataloaders.py

```
#sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  
sa, sb = f'{os.sep}JPEGImages{os.sep}', f'{os.sep}labels{os.sep}'
```

**Train**    
`python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-2-DCN2.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 100`

We can use tensorboard or Wandb to visualize the training curves. And all of training result will be saved on ./runs/.

**Test (model)**    
`python detect.py --source ./images/1_53_0.jpg --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --conf-thres 0.5 #(image)`

`python detect.py --source ./images  --imgsz 640 --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --conf-thres 0.5 #(image dir)`

`python detect.py --source test.mp4  --imgsz 640 --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --output test_result/3_video #(video)`

**Val (val_dataset)**   
`python val.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1`

`python split_val.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1 # split image to several equal parts`

`python sahi_split_val.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1 # resize 640x640, then split image into 256x256 with overlapped ratio 0.2 for val`

`python sahi_split_val_keep_src.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1 --sliced_height 256 --sliced_width 256 --overlapped_height_ratio 0.2 --overlapped_width_ratio 0.2 # keep the original size of images, then split image into 256x256 with overlapped ratio 0.2 for val`

**Test (test_dataset)**   
`python test.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-low/weights/best.pt --device '0' --batch-size 1` # **This is exactly test results on original yolov5**

`python split_test.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1 # split image to seveal equal parts`

`python sahi_split_test.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1 # resize 640x640, then split image into 256x256 with overlapped ratio 0.2 for test`. # **This is exactly test results on our improved yolov5** 

`python sahi_split_test_keep_src.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1 --sliced_height 256 --sliced_width 256 --overlapped_height_ratio 0.2 --overlapped_width_ratio 0.2 # keep the original size of images, then split image into 256x256 with overlapped ratio 0.2 for test`

**Inference**  

`python detect.py --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --source ./images --device 0 --save-txt`

`python split_detect.py --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --source ./images --device 0 --save-txt`

`python sahi_split_detect.py --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --source ./images --device 0 --save-txt --conf-thres 0.5 --iou-thres 0.6`

`python sahi_split_detect_keep_src.py --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --source ./images --device 0 --save-txt --sliced_height 256 --sliced_width 256 --overlapped_height_ratio 0.2 --overlapped_width_ratio 0.2 --conf-thres 0.5 --iou-thres 0.6`

**Notice**
Please note that you need to keep the values of conf-thres and iou-thres consistent whether in the code of from commands, when you conduct comparison experiments. Normally, we set --conf-thres 0.5 --iou-thres 0.6

**Extra instructions**

* We refer to the SAHI slicing idea (https://github.com/obss/sahi), and then modify the code of yolov5 to slice the input image into 256x256 small bolcks for test. Please check **def get_slice_bboxes()**  function in the our relevant python file, and add this part to yolov5 if you use the original yolov5 framework rather than ours. 


* All of  relevent code have NMS and Soft_nms, you can switch it through commented relevant part of codes. If you use Soft_nms, please refer to our code and add **def soft_nms()** function on ./utils/general.py

* It has two aphid datasets (aphid_dataset_1,aphid_dataset_2), '/dataset/voc2011' is aphid_dataset_1, '/dataset/voc2013' is aphid_dataset_2.

* we also provide a short tutorial how to run it on Colab https://colab.research.google.com/drive/1vL7kDD87bkYDHRseMB06YQNc5gR676oB#scrollTo=NsCtkhgY8lti , it will be easy to reproduce.



**3.2 Density map estimation network (CSRNet)**

**Train**

cd CSRNet-pytorch-master/

`python -m visdom.server` # Open visdom to watch the training curves

`python train.py train.json val.json 0 0` # Training

**Notice**

Please note that we have two datasets (aphid_dataset_1,aphid_dataset_2), before training, you need to configure some training files，for example:

if you train aphid_dataset_1,

(1) mofify 'CSRNet-pytorch-master/Shanghai/part_A_final_aphid_dataset_1' to 'CSRNet-pytorch-master/Shanghai/part_A_final'

(2) copy three json file from 'CSRNet-pytorch-master/aphid_dataset_1_json' to 'CSRNet-pytorch-master/'


if you train aphid_dataset_2,

(1) mofify 'CSRNet-pytorch-master/Shanghai/part_A_final_aphid_dataset_2' to 'CSRNet-pytorch-master/Shanghai/part_A_final'

(2) copy three json file from 'CSRNet-pytorch-master/aphid_dataset_2_json' to 'CSRNet-pytorch-master/'


**Test**

Please refer to '*4. Hybrid_network test*', you can use 'hybird_network_image.py' and 'hybird_network_test.py' to do test by setting the condition for the detection network to 'False' in the codes, in this case, only density map estimation network works. Also, please you need to modify the loaded model name in the codes which you want use as we have two density map estimation network model (for aphid_dataset_1, aphid_dataset_2 respectively).


**4. Hybrid_network test**
* The hybrid_network integrates the detection network and the density map estimation network. When the distribution density of aphids is low, it utilizes the improved Yolov5 to count aphids. Conversely, when the distribution density of aphids is high, it switches to CSRNet to count aphids.

`python hybird_network_image.py --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --source ./dataset/voc2013/JPEGImages/IMG_40_1.jpg --device 0` # For single image test

`python hybird_network_image.py --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --source ./images --device 0` # For image_file test

`python hybird_network_test.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1` # Test on the test set

`python hybird_network_test_Thresholds_Figure.py --data data/aphid_voc.yaml --weights ./runs/train/yolov5s-2-DCN2-low/weights/best.pt --device '0' --batch-size 1` # vary T from 0 to 200 with an interval of 5 and carry out counting test to get the optimal T
   

## Contact
If you have any issue, please contact us.


