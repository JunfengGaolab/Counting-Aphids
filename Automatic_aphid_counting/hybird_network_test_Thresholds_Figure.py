

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, soft_nms, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync

from sahi.utils.cv import read_image_as_pil
from sahi.slicing import slice_image
from typing import Dict, List, Optional, Union
from math import sqrt #MAE,MSE
from matplotlib.pyplot import MultipleLocator

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
from model import CSRNet # copy model.py image.py and utils.py from CSRNet-pytorch-master to yolov5,and rename
# utils as utils_crsnet in case to conflict with utils file of the yolov5's pakage.
import torch
#from matplotlib import cm as c
from torch.autograd import Variable
from torchvision import datasets, transforms

def get_slice_bboxes(
    image_height=1000,
    image_width=1000,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
):
    """Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio(float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0
    #print(type(overlap_height_ratio))
    #print(type(slice_height))
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                #print('xmin, ymin, xmax, ymax:',xmin, ymin, xmax, ymax)
                #print('xmax-xmin,ymax-ymin:', xmax-xmin, ymax-ymin)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                #print('x_max-x_min,y_max-y_min:', x_max - x_min, y_max - y_min)
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes



def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

# plot bar graph with horizontal direction
def Plot_v1(image_number, DS):
    fig = plt.figure()
    #x_major_locator = MultipleLocator(1)  # 把x轴的主刻度设置为1000的倍数
    #fig.xaxis.set_major_locator(x_major_locator)  # 把x轴的刻度间隔设置为1，并存在变量里
    #y_major_locator = MultipleLocator(1)  # 把y轴的刻度间隔设置为10，并存在变量里
    #fig.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数

    plt.plot(image_number, DS)
    plt.xlabel("Threshold")
    plt.ylabel("MAE")
    #plt.title("The MAE change curve with different thresholds of aphid number")
    plt.savefig("MAE.png")
    plt.show()

def Plot_v2(image_number, DS):
    fig = plt.figure()
    # x_major_locator = MultipleLocator(1)  # 把x轴的主刻度设置为1000的倍数
    # fig.xaxis.set_major_locator(x_major_locator)  # 把x轴的刻度间隔设置为1，并存在变量里
    #y_major_locator = MultipleLocator(1)  # 把y轴的刻度间隔设置为10，并存在变量里
    #fig.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.plot(image_number, DS)
    plt.xlabel("Threshold")
    plt.ylabel("RMSE")
    #plt.title("The RMSE change curve with different thresholds of aphid number")
    plt.savefig("RMSE.png")
    plt.show()


@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='test',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        sliced_height=256,
        sliced_width=256,
        overlapped_height_ratio=0.2,
        overlapped_width_ratio=0.2,
):



    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('test'), str) and data['test'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks

        rect = False

        task = task if task in ('train', 'val', 'test') else 'test'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = dict(enumerate(model.names if hasattr(model, 'names') else model.module.names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar



    number_list = []
    MAE_list = []
    RMSE_list = []

    for Thresh in range(0, 201, 5):
        print('Thresh:\n', Thresh)
        number_list.append(Thresh)

        target = []  # calculate MAE, MSE
        prediction = []  # calculate MAE, MSE


        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            callbacks.run('on_val_batch_start')
            t1 = time_sync()
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
            #print('height, width:',height, width)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            #out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs

            # split_merge straytage
            img = im
            mulpicplus = "2"  # 1 for normal_val,2 for SAHI_split_val
            assert (int(mulpicplus) >= 1)
            if mulpicplus == "1":
                pred,train_out = model(img, augment=augment, val=True) #have to add train_out,otherwise error

            else:
                '''
                xsz = img.shape[2]
                ysz = img.shape[3]
                mulpicplus = int(mulpicplus)
                x_smalloccur = int(xsz / mulpicplus * 1.2)
                y_smalloccur = int(ysz / mulpicplus * 1.2)
                #print('x_smalloccur,y_smalloccur', x_smalloccur, y_smalloccur)
                for i in range(mulpicplus):
                    x_startpoint = int(i * (xsz / mulpicplus))
                    for j in range(mulpicplus):
                        y_startpoint = int(j * (ysz / mulpicplus))
                        x_real = min(x_startpoint + x_smalloccur, xsz)
                        y_real = min(y_startpoint + y_smalloccur, ysz)
                        if (x_real - x_startpoint) % 64 != 0:
                            x_real = x_real - (x_real - x_startpoint) % 64
                        if (y_real - y_startpoint) % 64 != 0:
                            y_real = y_real - (y_real - y_startpoint) % 64
                        dicsrc = img[:, :, x_startpoint:x_real,
                                 y_startpoint:y_real]
                        pred_temp,train_out = model(dicsrc,
                                          augment=augment,
                                          val=True)
                        print('type(pred_temp):',type(pred_temp))
                        print('pred_temp:', pred_temp)
                        pred_temp[..., 0] = pred_temp[..., 0] + y_startpoint
                        pred_temp[..., 1] = pred_temp[..., 1] + x_startpoint
                        if i == 0 and j == 0:
                            out = pred_temp
                        else:
                            out = torch.cat([out, pred_temp], dim=1)
                '''
                #the input sise must large than the slice_size
                slice_bboxes = get_slice_bboxes(
                    image_height=height,
                    image_width=width,
                    slice_height = sliced_height,
                    slice_width = sliced_width,
                    overlap_height_ratio = overlapped_height_ratio,
                    overlap_width_ratio = overlapped_width_ratio,
                )

                for i in range(len(slice_bboxes)):
                    x_startpoint = slice_bboxes[i][0]
                    y_startpoint = slice_bboxes[i][1]
                    x_endpoint = slice_bboxes[i][2]
                    y_endpoint = slice_bboxes[i][3]

                    dicsrc = img[:, :, x_startpoint:x_endpoint,
                             y_startpoint:y_endpoint]

                    pred_temp, train_out = model(dicsrc,
                                                 augment=augment,
                                                 val=True)
                    #print('type(pred_temp):', type(pred_temp))
                    #print('pred_temp:', pred_temp)
                    pred_temp[..., 0] = pred_temp[..., 0] + y_startpoint
                    pred_temp[..., 1] = pred_temp[..., 1] + x_startpoint
                    if i == 0:
                        out = pred_temp
                    else:
                        out = torch.cat([out, pred_temp], dim=1)

            dt[1] += time_sync() - t2

            # Loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t3 = time_sync()

            #nms
            #out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)

            # soft_nms
            out = soft_nms(out, conf_thres, iou_thres, multi_label=True) #Soft DIoU-NMS

            dt[2] += time_sync() - t3



            # Metrics
            for si, pred in enumerate(out):
                print('path:', paths[si])
                print('len(det):',len(pred))


                if len(pred)<Thresh: # accpet the detection result of yolov5

                #print('paths[si]:',paths[si])

                    labels = targets[targets[:, 0] == si, 1:]
                    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions

                    # calculate MAE, MSE
                    print('nl, npr:', nl, npr, len(pred))
                    target.append(nl)
                    prediction.append(npr)

                    path, shape = Path(paths[si]), shapes[si][0]
                    correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                    seen += 1

                    if npr == 0:
                        if nl:
                            stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                            if plots:
                                confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                        continue

                    # Predictions
                    if single_cls:
                        pred[:, 5] = 0
                    predn = pred.clone()
                    scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                    # Evaluate
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                        scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = process_batch(predn, labelsn, iouv)
                        if plots:
                            confusion_matrix.process_batch(predn, labelsn)
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

                    # Save/log
                    if save_txt:
                        save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
                    if save_json:
                        save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
                    callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

                else: #CRSnet

                    transform = transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]),
                    ])

                    model_CSR = CSRNet()

                    # defining the model
                    model_CSR = model_CSR.cuda()

                    # loading the trained weights
                    # checkpoint = torch.load('0model_best.pth.tar')
                    #checkpoint = torch.load('/home/xumin/phd/CSRNet-pytorch-master/0checkpoint.pth.tar')
                    checkpoint = torch.load('./weights/CSRNet_Aphid_dataset_2/0model_best.pth.tar')

                    model_CSR.load_state_dict(checkpoint['state_dict'])
                    # checkpoint = torch.load('models/model_best.pth')#修改
                    # model.load_state_dict(checkpoint)

                    # img_path = 'Shanghai/part_A_final/test_data/images/IMG_123_3.jpg'
                    # h5_path = 'Shanghai/part_A_final/test_data/ground_truth/IMG_123_3.h5'

                    # img_path = '/home/xumin/phd/CSRNet-pytorch-master/Shanghai/part_A_final/test_data/images/IMG_344_3.jpg'
                    img_path = paths[si]
                    print('img_path:', img_path)
                    #h5_path = '/home/xumin/phd/CSRNet-pytorch-master/Shanghai/part_A_final/test_data/ground_truth/IMG_25_1.h5'
                    h5_path = paths[si].replace('.jpg', '.h5').replace('JPEGImages', 'ground_truth')
                    print('h5_path:',h5_path)

                    img = transform(Image.open(img_path).convert('RGB')).cuda()  # 修改


                    # split image to 1/4 for predict
                    with torch.no_grad():
                        output_src = model_CSR(img.unsqueeze(0))  # not use for predict, just for shape the size of temp
                    img = img.unsqueeze(0)
                    h, w = img.shape[2:4]
                    h_d = h // 2
                    w_d = w // 2
                    img_1 = Variable(img[:, :, :h_d, :w_d].cuda())
                    img_2 = Variable(img[:, :, :h_d, w_d:].cuda())
                    img_3 = Variable(img[:, :, h_d:, :w_d].cuda())
                    img_4 = Variable(img[:, :, h_d:, w_d:].cuda())
                    with torch.no_grad():
                        density_1 = model_CSR(img_1).data.cpu().numpy()
                        density_2 = model_CSR(img_2).data.cpu().numpy()
                        density_3 = model_CSR(img_3).data.cpu().numpy()
                        density_4 = model_CSR(img_4).data.cpu().numpy()
                    output = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
                    print("Predicted Count : ", round(output))
                    temp = np.asarray(output_src.detach().cpu().reshape(output_src.detach().cpu().shape[2],
                                                                        output_src.detach().cpu().shape[3]))
                    # print('output.detach().cpu().shape[2],output.detach().cpu().shape[3]', output_src.detach().cpu().shape[2], output_src.detach().cpu().shape[3])
                    temp[temp < 0.001] = 0
                    # print('temp',temp)

                    # True density map
                    true_temp = h5py.File(h5_path, 'r')  # 修改
                    temp_1 = np.asarray(true_temp['density'])
                    print("Original Count : ", round(np.sum(temp_1)))

                    target.append(round(np.sum(temp_1)))
                    prediction.append(round(output))



        # Compute metrics

        # calculate MAE, MSE
        # refer to: https://www.jb51.net/article/181171.htm
        error = []
        for i in range(len(target)):
            error.append(target[i] - prediction[i])
        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值
        print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
        print("MAE = ", sum(absError) / len(absError))  # 平均绝对误差MAE

        MAE_list.append(sum(absError) / len(absError))
        RMSE_list.append(sqrt(sum(squaredError) / len(squaredError)))

    print('number_list', number_list)
    print('MAE_list',MAE_list)
    print('RMSE_list',RMSE_list)
    Plot_v1(number_list, MAE_list)
    Plot_v2(number_list, RMSE_list)




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--sliced_height', type=int, default=256, help='sliced_height')
    parser.add_argument('--sliced_width', type=int, default=256, help='sliced_width')
    parser.add_argument('--overlapped_height_ratio', type=float, default=0.2, help='overlapped_height_ratio')
    parser.add_argument('--overlapped_width_ratio', type=float, default=0.2, help='overlapped_height_ratio')



    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(emojis(f'WARNING: confidence threshold {opt.conf_thres} > 0.001 produces invalid results ⚠️'))
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
