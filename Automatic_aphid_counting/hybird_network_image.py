import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, soft_nms, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

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
from model import CSRNet # copy model.py image.py and utils.py from CSRNet-pytorch-master to ./yolov5/,and rename
# utils as utils_crsnet in case to conflict with utils file of the yolov5's pakage.
import torch
#from matplotlib import cm as c
from torch.autograd import Variable
from torchvision import datasets, transforms

#due to qt problem on my linux env, so add these two lines,
envpath = '/home/xumin/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath



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








@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        sliced_height=256,
        sliced_width=256,
        overlapped_height_ratio=0.2,
        overlapped_width_ratio=0.2,
):

    source = str(source)
    img_pah = source
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    #print('imgsz',imgsz)

    #image = cv2.imread(img_pah)
    #imgsz = image.shape[0]
    #print('imgsz:',imgsz)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        nb, _, height, width = im.shape  # batch size, channels, height, width
        #print('height, width',height, width)

        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

        # split_merge straytage
        img = im
        mulpicplus = "2"  # 1 for normal_val,2 for SAHI_split_val
        assert (int(mulpicplus) >= 1)
        if mulpicplus == "1":
            pred, train_out = model(img, augment=augment, val=True)  # have to add train_out,otherwise error

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

            slice_bboxes = get_slice_bboxes(
                image_height=height,
                image_width=width,
                slice_height=sliced_height,
                slice_width=sliced_width,
                overlap_height_ratio=overlapped_height_ratio,
                overlap_width_ratio=overlapped_width_ratio,
            )
            #print('slice_bboxes:', slice_bboxes)

            #save the sliced pics
            #src = cv2.imread(img_pah)
            src = cv2.imread(path)
            src = cv2.resize(src, (width, height))
            cv2.imwrite('/home/xumin/yolov5/runs/slicing/' + '640640' + '.jpg', src)
            print('src.shape',src.shape)
            for i in range(len(slice_bboxes)):
                cropped_image = src[slice_bboxes[i][1]:slice_bboxes[i][3], slice_bboxes[i][0]:slice_bboxes[i][2]]
                cv2.imwrite('./runs/slicing/' + str(i) + '.jpg', cropped_image)




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
                # print('type(pred_temp):', type(pred_temp))
                # print('pred_temp:', pred_temp)
                pred_temp[..., 0] = pred_temp[..., 0] + y_startpoint
                pred_temp[..., 1] = pred_temp[..., 1] + x_startpoint
                if i == 0:
                    out = pred_temp
                else:
                    out = torch.cat([out, pred_temp], dim=1)

        pred = out
        #pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # nms

        #soft_nms
        pred = soft_nms(pred, conf_thres, iou_thres, multi_label=True)  # Soft DIoU-NMS
        #pred = soft_nms(pred, 0.1, 0.1, multi_label=True)  # Soft DIoU-NMS



        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        #pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            print('Len(det):',len(det))

            if 165 > len(det): # accpet the detection result of yolov5
            #if True:
            #if False:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #print('n:',n)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        label = None
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()



                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        # resize img and print number on the image
                        im0 = cv2.resize(im0, (2000, 2000))
                        count_number = "Aphids:{}".format(len(det))
                        cv2.putText(im0, count_number, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)

                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            else: #CSRNet

                transform = transforms.Compose([
                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                ])

                model_CSR = CSRNet()

                # defining the model
                model_CSR = model_CSR.cuda()

                # loading the trained weights
                # checkpoint = torch.load('0model_best.pth.tar')
                checkpoint = torch.load('./weights/CSRNet_Aphid_dataset_2/0model_best.pth.tar')
                model_CSR.load_state_dict(checkpoint['state_dict'])


                #img_path = '/home/xumin/phd/CSRNet-pytorch-master/Shanghai/part_A_final/test_data/images/IMG_344_3.jpg'
                img_path = path
                print('img_path:',img_path)
                h5_path = img_path.replace('.jpg', '.h5').replace('JPEGImages', 'ground_truth')
                print('h5_path:', h5_path)

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
                print("Predicted Count : ", int(output))
                temp = np.asarray(output_src.detach().cpu().reshape(output_src.detach().cpu().shape[2],
                                                                    output_src.detach().cpu().shape[3]))
                print('output.detach().cpu().shape[2],output.detach().cpu().shape[3]',output_src.detach().cpu().shape[2], output_src.detach().cpu().shape[3])
                temp[temp < 0.001] = 0
                print('temp',temp)

                # True density map
                ##true_temp = h5py.File(h5_path, 'r')  # 修改
                ##temp_1 = np.asarray(true_temp['density'])
                ##print("Original Count : ", round(np.sum(temp_1)))



                plt.figure(figsize=(200, 200), dpi=10)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)

                plt.xticks([])  # 去掉x轴
                plt.yticks([])  # 去掉y轴
                plt.imshow(temp, alpha=0.75)
                #plt.imshow(temp, cmap='gray')
                # plt.savefig("out.png", dpi=(640, 480),bbox_inches='tight', pad_inches=0)
                #plt.imshow(temp,cmap = CM.jet) #the BG is blue
                plt.savefig(save_path, pad_inches=0)
                plt.show()


                '''
                # No_add_weights
                im = cv2.imread(save_path)
                count_number = "Aphids:{}".format(round(output))
                cv2.putText(im, count_number, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 7)
                cv2.imwrite(save_path, im)
                '''

                # Add_weights_Generate_heatmap
                img = cv2.imread(img_path, 1)
                img = cv2.resize(img, (2000, 2000))
                img1 = cv2.imread(save_path, 1)
                dst = cv2.addWeighted(img, 0.2, img1, 0.8, 0)  # 图像融合
                count_number = "Aphids:{}".format(round(output))
                cv2.putText(dst, count_number, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
                cv2.imwrite(save_path, dst)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--sliced_height', type=int, default=256, help='sliced_height')
    parser.add_argument('--sliced_width', type=int, default=256, help='sliced_width')
    parser.add_argument('--overlapped_height_ratio', type=float, default=0.2, help='overlapped_height_ratio')
    parser.add_argument('--overlapped_width_ratio', type=float, default=0.2, help='overlapped_height_ratio')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
