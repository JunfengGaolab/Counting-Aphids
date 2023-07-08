
# refer to SAHI
import cv2
from matplotlib import pyplot as plt
from IPython import display
#%matplotlib inline

from sahi.utils.cv import read_image_as_pil
from sahi.slicing import slice_image
from typing import Dict, List, Optional, Union

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
    print(type(overlap_height_ratio))
    print(type(slice_height))
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
                print('xmax-xmin,ymax-ymin:', xmax-xmin, ymax-ymin)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                print('x_max-x_min,y_max-y_min:', x_max - x_min, y_max - y_min)
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


image= "/home/xumin/yolov5/dataset/voc2010/JPEGImages/IMG_1_2.jpg"
image_pil = read_image_as_pil(image)
print("image.shape: " + str(image_pil.size))

image_width, image_height = image_pil.size
if not (image_width != 0 and image_height != 0):
    raise RuntimeError(f"invalid image size: {image_pil.size} for 'slice_image'.")


slice_height = 256, #256
slice_width = 256, #256
overlap_height_ratio = 0.2,
overlap_width_ratio = 0.2



slice_bboxes = get_slice_bboxes(
    image_height=image_height,
    image_width=image_width,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

print('slice_bboxes:', slice_bboxes)
#print('type(slice_bboxes):',type(slice_bboxes))

image = cv2.imread(image)

for i in range(len(slice_bboxes)):
    clone = image.copy()
    #print('slice_bboxes[i]:',slice_bboxes[i])
    cv2.rectangle(clone, (slice_bboxes[i][0], slice_bboxes[i][1]), (slice_bboxes[i][2], slice_bboxes[i][3]), (0, 255, 0), 2)
    #cv2.imshow("Window", clone)
    #cv2.waitKey(1)
    #time.sleep(0.025)

    plt.imshow(clone)
    plt.pause(0.1)
    display.clear_output(wait=True)


'''
slice_image_result = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
'''






'''
# refer to https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
# import the necessary packages
import argparse
import time
import cv2
#import imutils
from matplotlib import pyplot as plt
from IPython import display

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



# load the image and define the window width and height
image = cv2.imread("pets.jpg")
(winW, winH) = (200, 200)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		#if window.shape[0] != winH or window.shape[1] != winW:
			#continue
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		#cv2.imshow("Window", clone)
		#cv2.waitKey(1)
		#time.sleep(0.025)

		plt.imshow(clone)
		plt.pause(0.1)
		display.clear_output(wait=True)
'''


'''
#refer to https://blog.csdn.net/submarineas/article/details/123347906
import cv2
from matplotlib import pyplot as plt
from IPython import display
#%matplotlib inline

def sliding_window(image, window, step):
    for y in range(0, image.shape[0] - window[1], step):
        for x in range(0, image.shape[1] - window[0], step):
            yield (x, y, image[y:y + window[1], x:x + window[0]])

image = cv2.imread("pets.jpg")
(window_w, window_h) = (300, 300)

for (x, y, window) in sliding_window(image, (window_w, window_h), 200):
    #if window.shape[0] != window_w or window.shape[1] != window_h:
        #continue

    clone = image.copy()
    cv2.rectangle(clone, (x, y), (x + window_w, y + window_h), (0, 255, 0), 2)
    clone = clone[:, :, ::-1]
    plt.imshow(clone)
    plt.pause(1)
    display.clear_output(wait=True)
'''
