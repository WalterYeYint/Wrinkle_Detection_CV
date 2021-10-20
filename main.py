### USAGE
### run in default
# python main.py
### To run with configs, enter the command below
# python main.py --input "path to input folder" --resize "percent to resize the image" --aperture "aperture size to be used in median blur"
# E.g. python main.py --input images --output outputs --resize 10 --aperture 3

import cv2
import os
import argparse
import numpy as np
from skimage.segmentation import clear_border

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="images",
	help="path to input folder")
ap.add_argument("-o", "--output", type=str, default="outputs",
	help="path to output folder")
ap.add_argument("-r", "--resize", type=int, default=10,
	help="percent to resize the image")
ap.add_argument("-a", "--aperture", type=int, default=3,
	help="aperture size to be used in median blur")
args = vars(ap.parse_args())

source_fldr = args["input"]
target_fldr = args["output"]
scale_percent = args["resize"] # percent of original size
aperture_size = args["aperture"]


for file in os.listdir(source_fldr):
	imgDir = source_fldr + "/" + file
	img = cv2.imread(imgDir)
	
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	# print(dim)
	resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	copied_img = resized_img.copy()

	gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	
	hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
	lower_white = np.array([0, 0, 100])
	upper_white = np.array([50, 40, 255])
	white_masked_img = cv2.inRange(hsv, lower_white, upper_white)

	ret, otsu_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)

	and_operated_img = cv2.bitwise_and(white_masked_img, otsu_img)

	contours, hierarchy = cv2.findContours(and_operated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	max_box = max(contours, key = cv2.contourArea)
	# max_box = np.array(max_box).reshape((-1,1,2)).astype(np.int32)
	# print(max_box)
	cv2.drawContours(copied_img, [max_box], -1, (0, 0, 255), 3) 

	mask = np.zeros_like(and_operated_img)
	cv2.fillPoly(mask, [max_box], 255)
	# cv2.drawContours(mask, box_dim, -1, (255, 255, 255), -1, cv2.LINE_AA)
	masked_img = cv2.bitwise_and(gray_img, mask)

	thresh = cv2.adaptiveThreshold(masked_img, 255,
		cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

	median_blur = cv2.medianBlur(thresh, aperture_size)
	no_border_img = clear_border(median_blur)

	output_img = resized_img.copy()
	contours, hierarchy = cv2.findContours(no_border_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# for c in contours:
	cv2.drawContours(output_img, contours, -1, (0, 0, 255), 3) 
	# cv2.imwrite()

	cv2.imshow('Input Image', resized_img)
	# cv2.imshow('Otsu Image', otsu_img)
	cv2.imshow('Image with largest contour', copied_img)
	# cv2.imshow('AND Operated Image',and_operated_img)
	# cv2.imshow('Poly Image', mask)
	# cv2.imshow('Masked Image', masked_img)
	# cv2.imshow('Thresholded Image', thresh)
	# cv2.imshow('Blurred Image', median_blur)
	cv2.imshow('No border Image', no_border_img)
	cv2.imshow('Output Image', output_img)
	cv2.waitKey()
