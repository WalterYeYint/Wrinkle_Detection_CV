### USAGE
### run in default
# python main.py
### To run with configs, enter the command below
# python main.py --input "path to input folder" --resize "percent to resize the image" --aperture "aperture size to be used in median blur"
# E.g. python main.py --input images --output outputs --resize 10 --aperture 3

import cv2
import os
import argparse

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

	gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

	# T, thresh = cv2.threshold(gray_img, 120, 255,
	# 	cv2.THRESH_BINARY)
	# T, thresh2 = cv2.threshold(gray_img, 70, 255,
	# 	cv2.THRESH_BINARY_INV)
	# masked_img = cv2.bitwise_and(thresh, thresh2)

	ret, otsu_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)
	masked_img = cv2.bitwise_and(gray_img, otsu_img)

	thresh = cv2.adaptiveThreshold(masked_img, 255,
		cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	
	median_blur = cv2.medianBlur(thresh, aperture_size)

	cv2.imshow('Input Image', resized_img)
	# cv2.imshow('Otsu Image', otsu_img)
	# cv2.imshow('Filtered Image', masked_img)
	# cv2.imshow('Thresholded Image', thresh)
	cv2.imshow('Blurred Image', median_blur)
	# cv2.imshow('Output Image 2', thresh2)
	# cv2.imshow('Masked Image', masked_img)
	cv2.waitKey()
