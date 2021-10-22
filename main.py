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
import random
import math
from skimage.segmentation import clear_border

def checker(a):
	num = int(a)
	if (num % 2) == 0:
		raise argparse.ArgumentTypeError('Block Size and Aperture Size must be odd numbers!!')
	return num

# def namestr(obj, namespace):
#     return [name for name in namespace if namespace[name] is obj]

def check_parameters_limit(curr_value, step_value, is_increment = True, is_sensitivity_percent = False):
	if is_increment == True:
		if is_sensitivity_percent == True and curr_value >= 100:
			return curr_value
		else:
			curr_value += step_value
	else:
		if curr_value <= step_value:
			return curr_value
		else:
			curr_value -= step_value
	return curr_value



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="images",
	help="path to input folder")
ap.add_argument("-o", "--output", type=str, default="outputs",
	help="path to output folder")
ap.add_argument("-r", "--resize", type=int, default=10,
	help="percent to resize the image")
ap.add_argument("-b", "--block_size", type=checker, default=11,
	help="Block Size value used in adaptive threshold")
ap.add_argument("-c", "--c_value", type=int, default=2,
	help="C value used in adaptive threshold")
ap.add_argument("-a", "--aperture", type=checker, default=3,
	help="aperture size to be used in median blur")
ap.add_argument("-s", "--sensitivity_percent", type=int, default=100,
	help="detection sensitivity percent of defects")
args = vars(ap.parse_args())

source_fldr = args["input"]
target_fldr = args["output"]
scale_percent = args["resize"] # percent of original size
block_size = args["block_size"]
c_value = args["c_value"]
aperture_size = args["aperture"]
sensitivity_percent = 100
quit_flag = False

print("The program has initialized...")
print("Block Size: {}, C Value: {}, Aperture Size: {}\n\n".format(block_size, c_value, aperture_size))
print("Adjust Block Size using ------- W key and S key")
print("Adjust C Value using ---------- E key and D key")
print("Adjust Aperture Size using ---- R key and F key")
print("Adjust Contour Quantity using - T key and G key")
print("To skip to next image, press -- N key")
print("To quit program, press -------- Q key\n\n")

for file in os.listdir(source_fldr):
	# print(namestr(c_value, globals())[0])
	while True:
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
		cv2.drawContours(copied_img, [max_box], -1, (0, 0, 255), 2) 

		mask = np.zeros_like(and_operated_img)
		cv2.fillPoly(mask, [max_box], 255)
		# cv2.drawContours(mask, box_dim, -1, (255, 255, 255), -1, cv2.LINE_AA)
		masked_img = cv2.bitwise_and(gray_img, mask)

		# perform adaptive threshold
		# cv2.adaptiveThreshold(src, maxValue, adaptiveMethd, thresholdType, blockSize(thickness), C value(quantity))
		thresh = cv2.adaptiveThreshold(masked_img, 255,
			cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c_value)

		median_blur = cv2.medianBlur(thresh, aperture_size)
		no_border_img = clear_border(median_blur)

		contours, hierarchy = cv2.findContours(no_border_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		random.shuffle(contours)

		# Calculating sensitivity percent
		contour_quantity = math.floor(len(contours) * sensitivity_percent / 100)

		contour_adjusted_img = np.zeros_like(no_border_img)
		if len(contours) > contour_quantity:
			contours = contours[0:contour_quantity]
		for contour in contours:
			cv2.fillPoly(contour_adjusted_img, [contour], 255)
		cv2.fillPoly(contour_adjusted_img, max_box, 255)
		# print(len(contours))
		
		output_img = resized_img.copy()
		# for c in contours:
		cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2) 
		# cv2.imwrite()


		cv2.imshow('Input Image', resized_img)
		# cv2.imshow('Otsu Image', otsu_img)
		# cv2.imshow('Image with largest contour', copied_img)
		# cv2.imshow('AND Operated Image',and_operated_img)
		# cv2.imshow('Poly Image', mask)
		# cv2.imshow('Masked Image', masked_img)
		# cv2.imshow('Thresholded Image', thresh)
		# cv2.imshow('Blurred Image', median_blur)
		# cv2.imshow('No border Image', no_border_img)
		cv2.imshow('Contour Adjusted Image', contour_adjusted_img)
		cv2.imshow('Output Image', output_img)
		key = cv2.waitKey()
		# if the `q` key was pressed, break from the loop
		if key == ord("n"):
			print("Skipping to next image...")
			break
		elif key == ord("w"):
			block_size = check_parameters_limit(block_size, 2, True)
		elif key == ord("s"):
			block_size = check_parameters_limit(block_size, 2, False)
		elif key == ord("e"):
			c_value = check_parameters_limit(c_value, 1, True)
		elif key == ord("d"):
			c_value = check_parameters_limit(c_value, 1, False)
		elif key == ord("r"):
			aperture_size = check_parameters_limit(aperture_size, 2, True)
		elif key == ord("f"):
			aperture_size = check_parameters_limit(aperture_size, 2, False)
		elif key == ord("t"):
			sensitivity_percent = check_parameters_limit(sensitivity_percent, 10, True, True)
		elif key == ord("g"):
			sensitivity_percent = check_parameters_limit(sensitivity_percent, 10, False, True)
		elif key == ord("q"):
			print("Quitting...")
			quit_flag = True
			break
		print("Block Size: {}, C Value: {}, Aperture Size: {}, Sensitivity Percent: {}".format(block_size, 
																														c_value, aperture_size, sensitivity_percent))
	if quit_flag == True:
		break
cv2.destroyAllWindows()

