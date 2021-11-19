### USAGE
### run in default
# python main.py
### To run with configs, enter the command below
# python main.py --input "path to input folder" --resize "percent to resize the image" --aperture "aperture size to be used in median blur"
# E.g. python main.py --input images --output outputs --resize 10 --block_size 11 --c_value 2 --aperture 3 --sensitivity_percent 100

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

def find_and_sort_contours(img, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE, reverse = True):
	contours, hierarchy = cv2.findContours(img, mode, method)
	contours = sorted(contours, key=cv2.contourArea, reverse = reverse)
	return contours

def fill_contours(img, contours):
	img = np.zeros_like(img)
	for contour in contours:
		cv2.fillPoly(img, [contour], 255)
	return img

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

def print_parameters(block_size, c_value, aperture_size, sensitivity_percent):
	print("Block Size: {}, C Value: {}, Aperture Size: {}, Sensitivity Percent: {}".format(block_size, 
																														c_value, aperture_size, sensitivity_percent))

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
ap.add_argument("-s", "--sensitivity_percent", type=int, default=70,
	help="detection sensitivity percent of defects")
args = vars(ap.parse_args())

source_fldr = args["input"]
target_fldr = args["output"]
scale_percent = args["resize"] # percent of original size
block_size = args["block_size"]
c_value = args["c_value"]
aperture_size = args["aperture"]
sensitivity_percent = args["sensitivity_percent"]
quit_flag = False
input_img_name = 'Input Image'
no_border_img_name = 'No Border Image'
output_img_name = 'Output Image'

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
		lower_white = np.array([0, 0, 0])
		upper_white = np.array([180, 60, 255])
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

		contours = find_and_sort_contours(median_blur)[2:]

		bordered_img = fill_contours(gray_img, contours)
		no_border_img = clear_border(bordered_img)

		contours = find_and_sort_contours(no_border_img)

		# Calculating sensitivity percent
		contour_quantity = math.floor(len(contours) * sensitivity_percent / 100)
		if len(contours) > contour_quantity:
			contours = contours[0:contour_quantity]
		
		no_border_img = fill_contours(gray_img, contours)
		cv2.fillPoly(no_border_img, max_box, 255)
		# print(len(contours))
		
		output_img = resized_img.copy()
		# for c in contours:
		cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2) 
		# cv2.imwrite()


		cv2.imshow(input_img_name, resized_img)
		# cv2.imshow('Otsu Image', otsu_img)
		# cv2.imshow('Image with largest contour', copied_img)
		# cv2.imshow('AND Operated Image',and_operated_img)
		# cv2.imshow('Poly Image', mask)
		# cv2.imshow('Masked Image', masked_img)
		# cv2.imshow('Thresholded Image', thresh)
		# cv2.imshow('Blurred Image', median_blur)	
		# cv2.imshow('Bordered Image', bordered_img)
		cv2.imshow(no_border_img_name, no_border_img)
		cv2.imshow(output_img_name, output_img)
		key = cv2.waitKey(1000)
		# if the `q` key was pressed, break from the loop
		if key == ord("n"):
			print("Skipping to next image...")
			break
		elif key == ord("w"):
			block_size = check_parameters_limit(block_size, 2, True)
			print_parameters(block_size, c_value, aperture_size, sensitivity_percent)
		elif key == ord("s"):
			block_size = check_parameters_limit(block_size, 2, False)
			print_parameters(block_size, c_value, aperture_size, sensitivity_percent)
		elif key == ord("e"):
			c_value = check_parameters_limit(c_value, 1, True)
			print_parameters(block_size, c_value, aperture_size, sensitivity_percent)
		elif key == ord("d"):
			c_value = check_parameters_limit(c_value, 1, False)
			print_parameters(block_size, c_value, aperture_size, sensitivity_percent)
		elif key == ord("r"):
			aperture_size = check_parameters_limit(aperture_size, 2, True)
			print_parameters(block_size, c_value, aperture_size, sensitivity_percent)
		elif key == ord("f"):
			aperture_size = check_parameters_limit(aperture_size, 2, False)
			print_parameters(block_size, c_value, aperture_size, sensitivity_percent)
		elif key == ord("t"):
			sensitivity_percent = check_parameters_limit(sensitivity_percent, 10, True, True)
			print_parameters(block_size, c_value, aperture_size, sensitivity_percent)
		elif key == ord("g"):
			sensitivity_percent = check_parameters_limit(sensitivity_percent, 10, False, True)
			print_parameters(block_size, c_value, aperture_size, sensitivity_percent)
		elif key == ord("q"):
			print("Quitting...")
			quit_flag = True
			break
		input_window = cv2.getWindowProperty(input_img_name,cv2.WND_PROP_VISIBLE)
		no_border_window = cv2.getWindowProperty(no_border_img_name,cv2.WND_PROP_VISIBLE)
		output_window = cv2.getWindowProperty(output_img_name,cv2.WND_PROP_VISIBLE)
		if input_window < 1 or no_border_window < 1 or output_window < 1:  
			print("Quitting...")
			quit_flag = True      
			break        
	if quit_flag == True:
		break
cv2.destroyAllWindows()

