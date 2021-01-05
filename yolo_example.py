from modules.position import *
from modules.color import *
import os
import cv2
import numpy as np
import random
from shutil import copyfile
from os import listdir
from os.path import isdir, isfile, join
import yaml


def mapPointsRotated(old_points, rotation_mat):
	### Map points to new points in rotated image
	points = np.array(old_points)
	# add ones
	ones = np.ones(shape=(len(points), 1))
	points_ones = np.hstack([points, ones])
	# transform points
	n_points = rotation_mat.dot(points_ones.T).T
	return n_points


def yoloCoordinates(points, width, height, offset_x=0, offset_y=0):
	X = []
	Y = []
	for p in points:
		x = p[0] + offset_x
		X.append(x)
		y = p[1] + offset_y
		Y.append(y)
	cx_p = ((min(X) + max(X)) / 2) / width
	cy_p = ((min(Y) + max(Y)) / 2) / height
	box_w_p = (max(X) - min(X)) / width
	box_h_p = (max(Y) - min(Y)) / height
	return cx_p, cy_p, box_w_p, box_h_p


def position_augmentation(obj_img, bg_img, dest_points, angle):
	obj_h, obj_w = obj_img.shape[:2]
	bg_h, bg_w = bg_img.shape[:2]

	### Rotate img
	rotated_img, rotation_mat = rotateImage(obj_img, angle)

	### Map points to new points in rotated image
	n_points = mapPointsRotated(dest_points, rotation_mat)

	### Add rotate to bg
	back_im, x_start, y_start = addBackground(rotated_img, bg_img)

	### Yolo format
	cx_p, cy_p, box_w_p, box_h_p = yoloCoordinates(n_points, back_im.shape[1], back_im.shape[0], offset_x=x_start, offset_y=y_start)

	return back_im, cx_p, cy_p, box_w_p, box_h_p



def color_augmentation(_img, params):
	result = {}
	if "is_invert" in params:
		channel = random.randint(10, 50)
		result["image_invert"] = invert_image(_img, channel)

	if "is_light" in params:
		gamma = random.choice(np.arange(0.1, 5.0, 0.1))
		result["image_light"] = add_light(_img, gamma)

	if "is_saturation" in params:
		saturation = random.randint(50, 80)
		result["image_saturation"] = saturation_image(_img, saturation)

	if "is_hue" in params:
		hue = random.randint(50, 80)
		result["image_hue"] = hue_image(_img, hue)

	if "is_blur" in params:
		blur = random.choice(np.arange(0.2, 4.0, 0.1))
		result["image_blur"] = gausian_blur(_img, blur)

	if "is_shift" in params:
		shift = random.randint(100, 500)
		result["image_shift"] = top_hat_image(_img, shift)

	if "is_sharpen" in params:
		result["image_sharpen"] = sharpen_image(_img)

	if "is_salt" in params:
		a = random.choice(np.arange(0.01, 0.09, 0.01))
		result["image_salt"] = salt_image(_img, 0.5, a)

	if "is_gray" in params:
		result["image_gray"] = gray_image(_img)

	return result


def createYoloImage(_img, class_id, result_path, points, file_name):
	_h, _w = _img.shape[:2]
	cx_p, cy_p, box_w_p, box_h_p = yoloCoordinates(points, _w, _h)
	coordinates = str(cx_p)+" "+str(cy_p)+" "+str(box_w_p)+" "+str(box_h_p)
	saveImageAndTxt(result_path, _img, file_name, class_id, coordinates)


def saveImageAndTxt(result_dir, img, filename, class_id, coordinates):
	yolo_txt = filename + ".txt"
	cv2.imwrite(result_dir+filename+".png", img)
	with open(result_dir+yolo_txt, 'w') as outfile:
		outfile.write(str(class_id) + " " + coordinates)

def copyYoloImage(_img, result_path, yolo_old_file, yolo_new_file):
	cv2.imwrite(result_path + yolo_new_file + ".png", _img)
	copyfile(result_path + yolo_old_file, result_path + yolo_new_file + ".txt")

if __name__ == "__main__":
	with open(r'./example_config.yaml') as file:
		config = yaml.load(file, Loader=yaml.FullLoader)

	object_path = "./example_data/object/"
	bg_dir = "./example_data/background/"
	result_path = config["result_path"]
	labels = config["labels"]

	background_images = [f for f in listdir(bg_dir) if isfile(join(bg_dir, f))]
	
	img1 = cv2.imread(object_path + labels[0] + "/front.jpg")
	img2 = cv2.imread(object_path + labels[0] + "/back.jpg")
	img3 = cv2.imread(object_path + labels[1] + "/front.jpg")
	img4 = cv2.imread(object_path + labels[1] + "/back.jpg")


	# Original Image
	coordinates = "0.5 0.5 1.0 1.0"
	saveImageAndTxt(result_path, img1, "two_front", 0, coordinates)
	saveImageAndTxt(result_path, img2, "two_back", 0, coordinates)
	saveImageAndTxt(result_path, img3, "twenty_front", 1, coordinates)
	saveImageAndTxt(result_path, img4, "twenty_back", 1, coordinates)

	img3_h, img3_w = img3.shape[:2]
	top_left = [0, 0]
	top_right = [img3_w, 0]
	bottom_right = [img3_w, img3_h]
	bottom_left = [0, img3_h]
	source_points_3 = [top_left, top_right, bottom_right, bottom_left]

	# Color augmentation
	result = color_augmentation(img1, {
		"is_light": True,
		"is_gray": True
	})
	for k in result:
		file_name_aug = "two_front" + k
		copyYoloImage(result[k], result_path, "two_front.txt", file_name_aug)


	# Information Missing
	img2_h, img2_w = img2.shape[:2]
	min_cut_x = int(img2_w * 10 / 100)
	max_cut_x = int(img2_w * 30 / 100)
	min_cut_y = int(img2_h * 10 / 100)
	max_cut_y = int(img2_h * 30 / 100)
	lst_x = range(min_cut_x, max_cut_x, 1)
	lst_y = range(min_cut_y, max_cut_y, 1)
	pt1 = [10, 10]
	pt2 = [random.choice(lst_x), 10]
	pt3 = [random.choice(lst_x), random.choice(lst_y)]
	pt4 = [10, random.choice(lst_y)]
	img_mising = missingInformation(img2,  [pt1, pt2, pt3, pt4])
	copyYoloImage(img_mising, result_path, "two_back.txt", "two_back_missing")


	# Transform image
	top_left = [0+randomOffset(img3_w), 0+randomOffset(img3_h)]
	top_right = [img3_w-randomOffset(img3_w), 0+randomOffset(img3_h)]
	bottom_right = [img3_w-randomOffset(img3_w), img3_h-randomOffset(img3_h)]
	bottom_left = [0+randomOffset(img3_w), img3_h-randomOffset(img3_h)]
	dest_points = [top_left, top_right, bottom_right, bottom_left]
	img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2RGBA)
	img_transform, n_points = perspectiveTransform(img3, source_points_3, dest_points)
	createYoloImage(img_transform, 1, result_path, n_points, "twenty_front_transform")

	# Add obj to background
	bg_name = random.choice(background_images)
	bg_path = bg_dir + bg_name
	bg_img = cv2.imread(bg_path)
	position_img, cx_p, cy_p, box_w_p, box_h_p = position_augmentation(img_transform, bg_img, dest_points=n_points, angle=45)
	new_file_name = "twenty_front_transform_background"
	coordinates = str(cx_p)+" "+str(cy_p)+" "+str(box_w_p)+" "+str(box_h_p)
	saveImageAndTxt(result_path, position_img, new_file_name, 1, coordinates)

	img4_h, img4_w = img4.shape[:2]
	top_left = [0, 0]
	top_right = [img4_w, 0]
	bottom_right = [img4_w, img4_h]
	bottom_left = [0, img4_h]
	source_points_4 = [top_left, top_right, bottom_right, bottom_left]
	bg_name = random.choice(background_images)
	bg_path = bg_dir + bg_name
	bg_img = cv2.imread(bg_path)
	img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2RGBA)
	position_img, cx_p, cy_p, box_w_p, box_h_p = position_augmentation(img4, bg_img, dest_points=source_points_4, angle=60)
	new_file_name = "twenty_back_transform_background"
	coordinates = str(cx_p)+" "+str(cy_p)+" "+str(box_w_p)+" "+str(box_h_p)
	saveImageAndTxt(result_path, position_img, new_file_name, 1, coordinates)


	# Rotate original img
	rotated_img, rotation_mat = rotateImage(img3, 30)
	rotated_h, rotated_w = rotated_img.shape[:2]
	n_points = mapPointsRotated(source_points_3, rotation_mat)
	createYoloImage(rotated_img, 1, result_path, n_points, "twenty_front_original_rotated")


