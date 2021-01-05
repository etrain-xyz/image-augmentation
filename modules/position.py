import random
import os
import cv2
import numpy as np

def missingInformation(img, points, color=(255,255,255)):
	contours = np.array(points)
	return cv2.fillPoly(img, pts=[contours], color=color)

def randomOffset(size):
	offset_percent = random.randint(1, 10)
	return int(size * offset_percent / 100)

def perspectiveTransform(image, source_points, dest_points):
	M = cv2.getPerspectiveTransform(np.float32(source_points), np.float32(dest_points))
	X_s = []
	X_d = []
	Y_s = []
	Y_d = []
	for point in dest_points:
		X_d.append(point[0])
		Y_d.append(point[1])
	for point in source_points:
		X_s.append(point[0])
		Y_s.append(point[1])

	dest_w = max(X_s+X_d) - min(X_s+X_d)
	dest_h = max(Y_s+Y_d) - min(Y_s+Y_d)

	x = min(X_d)
	y = min(Y_d)
	w = max(X_d) - min(X_d)
	h = max(Y_d) - min(Y_d)

	dst = cv2.warpPerspective(image, M, (dest_w, dest_h))
	cropped = dst[y:y+h, x:x+w]

	new_points = []
	for point in dest_points:
		new_points.append([point[0] - x, point[1] - y])
	return cropped, new_points

def addBackground(obj_img, bg_img):
	obj_h, obj_w = obj_img.shape[:2]
	bg_h, bg_w = bg_img.shape[:2]
	max_x = bg_w - obj_w
	max_y = bg_h - obj_h

	if max_x < 0 or max_y < 0:
		if max_x < 0:
			new_bg_w = bg_w - max_x + 10
			new_bg_h = int(new_bg_w * bg_h / bg_w)
			bg_h, bg_w = new_bg_h, new_bg_w
			max_y = bg_h - obj_h
		if max_y < 0:
			new_bg_h = bg_h - max_y + 10
			new_bg_w = int(new_bg_h * bg_w / bg_h)
			bg_h, bg_w = new_bg_h, new_bg_w

		newsize = (bg_w, bg_h)
		bg_img = cv2.resize(bg_img, newsize, cv2.INTER_AREA)
		bg_h, bg_w = bg_img.shape[:2]
		max_x = bg_w - obj_w
		max_y = bg_h - obj_h

	x_start = random.randint(0, max_x)
	y_start = random.randint(0, max_y)
	
	y1, y2 = y_start, y_start + obj_img.shape[0]
	x1, x2 = x_start, x_start + obj_img.shape[1]

	alpha_s = obj_img[:, :, 3] / 255.0
	alpha_l = 1.0 - alpha_s

	for c in range(0, 3):
		bg_img[y1:y2, x1:x2, c] = (alpha_s * obj_img[:, :, c] + alpha_l * bg_img[y1:y2, x1:x2, c])

	return bg_img, x_start, y_start


def rotateImage(obj_img, angle=90):
	obj_h, obj_w = obj_img.shape[:2]
	image_center = (int(obj_w / 2), int(obj_h / 2))
	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
	abs_cos = abs(rotation_mat[0, 0])
	abs_sin = abs(rotation_mat[0, 1])
	bound_w = int(obj_h * abs_sin + obj_w * abs_cos)
	bound_h = int(obj_h * abs_cos + obj_w * abs_sin)
	rotation_mat[0, 2] += bound_w / 2 - image_center[0]
	rotation_mat[1, 2] += bound_h / 2 - image_center[1]
	image_rotated = cv2.warpAffine(obj_img, rotation_mat, (bound_w, bound_h))
	return image_rotated, rotation_mat
