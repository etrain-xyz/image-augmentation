import cv2
import numpy as np


def invert_image(image,channel):
	# image=cv2.bitwise_not(image)
	return (channel-image)


def add_light(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
					  for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def saturation_image(image,saturation):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	v = img[:, :, 2]
	v = np.where(v <= 255 - saturation, v + saturation, 255)
	img[:, :, 2] = v
	return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def hue_image(image,saturation):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	v = img[:, :, 2]
	v = np.where(v <= 255 + saturation, v - saturation, 255)
	img[:, :, 2] = v
	return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	

def gausian_blur(image,blur):
	return cv2.GaussianBlur(image,(5,5),blur)

def top_hat_image(image, shift):
	kernel = np.ones((shift, shift), np.uint8)
	return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def sharpen_image(image):
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	return cv2.filter2D(image, -1, kernel)

def salt_image(image,p,a):
	noisy = image.copy()
	num_salt = np.ceil(a * image.size * p)
	coords = [np.random.randint(0, i - 1, int(num_salt))
			  for i in image.shape]
	noisy[tuple(coords)] = 1
	return noisy

def gray_image(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return gray