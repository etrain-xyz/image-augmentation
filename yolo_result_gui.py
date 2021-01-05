import cv2
import glob
import os
import random
import yaml

def draw_image(yolo_dir, labels, img_name, img_ext):
	img_path = yolo_dir + img_name + img_ext
	print(img_path)
	img = cv2.imread(img_path)
	dh, dw, _ = img.shape

	label_path = yolo_dir + img_name + ".txt"
	fl = open(label_path, 'r')
	data = fl.read().split("\n")
	fl.close()

	for dt in data:
		dt = dt.strip()
		fields = dt.split(' ')
		if len(fields) != 5:
			break
		# Split string to float
		li, x, y, w, h, = map(float, dt.split(' '))
		print( labels[int(li)] )

		l = int((x - w / 2) * dw)
		r = int((x + w / 2) * dw)
		t = int((y - h / 2) * dh)
		b = int((y + h / 2) * dh)
		
		if l < 0:
			l = 0
		if r > dw - 1:
			r = dw - 1
		if t < 0:
			t = 0
		if b > dh - 1:
			b = dh - 1	

		color = (0,0,255)
		img = cv2.rectangle(img, (l, t), (r, b), color, 2)
		img = cv2.putText(img, labels[int(li)], (int(r/2), int(b/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
	return img


def showImg(yolo_dir, yolo_labels, file):
	title = file["title"]
	ext = file["ext"]
	image = draw_image(yolo_dir, yolo_labels, title, ext)
	dh, dw, _ = image.shape
	new_width = 1000
	new_height = int(new_width * dh / dw)
	image = cv2.resize(image, (new_width, new_height))
	cv2.imshow("Show image",image)


def start(yolo_dir, yolo_labels):
	images = []
	for pathAndFilename in glob.iglob(os.path.join(yolo_dir, "*.png")):
		title, ext = os.path.splitext(os.path.basename(pathAndFilename))
		images.append({
			"title": title,
			"ext": ext
		})

	index = random.randrange(0, len(images))
	file = images.pop(index)
	showImg(yolo_dir, yolo_labels, file)

	while(1):
		k = cv2.waitKey(33)
		if k==27:    # Esc key to stop
			break
		elif k==32:  # Space to next
			if len(images) == 0:
				break
			else:
				index = random.randrange(0, len(images))
				file = images.pop(index)
				showImg(yolo_dir, yolo_labels, file)
				continue


if __name__ == "__main__":
	with open(r'./example_config.yaml') as file:
		config = yaml.load(file, Loader=yaml.FullLoader)

	print(config)
	start(config["result_path"], config["labels"])