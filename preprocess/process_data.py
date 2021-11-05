import sys
from typing import IO
import cv2
import numpy as np
import torch
import os

import face_preprocess

detector_path = 'preprocess/retina_mnet/'

sys.path.append(detector_path)
import retina



pth = f'{detector_path}model_detection/'
face_detector = retina.RetinaFace(pth, nms=0.4, worker=1)




root = '/home/allen/Documents/workplace/lips_reading/lips_reading/test'



for ID in os.listdir(root):
	for v in os.listdir(f'{root}/{ID}'):

		v_name = v.split('.')[0]

		if not os.path.exists(f'./data_aligned/{ID}_{v_name}'):
			os.mkdir(f'./data_aligned/{ID}_{v_name}')

		cap = cv2.VideoCapture(f'{root}/{ID}/{v}')

		fps = int(cap.get(cv2.CAP_PROP_FPS))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		print("fps", fps)
		print("frame_height", height)
		print("frame_width", width)



		def get_scale(img):
			im_shape = img.shape
			target_size = 480
			max_size = 640
			im_size_min = np.min(im_shape[0:2])
			im_size_max = np.max(im_shape[0:2])
			im_scale = float(target_size) / float(im_size_min)
			if np.round(im_scale * im_size_max) > max_size:
				im_scale = float(max_size) / float(im_size_max)
			return im_scale
		count = 0
		while True:
			ret, img = cap.read()




			if not ret or img is None:
				break

			if height == 1080:
				img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

			H,W,_ =img.shape

			cv2.imshow(f'crop.jpg',img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			x = img.copy()
			scale = get_scale(img)
			x = cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

			bboxes, landmark = face_detector.detect([x], [img])

			if landmark[0] is not None:
				aligned = face_preprocess.preprocess_112(img,np.array(landmark[0][0]))

				cv2.imwrite(f'./data_aligned/{ID}_{v_name}/{ID}_{v_name}_{count:04d}.jpg',aligned)

				count +=1

				print(aligned.shape)

				

