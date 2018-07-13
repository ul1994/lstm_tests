import scipy.ndimage
from scipy.ndimage import gaussian_filter as blur
import cv2

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data', one_hot=True)

from random import randint
import numpy as np

def place(img, im_size=0.5, frame_size=128):
	img = img.reshape((28, 28))
	img = cv2.resize(img, (0,0), fx=im_size, fy=im_size)
	ry, rx = randint(0, frame_size-img.shape[0]-1), randint(0, frame_size-img.shape[0]-1)
	return img, (rx, ry)

def get_direction():
	dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)]
	return dirs[randint(0, len(dirs)-1)]

def animate(initial, steps=10, speed=5, size=128):
	img, (x0, y0) = initial
	ih, iw = img.shape
	coords = [(x0, y0)]

	rx, ry = get_direction()

	bg = np.zeros((size, size))
	bg[y0:y0+ih, x0:x0+iw] = img

	frames = [bg]
	for ii in range(steps):
		dy, dx = ry* speed, rx* speed

		# redirect if OOB:
		while bg[y0+dy:y0+dy+ih, x0+dx:x0+dx+iw].shape != (ih, iw):
			rx, ry = get_direction()
			dy, dx = ry* speed, rx* speed

		y0 += dy
		x0 += dx
		coords.append((x0, y0))
		bg = np.zeros((size, size))
		bg[y0:y0+ih, x0:x0+iw] = img
		frames.append(bg)
	return frames, coords

def matching_heatmap(coords, imsize=14, framesize=128, scale=0.25):
	heats = []
	radius = int(imsize/2)
	for (xx, yy) in coords:
		bg = np.zeros((framesize, framesize))
		bg[yy+radius, xx+radius] = 255
		blurred = blur(bg, sigma=7)
		blurred /= np.max(blurred)
		heats.append(
			cv2.resize(blurred, (0,0), fx=scale, fy=scale))
		# heats.append((blurred * 255).astype(np.uint8))
	return heats