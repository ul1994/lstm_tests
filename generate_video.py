
DATA_DIR = "/beegfs/ua349/lstm/Penn_Action"

frames_dir = DATA_DIR + '/frames'
labels_dir = DATA_DIR + '/labels'

import os, sys
from scipy.io import loadmat
from random import shuffle

def gather_videos(SEQ_LEN = 4, still=False):
	refs = []


	# [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
	# coco_incl = [0, None, 1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12, None, None, None, None, None]
	coco_incl = [0, None, 1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12, None, None, None, None]
	# TODO: 19th entry should be a maskish thing...?

	frame_folders = os.listdir(frames_dir)
	for __fii, fldr in enumerate(sorted(frame_folders, key=lambda val: int(val))):

		sys.stdout.write('%d/%d ~ %d\r' % (__fii, len(frame_folders), len(refs)))
		sys.stdout.flush()

		flpath = '%s/%s' % (frames_dir, fldr)
		imgs = [fl for fl in os.listdir(flpath)]
		imgs = sorted(imgs, key=lambda val: int(val.split('.')[0]))

		lblpath = '%s/%s' % (labels_dir, fldr)
		mat = loadmat(lblpath)
		xs, ys, bbox = mat['x'], mat['y'], mat['bbox']
		vis, dim = mat['visibility'], mat['dimensions']

		for bii in range(0, len(imgs), SEQ_LEN):
			ref = {
				'frames': [],
	#             'labels': [],
				'boxes': [],
	#             'local_coords': [],
				'coords': [],
				'coco_coords': [],
				'visibility': [],
			}


			if bii + SEQ_LEN >= len(imgs):
				rng = range(len(imgs) - SEQ_LEN, len(imgs))
			else:
				rng = range(bii, bii+SEQ_LEN)
			assert rng is not None

			if still:
				rng = [bii] * SEQ_LEN # all the same ind
			for ii in rng:

				assert ii < len(xs)
				ref['frames'].append('%s/%s' % (flpath, imgs[ii]))

				try:
					# case where last bbox can be missing sometimes
					assert ii < len(bbox)
					box = bbox[ii]
				except:
					print('WARN: Missing bbox %s' % lblpath)
					print()
					box = bbox[ii-1] # just sub in last frame

				ref['boxes'].append(box)
				coords = []
				for (xx, yy) in zip(xs[ii], ys[ii]):
					xx = max(0, xx-1)
					yy = max(0, yy-1)

					if xx == 0 or yy == 0:
						# invalid coord
						coords.append((-1, -1))
					else:
						coords.append((xx, yy))

				ref['coords'].append(coords)
				coco_coords = [None if ind is None else coords[ind] for ind in coco_incl]
				ref['coco_coords'].append(coco_coords)
				ref['visibility'].append(vis[ii])
			assert len(ref['frames']) == SEQ_LEN
			assert len(ref['coords']) == SEQ_LEN
			assert len(ref['coco_coords']) == SEQ_LEN

			refs.append(ref)
	shuffed = refs
	shuffle(shuffed)
	return [shuffed, refs]

import cv2
from cv2 import imread
import numpy as np
import random
def size_image(img, size=368):
	img = img.astype(np.float32)

	maxdim = max(img.shape[:2])
	ratio = size / maxdim

	img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)

	random.uniform(0.5, 0.75)

	canvas = np.zeros((size, size, 3))
	dx, dy = 0, 0
	if img.shape[0] > img.shape[1]:
		dx = int((size - img.shape[1]) / 2)
	else:
		dy = int((size - img.shape[0]) / 2)

	canvas[dy:dy+img.shape[0], dx:dx+img.shape[1]] = img

	return canvas, (ratio, (dx, dy))

	# for frame in range()

from scipy.ndimage import gaussian_filter as blur

def create_heatmaps(coords, spec, size=46,dims=19):
	ratio, (dx, dy) = spec
	canvas = np.zeros((size, size, dims))

	for ii, coord in enumerate(coords):
		if coord is None:
			continue

		(xx, yy) = coord
		if xx == -1 or yy == -1:
			# invalid coord - DNE in image
			continue

		xx = int((dx + ratio * xx) * size/368)
		yy = int((dy + ratio * yy) * size/368)

		if xx >= size or yy >= size:
			continue

		canvas[yy, xx, ii] = 1
		img = blur(canvas[:, :, ii], sigma=1)
		canvas[:, :, ii] = img / np.max(img)

		canvas[yy, xx, -1] = 1 # sum mask

	assert np.max(canvas[:, :, -1]) <= 1.0
	img = blur(canvas[:, :, -1], sigma=1)
	img /= np.max(img)
	inv = 1 - img
	canvas[:, :, -1] = inv

	return canvas

def create_mask(bbox, spec, size=46, dims =19, coords=None, buffer=1):
	ratio, (dx, dy) = spec
	canvas = np.zeros((size, size, dims))

	x0, y0, xF, yF = bbox
	x0 = int((dx + ratio * x0) * size/368)
	xF = int((dx + ratio * xF) * size/368)
	y0 = int((dy + ratio * y0) * size/368)
	yF = int((dy + ratio * yF) * size/368)

	for ii in range(dims):
		# canvas[y0:yF, x0:xF, ii] = 1
		canvas[y0-buffer:yF+buffer, x0-buffer:xF+buffer, ii] = 1

	if coords is not None:
		# zero out joints that are missing from PENN via masking
		for ii, joint in enumerate(coords):
			if joint is None:
				canvas[:, :, ii] = 0

	# canvas[:, :, -1] = 0 # ignore final by masking it out

	return canvas

def next_video_batch(refs, bsize=6):
	brefs = refs[0][:bsize]
	refs[0] = refs[0][bsize:]

	if len(refs[0]) < bsize:
		shuffed = refs[1]
		shuffle(shuffed)
		refs[0] = shuffed

	videos = []
	masks = []
	targets = []

	for ref in brefs:
		sized, specs = zip(*[size_image(imread(path)) for path in ref['frames']])

		heats = [create_heatmaps(ref['coco_coords'][ii], specs[ii]) for ii in range(len(ref['frames']))]
		mask = [create_mask(ref['boxes'][ii], specs[ii], coords=ref['coco_coords'][ii]) for ii in range(len(ref['frames']))]

		# mask all except final
		for ii in range(len(ref['frames'])):
			heats[ii][:, :, :-1] = np.multiply(heats[ii][:, :, :-1], mask[ii][:, :, :-1])

		videos.append(sized)
		masks.append(mask)
		targets.append(heats)

	return (
		np.array(videos),
		np.array(masks),
		np.array(targets))




