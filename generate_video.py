import matplotlib as mpl
mpl.use('Agg')

import os, sys
import cv2
from scipy.io import loadmat
from random import shuffle
sys.path.append('../tf/ver1')
from training.label_maps import create_heatmap, create_paf
from training.dataflow import JointsLoader

DATA_DIR = "/beegfs/ua349/lstm/Penn_Action"

frames_dir = DATA_DIR + '/frames'
labels_dir = DATA_DIR + '/labels'

def gather_videos(SEQ_LEN = 4, still=False, speedup=2, shuffle=True):
	refs = []


	# [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
	coco_incl = [0, None, 1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12, None, None, None, None]

	def approx_neck(coords):
		NECK_IND = 1
		rsho, lsho = coords[2], coords[5]

		if rsho is None or lsho is None:
			return None

		for ent in [rsho[0], rsho[1], lsho[0], lsho[1]]:
			if ent == -1:
				return None

		coords[NECK_IND] = [
			(rsho[0] + lsho[0])/2,
			(rsho[1] + lsho[1])/2
		]

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

		# for bii in range(0, len(imgs), SEQ_LEN):
		end = int(len(imgs)/speedup)
		for bii in range(0, end, SEQ_LEN):
			ref = {
				'frames': [],
				'boxes': [],
				'coords': [],
				'coco_coords': [],
				'visibility': [],
			}


			if bii + SEQ_LEN >= end:
				rng = range(end - SEQ_LEN, end)
			else:
				rng = range(bii, bii+SEQ_LEN)
			assert rng is not None

			if still:
				rng = [bii] * SEQ_LEN # all the same ind
			for ii in rng:
				ii *= speedup

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
						coords.append(None)
					else:
						coords.append((xx, yy))

				ref['coords'].append(coords)

				coco_coords = [None if ind is None else coords[ind] for ind in coco_incl]
				approx_neck(coco_coords)
				ref['coco_coords'].append(coco_coords)

				ref['visibility'].append(vis[ii])
			assert len(ref['frames']) == SEQ_LEN
			assert len(ref['coords']) == SEQ_LEN
			assert len(ref['coco_coords']) == SEQ_LEN

			refs.append(ref)
	shuffed = refs
	if shuffle:
		shuffle(shuffed)
	return [shuffed, refs]

import cv2
from cv2 import imread
import numpy as np
import random

def size_image(img, bbox, zoom, size=368):
	img = img.astype(np.float32)

	targ = size * zoom
	half = targ / 2

	cx, cy = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2

	x0, y0 = max(0, cx - half), max(0, cy - half)
	region = img[int(y0):int(y0+targ), int(x0):int(x0+targ)]
	maxlen = max(region.shape[0], region.shape[1])

	region = cv2.resize(region, (0,0), fx=368 / maxlen, fy=368 / maxlen)
	assert region.shape[0] == size or region.shape[1] == size

	canvas = np.zeros((size, size, 3))

	# center image again b/c region is not always a square
	dx, dy = 0, 0
	if region.shape[0] > region.shape[1]:
		dx = int((size - region.shape[1]) / 2)
	else:
		dy = int((size - region.shape[0]) / 2)

	canvas[dy:dy+region.shape[0], dx:dx+region.shape[1]] = region

	return canvas, (zoom, targ, (x0, y0), (dx, dy))

	# for frame in range()

def create_mask(dims, height, width, bbox, stride=8, pad=1):
	x0, y0, xf, yf = (np.array(bbox) / stride)
	canvas = np.zeros((height, width, dims))
	canvas[int(y0)-pad:int(yf)+pad, int(x0)-pad:int(xf)+pad] = 1
	assert np.max(canvas) > 0
	return canvas

def augment(imgs, zoom, xoff, yoff, bbox, stride=1):
	imgs = np.array(imgs)
	# xoff = xoff / stride
	# yoff = yoff / stride
	FIRST = 0

	width = imgs[FIRST].shape[1]
	height = imgs[FIRST].shape[0]
	x0, y0, xf, yf = np.array(bbox[FIRST]) / stride
	boxX, boxY = (x0 + xf) / 2, (y0 + yf) / 2
	cX = width / 2 - boxX
	cY = height / 2 - boxY

	canvas = np.zeros(imgs.shape, dtype=imgs.dtype)
	fill = canvas[:, max(0, int(cY)):int(cY+height), max(0, int(cX)):int(cX+width), :]
	subj = imgs[:, :fill.shape[1], :fill.shape[2], :]

	# print(canvas.shape, fill.shape, subj.shape, (boxX, boxY), (width/2, height/2), (cX, cY))
	try:
		assert fill.shape == subj.shape
	except:
		raise Exception('ERR: %s -> %s' % (subj.shape, fill.shape))

	fill[:, :, :, :] = subj[:, :, :, :]

	return canvas

def next_video_batch(refs, bsize=6, format='heatpaf'):
	brefs = refs[0][:bsize]
	refs[0] = refs[0][bsize:]

	if len(refs[0]) < bsize:
		shuffed = refs[1]
		shuffle(shuffed)
		refs[0] = shuffed

	videos = []
	masks = [[], []]
	targets = [[], []]

	for ref in brefs:
		imgs, heats, pafs, mask_heats, mask_pafs = [], [], [], [], []
		for frame_ii in range(len(ref['frames'])):
			img = imread(ref['frames'][frame_ii])
			imgs.append(img)

			width, height = int(img.shape[0] / 8), int(img.shape[1] / 8)

			heats.append(create_heatmap(19, width, height, [ref['coco_coords'][frame_ii]], sigma=7.0, stride=8))
			pafs.append(create_paf(38, width, height, [ref['coco_coords'][frame_ii]], threshold=1.0, stride=8))

			mask_heats.append(create_mask(19, width, height, ref['boxes'][frame_ii]))
			mask_pafs.append(create_mask(38, width, height, ref['boxes'][frame_ii]))

		randZoom = random.uniform(0.5, 1.5)
		# randX = random.uniform(-92, 92)
		# randY = random.uniform(-92, 92)
		randX = 0
		randY = 0

		imgs = augment(imgs, randZoom, randX, randY, ref['boxes'], stride=1)
		mask_heats = augment(mask_heats, randZoom, randX, randY, ref['boxes'], stride=8)
		mask_pafs = augment(mask_pafs, randZoom, randX, randY, ref['boxes'], stride=8)
		heats = augment(heats, randZoom, randX, randY, ref['boxes'], stride=8)
		pafs = augment(pafs, randZoom, randX, randY, ref['boxes'], stride=8)

		videos.append(imgs)
		masks[0].append(mask_heats)
		masks[1].append(mask_pafs)
		targets[0].append(heats)
		targets[1].append(pafs)

	return (
		np.array(videos),
		np.array(masks[1]),
		np.array(masks[0]),
	), (
		np.array(targets[1]),
		np.array(targets[0]),
	)

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	dset = gather_videos(SEQ_LEN=4, still=False, shuffle=False)

	(frames, mask_pafs, mask_heats), (pafs, heats) = next_video_batch(dset, 1)

	FIRST = 0
	SEQ = 4
	plt.figure(figsize=(14, 10))

	for ii in range(SEQ):
		plt.subplot(3, SEQ, ii+1)
		plt.axis('off')
		img = frames[FIRST][ii].astype(np.float32)/256
		plt.imshow(img)
		msk = cv2.resize(mask_heats[FIRST][ii][:, :, FIRST], (0,0), fx=8, fy=8)
		plt.imshow(msk, alpha=0.1, vmin=0, vmax=1)

	for ii in range(SEQ):
		plt.subplot(3, SEQ, SEQ+ii+1)
		plt.axis('off')
		img = np.sum(heats[FIRST][ii][:, :, :-1], axis=-1).astype(np.float32)
		plt.imshow(img)
		plt.imshow(mask_heats[FIRST][ii][:, :, FIRST], alpha=0.15)

	for ii in range(SEQ):
		plt.subplot(3, SEQ, 2*SEQ+ii+1)
		plt.axis('off')
		sy, sx, _ = pafs[FIRST][ii].shape
		canvas = np.zeros((sy, sx))
		for dd in range(38):
			plane = pafs[FIRST][ii][:, :, dd]
			canvas[plane > 0] = plane[plane > 0]
		img = canvas.astype(np.float32)
		plt.imshow(img)
		plt.imshow(mask_heats[FIRST][ii][:, :, FIRST], alpha=0.15)

	plt.savefig('sample-dataset.png', bbox_inchex='tight')
	plt.close()

	plt.figure(figsize=(14, 10))
	for ii in range(38):
		plt.subplot(5, 8, ii+1)
		plt.axis('off')
		plt.imshow(mask_pafs[FIRST][FIRST][:, :, ii])

	plt.savefig('sample-paf-masks.png', bbox_inchex='tight')
	plt.close()
