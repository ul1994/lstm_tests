# import matplotlib as mpl
# mpl.use('Agg')

import os, sys, math
import cv2
from scipy.io import loadmat
from random import shuffle
sys.path.append('../tf/ver1')
from training.label_maps import create_heatmap, create_paf
from training.dataflow import JointsLoader
import scipy.ndimage as ndimage

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
				box = [val+1 for val in box]
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

def box_center(box):
	return (box[2] + box[0]) / 2, (box[3] + box[1]) / 2

def create_mask(dims, height, width, bbox, stride=8, pad=16):
	pad = int(pad / stride)
	x0, y0, xf, yf = (np.array(bbox) / stride)
	canvas = np.zeros((height, width, dims))
	canvas[int(y0)-pad:int(yf)+pad, int(x0)-pad:int(xf)+pad] = 1
	assert np.max(canvas) > 0
	return canvas

def shape_image(imgs, bbox, spec, stride=1):
	zoom, rotate, xoff, yoff = spec
	FIRST = 0
	xoff = xoff / stride
	yoff = yoff / stride

	imgs = np.array(imgs)
	sized = [cv2.resize(img.astype(np.float32), (0,0), fx=zoom, fy=zoom) for img in imgs]
	sized = np.array(sized).astype(imgs[FIRST].dtype)

	canvas = np.zeros(imgs.shape, dtype=imgs.dtype)
	canv_width = canvas[FIRST].shape[1]
	canv_height = canvas[FIRST].shape[0]
	sizedBox = np.array(bbox[FIRST]) / stride # bbox affected by zoom
	x0, y0, xf, yf = sizedBox

	boxX, boxY = (x0 + xf) / 2, (y0 + yf) / 2
	cX = -boxX # center p0 at origin
	cY = -boxY
	cX *= zoom # zoom at origin
	cY *= zoom
	cX += canv_width/2 # return p0 to canvas center
	cY += canv_height/2
	cX += xoff # apply offset
	cY += yoff

	width = sized[FIRST].shape[1]
	height = sized[FIRST].shape[0]

	fill_Y0 = max(0, cY)
	fill_YF = min(canv_height, cY+height)
	fill_X0 = max(0, cX)
	fill_XF = min(canv_width, cX+width)

	subj_Y0 = max(0, -cY)
	subj_X0 = max(0, -cX)

	fill = canvas[:, int(fill_Y0):int(fill_YF), int(fill_X0):int(fill_XF), :]
	subj = sized[:, int(subj_Y0):int(subj_Y0+fill.shape[1]), int(subj_X0):int(subj_X0+fill.shape[2]), :]

	try: assert fill.shape == subj.shape
	except:
		print(imgs[FIRST].shape, sized[FIRST].shape)
		print(cX, cY)
		print(xoff, yoff)
		print(fill_YF, fill_Y0, subj_YF, subj_Y0)
		print(fill_XF, fill_X0, subj_XF, subj_X0)
		raise Exception('ERR: %s -> %s' % (subj.shape, fill.shape))

	fill[:, :, :, :] = subj[:, :, :, :]


	imsize = canvas.shape[1:3]
	pivot = np.array(imsize) / 2 + np.array([yoff, xoff])
	padX = [imsize[1] - int(pivot[1]), int(pivot[1])]
	padY = [imsize[0] - int(pivot[0]), int(pivot[0])]
	padded = np.pad(canvas, [[0, 0], padY, padX, [0, 0]], 'constant')
	padded = ndimage.rotate(
		padded.astype(np.float32),
		rotate, axes=(2, 1), reshape=False).astype(imgs.dtype)
	# for frame_ii in range(len(padded)):
		# padded[frame_ii, :, :, :] = ndimage.rotate(
		# 	padded[frame_ii].astype(np.float32),
		# 	rotate, reshape=False).astype(imgs.dtype)

	canvas = padded[:, padY[0]:-padY[1], padX[0]:-padX[1], :]

	return canvas

def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
	"""Rotate a point around a given point.

	I call this the "high performance" version since we're caching some
	values that are needed >1 time. It's less readable than the previous
	function but it's faster.
	"""
	x, y = xy
	offset_x, offset_y = origin
	adjusted_x = (x - offset_x)
	adjusted_y = (y - offset_y)
	cos_rad = math.cos(radians)
	sin_rad = math.sin(radians)
	qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
	qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

	return qx, qy

def shape_coords(coords, bbox, imsize, spec):
	zoom, rotate, xoff, yoff = spec
	modded = []

	sizedBox = np.array(bbox) # bbox affected by zoom
	x0, y0, xf, yf = sizedBox

	boxX, boxY = (x0 + xf) / 2, (y0 + yf) / 2
	canv_height, canv_width = imsize[:2]
	cX = canv_width / 2 - boxX
	cY = canv_height / 2 - boxY

	for point in coords:
		if point is None:
			modded.append(None)
			continue

		jx, jy = point

		jx -= boxX # send coords to origin centered at box
		jy -= boxY
		jx *= zoom # scale at origin
		jy *= zoom
		jx, jy = rotate_around_point_highperf(
			(jx, jy),
			math.pi * rotate/180,
			(0, 0)) # rotate at origin

		jx += canv_width/2 # return it to canvas center
		jy += canv_height/2
		jx += xoff # offset
		jy += yoff

		modded.append([jx, jy])

	modded = np.array(modded)

	return modded

def next_video_batch(refs, bsize=6, format='heatpaf', stop=False):
	brefs = refs[0][:bsize]
	if not stop:
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

		randZoom = random.uniform(0.33, 1.0)
		# randDeg = random.uniform(-45, 45)
		randDeg = 45
		randX = random.uniform(-96, 96)
		randY = random.uniform(-96, 96)

		spec = (randZoom, randDeg, randX, randY)

		imgs = [imread(path) for path in ref['frames']]
		imsize = imgs[0].shape
		imgs = shape_image(imgs, ref['boxes'], spec)

		width, height = int(imgs[0].shape[0] / 8), int(imgs[0].shape[1] / 8)
		for frame_ii in range(len(ref['frames'])):
			coords = shape_coords(
				ref['coco_coords'][frame_ii],
				ref['boxes'][frame_ii],
				imsize, spec)
			heats.append(create_heatmap(19, width, height, [coords], sigma=7.0, stride=8))
			pafs.append(create_paf(38, width, height, [coords], threshold=1.0, stride=8))

			mask_heats.append(create_mask(19, width, height, ref['boxes'][frame_ii]))
			mask_pafs.append(create_mask(38, width, height, ref['boxes'][frame_ii]))

		mask_heats = shape_image(mask_heats, ref['boxes'], spec, stride=8)
		mask_pafs = shape_image(mask_pafs, ref['boxes'], spec, stride=8)

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

	for tag in range(10):
		(frames, mask_pafs, mask_heats), (pafs, heats) = next_video_batch(dset, 1, stop=True)

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

		plt.savefig('samples/%d.png' % tag, bbox_inchex='tight')
		plt.close()

		# plt.figure(figsize=(14, 10))
		# for ii in range(38):
		# 	plt.subplot(5, 8, ii+1)
		# 	plt.axis('off')
		# 	plt.imshow(mask_pafs[FIRST][FIRST][:, :, ii])

		# plt.savefig('sample-paf-masks.png', bbox_inchex='tight')
		# plt.close()
