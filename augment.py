
import os, sys, math
import numpy as np
import cv2
from cv2 import imread
from scipy.io import loadmat
import scipy.ndimage as ndimage
from random import shuffle, randint

sys.path.append('../tf/ver1')
from training.label_maps import create_heatmap, create_paf
from training.dataflow import JointsLoader

DATA_DIR = "/beegfs/ua349/lstm/Penn_Action"

frames_dir = DATA_DIR + '/frames'
labels_dir = DATA_DIR + '/labels'

def approx_neck(coords):
	NOSE_IND = 0
	NECK_IND = 1
	rsho, lsho, nose = coords[2], coords[5], coords[NOSE_IND]

	xavg, yavg = [], []
	if nose is None:
		if rsho is None or lsho is None:
			# one of the shoulders missing - too few points
			return

		xavg = [rsho[0], lsho[0]]
		yavg = [rsho[1], lsho[1]]
	else:
		if rsho is None and lsho is None:
			# both shoulders missing - neck alone not enough...
			return

		for point in [rsho, lsho]:
			if point is not None:
				xavg.append(point[0])
				yavg.append(point[1])
		xavg.append(nose[0])

	assert len(xavg) > 0 and len(yavg) > 0

	coords[NECK_IND] = [
		np.mean(xavg), # avg with nose X pos in x axis
		np.mean(yavg),
	]

NOSE_IND = 0
EYE_INDS = [14, 15]
EAR_INDS = [16, 17]
PENN_MISSING = EYE_INDS + EAR_INDS
# def approx_eyes(coords):
# 	if rsho is None or lsho is None:
# 		return None

# 	for ent in [rsho[0], rsho[1], lsho[0], lsho[1]]:
# 		if ent == -1:
# 			return None

# 	coords[NECK_IND] = [
# 		(rsho[0] + lsho[0])/2,
# 		(rsho[1] + lsho[1])/2
# 	]

def box_center(box):
	return (box[2] + box[0]) / 2, (box[3] + box[1]) / 2

def create_mask(dims, height, width, bbox, stride=8, pad=16):
	pad = int(pad / stride)
	x0, y0, xf, yf = (np.array(bbox) / stride)
	canvas = np.zeros((height, width, dims))
	canvas[max(int(y0)-pad, 0):int(yf)+pad, max(int(x0)-pad, 0):int(xf)+pad] = 1
	assert np.max(canvas) > 0
	return canvas

def shape_image(imgs, bbox, spec, stride=1):
	zoom, rotate, xoff, yoff, randFlip = spec
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


	if randFlip:
		canvas = np.flip(canvas, 2)

	# imsize = canvas.shape[1:3]
	# pivot = np.array(imsize) / 2 + np.array([yoff, xoff])
	# padX = [imsize[1] - int(pivot[1]), int(pivot[1])]
	# padY = [imsize[0] - int(pivot[0]), int(pivot[0])]
	# padded = np.pad(canvas, [[0, 0], padY, padX, [0, 0]], 'constant')
	# padded = ndimage.rotate(
	# 	padded.astype(np.float32),
	# 	rotate, axes=(2, 1), reshape=False).astype(imgs.dtype)

	# canvas = padded[:, padY[0]:-padY[1], padX[0]:-padX[1], :]

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
	zoom, rotate, xoff, yoff, randFlip = spec
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
		# jx, jy = rotate_around_point_highperf(
		# 	(jx, jy),
		# 	math.pi * rotate/180,
		# 	(0, 0)) # rotate at origin

		jx += canv_width/2 # return it to canvas center
		jy += canv_height/2
		jx += xoff # offset
		jy += yoff

		modded.append([jx, jy])

	modded = np.array(modded)

	if randFlip:
		for ii in range(len(modded)):
			if modded[ii] is not None:
				modded[ii][0] = canv_width - modded[ii][0] # flip the x coord across canvas

	return modded

def crop(imgs, tosize):
	imsize = imgs[0].shape

	canvas = np.zeros(
		(len(imgs), max(imsize[0], tosize), max(imsize[1], tosize), imsize[-1]),
		dtype=imgs[0].dtype)
	dy = int((canvas.shape[1] - imsize[0]) / 2)
	dx = int((canvas.shape[2] - imsize[1]) / 2)
	assert dy >= 0
	assert dx >= 0
	canvas[:, dy:dy+imsize[0], dx:dx+imsize[1], :] = imgs

	dy = int((canvas.shape[1] - tosize) / 2)
	dx = int((canvas.shape[2] - tosize) / 2)
	assert dy >= 0
	assert dx >= 0

	return canvas[:, dy:dy+tosize, dx:dx+tosize, :].copy()

def load_refs(batch_refs):
	videos = []
	masks = [[], []]
	targets = [[], []]

	for refs in batch_refs:
		zipped, augment = refs
		frames, boxes, _, coco_coords = zip(*zipped)
		imgs, heats, pafs, mask_heats, mask_pafs = [], [], [], [], []

		imgs = [imread(path) for path in frames]
		imsize = imgs[0].shape
		imgs = shape_image(imgs, boxes, augment)

		width, height = int(imgs[0].shape[0] / 8), int(imgs[0].shape[1] / 8)
		for frame_ii in range(len(frames)):
			coords = shape_coords(
				coco_coords[frame_ii],
				boxes[frame_ii],
				imsize, augment)
			assert len(coords) == len(coco_coords[frame_ii])
			heats.append(create_heatmap(19, width, height, [coords], sigma=7.0, stride=8))
			paf = create_paf(19, width, height, [coords], threshold=1.0, stride=8)
			assert paf.shape[-1] == 38
			pafs.append(paf)

			heat_mask = create_mask(19, width, height, boxes[frame_ii])
			mask_heats.append(heat_mask)

			for ind in PENN_MISSING:
				heat_mask[:, :, ind] = 0

			paf_mask = create_mask(38, width, height, boxes[frame_ii])
			mask_pafs.append(paf_mask)

			for ind, (j_idx1, j_idx2) in enumerate(JointsLoader.joint_pairs):
				if j_idx1 in PENN_MISSING or j_idx2 in PENN_MISSING:
					paf_mask[:, :, ind] = 0

		mask_heats = shape_image(mask_heats, boxes, augment, stride=8)
		mask_pafs = shape_image(mask_pafs, boxes, augment, stride=8)

		imgs = crop(imgs, 368)
		mask_heats = crop(mask_heats, 46)
		mask_pafs = crop(mask_pafs, 46)
		heats = crop(heats, 46)
		pafs = crop(pafs, 46)

		videos.append(imgs)
		masks[0].append(mask_heats)
		masks[1].append(mask_pafs)
		targets[0].append(heats)
		targets[1].append(pafs)

	return [
		np.array(videos),
		np.array(masks[1]),
		np.array(masks[0]),
	], [
		np.array(targets[1]),
		np.array(targets[0]),
	]