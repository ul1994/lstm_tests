DATA_DIR = "/beegfs/ua349/lstm/Penn_Action"

frames_dir = DATA_DIR + '/frames'
labels_dir = DATA_DIR + '/labels'

from .template import Dataset
import os, sys
from scipy.io import loadmat
import numpy as np
from numpy.random import shuffle as shuff
import random
import cv2
import scipy
from scipy.ndimage import gaussian_filter as blur
import json

SEQ_LEN = 6

# [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
coco_incl = [0, None, 1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12, None, None, None, None, None]
# TODO: 19th entry should be a maskish thing...?


def crop_heatmaps(coords, meta, size=46, image=None):
	ratio, (x0, y0) = meta

	shrink = size/368

	heats = np.zeros((size, size, len(coords)))

	for ii, (x, y) in enumerate(coords):
		canvas = np.zeros((size,size))

		if x <= 1 or y <= 1:
			# this coord does not exist in penn database
			continue

		yi = (y*ratio)+y0
		xi = (x*ratio)+x0
		try:
			assert yi < 368
			assert xi < 368
		except:
			print(image.shape)
			print(x, y)
			print(ratio, x0, y0)
			print(xi, yi)
			assert False

		canvas[int(yi * shrink), int(xi * shrink)] = 1

		canvas = blur(canvas, sigma=1)
		canvas /= np.max(canvas)

		heats[:, :, ii] = canvas

	return heats

def crop_vectmaps(coords, size=46, dims=38):
	canvas = np.zeros((size,size, dims))
	# for (x, y) in coords:
	# 	canvas[y, x] = 1
	# canvas = blur(canvas, sigma=4)
	# canvas /= np.max(canvas)
	return canvas


def crop_image(img, pad=368):

	maxdim = max(img.shape[0], img.shape[1])
	ratio = pad / maxdim
	img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)

	x0, y0 = 0, 0
	if img.shape[0] >= img.shape[1]:
		x0 = int((pad - img.shape[1])/2)
	else:
		y0 = int((pad - img.shape[0])/2)

	canvas = np.zeros((pad, pad, 3), dtype=np.uint8)
	canvas[y0:y0+img.shape[0], x0:x0+img.shape[1]] = img

	return canvas, (ratio, (x0, y0))

def get_images(batch):
	imgs = []
	heats = []
	vects = []
	for ref in batch:
		# ind = random.randint(0, len(ref['frames']) - 1)
		ind = 0
		image = cv2.imread(ref['frames'][ind])
		im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		coords = ref['coco_coords'][ind]
		for ii, (x, y) in enumerate(coords):
			try:
				assert x < image.shape[1]
				assert y < image.shape[0]
			except:
				print('OOB: (%d %d) in [%d %d] : %s' % (
					x, y, image.shape[1], image.shape[0],
					ref['frames'][ind]))
				coords[ii] = (min(x, image.shape[1]-1), min(y, image.shape[0]-1))

		im, meta = crop_image(im)
		imgs.append(im)
		heatmap = crop_heatmaps(coords, meta, image=image)
		heats.append(heatmap)
		vectmap = crop_vectmaps(coords)
		vects.append(vectmap)
	return imgs, heats, vects

def collect_refs(shuffle=True, reserve=64):
	refs = []
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
				# 'labels': [],
				'boxes': [],
				# 'local_coords': [],
				'coords': [],
				'coco_coords': [],
				'visibility': [],
			}


			if bii + SEQ_LEN >= len(imgs):
				rng = range(len(imgs) - SEQ_LEN, len(imgs))
			else:
				rng = range(bii, bii+SEQ_LEN)
			assert rng is not None

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
				box = [val.item() for val in list(box)] # just use listtype

				ref['boxes'].append(box)
				coords = [(x.item(), y.item()) for (x, y) in zip(list(xs[ii]), list(ys[ii]))]
				ref['coords'].append(coords)
				coco_coords = [(-1, -1) if ind is None else coords[ind] for ind in coco_incl]
				ref['coco_coords'].append(coco_coords)
				ref['visibility'].append([val.item() for val in list(vis[ii])])

			assert len(ref['frames']) == SEQ_LEN
			refs.append(ref)

	if shuffle:
		shuff(refs)

		if reserve is not None:
			tr = refs[:-reserve]
			tt = refs[-reserve:]
			dset = [tr, tt, tr, tt]
			for item in dset:
				assert len(dset) != 0
			return dset
		else:
			return refs

	return refs




class Penn(Dataset):

	refs = []

	def __init__(self):
		# if os.path.isfile('.cache_penn.json'):
		# 	with open('.cache_penn.json') as fl:
		# 		refs = json.load(fl)
		# 	print('> Loaded from json...')
		# else:
		self.refs = collect_refs()
			# with open('.cache_penn.json', 'w') as fl:
			# 	json.dump(refs, fl)

	def dequeue_batch(self, bsize=32, test=False):
		training, testing, __tr, __tt = self.refs
		if not test:
			batch = training[:bsize]
			self.refs[0] = training[bsize:]
			if len(self.refs[0]) < bsize:
				shuff(__tr)
				training = __tr
		else:
			batch = testing[:bsize]

		assert len(batch) != 0
		return batch


	def next_batch(self, bsize=32, test=False):
		batch = self.dequeue_batch(bsize, test)
		return get_images(batch)
