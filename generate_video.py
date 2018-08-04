# import matplotlib as mpl
# mpl.use('Agg')

import os, sys, math
import numpy as np
import cv2
from cv2 import imread
from scipy.io import loadmat
import scipy.ndimage as ndimage
from random import shuffle, randint, uniform
from augment import *

shuff = shuffle

sys.path.append('../tf/ver1')
from training.label_maps import create_heatmap, create_paf
from training.dataflow import JointsLoader

# [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
penn2coco = [0, None, 1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12, None, None, None, None]
#                      0 1 2 3 4  5 6 7  8 9 10 11 12 13
# order_to_pretrain = [2 1 3 7 11 4 8 12 5 9 13 6 10 14];

jhmdb2coco = [2, None, 1, 7, 4, 3, 11, 8, 12, 9, 6, 5, 13, 10, None, None, None, None]
class Video:

	def __init__(self, seqlen, speedup=1):
		self.index = 0
		self.limit = None

		self.source = None

		self.frames = []
		self.boxes = []
		self.masks = []
		self.coords = []
		self.coco_coords = []

		self.bucketid = -1
		self.speedup = speedup
		self.seqlen = seqlen
		self.augment = None

	def add_frame(self, img, box, mask, coords, coco_coords):
		self.frames.append(img)
		self.boxes.append(box)
		self.masks.append(mask)
		self.coords.append(coords)
		self.coco_coords.append(coco_coords)

	def ended(self):
		if self.limit is not None:
			print('==============================')
			print('LIMIT', self.index, self.limit)
			return self.index >= self.limit

		return self.index + self.seqlen * self.speedup >= len(self.zipped)

	def next_segment(self):
		assert self.augment is not None

		seg = self.zipped[self.index : self.index+self.seqlen * self.speedup : self.speedup]
		try:
			assert len(seg) == self.seqlen
		except:
			raise Exception('%d != %d' % (len(seg), self.seqlen))
		self.index += self.speedup

		return seg, self.augment

	def reset(self):
		self.index = 0

		randZoom = uniform(0.75, 1.25)
		randFlip = uniform(0, 1) > 0.5
		randDeg = uniform(-45, 45)
		randX = uniform(-96, 96)
		randY = uniform(-96, 96)

		maxbox = [float('inf'), float('inf'), 0, 0] # x0, y0, xf, yf

		for (x0, y0, xf, yf) in self.boxes:
			if x0 < maxbox[0]: maxbox[0] = x0
			if y0 < maxbox[1]: maxbox[1] = y0
			if xf > maxbox[2]: maxbox[2] = xf
			if yf > maxbox[3]: maxbox[3] = yf

		# TODO: Instead of fitting to the max bounds, conserve the origin
		#           and use frame-by-frame masks which are available
		self.boxes = [maxbox for _ in self.boxes]

		self.zipped = list(
			zip(self.frames,self.boxes,self.masks,self.coords,self.coco_coords))
		assert len(self.zipped) == len(self.frames)

		self.augment = (randZoom, randDeg, randX, randY, randFlip)

def fetch_penn(seqlen, speedup):
	PENN_DIR = "/beegfs/ua349/lstm/Penn_Action"

	frames_dir = PENN_DIR + '/frames'
	labels_dir = PENN_DIR + '/labels'
	masks_dir = PENN_DIR + '/masks'

	frame_folders = os.listdir(frames_dir)
	videos = []
	for __fii, fldr in enumerate(sorted(frame_folders, key=lambda val: int(val))):

		sys.stdout.write('%d/%d\r' % (__fii, len(frame_folders)))
		sys.stdout.flush()

		flpath = '%s/%s' % (frames_dir, fldr)
		imgs = [fl for fl in os.listdir(flpath)]
		imgs = sorted(imgs, key=lambda val: int(val.split('.')[0]))

		mspath = '%s/%s' % (masks_dir, fldr)
		maskpaths = ['%s/%s' % (mspath, fl) for fl in os.listdir(mspath)]

		lblpath = '%s/%s' % (labels_dir, fldr)
		mat = loadmat(lblpath)
		xs, ys, bbox = mat['x'], mat['y'], mat['bbox']
		vis, dim = mat['visibility'], mat['dimensions']

		vidobj = Video(speedup=speedup, seqlen=seqlen)
		vidobj.source = 'penn'
		videos.append(vidobj)

		assert len(vidobj.frames) == 0

		for frame_ii in range(len(imgs)):
			mask = maskpaths[frame_ii]

			if frame_ii <= len(bbox) - 1:
				box = bbox[frame_ii]
			else:
				box = bbox[-1] # just sub in last frame
			box = [val+1 for val in box]

			coords = []
			for (xx, yy) in zip(xs[frame_ii], ys[frame_ii]):
				if xx - 1 <= 0 or yy - 1 <= 0:
					coords.append(None)
				else:
					coords.append((xx-1, yy-1))

			coco_coords = [None if ind is None else coords[ind] for ind in penn2coco]
			approx_neck(coco_coords)

			vidobj.add_frame(
				'%s/%s' % (flpath, imgs[frame_ii]),
				box,
				mask,
				coords,
				coco_coords)

		assert len(vidobj.frames) == len(imgs)
		assert len(vidobj.boxes) == len(imgs)
	return videos

def just_files(ls):
    return [fl for fl in ls if fl[0] is not '.']

def fetch_jhmdb(seqlen, speedup):
	JHMDB_PATH = '/beegfs/ua349/lstm/JHMDB'

	imfolder = '%s/Rename_Images' % (JHMDB_PATH)
	maskfolder = '%s/puppet_mask' % (JHMDB_PATH)
	jointfolder = '%s/joint_positions' % (JHMDB_PATH)

	frame_folders = os.listdir(imfolder)
	videos = []

	cats = os.listdir(imfolder)
	for __catii, catfolder in enumerate(sorted(just_files(cats))):
		sys.stdout.write('%d/%d\r' % (__catii+1, len(cats)))
		sys.stdout.flush()

		vidnames = os.listdir('%s/%s' % (imfolder, catfolder))
		for __fii, fldr in enumerate(sorted(just_files(vidnames))):

			flpath = '%s/%s/%s' % (imfolder, catfolder, fldr)
			imgs = [fl for fl in just_files(os.listdir(flpath))]
			imgs = sorted(imgs, key=lambda name: int(name.split('.')[0]))

			maskpath = '%s/%s/%s/puppet_mask.mat' % (maskfolder, catfolder, fldr)
			try:
				maskmat = loadmat(maskpath)
				masks = np.swapaxes(np.array(maskmat['part_mask']), 0, 2)
			except:
				print('No mask: %s' % maskpath)
				continue
				# raise Exception()

			lblpath = '%s/%s/%s/joint_positions.mat' % (jointfolder, catfolder, fldr)
			mat = loadmat(lblpath)
			joints = np.swapaxes(np.array(mat['pos_img']), 0, 2)

			vidobj = Video(speedup=speedup, seqlen=seqlen)
			vidobj.source = 'jhmdb'
			videos.append(vidobj)

			# print(len(imgs), masks.shape)
			imgs = imgs[:len(masks)] # some vids are incomplete
			try:
				assert len(vidobj.frames) == 0
				assert len(imgs) == masks.shape[0]
				assert len(imgs) == joints.shape[0]
			except:
				print(len(imgs), mat.keys())
				print(masks.shape, joints.shape)
				raise Exception()

			for frame_ii in range(len(imgs)):

				mask = masks[frame_ii]
				coords = joints[frame_ii] # (15, 2)
				# print(coords)

				coco_coords = [None if ind is None else coords[ind] for ind in jhmdb2coco]
				approx_neck(coco_coords)

				vidobj.add_frame(
					'%s/%s' % (flpath, imgs[frame_ii]),
					mask,
					coords,
					coco_coords)

			assert len(vidobj.frames) == len(imgs)
			assert len(vidobj.boxes) == len(imgs)
	return videos

class MultiVideoDataset:
	def __init__(self,
					seqlen=4,
					limit_playback=None,
					speedup=2,
					shuffle=True,
					bins=7,
					plot_buckets=False,
					source='penn',
					vary_playback=2):

		self.speedup = speedup
		self.limit_playback = limit_playback
		self.vary_playback = vary_playback

		if source == 'jhmdb':
			videos = fetch_jhmdb(seqlen, speedup)
		elif source == 'penn':
			videos = fetch_penn(seqlen, speedup)
		else:
			raise Exception(' [!] FATAL: Unknown datasource')

		assert len(videos) > 0

		binsize = int(len(videos) / bins)

		if plot_buckets:
			vidlens = [len(obj.frames) for obj in videos]
			dist = np.zeros(np.max(vidlens) + 1)
			for ll in vidlens:
				dist[ll] += 1

			import matplotlib.pyplot as plt

			plt.figure(figsize=(14, 6))
			plt.title('Bucket size: %d  Videos: %d' % (binsize, len(vidlens)))
			plt.scatter(
				[ii for ii in range(len(dist)) if dist[ii] > 0],
				[val for val in dist if val > 0])

			agg = 0
			for ii, unit in enumerate(dist):
				agg += unit
				if agg > binsize:
					agg = 0
					plt.plot([ii, ii], [0, 50], color='red')

			plt.show()
			plt.close()

		buckets = []
		videos = sorted(videos, key=lambda obj: len(obj.frames))
		for ii in range(0, binsize * bins, binsize):
			end = ii + binsize
			if len(videos) - ii < binsize:
				end = len(videos)
			buckets.append(videos[ii:end])

		assert len(buckets) == bins

		# indicate which video belongs to which
		for bii in range(bins):
			for vii in range(len(buckets[bii])):
				buckets[bii][vii].bucketid = bii

		if shuffle: # all buckets are initially shuffled
			for ii in range(bins): shuff(buckets[ii])

		used = [list() for ii in range(bins)]
		streams = []

		self.streams = streams
		self.buckets = buckets
		self.used = used
		self.first_batch = True

	def sample_bucket(self, bind, bsize):
		if len(self.buckets[bind]) < bsize:
			# reset bucket if almost empty
			print(' [!] WARN: Reached end of bucket: %d!' % bind)
			ulen, blen = len(self.used[bind]), len(self.buckets[bind])
			full_bucket = self.used[bind] + self.buckets[bind]
			shuffle(full_bucket)
			self.buckets[bind] = full_bucket # refilling
			assert len(self.buckets[bind]) == ulen + blen
			self.used[bind] = []

		videos = self.buckets[bind][:bsize]
		self.buckets[bind] = self.buckets[bind][bsize:] # dequeue videos

		return videos

	def stream_status(self):
		fresh = 0
		for buck in self.buckets:
			fresh += len(buck)
		used = 0
		for buck in self.used:
			used += len(buck)
		print(' [*] Usage: %d/%d' % (used, used + fresh))

		for ii, video in enumerate(self.streams):
			print('     [*] Stream %d: %s' % (ii, video.frames[0]))

	def next_batch(self, bsize=6, format='heatpaf', stop=False):

		if len(self.streams) > 0 and bsize != len(self.streams):
			print(' [!] WARN: batch size changed!')
			self.clear()

		vid_ended = any([vid.ended() for vid in self.streams]) or len(self.streams) == 0
		if vid_ended:
			# clear the stream - save consumed videos
			for vid in self.streams:
				self.used[vid.bucketid].append(vid)

			# get a new set of videos
			bind = randint(0, len(self.buckets)-1)
			videos = self.sample_bucket(bind, bsize)

			# activate fetched videos
			# this will reset the frame index and also augmentation
			for vid in videos: vid.reset()

			# limit and jitter playback length
			if self.limit_playback is not None:
				limitlen = (self.limit_playback + randint(0, self.vary_playback)) * self.speedup
				for vid in videos:
					vidlen = len(vid.frames)
					randend = vidlen - limitlen
					randstart = randint(0, randend - 1)
					vid.index = int(randstart)
					vid.limit = int(randstart + limitlen)

			self.streams = videos

			if self.first_batch:
				vid_ended = False # don't trigger a reset w/ first batch
				self.first_batch = False

		# list of (segment, augment) tuples
		batch_refs = []
		for video in self.streams:
			batch_refs.append(video.next_segment())

		assert len(batch_refs) > 0

		ins, outs = load_refs(batch_refs)
		return ins, outs, vid_ended

	def clear(self):
		# clear the stream - save consumed videos
		for vid in self.streams:
			self.used[vid.bucketid].append(vid)

		self.streams = []


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	dset = gather_videos(SEQ_LEN=4, still=False, shuffle=False)

	batch = []
	for bucket in range(dset.buckets):
		for video in bucket:
			sample = video.frames[1]
			box = video.boxes[1]

			counter