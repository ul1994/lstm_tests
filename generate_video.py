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

sys.path.append('../tf/ver1')
from training.label_maps import create_heatmap, create_paf
from training.dataflow import JointsLoader


DATA_DIR = "/beegfs/ua349/lstm/Penn_Action"

frames_dir = DATA_DIR + '/frames'
labels_dir = DATA_DIR + '/labels'

class Video:

	def __init__(self, seqlen, speedup=1):
		self.index = 0

		self.frames = []
		self.boxes = []
		self.coords = []
		self.coco_coords = []

		self.bucketid = -1
		self.speedup = speedup
		self.seqlen = seqlen
		self.augment = None

	def add_frame(self, img, box, coords, coco_coords):
		self.frames.append(img)
		self.boxes.append(box)
		self.coords.append(coords)
		self.coco_coords.append(coco_coords)

	def ended(self):
		return self.index + self.seqlen * self.speedup >= len(self.frames)

	def next_segment(self):
		assert self.augment is not None

		seg = self.zipped[self.index : self.index+self.seqlen * self.speedup : self.speedup]
		try:
			assert len(seg) == self.seqlen
		except:
			raise Exception('%d != %d' % (len(seg), self.seqlen))
		self.index += self.seqlen * self.speedup

		return seg, self.augment

	def reset(self):
		self.index = 0

		randZoom = uniform(0.5, 1.0)
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
			zip(self.frames,self.boxes,self.coords,self.coco_coords))
		assert len(self.zipped) == len(self.frames)

		self.augment = (randZoom, randDeg, randX, randY, randFlip)

class MultiVideoDataset:
	def __init__(self, seqlen=4, speedup=2, shuffle=True, bins=7, plot_buckets=False):
		refs = []
		self.speedup = speedup

		frame_folders = os.listdir(frames_dir)
		videos = []
		for __fii, fldr in enumerate(sorted(frame_folders, key=lambda val: int(val))):

			sys.stdout.write('%d/%d\r' % (__fii, len(frame_folders)))
			sys.stdout.flush()

			flpath = '%s/%s' % (frames_dir, fldr)
			imgs = [fl for fl in os.listdir(flpath)]
			imgs = sorted(imgs, key=lambda val: int(val.split('.')[0]))

			lblpath = '%s/%s' % (labels_dir, fldr)
			mat = loadmat(lblpath)
			xs, ys, bbox = mat['x'], mat['y'], mat['bbox']
			vis, dim = mat['visibility'], mat['dimensions']

			vidobj = Video(speedup=speedup, seqlen=seqlen)
			videos.append(vidobj)

			assert len(vidobj.frames) == 0

			for frame_ii in range(len(imgs)):

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

				coco_coords = [None if ind is None else coords[ind] for ind in coco_incl]
				approx_neck(coco_coords)

				vidobj.add_frame(
					'%s/%s' % (flpath, imgs[frame_ii]),
					box,
					coords,
					coco_coords)

			assert len(vidobj.frames) == len(imgs)
			assert len(vidobj.boxes) == len(imgs)

		assert len(videos) > 0

		binsize = int(len(videos) / bins)

		if plot_buckets:
			vidlens = [len(obj.frames) for obj in videos]
			dist = np.zeros(np.max(vidlens) + 1)
			for ll in vidlens:
				dist[ll] += 1

			import matplotlib.pyplot as plt

			plt.figure(figsize=(14, 10))
			plt.title('Bucket size: %d' % binsize)
			plt.plot(dist)
			agg = 0
			for ii, unit in enumerate(dist):
				agg += unit
				if agg > binsize:
					agg = 0
					plt.plot([ii, ii], [0, 50], color='red')

			plt.show()

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
			for ii in range(bins): shuffle(buckets[ii])

		used = [list() for ii in range(bins)]
		streams = []

		self.streams = streams
		self.buckets = buckets
		self.used = used

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
		self.used[bind] += videos

		return videos

	def next_batch(self, bsize=6, format='heatpaf', stop=False):

		if len(self.streams) > 0 and bsize != len(self.streams):
			print(' [!] WARN: batch size changed!')
			self.clear()

		vid_ended = any([vid.ended() for vid in self.streams]) or len(self.streams) == 0
		if vid_ended:
			# TODO: collect the videos that are done to used buckets
			# if videos are done or stream is empty, get a new set of videos
			bind = randint(0, len(self.buckets)-1)
			videos = self.sample_bucket(bind, bsize)

			# this will reset the frame index and also augmentation
			for vid in videos: vid.reset()

			# clear the stream - save consumed videos
			for vid in self.streams:
				self.used[vid.bucketid].append(vid)

			# activate fetched videos
			self.streams = videos

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
