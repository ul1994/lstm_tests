
import os, sys
from interface import MaskRCNN
from cv2 import imread, imwrite
import numpy as np
import cv2
import time

if __name__ == '__main__':
	dpath = '/beegfs/ua349/lstm/Penn_Action/frames'

	vidfolders = os.listdir(dpath)
	vidfolders = sorted(vidfolders, key=lambda val: int(val))

	center_frames = []

	for folder in vidfolders:
		imgs = os.listdir('%s/%s' % (dpath, folder))
		imgs = sorted(imgs, key=lambda path: int(path.split('.')[0]))
		midind = int(len(imgs)/2)
		midfile = '%s/%s/%s' % (dpath, folder, imgs[midind])

		img = cv2.cvtColor(imread(midfile), cv2.COLOR_BGR2RGB)
		center_frames.append([img, midfile])

	model = MaskRCNN()

	for ii, (img, impath) in enumerate(center_frames):
		t0 = time.time()
		results = model.predict(img)
		npeople = np.sum([1 if one.name is 'person' else 0 for one in results])
		print('[%d/%d]: %.1fs' % (ii+1, len(center_frames), time.time() - t0), npeople, impath)