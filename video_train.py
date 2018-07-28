
import matplotlib as mpl
mpl.use('Agg')

import re
import math
import os, sys
import numpy as np
from functools import partial
import argparse

import tensorflow as tf
from tensorpack.dataflow.common import BatchData, MapData
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Lambda, LSTM, Reshape, TimeDistributed, Dense
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.utils import multi_gpu_model as multigpu

from components import *
sys.path.append('/scratch/ua349/pose/tf/ver1')
from training.optimizers import MultiSGD
from training.dataset import get_dataflow

from snapshot import Snapshot
from still_train import *
from models import *
from generate_video import *

def mix_gen(df, dset, batch_size, format='last', every=2):
	while True:
		for (inp, out) in df.get_data():
			(video_frames, mask_pafs, mask_heats), (video_pafs, video_heats) = next_video_batch(dset, batch_size)

			numtargs = 6 * 2
			videos, masks = [], [[], []]
			targets = [list() for ii in range(numtargs)]

			for batch_ii in range(batch_size):
				if batch_ii % every == 0:
					videos.append(inp[0][batch_ii])
					masks[0].append(inp[1][batch_ii])
					masks[1].append(inp[2][batch_ii])
					for jj in range(numtargs):
						targets[jj].append(out[jj][batch_ii])
				else:
					videos.append(video_frames[batch_ii])
					masks[0].append(mask_pafs[batch_ii])
					masks[1].append(mask_heats[batch_ii])
					for jj in range(numtargs):
						targets[jj].append(targs[batch_ii])

			videos = np.array(videos)
			masks[0] = np.array(masks[0])
			masks[1] = np.array(masks[1])
			for targ_ii in range(numtargs):
				targets[targ_ii] = np.array(targets[targ_ii])
				if format == 'last' and targ_ii > 4 * 2:
					targets[targ_ii] = targets[targ_ii][-1] # reduce to last frame

			assert len(targets) == numtargs
			yield [videos, masks[0], masks[1]], targets

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, required=True)
	parser.add_argument('--arch', default='conv_v1', type=str)
	parser.add_argument('--dataset', default='train', type=str)
	parser.add_argument('--format', default='sequence', required=True, type=str)

	parser.add_argument('--epochs', default=5, type=int)
	parser.add_argument('--time_steps', default=4, type=int)

	parser.add_argument('--gpus', default=1, type=int)
	parser.add_argument('--batch', default=6, type=int)
	parser.add_argument('--count', default=False, type=bool)
	args = parser.parse_args()

	__format = args.format.split('|')
	batch_size = args.batch

	model = MODELS[args.arch](time_steps=args.time_steps)

	if args.count:
		model.summary()
		exit()

	if args.gpus > 1:
		batch_size = args.batch * args.gpus
		model = multigpu(model, gpus=args.gpus)

	lr_multipliers = get_lr_multipliers(model)

	iterations_per_epoch = 110000 // batch_size
	_step_decay = partial(step_decay,
						iterations_per_epoch=iterations_per_epoch
						)
	lrate = LearningRateScheduler(_step_decay)

	# sgd optimizer with lr multipliers
	multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0,
						nesterov=False, lr_mult=lr_multipliers)


	loss_funcs = get_loss_funcs(batch_size)
	model.compile(loss=loss_funcs, optimizer=multisgd, metrics=["accuracy"])

	load_any_weights(model, multigpu=args.gpus > 1)


	df = get_dataflow(
		annot_path='%s/annotations/person_keypoints_%s2017.json' % (DATA_DIR, args.dataset),
		img_dir='%s/%s2017/' % (DATA_DIR, args.dataset))
	train_samples = df.size()
	print('Collected %d val samples...' % train_samples)
	train_df = batch_dataflow(df, batch_size, time_steps=args.time_steps, format=__format)

	dset = gather_videos(SEQ_LEN=args.time_steps, still=False)
	train_gen = mix_gen(train_df, dset, batch_size)

	model.fit_generator(train_gen,
		steps_per_epoch=train_samples // batch_size,
		epochs=args.epochs,
		callbacks=[lrate, Snapshot(args.name, train_gen, __format, stills=True)],
		use_multiprocessing=False,
		initial_epoch=0,
		verbose=1)

