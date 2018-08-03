
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
import keras
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Lambda, LSTM, Reshape, TimeDistributed, Dense
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.utils import multi_gpu_model as multigpu

from components import *
sys.path.append('../tf/ver1')
from training.optimizers import MultiSGD
from training.dataset import get_dataflow

from snapshot import Snapshot
from still_train import *
from models import *
from generate_video import *

DATA_DIR = '/beegfs/ua349/lstm/coco'

# if this is true, acknowledge it (set to False) and reset lstm layers
RESET_FLAG = False
SKIP_FLAG = False

class ManualLSTMReset(keras.callbacks.Callback):
	def __init__(self):
		pass

	def on_batch_begin(self, batch, logs=None):
		global RESET_FLAG
		if RESET_FLAG:
			print()
			print(' [*] WARN: Resetting internal LSTM states...')
			# flag ack and reset lstms
			self.model.reset_states()
			RESET_FLAG = False

class LossAbort(keras.callbacks.Callback):
	# because we train long sequences of images (videos),
	#   a bad video can completely throw off the model
	# If loss inflates drastically, skip video
	prevLoss = None
	def on_batch_end(self, batch, logs=None):
		global SKIP_FLAG

		loss = logs['loss']

		if self.prevLoss is not None and loss > self.prevLoss * 1.5:
			SKIP_FLAG = True
			print()
			print(' [*] WARN: Loss jumped from %.1f -> %.1f. Skipping batch...' % (self.prevLoss, loss))

		self.prevLoss = loss

def stack_video_outputs(pafs, heats, format, size=6, collapse=2):
	if format == 'join':
		# reduction to last timestep
		return [
			pafs, heats,
			pafs, heats,
			pafs[:, -1], heats[:, -1],

			pafs[:, -1], heats[:, -1],
			pafs[:, -1], heats[:, -1],
			pafs[:, -1], heats[:, -1],
		]
	elif format == 'last':
		# only matching the final timestep outputs
		return [
			pafs[:, -1], heats[:, -1],
			pafs[:, -1], heats[:, -1],
			pafs[:, -1], heats[:, -1],

			pafs[:, -1], heats[:, -1],
			pafs[:, -1], heats[:, -1],
			pafs[:, -1], heats[:, -1],
		]
	else:
		raise Exception('FATAL: Unknown format: %s' % format)

def mix_gen(df, dset, batch_size, outformat='join', every=2):
	global RESET_FLAG
	global SKIP_FLAG

	while True:
		if SKIP_FLAG:
			SKIP_FLAG = False # ack
			dset.clear()

		inp, out, video_ended = dset.next_batch(bsize=batch_size)

		if video_ended:
			# mix in coco image dataset in between videos
			# FIXME: what happens when df runs out of data?
			RESET_FLAG = True # reset lstm for static videos
			dset.stream_status()

		RESET_FLAG = video_ended

		out = stack_video_outputs(out[0], out[1], outformat)

		yield inp, out

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, required=True)
	parser.add_argument('--arch', default='conv_v1', type=str)
	parser.add_argument('--dataset', default='train', type=str)
	parser.add_argument('--format', default='sequence', required=True, type=str)

	parser.add_argument('--speedup', default=1, type=int)
	parser.add_argument('--epochs', default=5, type=int)
	parser.add_argument('--iters', default=100 * 1000, type=int)
	parser.add_argument('--time_steps', default=4, type=int)

	parser.add_argument('--gpus', default=1, type=int)
	parser.add_argument('--gpu', default=None, type=int)
	parser.add_argument('--batch', default=6, type=int)
	parser.add_argument('--count', default=False, type=bool)
	args = parser.parse_args()

	if args.gpu is not None:
		os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % int(args.gpu)

	__format = args.format.split('|')
	batch_size = args.batch * args.gpus

	model = MODELS[args.arch](bsize=batch_size, time_steps=args.time_steps)

	if args.count:
		model.summary()
		exit()

	if args.gpus > 1:
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


	# df = get_dataflow(
	# 	annot_path='%s/annotations/person_keypoints_%s2017.json' % (DATA_DIR, args.dataset),
	# 	img_dir='%s/%s2017/' % (DATA_DIR, args.dataset))
	# train_samples = df.size()
	# print('Collected %d val samples...' % train_samples)
	# train_df = batch_dataflow(df, batch_size, time_steps=args.time_steps, format=__format)
	train_df = None

	dset = MultiVideoDataset(shuffle=True, bins=7, seqlen=args.time_steps, speedup=args.speedup)
	train_gen = mix_gen(train_df, dset, batch_size, outformat=__format[1])

	cblist = [lrate, Snapshot(args.name, train_gen, __format), ManualLSTMReset(), LossAbort()]
	# cblist = [lrate, ManualLSTMReset()]

	tf.logging.set_verbosity(tf.logging.ERROR)
	model.fit_generator(train_gen,
		steps_per_epoch=args.iters,
		epochs=args.epochs,
		callbacks=cblist,
		use_multiprocessing=False,
		initial_epoch=0,
		verbose=1)

