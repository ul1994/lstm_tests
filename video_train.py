
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

from models import *

# base_lr = 2e-5 # 2e-5
base_lr = 0.00004
momentum = 0.9
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma

def batch_dataflow(df, batch_size, time_steps=4, num_stages=6):
		"""
		The function builds batch dataflow from the input dataflow of samples

		:param df: dataflow of samples
		:param batch_size: batch size
		:return: dataflow of batches
		"""
		df = BatchData(df, batch_size, use_list=False)
		df = MapData(df, lambda x: (
			[
				np.stack([x[0]] * time_steps, axis=1),
				np.stack([x[2]] * time_steps, axis=1)
			],
			[np.stack([x[4]] * time_steps, axis=1)] * num_stages))
		df.reset_state()
		return df

def get_lr_multipliers(model):
	"""
	Setup multipliers for stageN layers (kernel and bias)

	:param model:
	:return: dictionary key: layer name , value: multiplier
	"""
	lr_mult = dict()
	for layer in model.layers:

		if isinstance(layer, TimeDistributed) and isinstance(layer.layer, Conv2D):
			layer = layer.layer
			# stage = 1
			if re.match("Mconv\d_stage1.*", layer.name):
				kernel_name = layer.weights[0].name
				bias_name = layer.weights[1].name
				lr_mult[kernel_name] = 1
				lr_mult[bias_name] = 2

			# stage > 1
			elif re.match("Mconv\d_stage.*", layer.name):
				kernel_name = layer.weights[0].name
				bias_name = layer.weights[1].name
				lr_mult[kernel_name] = 4
				lr_mult[bias_name] = 8

			# vgg
			else:
				kernel_name = layer.weights[0].name
				bias_name = layer.weights[1].name
				lr_mult[kernel_name] = 1
				lr_mult[bias_name] = 2

	return lr_mult

def step_decay(epoch, iterations_per_epoch):
	"""
	Learning rate schedule - equivalent of caffe lr_policy =  "step"

	:param epoch:
	:param iterations_per_epoch:
	:return:
	"""
	initial_lrate = base_lr
	steps = epoch * iterations_per_epoch

	lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))

	return lrate

def get_loss_funcs(batch_size, time_steps=4):
	"""
	Euclidean loss as implemented in caffe
	https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
	:return:
	"""
	def _eucl_loss(x, y):
		return K.sum(K.square(x - y)) / batch_size / 2 / time_steps

	losses = {}
	losses["weight_stage1_L2"] = _eucl_loss
	losses["weight_stage2_L2"] = _eucl_loss
	losses["weight_stage3_L2"] = _eucl_loss
	losses["weight_stage4_L2"] = _eucl_loss
	losses["weight_stage5_L2"] = _eucl_loss
	losses["weight_stage6_L2"] = _eucl_loss

	return losses

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, required=True)
	# parser.add_argument('--arch', default='conv_v1', type=str)
	parser.add_argument('--dataset', default='train', type=str)

	parser.add_argument('--epochs', default=5, type=int)
	parser.add_argument('--time_steps', default=4, type=int)

	parser.add_argument('--gpus', default=1, type=int)
	parser.add_argument('--batch', default=6, type=int)
	parser.add_argument('--count', default=False, type=bool)
	args = parser.parse_args()

	batch_size = args.batch

	model = conv_v2(time_steps=args.time_steps)

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


	loss_funcs = get_loss_funcs(batch_size, time_steps=args.time_steps)
	model.compile(loss=loss_funcs, optimizer=multisgd, metrics=["accuracy"])

	model.load_weights('checkpoints/lstm-epoch_0.h5')

	from generate_video import *
	dset = gather_videos(SEQ_LEN=args.time_steps, still=True)
	# videos, masks, targets = next_video_batch(dset, batch_size)

	def gen(df):
		every = 4
		counter = 0

		while True:
			for (inp, out) in df.get_data():
				vids, mas, targs = next_video_batch(dset, batch_size)

				# yield inp, out
				videos, masks = [], []
				targets = [[], [], [], [], [], []]

				for ii in range(batch_size):
					if ii % 2 == 0:
						videos.append(inp[0][ii])
						masks.append(inp[1][ii])
						for jj in range(6):
							targets[jj].append(out[jj][ii])
					else:
						videos.append(vids[ii])
						masks.append(mas[ii])
						for jj in range(6):
							targets[jj].append(targs[ii])

				videos = np.array(videos)
				masks = np.array(masks)
				for ii in range(6):
					targets[ii] = np.array(targets[ii])

				assert len(targets) == 6
				yield [videos, masks], targets


	DATA_DIR = '/beegfs/ua349/lstm/coco'
	df = get_dataflow(
		annot_path='%s/annotations/person_keypoints_%s2017.json' % (DATA_DIR, args.dataset),
		img_dir='%s/%s2017/' % (DATA_DIR, args.dataset))
	train_samples = df.size()
	print('Collected %d val samples...' % train_samples)
	train_df = batch_dataflow(df, batch_size, time_steps=args.time_steps)
	train_gen = gen(train_df)

	from snapshot import Snapshot

	model.fit_generator(train_gen,
		steps_per_epoch=train_samples // batch_size,
		epochs=args.epochs,
		callbacks=[lrate, Snapshot(args.name, train_gen, time_series=True, time_steps=args.time_steps)],
		use_multiprocessing=False,
		initial_epoch=0,
		verbose=1)

