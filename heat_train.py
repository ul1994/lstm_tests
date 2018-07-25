
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
from keras.layers import Input, Lambda, LSTM, Reshape, TimeDistributed, Dense, Conv2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.utils import multi_gpu_model as multigpu

from components import *
sys.path.append('/scratch/ua349/pose/tf/ver1')
from training.optimizers import MultiSGD
from training.dataset import get_dataflow
from snapshot import Snapshot

from models import *

# base_lr = 1e-6 # 2e-5
# base_lr = 2e-5 # 2e-5
base_lr = 0.0001
# base_lr = 4e-5 # 2e-5
momentum = 0.9
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma

weights_best_file = "weights.best.h5"
training_log = "training.csv"
logs_dir = "./logs"

DATA_DIR = '/beegfs/ua349/lstm/coco'

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
				x[0],
				x[2]
			],
			[x[4]] * num_stages
		))
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
				lr_mult[kernel_name] = 1
				lr_mult[bias_name] = 2

			# vgg
			else:
				kernel_name = layer.weights[0].name
				bias_name = layer.weights[1].name
				lr_mult[kernel_name] = 1
				lr_mult[bias_name] = 2

	return lr_mult

def load_vgg_weights(model):
	from_vgg = {
		'conv1_1': 'block1_conv1',
		'conv1_2': 'block1_conv2',
		'conv2_1': 'block2_conv1',
		'conv2_2': 'block2_conv2',
		'conv3_1': 'block3_conv1',
		'conv3_2': 'block3_conv2',
		'conv3_3': 'block3_conv3',
		'conv3_4': 'block3_conv4',
		'conv4_1': 'block4_conv1',
		'conv4_2': 'block4_conv2'
	}
	vgg_model = VGG19(include_top=False, weights='imagenet')

	loaded = 0
	for layer in model.layers:
		if isinstance(layer, TimeDistributed) and isinstance(layer.layer, Conv2D):
			layer = layer.layer

			if layer.name in from_vgg:
				vgg_layer_name = from_vgg[layer.name]
				layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
				loaded += 1

	print('Loaded %d from vgg.' % loaded)
	assert loaded != 0

def load_any_weights(model):
	loaded = 0
	failed = 0
	badlist = []
	for layer in model.layers:
		if isinstance(layer, TimeDistributed) and isinstance(layer.layer, Conv2D):
			layer = layer.layer
			# print(layer.namse)
			wfile = layer.name + '_matrix.npy'
			bfile = layer.name + '_bias.npy'

			wmat = np.load('/scratch/ua349/pose/tf/ver1/model/weights/%s' % wfile)
			bias = np.load('/scratch/ua349/pose/tf/ver1/model/weights/%s' % bfile)

			try:
				layer.set_weights([wmat, bias])
				loaded += 1
			except:
				failed += 1
				badlist.append(layer.name)
	print('Loaded %d layers (mismatch: %d)' % (loaded, failed))
	for name in badlist:
		print('- %s' % name)


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

def get_loss_funcs(batch_size):
	"""
	Euclidean loss as implemented in caffe
	https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
	:return:
	"""
	def _eucl_loss(x, y):
		return K.sum(K.square(x - y)) / batch_size / 2

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

	parser.add_argument('--arch', default='heat_v1', type=str)
	parser.add_argument('--dataset', default='train', type=str)
	parser.add_argument('--last_epoch', default=1, type=int)

	parser.add_argument('--epochs', default=5, type=int)
	parser.add_argument('--batch', default=6, type=int)
	parser.add_argument('--gpus', default=1, type=int)

	parser.add_argument('--count', default=False, type=bool)
	args = parser.parse_args()


	batch_size = args.batch

	avail = {
		'heat_v1': heat_v1,
	}

	model = avail[args.arch]()


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

	if args.count:
		model.summary()
		exit()


	def gen(df):
		while True:
			for i in df.get_data():
				yield i

	df = get_dataflow(
		annot_path='%s/annotations/person_keypoints_%s2017.json' % (DATA_DIR, args.dataset),
		img_dir='%s/%s2017/' % (DATA_DIR, args.dataset))
	train_df = batch_dataflow(df, batch_size)
	train_gen = gen(train_df)

	train_samples = df.size()
	print(' [*] Collected %d val samples...' % train_samples)

	print(model.input)
	print(model.outputs)

	loaded = 0
	for layer in model.layers:
		if isinstance(layer, Conv2D):
			wmat = np.load('../tf/ver1/model/weights/%s_matrix.npy' % layer.name)
			bias = np.load('../tf/ver1/model/weights/%s_bias.npy' % layer.name)
			try:
				layer.set_weights([wmat, bias])
				loaded += 1
			except:
				print(' [x] failed to load: %s' % layer.name)

	assert loaded != 0
	print(' [*] Loaded %d weights' % loaded)

	model.fit_generator(train_gen,
		steps_per_epoch=train_samples // batch_size,
		epochs=args.epochs,
		callbacks=[lrate, Snapshot(args.name, train_gen)],
		# callbacks=[lrate],
		use_multiprocessing=False,
		initial_epoch=args.last_epoch,
		verbose=1)
