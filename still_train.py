
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
sys.path.append('../tf/ver1')
from training.optimizers import MultiSGD
from training.dataset import get_dataflow

from models import *

# base_lr = 2e-5 # 2e-5
base_lr = 0.00004
momentum = 0.9
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma

DATA_DIR = '/beegfs/ua349/lstm/coco'

def batch_dataflow(df, batch_size, time_steps=4, num_stages=6, format=['heatpaf', 'last']):
	informat, outformat = format


	df = BatchData(df, batch_size, use_list=False)

	def in_heat(x):
		return [
			np.stack([x[0]] * time_steps, axis=1),
			np.stack([x[2]] * time_steps, axis=1)
		]
	def in_heatpaf(x):
		return [
			np.stack([x[0]] * time_steps, axis=1),
			np.stack([x[1]] * time_steps, axis=1),
			np.stack([x[2]] * time_steps, axis=1)
		]
	def out_heat_last(x):
		return [np.stack([x[4]] * time_steps, axis=1)] * num_stages
	def out_heatpaf_last(x):
		return [
			np.stack([x[3]] * time_steps, axis=1),
			np.stack([x[4]] * time_steps, axis=1),
			np.stack([x[3]] * time_steps, axis=1),
			np.stack([x[4]] * time_steps, axis=1), # TD layers end here

			x[3], # TD layers are joined here by LSTM
			x[4],

			x[3], # these last outputs collapse to one timestep output
			x[4],

			x[3],
			x[4],

			x[3],
			x[4],
		]

	if informat == 'heat' and outformat == 'last':
		df = MapData(df, lambda x: (
			heat_only(x),
			out_heat_last(x)
		))
	elif informat == 'heatpaf' and outformat == 'last':
		df = MapData(df, lambda x: (
			in_heatpaf(x),
			out_heatpaf_last(x)
		))
	else:
		raise Exception('Unknown format requested: %s' % format)

	df.reset_state()
	return df

def gen(df):
		while True:
			for i in df.get_data():
				yield i

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
				lr_mult[kernel_name] = 2
				lr_mult[bias_name] = 4

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

def get_loss_funcs(batch_size):
	"""
	Euclidean loss as implemented in caffe
	https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
	:return:
	"""
	def _eucl_loss(x, y):
		return K.sum(K.square(x - y)) / batch_size / 2

	losses = {}
	losses["weight_stage1_L1"] = _eucl_loss
	losses["weight_stage1_L2"] = _eucl_loss
	losses["weight_stage2_L1"] = _eucl_loss
	losses["weight_stage2_L2"] = _eucl_loss
	losses["weight_stage3_L1"] = _eucl_loss
	losses["weight_stage3_L2"] = _eucl_loss
	losses["weight_stage4_L1"] = _eucl_loss
	losses["weight_stage4_L2"] = _eucl_loss
	losses["weight_stage5_L1"] = _eucl_loss
	losses["weight_stage5_L2"] = _eucl_loss
	losses["weight_stage6_L1"] = _eucl_loss
	losses["weight_stage6_L2"] = _eucl_loss
	return losses

def load_any_weights(model, multigpu=True):
	loaded = 0
	failed = 0
	badlist = []
	if multigpu:
		model_find = [layer for layer in model.layers if type(layer) == Model]
		assert len(model_find) == 1
		model = model_find[0]

	for layer in model.layers:
		if type(layer) == Conv2D or (hasattr(layer, 'layer') and type(layer.layer) == Conv2D):
			if hasattr(layer, 'layer'): layer = layer.layer

			wfile = layer.name + '_matrix.npy'
			bfile = layer.name + '_bias.npy'

			wmat = np.load('../tf/ver1/model/weights/%s' % wfile)
			bias = np.load('../tf/ver1/model/weights/%s' % bfile)

			try:
				layer.set_weights([wmat, bias])
				loaded += 1
			except:
				failed += 1
				# print(layer.get_weights()[0].shape, wmat.shape)
				badlist.append((type(layer), layer.name))
		# else:
		# 	failed += 1
		# 	badlist.append((type(layer), layer.name))
	print('Loaded %d layers (mismatch: %d)' % (loaded, failed))
	assert loaded > 0

	for name in badlist:
		print('-', name)
	# exit(1)

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
	train_gen = gen(train_df)

	print(model.inputs[0].get_shape())
	# print(model.outputs)

	from snapshot import Snapshot

	model.fit_generator(train_gen,
		steps_per_epoch=train_samples // batch_size,
		epochs=args.epochs,
		callbacks=[lrate, Snapshot(args.name, train_gen, __format, stills=True)],
		use_multiprocessing=False,
		initial_epoch=0,
		verbose=1)

