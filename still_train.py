
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
from keras.layers import Input, Lambda, LSTM, Reshape, TimeDistributed, Dense
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.utils import multi_gpu_model as multigpu

from components import *
sys.path.append('/scratch/ua349/pose/tf/ver1')
from training.optimizers import MultiSGD
from training.dataset import get_dataflow, batch_dataflow

weight_decay = 5e-4
TEST_EVERY = 10
VISUALIZE_EVERY = 100
ITERS = 10000

base_lr = 4e-5 # 2e-5
momentum = 0.9
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000

TIME_STEPS = 4

stages = 6
np_branch1 = 38
np_branch2 = 19
IMSIZE = 368
OUTSIZE = 46

weights_best_file = "weights.best.h5"
training_log = "training.csv"
logs_dir = "./logs"

DATA_DIR = '/beegfs/ua349/lstm/coco'

def get_recurrent_model():
	img_input_shape = (TIME_STEPS, IMSIZE, IMSIZE, 3)
	heat_mask_shape = (TIME_STEPS, OUTSIZE, OUTSIZE, 19)
	flat_shape = (OUTSIZE * OUTSIZE * np_branch2,)


	outputs = []

	img_input = Input(shape=img_input_shape)
	heat_mask = Input(shape=heat_mask_shape)
	inputs = [img_input, heat_mask]

	img_normalized = Lambda(lambda x: x / 256 - 0.5)
	normed = TimeDistributed(img_normalized)(img_input)

	# Vgg layers
	stage0_out = vgg_block(normed, weight_decay)

	carry = stage1_block(stage0_out, np_branch2, 2, weight_decay)
	# print(carry.get_shape(), heat_mask.get_shape())
	masked = apply_mask(carry, heat_mask, np_branch2, 1, 2)
	outputs.append(masked)

	carry = TimeDistributed(Reshape(flat_shape))(carry)

	carry = TimeDistributed(Dense(512))(carry)
	carry = LSTM(512, input_shape=(512,), return_sequences=True)(carry)
	carry = TimeDistributed(Dense(flat_shape[0]))(carry)

	carry = TimeDistributed(Reshape(heat_mask_shape[1:]))(carry)

	FILTER_STAGES = 6
	for sn in range(2, FILTER_STAGES + 1):
		if sn != FILTER_STAGES - 1:
			carry = Concatenate()([carry, stage0_out])

		carry = stageT_block(carry, np_branch2, sn, 2, weight_decay)
		masked = apply_mask(carry, heat_mask, np_branch2, sn, 2)

		outputs.append(masked)

	model = Model(inputs=inputs, outputs = outputs)
	model.summary()

	return model

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
	parser.add_argument('--gpus', default=1, type=int)
	parser.add_argument('--batch', default=6, type=int)
	args = parser.parse_args()

	batch_size = args.batch
	model = get_recurrent_model()

	if args.gpus > 1:
		batch_size = args.batch * args.gpus
		model = multigpu(model, gpus=args.gpus)

	lr_multipliers = get_lr_multipliers(model)


	iterations_per_epoch = 110000 // batch_size
	_step_decay = partial(step_decay,
						iterations_per_epoch=iterations_per_epoch
						)
	lrate = LearningRateScheduler(_step_decay)
	checkpoint = ModelCheckpoint('heat.best.h5', monitor='loss',
								verbose=0, save_best_only=False,
								save_weights_only=True, mode='min', period=1)
	csv_logger = CSVLogger(training_log, append=True)
	tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
					write_images=False)

	callbacks_list = [lrate, checkpoint, csv_logger, tb]

	# sgd optimizer with lr multipliers
	multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0,
						nesterov=False, lr_mult=lr_multipliers)


	loss_funcs = get_loss_funcs(batch_size)
	model.compile(loss=loss_funcs, optimizer=multisgd, metrics=[])

	def batch_dataflow(df, batch_size):
		"""
		The function builds batch dataflow from the input dataflow of samples

		:param df: dataflow of samples
		:param batch_size: batch size
		:return: dataflow of batches
		"""
		df = BatchData(df, batch_size, use_list=False)
		df = MapData(df, lambda x: (
			[
				np.stack([x[0]] * TIME_STEPS, axis=1),
				np.stack([x[2]] * TIME_STEPS, axis=1)
			],
			[np.stack([x[4]] * TIME_STEPS, axis=1)] * 6))
		df.reset_state()
		return df


	def gen(df):
		while True:
			for i in df.get_data():
				yield i

	df = get_dataflow(
		annot_path='%s/annotations/person_keypoints_train2017.json' % DATA_DIR,
		img_dir='%s/train2017/' % DATA_DIR)
	train_samples = df.size()
	print('Collected %d val samples...' % train_samples)
	train_df = batch_dataflow(df, batch_size)
	train_gen = gen(train_df)

	print(model.inputs)
	print(model.outputs)

	for batch_in, batch_out in train_gen:
		print(batch_in[0].shape)
		break


	model.fit_generator(train_gen,
		steps_per_epoch=train_samples // batch_size,
		epochs=max_iter,
		callbacks=callbacks_list,
		# validation_data=val_di,
		# validation_steps=val_samples // batch_size,
		use_multiprocessing=False,
		initial_epoch=0,
		verbose=1)

