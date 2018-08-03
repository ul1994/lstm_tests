
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
from keras.layers import Input, Lambda, LSTM, Reshape, TimeDistributed, Dense, ConvLSTM2D, Dropout
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.utils import multi_gpu_model as multigpu

from components import *
sys.path.append('/scratch/ua349/pose/tf/ver1')


stages = 6
np_branch1 = 38
np_branch2 = 19

def heat_v1(imsize=368, outsize=46, weight_decay=5e-4):
	img_input_shape = (imsize, imsize, 3)
	heat_mask_shape = (outsize, outsize, 19)
	vect_mask_shape = (outsize, outsize, 19)

	inputs = []
	outputs = []

	img_input = Input(shape=img_input_shape)
	heat_weight_input = Input(shape=heat_mask_shape)
	# vec_weight_input = Input(shape=vect_mask_shape)
	vec_weight_input = None

	inputs.append(img_input)
	inputs.append(heat_weight_input)
	# inputs.append(vec_weight_input)

	img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

	# VGG
	stage0_out = vgg_block(img_normalized, weight_decay, rnn=False)

	# stage 1 - branch 2 (confidence maps)
	stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay, rnn=False)
	w2 = apply_mask(stage1_branch2_out, heat_weight_input, np_branch2, 1, 2)

	x = Concatenate()([stage1_branch2_out, stage0_out])

	# print('Concat shape:', x.get_shape())
	outputs.append(w2)

	# stage sn >= 2
	for sn in range(2, stages + 1):

		# stage SN - branch 2 (confidence maps)
		stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay, rnn=False)
		w2 = apply_mask(stageT_branch2_out, heat_weight_input, np_branch2, sn, 2)

		outputs.append(w2)

		if (sn < stages):
			x = Concatenate()([stageT_branch2_out, stage0_out])

	model = Model(inputs=inputs, outputs=outputs)

	return model


def lstm_v1(time_steps=4, imsize=368, outsize=46, weight_decay = 5e-4):
	img_input_shape = (time_steps, imsize, imsize, 3)
	heat_mask_shape = (time_steps, outsize, outsize, 19)
	flat_shape = (outsize * outsize * np_branch2,)


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

	carry = TimeDistributed(Dense(1024, activation='relu'))(carry)
	carry = LSTM(512, input_shape=(1024,), return_sequences=True)(carry)
	carry = TimeDistributed(Dense(flat_shape[0], activation='relu'))(carry)

	carry = TimeDistributed(Reshape(heat_mask_shape[1:]))(carry)

	FILTER_STAGES = 6
	for sn in range(2, FILTER_STAGES + 1):
		if sn != FILTER_STAGES - 1:
			carry = Concatenate()([carry, stage0_out])

		carry = stageT_block(carry, np_branch2, sn, 2, weight_decay)
		masked = apply_mask(carry, heat_mask, np_branch2, sn, 2)

		outputs.append(masked)

	model = Model(inputs=inputs, outputs = outputs)
	# model.summary()

	return model


def conv_v1(time_steps=4, imsize=368, outsize=46, weight_decay = 5e-4):
	img_input_shape = (time_steps, imsize, imsize, 3)
	heat_mask_shape = (time_steps, outsize, outsize, 19)

	outputs = []

	img_input = Input(shape=img_input_shape)
	heat_mask = Input(shape=heat_mask_shape)
	inputs = [img_input, heat_mask]

	img_normalized = Lambda(lambda x: x / 256 - 0.5)
	normed = TimeDistributed(img_normalized)(img_input)

	# Vgg layers
	stage0_out = vgg_block(normed, weight_decay, rnn=True)

	carry = stage1_block(stage0_out, np_branch2, 2, weight_decay)
	masked = apply_mask(carry, heat_mask, np_branch2, 1, 2)
	outputs.append(masked)

	carry = Concatenate()([carry, stage0_out])

	carry = ConvLSTM2D(
		filters=np_branch2,
		kernel_size=(3, 3),
		padding='same',
		return_sequences=True)(carry)

	carry = ConvLSTM2D(
		filters=np_branch2,
		kernel_size=(3, 3),
		padding='same',
		return_sequences=True)(carry)

	START_AT = 2
	FILTER_STAGES = 6
	for sn in range(START_AT, FILTER_STAGES + 1):

		carry = stageT_block(carry, np_branch2, sn, 2, weight_decay)

		masked = apply_mask(carry, heat_mask, np_branch2, sn, 2)
		outputs.append(masked)

		if sn < FILTER_STAGES:
			# don't concatenate final output
			carry = Concatenate()([carry, stage0_out])

	model = Model(inputs=inputs, outputs = outputs)
	# model.summary()

	return model

def conv_v2(time_steps=4, imsize=368, outsize=46, weight_decay = 5e-4):
	img_input_shape = (time_steps, imsize, imsize, 3)
	heat_mask_shape = (time_steps, outsize, outsize, 19)

	inputs = []
	outputs = []

	img_input = Input(shape=img_input_shape)
	heat_weight_input = Input(shape=heat_mask_shape)

	inputs.append(img_input)
	inputs.append(heat_weight_input)

	img_normalized = TimeDistributed(Lambda(lambda x: x / 256 - 0.5))(img_input) # [-0.5, 0.5]

	# VGG
	stage0_out = vgg_block(img_normalized, weight_decay, rnn=True)

	# stage 1 - branch 2 (confidence maps)
	stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay, rnn=True)
	w2 = apply_mask(stage1_branch2_out, heat_weight_input, np_branch2, 1, 2)
	outputs.append(w2)

	x = Concatenate()([stage1_branch2_out, stage0_out])
	# print('Concat shape:', x.get_shape())

	x = Dropout(0.5)(x)


	# # stage sn >= 2
	for sn in range(2, stages + 1):

		# stage SN - branch 2 (confidence maps)
		stageT_branch2_out = stageT_block_lstm(x, np_branch2, sn, 2, weight_decay)
		w2 = apply_mask(stageT_branch2_out, heat_weight_input, np_branch2, sn, 2)

		outputs.append(w2)

		if (sn < stages):
			x = Concatenate()([stageT_branch2_out, stage0_out])

	model = Model(inputs=inputs, outputs=outputs)

	return model

def mod_v1(time_steps=4, imsize=368, outsize=46, weight_decay = 5e-4):
	img_input_shape = (time_steps, imsize, imsize, 3)
	heat_mask_shape = (time_steps, outsize, outsize, 19)

	inputs = []
	outputs = []

	img_input = Input(shape=img_input_shape)
	heat_weight_input = Input(shape=heat_mask_shape)

	inputs.append(img_input)
	inputs.append(heat_weight_input)

	img_normalized = TimeDistributed(Lambda(lambda x: x / 256 - 0.5))(img_input) # [-0.5, 0.5]

	# VGG
	stage0_out = vgg_block(img_normalized, weight_decay, rnn=True)

	# stage 1 - branch 2 (confidence maps)
	stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay, rnn=True)
	w2 = apply_mask(stage1_branch2_out, heat_weight_input, np_branch2, 1, 2)
	outputs.append(w2)

	x = Concatenate()([stage1_branch2_out, stage0_out])
	# print('Concat shape:', x.get_shape())

	x = Dropout(0.5)(x)

	get_last = Lambda(lambda tensor: tensor[:, time_steps-1, :, :, :])
	for sn in range(2, stages + 1):
		if sn < stages - 1:
			stageT_branch2_out = stageT_block_lstm(x, np_branch2, sn, 2, weight_decay)
			x = Concatenate()([stageT_branch2_out, stage0_out])
			w2 = apply_mask(stageT_branch2_out, heat_weight_input, np_branch2, sn, 2)
			outputs.append(w2)

		if sn == stages - 1:
			stageT_branch2_out = stageJoin_block_lstm(x, np_branch2, sn, 2, weight_decay)
			x = Concatenate()([stageT_branch2_out, get_last(stage0_out)])
			w2 = apply_mask(stageT_branch2_out, get_last(heat_weight_input), np_branch2, sn, 2)
			outputs.append(w2)

		elif sn == stages:
			stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay, rnn=False)
			w2 = apply_mask(stageT_branch2_out, get_last(heat_weight_input), np_branch2, sn, 2)
			outputs.append(w2)

	model = Model(inputs=inputs, outputs=outputs)

	return model

def mod_v2(time_steps=4, imsize=368, outsize=46, weight_decay = 5e-4, trainable=True):
	img_input_shape = (time_steps, None, None, 3)
	heat_mask_shape = (time_steps, None, None, 19)

	inputs = []
	outputs = []

	img_input = Input(shape=img_input_shape)
	heat_weight_input = Input(shape=heat_mask_shape)

	inputs.append(img_input)
	if trainable:
		inputs.append(heat_weight_input)

	img_normalized = TimeDistributed(Lambda(lambda x: x / 256 - 0.5))(img_input) # [-0.5, 0.5]

	# VGG
	stage0_out = vgg_block(img_normalized, weight_decay, rnn=True)

	# stage 1 - branch 2 (confidence maps)
	stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay, rnn=True)
	if trainable:
		w2 = apply_mask(stage1_branch2_out, heat_weight_input, np_branch2, 1, 2)
		outputs.append(w2)

	x = Concatenate()([stage1_branch2_out, stage0_out])
	# print('Concat shape:', x.get_shape())

	x = Dropout(0.5)(x)

	get_last = Lambda(lambda tensor: tensor[:, time_steps-1, :, :, :])
	for sn in range(2, stages + 1):
		if sn < stages - 1:
			stageT_branch2_out = stageT_block_lstm(x, np_branch2, sn, 2, weight_decay)
			x = Concatenate()([stageT_branch2_out, stage0_out])
			if trainable:
				w2 = apply_mask(stageT_branch2_out, heat_weight_input, np_branch2, sn, 2)
				outputs.append(w2)

		if sn == stages - 1:
			stageT_branch2_out = stageJoin_block_lstm(x, np_branch2, sn, 2, weight_decay)
			x = Concatenate()([stageT_branch2_out, get_last(stage0_out)])
			if trainable:
				w2 = apply_mask(stageT_branch2_out, get_last(heat_weight_input), np_branch2, sn, 2)
				outputs.append(w2)

		elif sn == stages:
			stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay, rnn=False)
			if trainable:
				w2 = apply_mask(stageT_branch2_out, get_last(heat_weight_input), np_branch2, sn, 2)
				outputs.append(w2)

	if trainable:
		model = Model(inputs=inputs, outputs=outputs)
	else:
		model = Model(inputs=inputs, outputs=[stageT_branch2_out])

	return model

def apply_mask_full(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w

def mod_v3(trainable=True, bsize=6, time_steps=4, imsize=368, outsize=46, weight_decay=5e-4):

	stages = 6
	np_branch1 = 38
	np_branch2 = 19

	img_input_shape = (bsize, time_steps, imsize, imsize, 3)
	vec_input_shape = (bsize, time_steps, outsize, outsize, 38)
	heat_input_shape = (bsize, time_steps, outsize, outsize, 19)

	inputs = []
	outputs = []

	# img_input = Input(shape=img_input_shape[1:], batch_shape=img_input_shape)
	img_input = Input(batch_shape=img_input_shape)
	inputs.append(img_input)
	if trainable:
		vec_mask = Input(shape=vec_input_shape[1:], batch_shape=vec_input_shape)
		heat_mask = Input(shape=heat_input_shape[1:], batch_shape=heat_input_shape)
		inputs.append(vec_mask)
		inputs.append(heat_mask)


	img_normalized = Lambda(lambda x: x / 256 - 0.5)
	normed = TimeDistributed(img_normalized)(img_input)

	# VGG
	stage0_out = vgg_block(normed, weight_decay, rnn=True)

	# stage 1 - branch 1 (PAF)
	stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay, rnn=True)
	stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay, rnn=True)

	if trainable:
		w1 = apply_mask_full(stage1_branch1_out, vec_mask, heat_mask, np_branch1, 1, 1)
		w2 = apply_mask_full(stage1_branch2_out, vec_mask, heat_mask, np_branch2, 1, 2)
		outputs.append(w1)
		outputs.append(w2)

	x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])


	# stage sn >= 2
	get_last = Lambda(lambda tensor: tensor[:, time_steps-1, :, :, :])
	for sn in range(2, stages + 1):
		if sn == 2:
			stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay, rnn=True)
			stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay, rnn=True)
			if trainable:
				w2 = apply_mask_full(stageT_branch2_out, vec_mask, heat_mask, np_branch2, sn, 2)
				w1 = apply_mask_full(stageT_branch1_out, vec_mask, heat_mask, np_branch1, sn, 1)
				outputs.append(w1)
				outputs.append(w2)
			x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

		elif sn == 3:
			stageT_branch1_out = stageJoin_block_lstm(x, np_branch1, sn, 1, weight_decay, stateful=True)
			stageT_branch2_out = stageJoin_block_lstm(x, np_branch2, sn, 2, weight_decay, stateful=True)
			if trainable:
				w1 = apply_mask_full(stageT_branch1_out, get_last(vec_mask), get_last(heat_mask), np_branch1, sn, 1)
				w2 = apply_mask_full(stageT_branch2_out, get_last(vec_mask), get_last(heat_mask), np_branch2, sn, 2)
				outputs.append(w1)
				outputs.append(w2)
			x = Concatenate()([stageT_branch1_out, stageT_branch2_out, get_last(stage0_out)])

		elif sn > 3:
			stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay, rnn=False)
			stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay, rnn=False)
			if trainable:
				w1 = apply_mask_full(stageT_branch1_out, get_last(vec_mask), get_last(heat_mask), np_branch1, sn, 1)
				w2 = apply_mask_full(stageT_branch2_out, get_last(vec_mask), get_last(heat_mask), np_branch2, sn, 2)
				outputs.append(w1)
				outputs.append(w2)

			if sn != stages:
				x = Concatenate()([stageT_branch1_out, stageT_branch2_out, get_last(stage0_out)])
			else:
				if not trainable:
					outputs.append(stageT_branch1_out)
					outputs.append(stageT_branch2_out)

	model = Model(inputs=inputs, outputs=outputs)

	return model

def mod_v4(trainable=True, bsize=6, time_steps=None, imsize=368, outsize=46, weight_decay=5e-4):
	STATEFUL_TIMESTEP = 1

	stages = 6
	np_branch1 = 38
	np_branch2 = 19

	img_input_shape = (bsize, STATEFUL_TIMESTEP, imsize, imsize, 3)
	vec_input_shape = (bsize, STATEFUL_TIMESTEP, outsize, outsize, 38)
	heat_input_shape = (bsize, STATEFUL_TIMESTEP, outsize, outsize, 19)

	inputs = []
	outputs = []

	img_input = Input(batch_shape=img_input_shape)
	inputs.append(img_input)
	if trainable:
		vec_mask = Input(shape=vec_input_shape[1:], batch_shape=vec_input_shape)
		heat_mask = Input(shape=heat_input_shape[1:], batch_shape=heat_input_shape)
		inputs.append(vec_mask)
		inputs.append(heat_mask)


	img_normalized = Lambda(lambda x: x / 256 - 0.5)
	normed = TimeDistributed(img_normalized)(img_input)

	# VGG
	stage0_out = vgg_block(normed, weight_decay, rnn=True)

	# stage 1 - branch 1 (PAF)
	stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay, rnn=True)
	stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay, rnn=True)

	flat = Lambda(lambda tensor: tensor[:, 0, :, :, :])
	if trainable:
		w1 = apply_mask_full(flat(stage1_branch1_out), flat(vec_mask), flat(heat_mask), np_branch1, 1, 1)
		w2 = apply_mask_full(flat(stage1_branch2_out), flat(vec_mask), flat(heat_mask), np_branch2, 1, 2)
		outputs.append(w1)
		outputs.append(w2)
	x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

	# stage sn >= 2
	for sn in range(2, stages + 1):
		if sn == 2:
			stageT_branch1_out = stageJoin_block_lstm(x, np_branch1, sn, 1, weight_decay, stateful=True)
			stageT_branch2_out = stageJoin_block_lstm(x, np_branch2, sn, 2, weight_decay, stateful=True)
			if trainable:
				w1 = apply_mask_full(stageT_branch1_out, flat(vec_mask), flat(heat_mask), np_branch1, sn, 1)
				w2 = apply_mask_full(stageT_branch2_out, flat(vec_mask), flat(heat_mask), np_branch2, sn, 2)
				outputs.append(w1)
				outputs.append(w2)
			x = Concatenate()([stageT_branch1_out, stageT_branch2_out, flat(stage0_out)])

		if sn > 2:
			stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay, rnn=False)
			stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay, rnn=False)
			if trainable:
				w1 = apply_mask_full(stageT_branch1_out, flat(vec_mask), flat(heat_mask), np_branch1, sn, 1)
				w2 = apply_mask_full(stageT_branch2_out, flat(vec_mask), flat(heat_mask), np_branch2, sn, 2)
				outputs.append(w1)
				outputs.append(w2)

			if sn != stages:
				x = Concatenate()([stageT_branch1_out, stageT_branch2_out, flat(stage0_out)])
			else:
				if not trainable:
					outputs.append(stageT_branch1_out)
					outputs.append(stageT_branch2_out)

	model = Model(inputs=inputs, outputs=outputs)

	return model

MODELS = {
	'mod_v3': mod_v3,
	'mod_v4': mod_v4,
}

if __name__ == '__main__':
	import os, sys
	os.environ["CUDA_VISIBLE_DEVICES"]= "1"

	mdl = MODELS['mod_v4'](bsize=1, time_steps=4)
	mdl.summary()

