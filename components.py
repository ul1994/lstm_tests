from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers import TimeDistributed, ConvLSTM2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant


def relu(): return Activation('relu')

def conv_lstm(nf, ks, name, weight_decay, first=False):
	kernel_reg = l2(weight_decay[0]) if weight_decay else None
	bias_reg = l2(weight_decay[1]) if weight_decay else None

	return ConvLSTM2D(nf, (ks, ks), padding='same', name=name,
					kernel_regularizer=kernel_reg,
					bias_regularizer=bias_reg,
					kernel_initializer=random_normal(stddev=0.01),
					bias_initializer=constant(0.0),
					return_sequences=True)

def conv(nf, ks, name, weight_decay, first=False):
	kernel_reg = l2(weight_decay[0]) if weight_decay else None
	bias_reg = l2(weight_decay[1]) if weight_decay else None

	return Conv2D(nf, (ks, ks), padding='same', name=name,
					kernel_regularizer=kernel_reg,
					bias_regularizer=bias_reg,
					kernel_initializer=random_normal(stddev=0.01),
					bias_initializer=constant(0.0))


def pooling(ks, st, name):
	return MaxPooling2D((ks, ks), strides=(st, st), name=name)
	# return x


def vgg_block(x, weight_decay, rnn=True):
	# Block 1
	if rnn is False: TD = lambda x: x
	else: TD = TimeDistributed

	x = TD(conv(64, 3, "conv1_1", (weight_decay, 0), first=True))(x)
	x = TD(relu())(x)
	x = TD(conv(64, 3, "conv1_2", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(pooling(2, 2, "pool1_1"))(x)

	# Block 2
	x = TD(conv(128, 3, "conv2_1", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 3, "conv2_2", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(pooling(2, 2, "pool2_1"))(x)

	# Block 3
	x = TD(conv(256, 3, "conv3_1", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(256, 3, "conv3_2", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(256, 3, "conv3_3", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(256, 3, "conv3_4", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(pooling(2, 2, "pool3_1"))(x)

	# Block 4
	x = TD(conv(512, 3, "conv4_1", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(512, 3, "conv4_2", (weight_decay, 0)))(x)
	x = TD(relu())(x)

	# Additional non vgg layers
	x = TD(conv(256, 3, "conv4_3_CPM", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 3, "conv4_4_CPM", (weight_decay, 0)))(x)
	x = TD(relu())(x)
	return x


def stage1_block(x, num_p, branch, weight_decay, rnn=True):
	if rnn is False: TD = lambda x: x
	else: TD = TimeDistributed

	# Block 1
	x = TD(conv(128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0)))(x)
	return x


def stageT_block(x, num_p, stage, branch, weight_decay, rnn=True):
	if rnn is False: TD = lambda x: x
	else: TD = TimeDistributed

	# Block 1
	x = TD(conv(128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)

	return x


def apply_mask(x, mask2, num_p, stage, branch):
	w_name = "weight_stage%d_L%d" % (stage, branch)
	w = Multiply(name=w_name) ([x, mask2])  # vec_heat
	return w

def stage1_block_lstm(x, num_p, branch, weight_decay):
	# Block 1
	x = conv_lstm(128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))(x)
	x = TimeDistributed(relu())(x)
	x = conv_lstm(128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))(x)
	x = TimeDistributed(relu())(x)
	x = conv_lstm(128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))(x)
	x = TimeDistributed(relu())(x)
	x = conv_lstm(512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))(x)
	x = TimeDistributed(relu())(x)
	x = conv_lstm(num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))(x)
	return x


def stageT_block_lstm(x, num_p, stage, branch, weight_decay):
	# Block 1
	TD = TimeDistributed

	# print(x.get_shape())
	x = conv_lstm(128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))(x)
	x = TimeDistributed(relu())(x)
	# print(x.get_shape())
	# exit()
	x = conv_lstm(128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))(x)
	x = TimeDistributed(relu())(x)
	x = TD(conv(128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)
	x = TD(relu())(x)
	x = TD(conv(num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0)))(x)

	return x