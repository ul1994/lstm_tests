import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.backend as K
import cv2

class Snapshot(keras.callbacks.Callback):
	batch = None
	functor = None
	bcount = 0
	ecount = 0

	def eval(self, save_name):

		# images, heat_masks, targets = self.pbatch
		_, outform = self.format
		stills = self.stills
		ins, outs = self.batch

		results = self.model.predict(ins)

		if stills:
			NSHOW = 5
			plt.figure(figsize=(14, 10))
			for ii in range(NSHOW):
				plt.subplot(3, NSHOW, ii+1)
				if ii == 0: plt.gca().set_title('Epoch: %d   Batch: %d' % (self.ecount, self.bcount))
				plt.axis('off')
				img = ins[0][ii][-1] # ii-th batch, last img in sequence
				img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
				plt.imshow(img.astype(np.float32)/256)
			for ii in range(NSHOW):
				plt.subplot(3, NSHOW, NSHOW+ii+1)
				plt.axis('off')
				if outform == 'last':
					heat = results[-1][ii]
				else:
					raise Exception('Not supported')
				plt.imshow(np.sum(heat[:, :, :-1].astype(np.float32), axis=-1), vmin=0, vmax=1)
			for ii in range(NSHOW):
				plt.subplot(3, NSHOW, NSHOW*2+ii+1)
				plt.axis('off')
				if outform == 'last':
					target = outs[-1][ii]
				else:
					raise Exception('Not supported')
				plt.imshow(np.sum(target[:, :, :-1].astype(np.float32), axis=-1), vmin=0, vmax=1)
		else:
			raise Exception('Not implemented')

		# if not self.ts or self.time_format == 'last':
		# 	plt.figure(figsize=(14, 7))
		# 	for ii in range(self.time_steps):
		# 		if ii == 0: plt.gca().set_title('Epoch: %d   Batch: %d' % (self.ecount, self.bcount))
		# 		plt.subplot(3, self.time_steps, ii+1)
		# 		plt.axis('off')
		# 		img = images[ii]
		# 		if self.ts: # will be a tseries for last format
		# 			img = img[0]
		# 		plt.imshow(img.astype(np.float32)/255)

		# 	for ii in range(self.time_steps):
		# 		plt.subplot(3, self.time_steps, self.time_steps*1+ii+1)
		# 		plt.axis('off')
		# 		res = results[ii]
		# 		# in last-format, output will be for 1 frame even though it is a tseries
		# 		plt.imshow(np.sum(res[:, :, :-1], axis=-1))

		# 	for ii in range(self.time_steps):
		# 		plt.subplot(3, self.time_steps, self.time_steps*2+ii+1)
		# 		plt.axis('off')
		# 		res = targets[ii]
		# 		plt.imshow(np.sum(res[:, :, :-1].astype(np.float32), axis=-1), vmin=0, vmax=1)

		# else:
		# 	plt.figure(figsize=(14, 10))
		# 	for ii in range(self.time_steps):
		# 		plt.subplot(3, self.time_steps, ii+1)
		# 		if ii == 0: plt.gca().set_title('Epoch: %d   Batch: %d' % (self.ecount, self.bcount))
		# 		plt.axis('off')
		# 		img = images[1][ii]
		# 		plt.imshow(img.astype(np.float32)/256)
		# 	for ii in range(self.time_steps):
		# 		plt.subplot(3, self.time_steps, self.time_steps+ii+1)
		# 		plt.axis('off')
		# 		res = results[1][ii]
		# 		plt.imshow(np.sum(res[:, :, :-1].astype(np.float32), axis=-1), vmin=0, vmax=1)
		# 	for ii in range(self.time_steps):
		# 		plt.subplot(3, self.time_steps, self.time_steps*2+ii+1)
		# 		plt.axis('off')
		# 		res = targets[1][ii]
		# 		plt.imshow(np.sum(res[:, :, :-1].astype(np.float32), axis=-1), vmin=0, vmax=1)

		plt.savefig('previews/%s' % save_name, bbox_inches='tight')
		plt.close()
		exit()

	def __init__(self, tag, data_gen, format, every=100, stills=False):
		self.tag = tag
		self.validation_data = None
		self.model = None
		self.format = format

		self.every = every
		self.counter = 0
		self.stills = stills

		for ii, batch in enumerate(data_gen):
			self.batch = batch
			break

	def on_batch_end(self, batch, logs=None):
		if self.bcount < 50:
			self.eval('%s-%s_%s.png' % (self.tag, self.ecount, self.bcount))
			self.bcount += 1
			return

		if self.counter % self.every != 0:
			self.counter += 1
			self.bcount += 1
			return


		self.eval('%s-%s_%s.png' % (self.tag, self.ecount, self.bcount))
		self.counter = 1
		self.bcount += 1

	def on_epoch_end(self, epoch, logs=None):
		self.model.save_weights('checkpoints/%s-epoch_%d.h5' % (self.tag, self.ecount))
		self.ecount += 1
		self.bcount = 0


