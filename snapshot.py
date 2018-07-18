import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.backend as K

class Snapshot(keras.callbacks.Callback):
	pbatch = None
	functor = None
	bcount = 0
	ecount = 0

	def eval(self, save_name):
		if self.functor is None:
			inp = self.model.input
			outputs = self.model.outputs[-2:]
			self.functor = K.function(inp, outputs ) # evaluation function

		images, heat_masks, targets = self.pbatch

		layer_outs = self.functor(
			[images, heat_masks]
		)

		results = layer_outs[-1]


		if not self.ts:
			plt.figure(figsize=(14, 7))
			for ii in range(5):
				plt.subplot(2, 5, ii+1)
				plt.axis('off')
				img = images[ii]
				if self.ts:
					img = img[0]

				plt.imshow(img.astype(np.float32)/255)
			for ii in range(5):
				plt.subplot(2, 5, 5+ii+1)
				plt.axis('off')
				res = results[ii]
				if self.ts:
					res = res[0]
				plt.imshow(np.sum(res[:, :, :-1], axis=-1), vmin=0, vmax=1)
		else:
			plt.figure(figsize=(14, 10))
			for ii in range(self.time_steps):
				plt.subplot(3, self.time_steps, ii+1)
				plt.axis('off')
				img = images[1][ii]
				plt.imshow(img.astype(np.float32)/256)
			for ii in range(self.time_steps):
				plt.subplot(3, self.time_steps, self.time_steps+ii+1)
				plt.axis('off')
				res = results[1][ii]
				plt.imshow(np.sum(res[:, :, :-1].astype(np.float32), axis=-1), vmin=0, vmax=1)
			for ii in range(self.time_steps):
				plt.subplot(3, self.time_steps, self.time_steps*2+ii+1)
				plt.axis('off')
				res = targets[1][ii]
				plt.imshow(np.sum(res[:, :, :-1].astype(np.float32), axis=-1), vmin=0, vmax=1)

		plt.savefig('previews/%s' % save_name, bbox_inches='tight')
		plt.close()
		# exit()

	def __init__(self, tag, data_gen, every=100, time_series=False, time_steps=None):
		self.tag = tag
		self.validation_data = None
		self.model = None

		self.ts = time_series
		self.every = every
		self.counter = 0
		self.time_steps = time_steps

		images = []
		heat_masks = []
		targets = []
		for ii, (inp, out) in enumerate(data_gen):
			img, heat = inp
			images = np.array(img)
			heat_masks = np.array(heat)
			targets = out[-1]

			# self.pbatch = (images, masks, targets)
			break

		self.pbatch = (images, heat_masks, targets)

	def on_train_begin(self, logs=None):
		pass

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


