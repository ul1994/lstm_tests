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

		if stills or outform == 'last':
			NSHOW = len(ins[0]) # batch size
			plt.figure(figsize=(14, 10))
			for ii in range(NSHOW):
				plt.subplot(3, NSHOW, ii+1)
				if ii == 0: plt.gca().set_title('Epoch: %d   Batch: %d' % (self.ecount, self.bcount))
				plt.axis('off')
				img = ins[0][ii][-1] # ii-th batch, last img in sequence
				img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
				plt.imshow(img.astype(np.float32)/256)

				mask = ins[2][ii][-1][:, :, -1] # paf mask of iith batch (applies to all)
				mask = cv2.resize(mask, (0,0), fx=8, fy=8)
				plt.imshow(mask, alpha=0.5)

			for ii in range(NSHOW):
				plt.subplot(3, NSHOW, NSHOW+ii+1)
				plt.axis('off')
				if outform == 'last':
					LAST_HEAT = -1
					heat = results[LAST_HEAT][ii] # (46, 46, 19)
				else:
					raise Exception('Not supported')
				plt.imshow(np.sum(heat[:, :, :-1].astype(np.float32), axis=-1))
			for ii in range(NSHOW):
				plt.subplot(3, NSHOW, NSHOW*2+ii+1)
				plt.axis('off')
				if outform == 'last':
					LAST_PAF = -2
					paf = results[LAST_PAF][ii] # (46, 46, X)
				else:
					raise Exception('Not supported')
				plt.imshow(np.sum(paf.astype(np.float32), axis=-1))
		else:
			FIRST_BATCH = 0
			SEQLEN = len(ins[0][0])
			plt.figure(figsize=(14, 10))
			for tii in range(SEQLEN):
				plt.subplot(3, SEQLEN, tii+1)
				if tii == 0: plt.gca().set_title('Epoch: %d   Batch: %d' % (self.ecount, self.bcount))
				plt.axis('off')
				img = ins[0][FIRST_BATCH][tii] # first batch, iterate over seq
				img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
				plt.imshow(img.astype(np.float32)/256)
				mask = ins[2][FIRST_BATCH][tii][:, :, -1] # first batch, iterate over seq
				mask = cv2.resize(mask, (0,0), fx=8, fy=8)
				plt.imshow(mask, alpha=0.5)

			for tii in range(SEQLEN):
				plt.subplot(3, SEQLEN, SEQLEN+tii+1)
				plt.axis('off')
				if outform == 'join':
					if tii == SEQLEN-1: # last
						# get the actual last heat out ~ network output
						LAST_LAYER = -1
						heat = results[LAST_LAYER][FIRST_BATCH]
					else:
						# get intermediate sequential outs from middle layers
						SECOND_LAYER = 1
						STACK_PATTERN = 2
						heat = results[SECOND_LAYER * STACK_PATTERN + 1][FIRST_BATCH][tii]
				elif outform == 'last':
					if tii == SEQLEN-1: # last
						# get the actual last heat out ~ network output
						LAST_LAYER = -1
						heat = results[LAST_LAYER][FIRST_BATCH]
					else:
						continue
				else: raise Exception('Not supported')
				plt.imshow(np.sum(heat[:, :, :-1].astype(np.float32), axis=-1))
			for tii in range(SEQLEN):
				plt.subplot(3, SEQLEN, 2*SEQLEN+tii+1)
				plt.axis('off')
				if outform == 'join':
					if tii == SEQLEN-1: # last
						# get the actual last heat out ~ network output
						LAST_LAYER = -2
						limbs = results[LAST_LAYER][FIRST_BATCH]
					else:
						# get intermediate sequential outs from middle layers
						SECOND_LAYER = 1
						STACK_PATTERN = 2
						limbs = results[SECOND_LAYER * STACK_PATTERN][FIRST_BATCH][tii]
				elif outform == 'last':
					if tii == SEQLEN-1: # last
						# get the actual last heat out ~ network output
						LAST_LAYER = -2
						limbs = results[LAST_LAYER][FIRST_BATCH]
					else:
						continue
				else: raise Exception('Not supported')
				plt.imshow(np.sum(limbs.astype(np.float32), axis=-1))

		plt.savefig('previews/%s' % save_name, bbox_inches='tight')
		plt.close()
		# exit()

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
		# keep saving weights every eval
		self.model.save_weights('checkpoints/%s-epoch_%d.h5' % (self.tag, self.ecount))
		self.counter = 1
		self.bcount += 1

	def on_epoch_end(self, epoch, logs=None):
		self.model.save_weights('checkpoints/%s-epoch_%d.h5' % (self.tag, self.ecount))
		self.ecount += 1
		self.bcount = 0


