
import os
import sys
import random
import math
import numpy as np
import skimage.io

# Root directory of the project
MASK_RCNN_PATH = os.path.abspath("../../rcnn")

# Import Mask RCNN
sys.path.append(MASK_RCNN_PATH)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(MASK_RCNN_PATH, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(MASK_RCNN_PATH, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MASK_RCNN_PATH, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(MASK_RCNN_PATH, "images")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
			'bus', 'train', 'truck', 'boat', 'traffic light',
			'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
			'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
			'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
			'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
			'kite', 'baseball bat', 'baseball glove', 'skateboard',
			'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
			'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
			'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
			'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
			'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
			'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
			'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
			'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

class MaskRCNNResult:
	def __init__(self, roi, mask, cid, score):
		self.roi = roi
		self.mask = mask
		self.cid = cid
		self.name = class_names[cid]
		self.score = score

class MaskRCNN:

	def __init__(self):
		config = InferenceConfig()

		# Create model object in inference mode.
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

		# Load weights trained on MS-COCO
		model.load_weights(COCO_MODEL_PATH, by_name=True)

		self.model = model

	def predict(self, image):
		# Run detection
		results = self.model.detect([image], verbose=0)

		# Visualize results
		r = results[0]
		# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

		resobjs = []
		masks = np.swapaxes(np.swapaxes(r['masks'], 0, 2), 1, 2)
		for roi, mask, cid, score in zip(r['rois'], masks, r['class_ids'], r['scores']):
			# print(mask.shape)
			resobjs.append(MaskRCNNResult(roi, mask, cid, score))

		return resobjs

	def visualize(self, image, results, ax=None):
		from mrcnn.visualize import random_colors, apply_mask
		from matplotlib import patches, lines
		from matplotlib.patches import Polygon
		from skimage.measure import find_contours

		figsize=(16, 16)
		scores=None
		title=""

		show_mask=True
		show_bbox=True
		colors=None
		captions=None

		boxes, masks, class_ids = [], [], []
		for res in results:
			boxes.append(res.roi)
			masks.append(res.mask)
			class_ids.append(res.cid)

		boxes = np.array(boxes)
		masks = np.array(masks)
		masks = np.swapaxes(np.swapaxes(masks, 0, 2), 0, 1)

		# print(boxes.shape, masks.shape)
		class_ids = np.array(class_ids)

		# print(masks.shape, image.shape)
		# Number of instances
		N = boxes.shape[0]
		if not N:
			print("\n*** No instances to display *** \n")
		else:
			assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

		# If no axis is passed, create one and automatically call show()
		auto_show = False
		if not ax:
			_, ax = plt.subplots(1, figsize=figsize)
			auto_show = True

		# Generate random colors
		colors = colors or random_colors(N)

		# Show area outside image boundaries.
		height, width = image.shape[:2]
		ax.set_ylim(height + 10, -10)
		ax.set_xlim(-10, width + 10)
		ax.axis('off')
		ax.set_title(title)

		masked_image = image.astype(np.uint32).copy()
		for i in range(N):
			color = colors[i]

			# Bounding box
			if not np.any(boxes[i]):
				# Skip this instance. Has no bbox. Likely lost in image cropping.
				continue
			y1, x1, y2, x2 = boxes[i]
			if show_bbox:
				p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
									alpha=0.7, linestyle="dashed",
									edgecolor=color, facecolor='none')
				ax.add_patch(p)

			# Label
			if not captions:
				class_id = class_ids[i]
				score = scores[i] if scores is not None else None
				label = class_names[class_id]
				x = random.randint(x1, (x1 + x2) // 2)
				caption = "{} {:.3f}".format(label, score) if score else label
			else:
				caption = captions[i]
			ax.text(x1, y1 + 8, caption,
					color='w', size=11, backgroundcolor="none")

			# Mask
			mask = masks[:, :, i]
			if show_mask:
				masked_image = apply_mask(masked_image, mask, color)

			# Mask Polygon
			# Pad to ensure proper polygons for masks that touch image edges.
			padded_mask = np.zeros(
				(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
			padded_mask[1:-1, 1:-1] = mask
			contours = find_contours(padded_mask, 0.5)
			for verts in contours:
				# Subtract the padding and flip (y, x) to (x, y)
				verts = np.fliplr(verts) - 1
				p = Polygon(verts, facecolor="none", edgecolor=color)
				ax.add_patch(p)
		ax.imshow(masked_image.astype(np.uint8))
		if auto_show:
			plt.show()