import os, math, random, re
from os.path import *
from glob import glob
import numpy as np
import cv2

import torch
import torch.utils.data as data


'''
4:benign-calcification,
3:malignant-calcification,
2:benign-mass,
1:malignant-mass,
0:background
'''


class CBIS_DDSM(data.Dataset):
	def __init__(self, is_augment=False, is_train=True, root='/rds/project/t2_vol2/rds-t2-cs056/yy475/data'):
		self.root = root
		self.is_augment = is_augment
		self.is_train = is_train

		if self.is_train:
			self.img_list = sorted(glob(self.root + '/training/image/*.npy'))
			self.mask_list = sorted(glob(self.root + '/training/mask/*.npy'))
			assert len(self.img_list) == len(self.mask_list)
		else:
			self.img_list = sorted(glob(self.root + '/test/image/*.npy'))
			self.mask_list = sorted(glob(self.root + '/test/mask/*.npy'))
			assert len(self.img_list) == len(self.mask_list)

		self.size = len(self.img_list)

	def __getitem__(self, index):
		index = index % self.size

		image = np.load(self.img_list[index])
		if self.is_train:
			mask = np.load(self.root + '/training/mask/' + basename(self.img_list[index])[:-4] + '_mask.npy')
		else:
			mask = np.load(self.root + '/test/mask/' + basename(self.img_list[index])[:-4] + '_mask.npy')
		assert image.shape == mask.shape

		if self.is_augment:
			if random.random() < 0.5:
				image = np.copy(np.flipud(image))
				mask = np.copy(np.flipud(mask))

			if random.random() < 0.5:
				image = np.copy(np.fliplr(image))
				mask = np.copy(np.fliplr(mask))

			if random.random() < 0.5:
				scale_factor = np.random.uniform(1, 1.1)
				image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
				mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
				h, w = image.shape
				h1 = random.randint(0, h - 1152)
				w1 = random.randint(0, w - 896)
				image = np.copy(image[h1:(h1+1152), w1:(w1+896)])
				mask = np.copy(mask[h1:(h1+1152), w1:(w1+896)])
			assert image.shape == mask.shape == (1152, 896)

		if (3 in mask) or (1 in mask):
			label = torch.tensor(1, dtype=torch.float32)

		image = np.copy(np.expand_dims(image, axis=0))
		mask = np.copy(mask)

		image = torch.from_numpy(image)
		mask = torch.from_numpy(mask).long()

		return image, mask, label

	def __len__(self):
		return self.size



