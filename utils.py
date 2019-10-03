from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

class FullModel(nn.Module):
	def __init__(self, model, loss):
		super(FullModel, self).__init__()
		self.model = model
		self.loss = loss

	def forward(self, inputs, labels):
		outputs, _ = self.model(inputs)
		loss = self.loss(outputs, labels.long())
		return torch.unsqueeze(loss,0), outputs

class AverageMeter(object):

	def __init__(self):
		self.initialized = False
		self.val = None
		self.avg = None
		self.sum = None
		self.count = None

	def initialize(self, val, weight):
		self.val = val
		self.avg = val
		self.sum = val * weight
		self.count = weight
		self.initialized = True

	def update(self, val, weight=1):
		if not self.initialized:
			self.initialize(val, weight)
		else:
			self.add(val, weight)

	def add(self, val, weight):
		self.val = val
		self.sum += val * weight
		self.count += weight
		self.avg = self.sum / self.count

	def value(self):
		return self.val

	def average(self):
		return self.avg

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
	output = pred.cpu().numpy().transpose(0, 2, 3, 1)
	seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
	seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

	ignore_index = seg_gt != ignore
	seg_gt = seg_gt[ignore_index]
	seg_pred = seg_pred[ignore_index]

	index = (seg_gt * num_class + seg_pred).astype('int32')
	label_count = np.bincount(index)
	confusion_matrix = np.zeros((num_class, num_class))

	for i_label in range(num_class):
		for i_pred in range(num_class):
			cur_index = i_label * num_class + i_pred
			if cur_index < len(label_count):
				confusion_matrix[i_label, i_pred] = label_count[cur_index]
	return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
		cur_iters, power=0.9):
	lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
	optimizer.param_groups[0]['lr'] = lr
	return lr