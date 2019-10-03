import time
import numpy as np
import numpy.ma as ma
import argparse
import cv2

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

import _init_paths
import models
import dataset
from loss import CrossEntropy, OhemCrossEntropy
from config import config, update_config
from utils import FullModel, AverageMeter, get_confusion_matrix, adjust_learning_rate


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', required=True, type=str)
	parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
	args = parser.parse_args()
	update_config(config, args)

	print('Number of GPU: %d'%len(config.GPUS))
	gpus = list(config.GPUS)

	model, classheader = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

	print('Data root: ' + config.DATASET.ROOT)
	print('Batch size: %d'%(config.TRAIN.BATCH_SIZE_PER_GPU*len(config.GPUS)))

	trainset = dataset.CBIS_DDSM(
		is_augment=True,
		is_train=True,
		root=config.DATASET.ROOT
		)
	print('Number of images: %d'%len(trainset))
	trainloader = Data.DataLoader(
		trainset,
		batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(config.GPUS),
		shuffle=config.TRAIN.SHUFFLE,
		num_workers=config.WORKERS
		)

	epoch_size = config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH
	num_batches = len(trainloader)
	batch_total = num_batches * epoch_size
	print('Epoch size: %d'%epoch_size)
	print('Number of batch: %d'%batch_total)

	if config.LOSS.USE_OHEM:
		print('Loss function: OHEM')
		criterion = OhemCrossEntropy(
			thres=config.LOSS.OHEMTHRES,
			min_kept=config.LOSS.OHEMKEEP,
			weight=torch.FloatTensor(
				[0.81588661, 1.06673064, 1.09130692, 0.98948018, 1.03659565]
				)
			)
	else:
		print('Loss function: cross entropy')
		criterion = CrossEntropy()

	model = FullModel(model, criterion)
	model = nn.DataParallel(model, device_ids=gpus).cuda()

	optimizer = torch.optim.SGD(
		[{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': config.TRAIN.LR}],
		lr=config.TRAIN.LR,
		momentum=config.TRAIN.MOMENTUM,
		weight_decay=config.TRAIN.WD
		)
	print('Optimizer: SGD')

	last_epoch = 0
	# if config.TRAIN.RESUME:
	# 	checkpoint = torch.load(config.MODEL.PRETRAINED)
	# 	last_epoch = checkpoint['epoch']
	# 	model.module.load_state_dict(checkpoint['state_dict'])
	# 	optimizer.load_state_dict(checkpoint['optimizer'])

	# Pretrained on Cityscapes
	# if config.TRAIN.RESUME:
	# 	pretrained_dict = torch.load(config.MODEL.PRETRAINED)
	# 	model_dict = model.module.state_dict()
	# 	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 	model_dict.update(pretrained_dict)
	# 	model_dict['model.last_layer.3.weight'] = model.module.state_dict()['model.last_layer.3.weight']
	# 	model_dict['model.last_layer.3.bias'] = model.module.state_dict()['model.last_layer.3.bias']
	# 	model.module.load_state_dict(model_dict)
	# 	print('Checkpoint resumed')

	# Pretrained on Imagenet
	if config.TRAIN.RESUME:
		pretrained_dict = torch.load(config.MODEL.PRETRAINED)
		model_dict = model.module.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		# model_dict['model.last_layer.3.weight'] = model.module.state_dict()['model.last_layer.3.weight']
		# model_dict['model.last_layer.3.bias'] = model.module.state_dict()['model.last_layer.3.bias']
		model.module.load_state_dict(model_dict)
		print('Checkpoint resumed')

	start = time.time()
	print('Start training')
	for epoch_idx in range(last_epoch, epoch_size):
		print('Epoch: %d'%epoch_idx)
		for batch_idx, (img, label, _) in enumerate(trainloader):
			print('Batch: %d'%batch_idx)

			model.train()
			train_loss = AverageMeter()
			batch_processed = (epoch_idx - last_epoch) * num_batches + batch_idx + 1

			img = Variable(img).cuda()
			label = Variable(label).cuda()

			model.zero_grad()
			loss, _ = model(img, label)
			loss = loss.mean()
			loss.backward()
			optimizer.step()

			train_loss.update(loss.item())

			lr = adjust_learning_rate(
				optimizer,
				config.TRAIN.LR,
				batch_total,
				batch_processed
				)

			speed = batch_processed / (time.time() - start)
			remain_time = (batch_total - batch_processed - (last_epoch * num_batches)) / speed / 3600
			print('Progress: %d/%d Batch: %d/%d Loss: %f LR: %f Remaining time: %.2f hrs'%(
				epoch_idx,
				epoch_size,
				batch_processed + last_epoch * num_batches,
				batch_total,
				train_loss.average(),
				lr,
				remain_time
				))

		torch.save({
			'epoch': epoch_idx+1,
			'state_dict': model.module.state_dict(),
			'optimizer': optimizer.state_dict()
			}, config.OUTPUT_DIR + '/checkpoint/checkpoint_%02d.pth.tar'%(epoch_idx+1))
