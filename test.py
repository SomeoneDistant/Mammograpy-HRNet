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


def Get_mIOU(confusion_matrix):
	pos = confusion_matrix.sum(1)
	res = confusion_matrix.sum(0)
	tp = np.diag(confusion_matrix)
	IoU_array = (tp / np.maximum(1.0, pos + res - tp))
	mean_IoU = IoU_array.mean()
	return mean_IoU

class TestModel(nn.Module):
	def __init__(self, model):
		super(TestModel, self).__init__()
		self.model = model

	def forward(self, inputs):
		outputs = self.model(inputs)
		return outputs

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', required=True, type=str)
	parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
	args = parser.parse_args()
	update_config(config, args)

	model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

	print('Data root: ' + config.DATASET.ROOT)

	testset = dataset.CBIS_DDSM(
		is_augment=False,
		is_train=False,
		root=config.DATASET.ROOT
		)
	print('Number of images: %d'%len(testset))
	testloader = Data.DataLoader(
		testset,
		batch_size=1,
		shuffle=False,
		num_workers=2
		)

	num_batches = len(testloader)
	print('Number of batch: %d'%num_batches)

	model = TestModel(model)
	model = model.cuda()
	checkpoint = torch.load(config.MODEL.PRETRAINED)
	model.load_state_dict(checkpoint['state_dict'])
	model.eval()

	confusion_matrix_b = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
	confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

	print('Start testing')
	start = time.time()
	with torch.no_grad():
		for batch_idx, (img, label) in enumerate(testloader):
			torch.cuda.empty_cache()
			print('Batch: %d'%batch_idx)

			img = Variable(img).cuda()
			label = Variable(label)
			size = label.size()

			try:
				pred = model(img)
				pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear')
				np.save('/rds/project/t2_vol2/rds-t2-cs056/yy475/data/work/%04d.npy'%batch_idx, pred.cpu().numpy())

				confusion_matrix_b += get_confusion_matrix(
					label,
					pred,
					size,
					config.DATASET.NUM_CLASSES,
					0
					)
				confusion_matrix += get_confusion_matrix(
					label,
					pred,
					size,
					config.DATASET.NUM_CLASSES,
					-1
					)

				del img, label, size, pred

			except:
				pass

			mIOU_b = Get_mIOU(confusion_matrix_b)
			mIOU = Get_mIOU(confusion_matrix)

			batch_processed = batch_idx + 1
			speed = batch_processed / (time.time() - start)
			remain_time = (num_batches - batch_processed) / speed / 3600
			print('Progress: %d/%d Remaining time: %.2fhrs mIOU(without BG): %.4f mIOU: %.4f'%(
				batch_processed,
				num_batches,
				remain_time,
				mIOU_b,
				mIOU
				))

		np.save('/home/yy475/MammoProject/MammoNet/work/confusion_matrix.npy', confusion_matrix)
		np.save('/home/yy475/MammoProject/MammoNet/work/confusion_matrix_b.npy', confusion_matrix_b)

		mIOU_b = Get_mIOU(confusion_matrix_b)
		mIOU = Get_mIOU(confusion_matrix)
		print('Final mIOU(without BG): %f'%mIOU_b)
		print('Final mIOU: %f'%mIOU)
