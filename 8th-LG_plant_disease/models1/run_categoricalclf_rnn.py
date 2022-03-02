import os
import torch

import random
import pandas as pd
import torch.nn as nn

from src.rnn import LSTM
from efficientnet_pytorch import EfficientNet
from src.ensemble_model import EnsembleModel

from src.classifier import CategoricalClassifier
from src.data_description import labels

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


if __name__ == '__main__':
	# user's parameters
	os.environ['CUDA_VISIBLE_DIVICES'] = '0'
	_base_dir = '../'
	_num_gpus = 1
	_batch_size = 16
	_trainset_ratio = 0.8
	_sample_dir = os.path.join(_base_dir, 'data')

	_csv_dir = _sample_dir
	_images_dir = os.path.join(
		_sample_dir,
		'train'
	)
	_log_dir = os.path.join(_base_dir, 'logs_disease')
	_imbalanced_ds = True
	_data_augmentation = True
	_data_preprocessing = True
	_logits = True
	_num_classes = 25
	_epochs = 100
	_lr_begin = 0.0005
	_onehot = False
	_max_seq_length = 60

	print("START TRAINING")

	_model_rnn = LSTM(
		input_dim=9,
		output_dim=_num_classes,
		hidden_dim=200,
		num_layers=1
	)

	categorical_classifier = CategoricalClassifier(
		model=_model_rnn,
		device='cuda',
		num_gpus=_num_gpus,
		mode='rnn' # 'cnn', 'rnn', 'combined'
		# weight_path='./best_model.pth'
	)

	_train_dataloader, _val_dataloader = categorical_classifier.data_loader(
		images_dir=_images_dir,
		csv_dir=_csv_dir,
		batch_size=_batch_size,
		trainset_ratio=_trainset_ratio,
		imbalanced_ds=_imbalanced_ds,
		data_augmentation=_data_augmentation,
		data_preprocessing=_data_preprocessing,
		num_classes=_num_classes,
		onehot=_onehot,
		max_seq_length=_max_seq_length
	)

	categorical_classifier.train(
		train_dataloader=_train_dataloader,
		val_dataloader=_val_dataloader,
		log_dir=_log_dir,
		epochs=_epochs,
		lr_begin=_lr_begin,
		logits=_logits,
		onehot=_onehot
	)
