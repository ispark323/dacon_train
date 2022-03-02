import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import numpy as np
import argparse
from efficientnet_pytorch import EfficientNet
from src import DaconLSTM

from src.data_description import labels

from src.dataloader import get_preprocessing_fn
from src.dataloader import get_validation_augmentation
from src.dataloader import get_preprocessing
from src.dataloader import Dataset
from src.dataloader import create_single_dataloader

import csv
from tqdm import tqdm as tqdm
import sys
import time

torch.backends.cudnn.benchmark = True


class EnsembleModel(nn.Module):
    def __init__(self, model_cnn, model_rnn, num_classes):
        super(EnsembleModel, self).__init__()
        self.model_cnn = model_cnn
        self.model_rnn = model_rnn

        self.classifier = nn.Linear(int(num_classes * 2), num_classes)

    def forward(self, x_cnn, x_rnn):
        out_cnn = self.model_cnn(x_cnn)
        out_rnn = self.model_rnn(x_rnn)
        out = torch.cat((out_cnn, out_rnn), dim=1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
	start_time = time.time()

	torch.autograd.set_detect_anomaly(False)
	torch.autograd.profiler.profile(False)
	torch.autograd.profiler.emit_nvtx(False)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	parser = argparse.ArgumentParser()
	parser.add_argument('--base_folder', type=str, default='./data/test')
	parser.add_argument('--save_folder', type=str, default='./submission')
	parser.add_argument('--weight_fn', type=str, default='./weights/submission_0201_001/best_model.pth')
	# parser.add_argument('--label_fn', type=str, default='./data/sample_submission.csv')

	parser.add_argument('--device', type=str, default=device)

	args = parser.parse_args()

	# _base_dir = '/home/mongmong/Desktop/dacon/plant_disease/'
	# _base_dir = './'

	# _sample_dir = os.path.join(_base_dir, 'data', 'test')
	# _csv_saved_dir = os.path.join(_base_dir, 'submission')

	# _log_dir = os.path.join(_base_dir, 'checkpoint','submission_0201_001')
	# _model_path = os.path.join(_log_dir, 'best_model.pth')

	_sample_dir = args.base_folder
	_csv_saved_dir = args.save_folder
	_model_path = args.weight_fn

	_num_classes = 25
	_max_seq_length = 60
	_batch_size = 32
	_onehot = False
	_device = 'cuda'
	_mode = 'combined'

	_model_cnn = EfficientNet.from_name('efficientnet-b0', num_classes=_num_classes)
	_model_rnn = DaconLSTM(
		input_dim=9,
		output_dim=_num_classes,
		hidden_dim=200,
		num_layers=1
	)
	_model = EnsembleModel(
		model_cnn=_model_cnn,
		model_rnn=_model_rnn,
		num_classes=_num_classes
	)

	_model.load_state_dict(torch.load(_model_path))
	_model.to(_device)
	_model.eval()

	print("MODEL is LOADED")
	info_dict = []

	test_fn_list = os.listdir(_sample_dir)
	test_fn_list.sort()

	print("GET CSV FILES")
	for test_fn in test_fn_list:
		tmp = {'filename': str(test_fn), 'disease_label': 0}
		info_dict.append(tmp)

	print("MAKE a INFO_DICT")
	preprocessing_fn = get_preprocessing_fn(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225],
		input_range=[0, 1],
		input_space='RGB'
	)
	augmentation_fn_val = get_validation_augmentation(512, 512)

	print("AUGMENTATION COMPLETE")
	test_dataset = Dataset(
		images_dir=_sample_dir,
		infodict=info_dict,
		preprocessing=get_preprocessing(preprocessing_fn),
		augmentation=augmentation_fn_val,
		onehot=_onehot,
		num_classes=_num_classes,
		max_seq_length=_max_seq_length
	)

	test_dataloader = create_single_dataloader(
		test_dataset,
		batch_size=_batch_size,
		train=False, num_workers=12,
		sampler=None
	)
	print("DATALOADER is READY")

	stage_name = 'test'
	verbose = True

	print("START PREDICTIONS")
	labels_np = np.empty(0)
	with tqdm(test_dataloader, desc=stage_name, file=sys.stdout, disable=not (verbose)) as iterator:
		for x_cnn, x_rnn, y in iterator:
			x_cnn, x_rnn, y = x_cnn.to(_device), x_rnn.to(_device), y.to(_device)

			with torch.no_grad():
				if _mode == 'cnn':
					prediction = _model.forward(x_cnn)
				elif _mode == 'rnn':
					prediction = _model.forward(x_rnn)
				else:
					prediction = _model.forward(x_cnn, x_rnn)

			prediction = (prediction.squeeze().cpu().numpy())
			labels_np = np.append(labels_np, np.argmax(prediction, axis=1))

	print("PREDICTION COMPLETE")
	with open(os.path.join(_csv_saved_dir, 'model_003.csv'), 'wt') as output_file:
		writer = csv.DictWriter(output_file, fieldnames=["image", "label"])
		writer.writeheader()

		csv_writer = csv.writer(output_file, delimiter=',')

		for _info, pred in zip(info_dict, labels_np):
			disease_label = labels[pred]
			csv_writer.writerow([_info["filename"], disease_label])

	end_time = time.time()

	print("SPENDING TIME: ", end_time - start_time)
