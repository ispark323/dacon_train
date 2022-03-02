import os
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import torch
import numpy as np

from src.dataloader import create_dataloaders
from src.dataloader import get_preprocessing_fn
from src.metrics import Accuracy
from src.metrics import Recall
from src.metrics import Precision
from src.metrics import Fscore
from src.metrics import ConfusionMatrix
from src.trainer import TrainEpoch
from src.trainer import ValidEpoch
from src.data_description import csv_col_list


class CategoricalClassifier:
    def __init__(self, model, num_gpus=0,
                 device='cpu', weight_path=None, mode='combined'):
        self.model = model
        self.weight_path = weight_path
        self.device = device
        self.num_gpus = num_gpus
        self.mode = mode
        if self.weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))
            # self.model = torch.load(weight_path)
        if num_gpus > 1:
            print("Let's use", num_gpus, "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)

    @staticmethod
    def data_loader(images_dir, csv_dir,
                    batch_size=32,
                    trainset_ratio=0.8,
                    imbalanced_ds=False,
                    data_augmentation=False,
                    data_preprocessing=False,
                    onehot=True,
                    num_classes=2,
                    max_seq_length=100,
                    train_dict=None,
                    val_dict=None
                    ):
        __train_dataloader, __val_dataloader = create_dataloaders(
            csv_dir=csv_dir,
            trainset_ratio=trainset_ratio,
            images_dir=images_dir,
            batch_size=batch_size,
            imbalanced_ds=imbalanced_ds,
            data_augmentation=data_augmentation,
            data_preprocessing=data_preprocessing,
            onehot=onehot,
            num_classes=num_classes,
            max_seq_length=max_seq_length,
            train_dict=train_dict,
            val_dict=val_dict
        )
        return __train_dataloader, __val_dataloader

    def train(self,
              train_dataloader,
              val_dataloader,
              log_dir,
              epochs,
              logits=True,
              lr_begin=0.0001,
              onehot=False
              ):

        now = datetime.now()
        date_time = now.strftime("%m%d%H%M%S")
        savemodel_dir = os.path.join(log_dir, date_time)
        Path(savemodel_dir).mkdir(parents=True, exist_ok=True)

        loss = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam([
            dict(params=self.model.parameters(), lr=lr_begin),
        ])
        metrics = [
            Accuracy(activation='softmax2d', onehot=onehot),
            Fscore(activation='softmax2d', onehot=onehot),
            ConfusionMatrix(activation='softmax2d', onehot=onehot),
        ]

        train_epoch = TrainEpoch(
            self.model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=self.device,
            verbose=True,
            mode=self.mode
        )

        valid_epoch = ValidEpoch(
            self.model,
            loss=loss,
            metrics=metrics,
            device=self.device,
            verbose=True,
            mode=self.mode
        )

        # train model for 100 epochs
        max_score = 0
        train_logs_filename = os.path.join(savemodel_dir, 'trainlogs.txt')
        valid_logs_filename = os.path.join(savemodel_dir, 'validlogs.txt')
        best_logs_filename = os.path.join(savemodel_dir, 'bestlogs.txt')

        for i in range(epochs):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(val_dataloader)

            with open(train_logs_filename, 'a') as train_logs_file:
                train_logs_file.write('Epochs: {} \n'.format(i))
                for key in train_logs.keys():
                    if key == 'confusion_matrix':
                        values = train_logs[key].astype('int')

                        train_logs_file.write(
                            '\n {}: \n {} \n'.format(key, values)
                        )
                    else:
                        train_logs_file.write(
                            '{}: {} \t'.format(key, train_logs[key])
                        )
                train_logs_file.write('\n')

            with open(valid_logs_filename, 'a') as valid_logs_file:
                valid_logs_file.write('Epochs: {} \n'.format(i))

                for key in valid_logs.keys():
                    if key == 'confusion_matrix':
                        values = valid_logs[key].astype('int')

                        valid_logs_file.write(
                            '\n {}: \n {} \n'.format(key, values)
                        )
                    else:
                        valid_logs_file.write(
                            '{}: {} \t'.format(key, valid_logs[key])
                        )
                valid_logs_file.write('\n')

            if max_score <= valid_logs['fscore']:
                max_score = valid_logs['fscore']

                if self.num_gpus > 1:
                    torch.save(
                        self.model.module.state_dict(),
                        # self.model.state_dict(),
                        os.path.join(savemodel_dir, 'best_model.pth')
                    )
                else:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(savemodel_dir, 'best_model.pth')
                    )
                with open(best_logs_filename, 'w') as best_logs_file:
                    best_logs_file.write('Epochs: {} \n'.format(i))
                    for key in train_logs.keys():
                        if key == 'confusion_matrix':
                            values = train_logs[key].astype('int')
                            best_logs_file.write(
                                '\n {}: \n {} \n'.format(key, values)
                            )
                        else:
                            best_logs_file.write(
                                '{}: {} \t'.format(key, train_logs[key])
                            )
                    best_logs_file.write('\n')
                    for key in valid_logs.keys():
                        if key == 'confusion_matrix':
                            values = valid_logs[key].astype('int')

                            best_logs_file.write(
                                '\n {}: \n {} \n'.format(key, values)
                            )
                        else:
                            best_logs_file.write(
                                '{}: {} \t'.format(key, valid_logs[key])
                            )
                    best_logs_file.write('\n')

                print('Model saved!')

            if i == epochs // 2:
                optimizer.param_groups[0]['lr'] = lr_begin * 0.1
                print('Decrease decoder learning rate to ', lr_begin * 0.1)
            if i == int(epochs // 4 * 3):
                optimizer.param_groups[0]['lr'] = lr_begin * 0.01
                print('Decrease decoder learning rate to ', lr_begin * 0.01)

    def predict(self, images_path, image_dir, data_preprocessing=True,
                mode='combined', max_seq_length=100):
        _labels = []
        _results = dict()
        for img_path in images_path:
            image_filename = os.path.join(
                image_dir,
                str(img_path),
                str(img_path) + '.jpg'
            )
            _image = cv2.imread(image_filename)
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
            _image = cv2.resize(_image, dsize=(512, 512))

            csv_filename = os.path.join(
                image_dir,
                str(img_path),
                str(img_path) + '.csv'
            )

            df = pd.read_csv(csv_filename)
            df = df[csv_col_list]
            for col in csv_col_list:
                df.drop(df.loc[df[col] == '-'].index, inplace=True)

            seq_full_length = np.zeros((max_seq_length, 9), dtype=np.float32)
            seq = df.to_numpy().astype(np.float32)

            if seq.shape[0] > max_seq_length:
                seq_full_length[:] = seq[-max_seq_length:]
            else:
                seq_full_length[-seq.shape[0]:] = seq[:seq.shape[0]]

            if data_preprocessing:
                preprocessing_fn = get_preprocessing_fn(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    # mean=[0.5, 0.5, 0.5],
                    # std=[0.5, 0.5, 0.5],
                    input_range=[0, 1],
                    input_space='RGB'
                )
                _image = preprocessing_fn(_image)

            _image = _image.transpose(2, 0, 1).astype('float32')
            _x_cnn = torch.from_numpy(_image).to(self.device).unsqueeze(0)
            _x_rnn = torch.from_numpy(seq_full_length).to(self.device).unsqueeze(0)

            # label = self.model.predict(x_tensor)
            if self.model.training:
                self.model.eval()

            with torch.no_grad():
                if mode == 'cnn':
                    _label = torch.softmax(
                        self.model.forward(_x_cnn), dim=1
                    )
                elif mode == 'rnn':
                    _label = torch.softmax(
                        self.model.forward(_x_rnn), dim=1
                    )
                else:
                    _label = torch.softmax(
                        self.model.forward(_x_cnn, _x_rnn), dim=1
                    )
            _label = (_label.squeeze().cpu().numpy())
            _label = np.argmax(_label)

            _labels.append(_label)

            _results[img_path] = _label
        return _results

    def predict2(self, images_path, image_dir, _results, data_preprocessing=True):
        _labels = []
        for img_path in images_path:

            _image = cv2.imread(os.path.join(image_dir, img_path))
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

            if data_preprocessing:
                preprocessing_fn = get_preprocessing_fn(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    input_range=[0, 1],
                    input_space='RGB'
                )
                _image = preprocessing_fn(_image)

            _image = _image.transpose(2, 0, 1).astype('float32')
            _x_tensor = torch.from_numpy(_image).to(self.device).unsqueeze(0)

            # label = self.model.predict(x_tensor)
            if self.model.training:
                self.model.eval()

            with torch.no_grad():
                _label = torch.softmax(
                    self.model.forward(_x_tensor), dim=1
                )
            _label = (_label.squeeze().cpu().numpy())
            _label = np.argmax(_label)

            _labels.append(_label)

            _results[img_path][1] = _label
        return _results
