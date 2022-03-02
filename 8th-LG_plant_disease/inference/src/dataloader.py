import cv2
import os
import csv
import random
import functools

import pandas as pd
import numpy as np
import albumentations as albu

import torch
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.data_description import labels
from src.data_description import csv_col_list
from src.data_description import csv_feature_dict


def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None,  **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x


def get_preprocessing_fn(mean, std, input_space="RGB", input_range=[0, 1]):
    # params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(
        preprocess_input,
        mean=mean,
        std=std,
        input_space=input_space,
        input_range=input_range
    )


def get_training_augmentation(height, width):
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=height, width=width, always_apply=True),

        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
        #         # albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                # albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        albu.Resize(
            height=height,
            width=width,
            always_apply=True
        )
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(
            height=height,
            width=width,
            always_apply=True
        )
        # albu.PadIfNeeded(384, 480)
        # albu.PadIfNeeded(height, width)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def create_single_dataloader(ds, batch_size,
                             train=False, num_workers=16,
                             sampler=None):
    if sampler:
        data_loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
        )
    else:
        if train:
            shuffle = True
        else:
            shuffle = False
        data_loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True
        )
    return data_loader


def create_dataloaders(images_dir, csv_dir, batch_size,
                       imbalanced_ds=True,
                       trainset_ratio=0.7,
                       data_augmentation=False,
                       data_preprocessing=False,
                       onehot=True,
                       num_classes=2,
                       max_seq_length=100,
                       train_dict=None,
                       val_dict=None):

    if train_dict is None or val_dict is None:
        print("TRAIN DICT IS NOT GIVEN!!!!!!")
        info_dict = []
        dataset = pd.read_csv(os.path.join(csv_dir, 'train.csv'), delimiter=',', header=0)

        # print(dataset)
        dict_keys = list(labels.keys())
        dict_values = list(labels.values())

        for k in range(dataset.__len__()):
            labels_str = dataset['label'][k]
            labels_int = dict_keys[dict_values.index(labels_str)]
            tmp = {'filename': str(dataset['image'][k]),
                   'disease_label': int(labels_int)
                   }
            info_dict.append(tmp)

        # print(info_dict)

        # print(info_dict)
        trainset_num = int(len(info_dict) * trainset_ratio)
        random.shuffle(info_dict)

        train_dict = info_dict[:trainset_num]
        val_dict = info_dict[trainset_num:]

    preprocessing_fn = None
    augmentation_fn_train = None
    augmentation_fn_val = None

    if data_preprocessing:
        preprocessing_fn = get_preprocessing_fn(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            input_range=[0, 1],
            input_space='RGB'
        )

    if data_augmentation:
        augmentation_fn_train = get_training_augmentation(512, 512)
        augmentation_fn_val = get_validation_augmentation(512, 512)

    train_dataset = Dataset(
        images_dir=images_dir,
        augmentation=augmentation_fn_train,
        preprocessing=get_preprocessing(preprocessing_fn),
        infodict=train_dict,
        onehot=onehot,
        num_classes=num_classes,
        max_seq_length=max_seq_length
    )
    val_dataset = Dataset(
        images_dir=images_dir,
        infodict=val_dict,
        preprocessing=get_preprocessing(preprocessing_fn),
        augmentation=augmentation_fn_val,
        onehot=onehot,
        num_classes=num_classes,
        max_seq_length=max_seq_length
    )

    if imbalanced_ds:
        cnt = np.zeros(num_classes)
        for train_info in train_dict:
            cnt[train_info['disease_label']] += 1

        print(cnt)
        weights = np.zeros(num_classes)
        for k in range(num_classes):
            if cnt[k] != 0:
                weights[k] = 1.0 / cnt[k]

        weights = weights / np.sum(weights)

        sample_weight = np.zeros(len(train_dict))
        for k, train_info in enumerate(train_dict):
            sample_weight[k] = weights[train_info['disease_label']]

        sample_weight = torch.from_numpy(sample_weight)

        sampler = WeightedRandomSampler(
            sample_weight.type('torch.DoubleTensor'),
            len(sample_weight)
        )
    else:
        sampler = None

    train_dataloader = create_single_dataloader(
        train_dataset, batch_size,
        train=True, num_workers=8,
        sampler=sampler
    )

    val_dataloader = create_single_dataloader(
        val_dataset, batch_size,
        train=False, num_workers=4,
        sampler=None
    )

    return train_dataloader, val_dataloader


class Dataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            infodict,
            augmentation=None,
            preprocessing=None,
            onehot=True,
            num_classes=2,
            max_seq_length=100
    ):
        self.infodict = infodict
        self.images_dir = images_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.onehot = onehot
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        image_filename = os.path.join(
            self.images_dir,
            self.infodict[idx]['filename'],
            self.infodict[idx]['filename']+'.jpg'
        )

        csv_filename = os.path.join(
            self.images_dir,
            self.infodict[idx]['filename'],
            self.infodict[idx]['filename']+'.csv'
        )

        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        df = pd.read_csv(csv_filename)
        df = df[csv_col_list]
        df = df.replace('-', 0)
        # for col in csv_col_list:
        #    df.drop(df.loc[df[col] == '-'].index, inplace=True)

        for col in df.columns:
            df[col] = df[col].astype(float) - csv_feature_dict[col][0]
            df[col] = df[col] / (csv_feature_dict[col][1] - csv_feature_dict[col][0])

        seq_full_length = np.zeros((self.max_seq_length, 9), dtype=np.float32)
        seq = df.to_numpy().astype(np.float32)

        if seq.shape[0] > self.max_seq_length:
            seq_full_length[:] = seq[-self.max_seq_length:]
        else:
            seq_full_length[-seq.shape[0]:] = seq[:]

        # print(seq_full_length)
        # seq_full_length = seq_full_length / 100.0

        # print(seq_full_length.shape)
        label_idx = self.infodict[idx]['disease_label']

        if self.onehot:
            label = np.zeros(self.num_classes)
            label[label_idx] = 1
            label = label.astype('float32')
        else:
            label = label_idx

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        # print(seq.shape)
        # print(seq_full_length.shape)
        return image, seq_full_length, label

    def __len__(self):
        return len(self.infodict)
