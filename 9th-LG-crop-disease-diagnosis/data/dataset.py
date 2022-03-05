from typing import List, Optional

import cv2
import torch
from albumentations import (
    Compose,
    Normalize,
    Resize,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    RandomRotate90,
)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from data.utils import get_feature_from_csv, MakeLabelFromJson


# E1102: torch.tensor is not callable (not-callable)
# pylint: disable=not-callable
def _get_transforms(use_augmentation: bool, img_size: int):
    if use_augmentation:
        return Compose(
            [
                Rotate(30, p=0.5),
                RandomRotate90(p=0.5),
                Resize(img_size, img_size),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    return Compose(
        [
            Resize(img_size, img_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


class ImageClassificationDataset(Dataset):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        img_paths: List,
        json_paths: Optional[List] = None,
        training: bool = True,
        img_size: int = 224,
        use_augmentation: bool = True,
    ):
        """
        Torch dataset for image classification
        [Args]
        img_paths: input dataset path
        json_paths: label dataset path
        """
        if not training:
            assert json_paths is None
        else:
            assert json_paths

        self.img_paths = img_paths
        self.json_paths = json_paths
        self.training = training
        self.img_size = img_size
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = cv2.imread(self.img_paths[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = _get_transforms(self.use_augmentation, self.img_size)(image=img)
        img = augmented["image"]

        if self.training:
            label = MakeLabelFromJson.get_risk_label_from_json(self.json_paths[item])

            return {
                "input": img,
                "target": torch.tensor(label, dtype=torch.long),
            }

        return {"input": img}


class UsingCSVDataset(Dataset):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        img_paths: List,
        csv_paths: List,
        json_paths: Optional[List] = None,
        training: bool = True,
        img_size: int = 224,
        use_augmentation: bool = True,
        scale_type: str = "minmax",
    ):
        """
        Torch dataset for image classification using CSV
        [Args]
        img_paths: encoder input dataset path
        csv_paths: decoder input dataset path
        json_paths: label dataset path
        """
        if not training:
            assert json_paths is None
        else:
            assert json_paths

        self.img_paths = img_paths
        self.csv_paths = csv_paths
        self.json_paths = json_paths
        self.training = training
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.scale_type = scale_type

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = cv2.imread(self.img_paths[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = _get_transforms(self.use_augmentation, self.img_size)(image=img)
        img = augmented["image"]

        feature = get_feature_from_csv(self.csv_paths[item], scale_type=self.scale_type)

        if self.training:
            label = MakeLabelFromJson.get_crop_disease_risk_label_from_json(
                self.json_paths[item]
            )
            return {
                "input": img,
                "decoder_input": torch.tensor(feature, dtype=torch.float32),
                "target": torch.tensor(label, dtype=torch.long),
            }

        return {
            "input": img,
            "decoder_input": torch.tensor(feature, dtype=torch.float32),
        }
