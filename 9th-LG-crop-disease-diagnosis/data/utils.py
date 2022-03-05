import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from constant import (
    SELECT_NUMBER_OF_ROW,
    CROP_DISEASE_DICT,
    LABEL_DICT,
)


def feature_scaling(feature: np.array, scale_type: Optional[str]):
    if scale_type is None:
        return feature

    scaler = None
    if scale_type == "minmax":
        scaler = MinMaxScaler()

    elif scale_type == "standard":
        scaler = StandardScaler()

    scaler.fit(feature)

    return scaler.transform(feature)


def get_feature_from_csv(csv_path: str, scale_type: Optional[str] = None):
    feature = pd.read_csv(csv_path).to_numpy()

    feature = feature_scaling(feature, scale_type)

    feature = feature[:SELECT_NUMBER_OF_ROW, :].T

    if feature.shape[1] < SELECT_NUMBER_OF_ROW:
        pad_num = SELECT_NUMBER_OF_ROW - feature.shape[1]
        feature = np.pad(
            feature, ((0, 0), (0, pad_num)), "constant", constant_values=0.0
        )

    return feature


class MakeLabelFromJson:
    @staticmethod
    def get_risk_label_from_json(json_file_path) -> int:
        with open(json_file_path, "r") as path:
            json_data = json.load(path)
            label = json_data["annotations"]["risk"]

        return label

    @staticmethod
    def get_crop_disease_label_from_json(json_file_path) -> int:
        with open(json_file_path, "r") as path:
            json_data = json.load(path)
            crop = json_data["annotations"]["crop"]
            disease = json_data["annotations"]["disease"]
            label = CROP_DISEASE_DICT[f"{crop}_{disease}"]

        return label

    @staticmethod
    def get_crop_disease_risk_label_from_json(json_file_path) -> int:
        with open(json_file_path, "r") as path:
            json_data = json.load(path)
            crop = json_data["annotations"]["crop"]
            disease = json_data["annotations"]["disease"]
            risk = json_data["annotations"]["risk"]
            label = LABEL_DICT[f"{crop}_{disease}_{risk}"]

        return label
