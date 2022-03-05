"""Create a new csv through feature selection and feature extraction"""
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np

from constant import (
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    NEW_TRAIN_CSV_PATH,
    NEW_TEST_CSV_PATH,
)
from csv_preprocessing.preprocessing import feature_selection_and_save
from utils.get_path import get_csv_paths
from utils.translation import str2bool


def _cal_missing_value_ratio(
    csv_paths: List, minimum_rate: int
) -> List[Tuple[str, int]]:
    """
    missing value ratio EDA
    입력되는 csv_path list를 통해 전체 csv의 결측치 비율을 구함
    minimum_rate% 미만의 결측치를 가진 column을 return
    """
    print("=" * 100)
    print("Calculate missing value ratio")

    minimum_rate /= 100
    dataset_dict: Dict = {}
    total_data_num: int = 0
    low_missing_value_ratio_column_list: List = []

    for enum, csv_path in enumerate(tqdm(csv_paths)):
        data_frame = pd.read_csv(csv_path)
        data_frame = data_frame.replace("-", np.NaN)
        total_data_num += len(data_frame)

        for key, not_nan_data_num in zip(data_frame.keys(), data_frame.count()):
            if enum == 0:
                dataset_dict[key] = not_nan_data_num
            else:
                dataset_dict[key] += not_nan_data_num

    for key, data_num in dataset_dict.items():
        ratio = data_num / total_data_num
        if ratio >= minimum_rate:
            low_missing_value_ratio_column_list.append((key, round(ratio, 4)))

    print("low missing value ratio column")
    for column_and_ratio in low_missing_value_ratio_column_list:
        print(column_and_ratio)
    print(f"total data num : {total_data_num}")
    print("Done.\n")

    return low_missing_value_ratio_column_list


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--trainset", type=str2bool, default="True")
    args.add_argument("--testset", type=str2bool, default="True")

    args = args.parse_args()

    if args.trainset:
        if not get_csv_paths(NEW_TRAIN_CSV_PATH):
            print("CSV file preprocessing in Train Dataset")
            train_csv_paths = get_csv_paths(TRAIN_CSV_PATH)
            low_missing_value_ratio_column_list = _cal_missing_value_ratio(
                csv_paths=train_csv_paths, minimum_rate=90
            )
            feature_selection_and_save(
                csv_paths=train_csv_paths,
                low_missing_value_ratio_column_list=low_missing_value_ratio_column_list,
            )

    if args.testset:
        if not get_csv_paths(NEW_TEST_CSV_PATH):
            print("CSV file preprocessing in Test Dataset")
            test_csv_paths = get_csv_paths(TEST_CSV_PATH)
            low_missing_value_ratio_column_list = _cal_missing_value_ratio(
                csv_paths=test_csv_paths, minimum_rate=90
            )
            feature_selection_and_save(
                csv_paths=test_csv_paths,
                low_missing_value_ratio_column_list=low_missing_value_ratio_column_list,
            )
