"""preprocessing csv files"""

import os
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm


def _data_frame_to_save(data_frame, csv_path: str):
    # window 에서 path 안에 \\가 있는 경우가 있음
    csv_path = csv_path.replace("\\", "/")

    split_csv_path = csv_path.split("/")
    save_csv_folder_name = "/".join(split_csv_path[:-1])
    save_csv_file_name = split_csv_path[-1]

    save_path = os.path.join(save_csv_folder_name, "new_" + save_csv_file_name)

    data_frame = data_frame.replace("-", "0")
    data_frame.to_csv(save_path, index=False)


def feature_selection(csv_path: List, select_column_list: List):
    """결측치가 적은 column을 추출해서 feature engineering"""
    data_frame = pd.read_csv(csv_path)
    selection_df = data_frame.loc[:, select_column_list]
    return selection_df


def feature_selection_and_save(
    csv_paths: List[str], low_missing_value_ratio_column_list: List[Tuple[str, int]]
):
    print("=" * 100 + "\nStart Feature Selection And Save New Dataset")
    column_list: List = [
        column[0] for column in low_missing_value_ratio_column_list[1:]
    ]

    for csv_path in tqdm(csv_paths):
        selection_df = feature_selection(csv_path, column_list)
        _data_frame_to_save(selection_df, csv_path)

    print("Done.\n")
