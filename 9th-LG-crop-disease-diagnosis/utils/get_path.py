"""get files paths and get save path"""

import os
from glob import glob
from typing import List


def get_csv_paths(path: str) -> List:
    assert path[-3:] == "csv"
    return sorted(glob(path), key=lambda x: x.split(".csv")[0][-5:])


def get_json_paths(path: str) -> List:
    assert path[-4:] == "json"
    return sorted(glob(path), key=lambda x: x.split(".json")[0][-5:])


def get_image_paths(path: str) -> List:
    assert path[-3:] == "jpg"
    return sorted(glob(path), key=lambda x: x.split(".jpg")[0][-5:])


def get_save_model_path(save_path: str, save_model_name: str):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    num_folder = len(glob(save_path + "*"))
    save_folder_path = os.path.join(save_path, str(num_folder + 1))

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    save_model_path = os.path.join(save_folder_path, save_model_name)
    print(f"Model Save Path : {save_folder_path}")

    return save_model_path, save_folder_path


def get_save_kfold_model_path(save_path: str, save_model_name: str, fold_num: int):
    # fold 저장할 폴더
    save_folder_path = os.path.join(save_path, str(fold_num + 1))

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    save_model_path = os.path.join(save_folder_path, save_model_name)
    print(f"Model Save Path : {save_folder_path}")

    return save_model_path, save_folder_path
