import pandas as pd

from torch import Tensor

from constant import TRAIN_LABEL_CSV_PATH


def get_class_weights():
    """
    각 Class의 가중치별 계산한 값을 Return합니다.
    Pytorch의 CrossEntropyLoss의 weight 인자값입니다.
    """
    train_dataframe = pd.read_csv(TRAIN_LABEL_CSV_PATH)
    sorted_train_labels_values = (
        train_dataframe["label"].value_counts().sort_index().values
    )

    return Tensor(1 / sorted_train_labels_values)
