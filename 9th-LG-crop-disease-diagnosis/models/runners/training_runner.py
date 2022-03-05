import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constant import LABELS
from models.runners.runner import Runner
from utils.mixup import mixup_data, mixup_criterion


def _save_loss_graph(save_folder_path: str, train_loss: List, valid_loss: List):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_loss, label="train_loss")
    plt.plot(valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "loss.png")
    plt.savefig(save_path)


def _save_acc_graph(
    save_folder_path: str,
    train_acc: List,
    train_f1: List,
    valid_acc: List,
    valid_f1: List,
):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.plot(train_acc, label="train_acc")
    plt.plot(valid_acc, label="valid_acc")
    plt.plot(train_f1, label="train_f1_score")
    plt.plot(valid_f1, label="valid_f1_score")
    plt.xlabel("epoch")
    plt.ylabel("acc, f1-score")
    plt.title("ACC, F1-score", fontsize=15)
    plt.legend()
    save_path = os.path.join(save_folder_path, "acc_f1_score.png")
    plt.savefig(save_path)


def _calc_accuracy(prediction, label):
    _, max_indices = torch.max(prediction, 1)
    accuracy = (max_indices == label).sum().data.cpu().numpy() / max_indices.size()[0]

    prediction = torch.argmax(prediction, dim=-1).data.cpu().numpy()
    label = label.data.cpu().numpy()
    # pylint: disable=invalid-name
    f1 = np.mean(
        f1_score(
            prediction,
            label,
            average=None,
        )
    )
    return accuracy, f1, prediction, label


class TrainingRunner(Runner):
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments
    def __init__(
        self, model: nn.Module, optimizer, scheduler, loss_func, device, max_grad_norm
    ):
        super().__init__(model, optimizer, scheduler, loss_func, device, max_grad_norm)
        self._valid_predict: List = []
        self._valid_label: List = []

    def forward(self, item):
        if len(item) == 2:
            inp = item["input"].to(self._device)
            target = item["target"].to(self._device)
            output = self._model.forward(inp)

            acc, f1, prediction, label = _calc_accuracy(output, target)
            return self._loss_func(output, target), acc, f1, prediction, label

        inp = item["input"].to(self._device)
        decoder_inp = item["decoder_input"].to(self._device)
        target = item["target"].to(self._device)
        output = self._model.forward(inp, decoder_inp)

        acc, f1, prediction, label = _calc_accuracy(output, target)
        return self._loss_func(output, target), acc, f1, prediction, label

    def _mixup_forward(self, item):
        if len(item) == 2:
            inp = item["input"].to(self._device)
            target = item["target"].to(self._device)

            inp, target_a, target_b, lam = mixup_data(inp, target, self._device)

            output = self._model.forward(inp)

            acc, f1, prediction, label = _calc_accuracy(output, target)
            return self._loss_func(output, target), acc, f1, prediction, label

        inp = item["input"].to(self._device)
        decoder_inp = item["decoder_input"].to(self._device)
        target = item["target"].to(self._device)

        inp, target_a, target_b, lam = mixup_data(inp, target, self._device)

        output = self._model.forward(inp, decoder_inp)

        acc, f1, prediction, label = _calc_accuracy(output, target)

        return (
            mixup_criterion(self._loss_func, output, target_a, target_b, lam),
            acc,
            f1,
            prediction,
            label,
        )

    def _backward(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()

    def run(self, data_loader: DataLoader, epoch: int, training=True, mixup=False):
        total_loss: float = 0.0
        total_acc: float = 0.0
        total_f1_score: float = 0.0
        total_batch: int = 0
        self._valid_predict = []
        self._valid_label = []

        if training:
            print("=" * 25 + f"Epoch {epoch} Train" + "=" * 25)
            self._model.train()
            for item in tqdm(data_loader):
                self._optimizer.zero_grad()
                if mixup and total_batch % 10 == 0:
                    loss, acc, f1, prediction, label = self._mixup_forward(item)
                else:
                    loss, acc, f1, prediction, label = self.forward(item)
                del prediction, label
                total_loss += loss.item()
                total_acc += acc
                total_f1_score += f1
                total_batch += 1

                # if batch % 5 == 0:
                #     print(f"avg loss : {total_loss / (batch + 1)}")
                #     print(f"avg acc : {total_acc / (batch + 1)}")
                self._backward(loss)
        else:
            print("=" * 25 + f"Epoch {epoch} Valid" + "=" * 25)
            self._model.eval()
            with torch.no_grad():
                for item in tqdm(data_loader):
                    loss, acc, f1, prediction, label = self.forward(item)
                    self._valid_predict.extend(prediction)
                    self._valid_label.extend(label)

                    total_loss += loss.item()
                    total_acc += acc
                    total_f1_score += f1
                    total_batch += 1

        return (
            round((total_loss / total_batch), 4),
            round((total_acc / total_batch), 4),
            round((total_f1_score / total_batch), 4),
        )

    def save_model(self, save_path):
        torch.save(
            {
                "model": self._model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler.state_dict(),
            },
            save_path,
        )

    @staticmethod
    # pylint: disable=too-many-arguments
    def save_result(
        epoch: int,
        save_folder_path: str,
        train_f1_score: float,
        valid_f1_score: float,
        train_loss: float,
        valid_loss: float,
        train_acc: float,
        valid_acc: float,
        args,
    ):

        save_json_path = os.path.join(save_folder_path, "model_spec.json")
        with open(save_json_path, "w") as json_file:
            save_json = args.__dict__
            json.dump(save_json, json_file)

        save_result_path = os.path.join(save_folder_path, "result.json")
        with open(save_result_path, "w") as json_file:
            save_result_dict: Dict = {
                "best_epoch": epoch + 1,
                "train_f1_score": train_f1_score,
                "valid_f1_score": valid_f1_score,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
            }

            json.dump(save_result_dict, json_file)
        print("Save Model and Graph\n")

    @staticmethod
    # pylint: disable=too-many-arguments
    def save_graph(
        save_folder_path: str,
        train_loss: List,
        train_acc: List,
        train_f1_score: List,
        valid_loss: List,
        valid_acc: List,
        valid_f1_score: List,
    ):
        _save_loss_graph(save_folder_path, train_loss, valid_loss)
        _save_acc_graph(
            save_folder_path, train_acc, train_f1_score, valid_acc, valid_f1_score
        )

    def save_confusion_matrix(self, save_folder_path: str):
        matrix = confusion_matrix(self._valid_label, self._valid_predict)

        data_frame = pd.DataFrame(matrix, columns=LABELS, index=LABELS)

        plt.figure(figsize=(20, 12))
        sns.heatmap(
            data_frame,
            cmap="Blues",
            annot_kws={"size": 8},
            annot=True,
            linecolor="grey",
            linewidths=0.3,
        )
        plt.xticks(rotation=-40, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Prediction")
        plt.ylabel("Answer")

        save_path = os.path.join(save_folder_path, "confusion_matrix.png")
        plt.savefig(save_path)
