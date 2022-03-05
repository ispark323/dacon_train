"""Using Multi-Processing In Inference"""
import argparse
from typing import List
import os
from glob import glob
from multiprocessing import Process, Queue
from time import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from constant import (
    TEST_IMAGE_PATH,
    NEW_TEST_CSV_PATH,
    SELECT_NUMBER_OF_ROW,
    LABEL_DICT,
    LABEL_DECODE_DICT,
    SUBMISSION_CSV_PATH,
)
from data import UsingCSVDataset
from models.model.models import Model
from models.runners.inference_runner import InferenceRunner
from utils.class_weight import get_class_weights
from utils.fix_seed import seed_torch
from utils.get_path import (
    get_image_paths,
    get_csv_paths,
)
from utils.translation import str2bool


def get_inference_runner(args, save_model_path):
    # ===========================================================================
    model = Model.get_model(args.model_name, args.__dict__).to(args.device)

    # ===========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    class_weight = get_class_weights()

    if args.class_weighted_loss:
        loss_func = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, weight=class_weight.cuda()
        )
    else:
        loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.T_max, eta_min=args.eta_min
    )

    # ===========================================================================
    inference_runner = InferenceRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=args.device,
        max_grad_norm=args.max_grad_norm,
    )
    inference_runner.load_model(save_model_path)
    return inference_runner


def kfold_main_loop(mp_q, args, test_img_paths, test_csv_paths, save_model_path):
    test_dataset = UsingCSVDataset(
        img_paths=test_img_paths,
        csv_paths=test_csv_paths,
        training=False,
        img_size=args.img_size,
        use_augmentation=False,
        scale_type=args.scale_type,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    inference_runner = get_inference_runner(args, save_model_path)

    print(f"Process ID : {os.getpid()}")
    prediction = inference_runner.infer(test_data_loader)
    print("len(prediction) :", prediction.shape)
    print("Done.")

    mp_q.put([os.getpid(), prediction])


def save_submission(prediction, save_folder_path):
    decoded_prediction = [LABEL_DECODE_DICT[int(pred)] for pred in prediction]

    submission = pd.read_csv(SUBMISSION_CSV_PATH)
    submission["label"] = decoded_prediction

    print("Save submission")
    save_csv_path = os.path.join(save_folder_path, "submission.csv")
    submission.to_csv(save_csv_path, index=False)
    print("Done.")


def get_index_info_to_allocate_per_process(proc_num: int, test_img_paths: List):
    number_of_data_per_process: int = int(len(test_img_paths) / proc_num)

    if proc_num == 1:
        return [0], [len(test_img_paths)]

    index_list: List = []
    index: int = 0
    for _ in range(proc_num - 1):
        index_list.append(index)
        index += int(number_of_data_per_process)
    index_list.append(index)
    index_list.append(len(test_img_paths))
    print(f"index list : {index_list}\n")

    start_index = index_list[:-1]
    end_index = index_list[1:]

    return start_index, end_index


def spawn_process(args, test_img_paths, test_csv_paths, save_model_path):
    start_index, end_index = get_index_info_to_allocate_per_process(
        args.proc_num, test_img_paths
    )

    processes: List = []
    prediction_per_proc: List = []
    mp_q = Queue()

    for start, end in zip(start_index, end_index):
        proc = Process(
            target=kfold_main_loop,
            args=(
                mp_q,
                args,
                test_img_paths[start:end],
                test_csv_paths[start:end],
                save_model_path,
            ),
        )
        processes.append(proc)
        proc.start()

    for num in range(args.proc_num):
        print(f"Get 'Process {num}' Prediction Data From Queue")
        prediction_per_proc.append(mp_q.get())
        print(f"Get in Q {num} Done.")

    for proc in processes:
        print(f"Join {proc}")
        proc.join()
        print(f"Join {proc} Done.")

    # ===========================================================================
    print("Post Processing")
    prediction_per_proc = sorted(prediction_per_proc, key=lambda x: x[0])
    prediction_per_proc = [prediction[1] for prediction in prediction_per_proc]

    prediction = None

    if len(prediction_per_proc) == 1:
        prediction = np.array(prediction_per_proc[0])
    else:
        for i in range(1, len(prediction_per_proc)):
            if i == 1:
                prediction = np.concatenate(
                    (prediction_per_proc[0], prediction_per_proc[i]), axis=0
                )
            else:
                prediction = np.concatenate(
                    (prediction, prediction_per_proc[i]), axis=0
                )
    print("prediction shape", prediction.shape)
    print("Done.\n")
    return prediction


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--backbone", type=str, default="resnet50d")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--num_classes", type=int, default=len(LABEL_DICT))
    args.add_argument("--T_max", type=int, default=10)
    args.add_argument("--eta_min", type=float, default=1e-6)
    args.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    args.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    args.add_argument("--eps", type=float, default=1e-8)
    args.add_argument("--weight_decay", type=float, default=1e-3)
    args.add_argument("--num_layer", type=int, default=1)
    args.add_argument("--d_model", type=int, default=SELECT_NUMBER_OF_ROW)
    args.add_argument("--nhead", type=int, default=8)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--max_grad_norm", type=float, default=1.0)
    args.add_argument("--img_size", type=int, default=224)
    args.add_argument("--fc", type=int, default=2048)
    args.add_argument("--num_worker", type=int, default=0)
    args.add_argument("--model_name", type=str, default="image_caption")
    args.add_argument("--label_smoothing", type=float, default=0.1)
    args.add_argument("--use_csv", type=str2bool, default="True")
    args.add_argument(
        "--scale_type", type=str, default="minmax", help="csv data scaling method"
    )
    args.add_argument("--device", type=int, default=0)
    args.add_argument(
        "--class_weighted_loss",
        type=str2bool,
        default="False",
        help="class weight loss",
    )
    args.add_argument("--proc_num", type=int, default=5)
    args.add_argument(
        "--save_model_path",
        type=str,
        default="models/saved_model/1/*/model.pt",
    )
    args.add_argument(
        "--save_folder_path",
        type=str,
        default="models/saved_model/1/",
    )

    args = args.parse_args()

    infer_start_time = time()
    torch.multiprocessing.set_start_method("spawn")
    seed_torch(args.seed)

    args.device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    # ===========================================================================
    test_img_paths = get_image_paths(TEST_IMAGE_PATH)
    test_csv_paths = get_csv_paths(NEW_TEST_CSV_PATH)

    print(f"test_img_paths : {len(test_img_paths)}")
    print(f"test_csv_paths : {len(test_csv_paths)}")

    infer_results = []

    save_model_paths = sorted(glob(args.save_model_path))
    print("saved model paths", save_model_paths)

    for fold_num, save_model_path in enumerate(save_model_paths):
        print("=" * 100)
        print(f"Model trained fold : {fold_num + 1}")
        print(f"Saved Model path : {save_model_path}")
        infer_result = spawn_process(
            args,
            test_img_paths,
            test_csv_paths,
            save_model_path,
        )
        infer_results.append(infer_result)

    print("Soft Voting")
    prediction = (
        infer_results[0]
        + infer_results[1]
        + infer_results[2]
        + infer_results[3]
        + infer_results[4]
    )
    prediction = prediction / 5
    prediction = [np.argmax(i) for i in prediction]
    print("Done.")

    save_submission(prediction, args.save_folder_path)

    print(f"Inference Time : {time() - infer_start_time} sec")
