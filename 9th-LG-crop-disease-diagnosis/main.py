import argparse
import os
from glob import glob

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from constant import (
    TRAIN_IMAGE_PATH,
    TRAIN_JSON_PATH,
    NEW_TRAIN_CSV_PATH,
    TEST_IMAGE_PATH,
    NEW_TEST_CSV_PATH,
    SAVE_MODEL_NAME,
    SELECT_NUMBER_OF_ROW,
    LABEL_DICT,
    LABEL_DECODE_DICT,
    SUBMISSION_CSV_PATH,
)
from data import UsingCSVDataset
from data.data_loader import get_data_loader
from data.utils import MakeLabelFromJson
from models.model.models import Model
from models.runners.training_runner import TrainingRunner
from models.runners.inference_runner import InferenceRunner
from utils.class_weight import get_class_weights
from utils.fix_seed import seed_torch
from utils.get_path import (
    get_image_paths,
    get_json_paths,
    get_csv_paths,
    get_save_kfold_model_path,
)

from utils.translation import str2bool


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def kfold_main_loop(
    args,
    train_img_paths,
    train_csv_paths,
    train_json_paths,
    valid_img_paths,
    valid_csv_paths,
    valid_json_paths,
    test_img_paths,
    test_csv_paths,
    fold_num,
):

    train_dataset = UsingCSVDataset(
        img_paths=train_img_paths,
        csv_paths=train_csv_paths,
        json_paths=train_json_paths,
        training=True,
        img_size=args.img_size,
        use_augmentation=True,
        scale_type=args.scale_type,
    )
    valid_dataset = UsingCSVDataset(
        img_paths=valid_img_paths,
        csv_paths=valid_csv_paths,
        json_paths=valid_json_paths,
        training=True,
        img_size=args.img_size,
        use_augmentation=False,
        scale_type=args.scale_type,
    )
    test_dataset = UsingCSVDataset(
        img_paths=test_img_paths,
        csv_paths=test_csv_paths,
        training=False,
        img_size=args.img_size,
        use_augmentation=False,
        scale_type=args.scale_type,
    )

    # ===========================================================================
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(
        train_dataset, valid_dataset, test_dataset, args.batch_size, args.num_worker
    )

    # ===========================================================================
    model = Model.get_model(args.model_name, args.__dict__).to(device)

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
    save_model_path, save_folder_path = get_save_kfold_model_path(
        args.save_path, SAVE_MODEL_NAME, fold_num
    )

    # ===========================================================================
    train_runner = TrainingRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=device,
        max_grad_norm=args.max_grad_norm,
    )

    # ===========================================================================
    prev_valid_loss: float = 1e4
    t_loss, t_acc, t_f1_score = [], [], []
    v_loss, v_acc, v_f1_score = [], [], []

    for epoch in range(args.epochs):
        print(f"Epoch : {epoch + 1}")
        train_loss, train_acc, train_f1_score = train_runner.run(
            train_data_loader, epoch + 1, mixup=args.mixup
        )
        t_loss.append(train_loss)
        t_acc.append(train_acc)
        t_f1_score.append(train_f1_score)
        print(
            f"Train loss : {train_loss}, Train acc : {train_acc}, F1-score : {train_f1_score}"
        )

        valid_loss, valid_acc, valid_f1_score = train_runner.run(
            valid_data_loader, epoch + 1, training=False, mixup=False
        )
        v_loss.append(valid_loss)
        v_acc.append(valid_acc)
        v_f1_score.append(valid_f1_score)
        print(
            f"Valid loss : {valid_loss}, Valid acc : {valid_acc}, F1-score : {valid_f1_score}"
        )

        train_runner.save_graph(
            save_folder_path, t_loss, t_acc, t_f1_score, v_loss, v_acc, v_f1_score
        )
        if prev_valid_loss > valid_loss:
            prev_valid_loss = valid_loss

            train_runner.save_model(save_path=save_model_path)
            train_runner.save_result(
                epoch,
                save_folder_path,
                train_f1_score,
                valid_f1_score,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc,
                args,
            )
            train_runner.save_confusion_matrix(save_folder_path)

    inference_runner = InferenceRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=device,
        max_grad_norm=args.max_grad_norm,
    )
    inference_runner.load_model(save_model_path)
    prediction = inference_runner.run(test_data_loader)
    print("len(prediction) :", len(prediction))
    print("Done.")

    decoded_prediction = [LABEL_DECODE_DICT[int(pred)] for pred in prediction]

    submission = pd.read_csv(SUBMISSION_CSV_PATH)
    submission["label"] = decoded_prediction

    print("Save submission")
    save_csv_path = os.path.join(save_folder_path, f"fold{fold_num+1}_submission.csv")
    submission.to_csv(save_csv_path, index=False)
    print("Done.")


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
    args.add_argument("--save_path", type=str, default="./models/saved_model/")
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
    args.add_argument("--mixup", type=str2bool, default="True")

    args = args.parse_args()

    print(args)

    seed_torch(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===========================================================================
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    num_folder = len(glob(args.save_path + "*"))
    args.save_path = os.path.join(args.save_path, str(num_folder + 1))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # ===========================================================================
    img_paths = get_image_paths(TRAIN_IMAGE_PATH)
    csv_paths = get_csv_paths(NEW_TRAIN_CSV_PATH)
    json_paths = get_json_paths(TRAIN_JSON_PATH)
    test_img_paths = get_image_paths(TEST_IMAGE_PATH)
    test_csv_paths = get_csv_paths(NEW_TEST_CSV_PATH)

    labels = [
        MakeLabelFromJson.get_crop_disease_risk_label_from_json(i) for i in json_paths
    ]

    print(f"img_paths : {len(img_paths)}")
    print(f"csv_paths : {len(csv_paths)}")
    print(f"json_paths : {len(json_paths)}\n")

    print(f"labels : {len(labels), labels[:10]}")

    fold_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, valid in skf.split(img_paths, labels):
        fold_list.append([train, valid])
        print("train", len(train), train)
        print("valid", len(valid), valid)
        print()

    for fold_num, fold in enumerate(fold_list):
        print(f"Fold num : {str(fold_num + 1)}, fold : {fold}")
        train_img_paths = [img_paths[i] for i in fold[0]]
        train_csv_paths = [csv_paths[i] for i in fold[0]]
        train_json_paths = [json_paths[i] for i in fold[0]]

        valid_img_paths = [img_paths[i] for i in fold[1]]
        valid_csv_paths = [csv_paths[i] for i in fold[1]]
        valid_json_paths = [json_paths[i] for i in fold[1]]

        kfold_main_loop(
            args,
            train_img_paths,
            train_csv_paths,
            train_json_paths,
            valid_img_paths,
            valid_csv_paths,
            valid_json_paths,
            test_img_paths,
            test_csv_paths,
            fold_num,
        )
