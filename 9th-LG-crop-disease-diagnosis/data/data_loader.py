from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constant import (
    TRAIN_IMAGE_PATH,
    TRAIN_JSON_PATH,
    NEW_TRAIN_CSV_PATH,
    TEST_IMAGE_PATH,
    NEW_TEST_CSV_PATH,
)
from data import ImageClassificationDataset, UsingCSVDataset
from utils.get_path import (
    get_image_paths,
    get_json_paths,
    get_csv_paths,
)


def _image_classification_dataset(args):
    img_paths = get_image_paths(TRAIN_IMAGE_PATH)
    json_paths = get_json_paths(TRAIN_JSON_PATH)
    test_img_paths = get_image_paths(TEST_IMAGE_PATH)

    (
        train_img_paths,
        valid_img_paths,
        train_json_paths,
        valid_json_paths,
    ) = train_test_split(
        img_paths,
        json_paths,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    train_dataset = ImageClassificationDataset(
        img_paths=train_img_paths,
        json_paths=train_json_paths,
        training=True,
        img_size=args.img_size,
        use_augmentation=True,
    )
    valid_dataset = ImageClassificationDataset(
        img_paths=valid_img_paths,
        json_paths=valid_json_paths,
        training=True,
        img_size=args.img_size,
        use_augmentation=False,
    )
    test_dataset = ImageClassificationDataset(
        img_paths=test_img_paths,
        training=False,
        img_size=args.img_size,
        use_augmentation=False,
    )

    return train_dataset, valid_dataset, test_dataset


def _use_csv_classification_dataset(args):
    img_paths = get_image_paths(TRAIN_IMAGE_PATH)
    csv_paths = get_csv_paths(NEW_TRAIN_CSV_PATH)
    json_paths = get_json_paths(TRAIN_JSON_PATH)
    test_img_paths = get_image_paths(TEST_IMAGE_PATH)
    test_csv_paths = get_csv_paths(NEW_TEST_CSV_PATH)

    (
        train_img_paths,
        valid_img_paths,
        train_csv_paths,
        valid_csv_paths,
        train_json_paths,
        valid_json_paths,
    ) = train_test_split(
        img_paths,
        csv_paths,
        json_paths,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

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

    return train_dataset, valid_dataset, test_dataset


def make_dataset(args):
    if args.use_csv:
        return _use_csv_classification_dataset(args)
    return _image_classification_dataset(args)


def get_data_loader(
    train_dataset, valid_dataset, test_dataset, batch_size: int, num_worker: int
):
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=True,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=False,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=False,
    )

    return train_data_loader, valid_data_loader, test_data_loader
