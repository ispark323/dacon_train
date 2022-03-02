import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import timm
import random
from src import (
    seed_everything,
    get_sampler,
    ModelTrainer, 
    DaconDataset,
    accuracy_function, 
    get_train_transforms, 
    get_valid_transforms,
)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='./data/train')
    parser.add_argument('--label_fn', type=str, default='./data/train.csv')
    parser.add_argument('--save_folder', type=str, default='./checkpoint')
    parser.add_argument('--kfold_idx', type=int, default=0)

    parser.add_argument('--model', type=str, default='tf_efficientnetv2_s')    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--comments', type=str, default=None)

    args = parser.parse_args()
    
    assert os.path.isdir(args.base_folder), 'wrong path'
    assert os.path.isfile(args.label_fn), 'wrong path'

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)  
    
    label_df = pd.read_csv(args.label_fn)
    train_idx = random.sample(range(0, len(label_df)), int(len(label_df) * 0.7))

    train_df = label_df.iloc[train_idx]
    valid_df = label_df.drop(train_idx)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for fold_idx, (train_idx, valid_idx) in enumerate(sss.split(X=label_df['image'], y=label_df['label'])):
        if args.kfold_idx == fold_idx:
            train_df = label_df.iloc[train_idx]
            valid_df = label_df.iloc[valid_idx]

    train_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=train_df,
        transforms=get_train_transforms(),
    )

    valid_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=valid_df,
        transforms=get_valid_transforms(),
    )

    train_sampler = get_sampler(
        df=train_df,
        dataset=train_dataset
    )

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            # shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

    valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
        )

    model = timm.create_model(args.model, pretrained=True, num_classes=1000)

    loss = torch.nn.CrossEntropyLoss()
    metric = accuracy_function
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=args.lr / 1e3)

    trainer = ModelTrainer(
            model=model,
            train_loader=train_data_loader,
            valid_loader=valid_data_loader,
            loss_func=loss,
            metric_func=metric,
            optimizer=optimizer,
            device=args.device,
            save_dir=args.save_folder,
            mode='max', 
            scheduler=scheduler, 
            num_epochs=args.epochs,
            num_snapshops=None,
            parallel=False,
            use_csv=False,
            use_amp=True,
            use_wandb=False,            
        )

    # trainer.initWandb(
    #     project_name='dacon_farm',
    #     run_name=args.comments,
    #     args=args,
    # )

    trainer.train()
    
    with open(os.path.join(trainer.save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value)) 


if __name__ == '__main__':
    main()