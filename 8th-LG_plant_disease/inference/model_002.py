import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import timm
import torch.nn as nn
from src import (
    seed_everything,
    DaconDataset,
    DaconLSTM, 
    get_test_transforms, 
    load_model_weights,
)
torch.backends.cudnn.benchmark = True


class DaconModel(nn.Module):
    def __init__(self, model_cnn, model_rnn, num_classes=25):
        super().__init__()
        self.model_cnn = model_cnn
        self.model_rnn = model_rnn
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.classifier = nn.Linear(int(num_classes * 2), num_classes)

    def forward(self, img, csv):
        out_cnn = self.model_cnn(img)
        out_cnn = self.dropout1(out_cnn)
        out_rnn = self.model_rnn(csv)
        out_rnn = self.dropout2(out_rnn)
        out = torch.cat((out_cnn, out_rnn), dim=1)
        out = self.classifier(out)

        return out

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='./data/test')
    parser.add_argument('--save_folder', type=str, default='./submission')
    parser.add_argument('--weight_folder', type=str, default='./weights/0203180850')
    parser.add_argument('--label_fn', type=str, default='./data/sample_submission.csv')

    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    assert os.path.isdir(args.base_folder), 'wrong path'
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    assert os.path.isdir(args.weight_folder), 'wrong path'
    assert os.path.isfile(args.label_fn), 'wrong path'

    test_df = pd.read_csv(args.label_fn)

    test_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=test_df,
        phase='test',
        max_len=320, 
        transforms=get_test_transforms(),
    )

    test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True
        )

    model_cnn = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=25)
    
    rnn_model = DaconLSTM()
    model = DaconModel(
        model_cnn=model_cnn,
        model_rnn=rnn_model
    )

    weight_fns = glob.glob(args.weight_folder + '/*.pth')
    results = np.zeros(shape=(len(test_df), 25))

    for weight_fn in weight_fns:
        model = load_model_weights(model, weight_fn)
        model.to(device)
        model.eval()

        for idx, sample in enumerate(tqdm(test_data_loader)):
            img, csv = sample['image'].to(device), sample['csv'].to(device)
            with torch.no_grad():
                output = model(img, csv)

            batch_index = idx * args.batch_size
            results[batch_index:batch_index+args.batch_size] += output.clone().detach().cpu().numpy() ## soft-vote

    voting_results = np.array([test_dataset.decode(np.argmax(result)) for result in results])
    test_df['label'] = voting_results
    safe_fn = str(os.path.join(args.save_folder, 'model_002')) + '.csv'
    test_df.to_csv(safe_fn, index=False)

if __name__ == '__main__':
    main()