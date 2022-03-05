from typing import List
from torch.utils.data import DataLoader
import torch
import numpy as np

# import ttach as tta
from tqdm import tqdm
from models.runners.runner import Runner


class InferenceRunner(Runner):
    def forward(self, item):
        self._optimizer.zero_grad()
        if len(item) == 1:
            inp = item["input"].to(self._device)
            output = self._model.forward(inp)
            return output

        inp = item["input"].to(self._device)
        decoder_inp = item["decoder_input"].to(self._device)
        output = self._model.forward(inp, decoder_inp)
        return output

    def run(self, data_loader: DataLoader, epoch=None, training=False, mixup=False):
        print("=" * 25 + "Start Inference" + "=" * 25)
        prediction: List = []
        assert not training
        assert not mixup

        self._model.eval()
        with torch.no_grad():
            for item in tqdm(data_loader):
                output = self.forward(item)
                prediction.extend(torch.argmax(output, dim=-1).data.cpu().numpy())

        return prediction

    def infer(self, data_loader: DataLoader, training=False):
        print("=" * 25 + "Start Inference" + "=" * 25)
        prediction: List = []
        assert not training

        self._model.eval()
        with torch.no_grad():
            for item in tqdm(data_loader):
                output = self.forward(item)
                prediction.extend(output.data.cpu().numpy())
        prediction = np.array(prediction)

        return prediction

    def load_model(self, load_path):
        status = torch.load(load_path, map_location=torch.device("cpu"))
        self._model.load_state_dict(status["model"])
        self._model.to(self._device)
        self._optimizer.load_state_dict(status["optimizer"])
        self._scheduler.load_state_dict(status["scheduler"])

        # self._model = tta.ClassificationTTAWrapper(
        #     self._model.to(self._device),
        #     tta.aliases.five_crop_transform(224, 224),
        #     merge_mode="mean",
        # )
