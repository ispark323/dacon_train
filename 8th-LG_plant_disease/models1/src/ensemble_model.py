import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, model_cnn, model_rnn, num_classes):
        super(EnsembleModel, self).__init__()
        self.model_cnn = model_cnn
        self.model_rnn = model_rnn

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(int(num_classes * 2), num_classes)

    def forward(self, x_cnn, x_rnn):
        # out_cnn = self.dropout1(self.model_cnn(x_cnn))
        # out_rnn = self.dropout2(self.model_rnn(x_rnn))
        out_cnn = self.model_cnn(x_cnn)
        out_rnn = self.model_rnn(x_rnn)
        out = torch.cat((out_cnn, out_rnn), dim=1)
        # out = (out_cnn + out_rnn) * 0.5
        # out = 0.1 * out_cnn + 0.9 * out_rnn
        out = self.classifier(out)
        return out
