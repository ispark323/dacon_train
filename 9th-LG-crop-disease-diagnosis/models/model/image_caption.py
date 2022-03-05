import timm
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from models.model.models import Model


# pylint: disable=invalid-name
@Model.register("image_caption")
class ImageCaptionModel(Model):
    def __init__(self, **kwargs):
        super().__init__()

        backbone = kwargs["backbone"]
        num_layer = kwargs["num_layer"]
        d_model = kwargs["d_model"]
        nhead = kwargs["nhead"]
        fc = kwargs["fc"]
        num_classes = kwargs["num_classes"]

        self.encoder = timm.create_model(backbone, pretrained=True)
        self.encoder_fc = nn.Linear(fc, d_model)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layer)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, tgt):
        x = self.encoder.forward_features(src)
        x = x.view([-1, (x.shape[2] * x.shape[3]), x.shape[1]])
        memory = self.encoder_fc(x)

        x = self.decoder(tgt, memory)
        x = x.view([-1, x.shape[2], x.shape[1]])
        x = self.gap(x)
        x = x.view([-1, x.shape[1]])

        output = self.fc(x)
        return output
