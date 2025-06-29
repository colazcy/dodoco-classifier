import torch.nn as nn
from models.ECAPA_TDNN import ECAPA_TDNN
from models.MLP import MLP


class LangID(nn.Module):
    def __init__(
        self,
        num_lang: int,
        sample_rate: int,
    ):
        super(LangID, self).__init__()
        self.encoder = ECAPA_TDNN(
            sample_rate=sample_rate,
        )
        self.mlp = MLP(
            input_size=192,
            hidden_size=192,
            output_size=num_lang,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x
