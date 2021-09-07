
from argparse import Namespace
from distutils.util import strtobool

import logging
import math

import torch
import pdb

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
# from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
# from espnet.nets.pytorch_backend.e2e_asr import Reporter
# from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
# from espnet.nets.pytorch_backend.nets_utils import th_accuracy
# from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
# from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
# from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
# from espnet.nets.pytorch_backend.transformer.initializer import initialize
# from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
# from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
# from espnet.nets.pytorch_backend.transformer.mask import target_mask
# from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
# from espnet.nets.scorers.ctc import CTCPrefixScorer
from einops import rearrange
from model.loss_functions import AngularPenaltySMLoss, FocalLoss
from model.loss_functions import AngleSimpleLinear, AMSoftmaxLoss



class MyTransformer(torch.nn.Module):
    # def __init__(self, idim, odim, adim, bdim=256, elayers=6, linear_units=1024):
    def __init__(self, idim, odim, adim, bdim=256, elayers=6, linear_units=1024,
                 loss_function="ce", m=2.0, gamma=2.0):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, bdim, kernel_size=1, padding=0),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
        )

        self.encoder = Encoder(
            idim=bdim,
            attention_dim=adim,
            linear_units=linear_units,
            num_blocks=elayers,
            input_layer="linear",
            dropout_rate=0.5,
            attention_dropout_rate=0.5,
            positional_dropout_rate=0.5
        )
        self.loss_function = loss_function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.focal_criterion = FocalLoss(gamma=gamma)

        self.output_prj = torch.nn.Linear(2*adim, odim)
        if self.loss_function == "asoftmax":
            self.output_prj_no_b = AngleSimpleLinear(2 * adim, odim)
            # self.output_prj_no_b = AngleSimpleLinear(odim, odim)
            self.a_criterion = AMSoftmaxLoss(m=m)
            # self.output_prj_no_b = torch.nn.Linear(2 * adim, odim, bias=False)
            # self.a_criterion = AngularPenaltySMLoss(None, None, "sphereface", m=m, fc=self.output_prj_no_b)

    def forward(self, x_btd, len_b, y_b=None):
        x_btd = x_btd[:, :max(len_b)]
        # print(x_btd.shape)
        src_mask = make_non_pad_mask(len_b.tolist()).unsqueeze(-2).to(x_btd.device)
        x_bdt = rearrange(x_btd, "b t d -> b d t")
        x_bdt = self.conv(x_bdt)

        x_btd = rearrange(x_bdt, "b d t -> b t d")
        # print(x_btd.shape)
        h_btd, h_mask = self.encoder(x_btd, src_mask)
        mean = torch.mean(h_btd, dim=1).unsqueeze(1)
        std = torch.std(h_btd, dim=1).unsqueeze(1)
        h_bd = torch.cat((mean, std), dim=-1).squeeze(1)  # (B, D)
        # output layer
        # print(pred_bd.shape)
        if y_b is not None:
            if self.loss_function == "ce":
                pred_bd = self.output_prj(h_bd)
                loss = self.criterion(pred_bd, y_b)
            elif self.loss_function == "asoftmax":
                pred_bd = self.output_prj_no_b(h_bd)
                loss = self.a_criterion(pred_bd, y_b)
            else:
                pred_bd = self.output_prj(h_bd)
                loss = self.focal_criterion(pred_bd, y_b)

        else:
            loss = 0

        return {"loss":loss, "pred":pred_bd, "hidden":h_bd}


if __name__ == "__main__":
    model = MyTransformer(idim=1024, odim=8, adim=512, bdim=256, elayers=3)
    input = torch.rand(4, 200, 1024)
    labels = torch.randint(0, 9, (4, ))
    lens = torch.randint(100, 200, (4,))
    out = model(input, lens, labels)
    print(out["pred"].shape)
