import os
import torch
import kaldiio as kio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from config import DataConfig
from dataset.transforms import *
from dataset.wav_data import WavDataset


class SpmDataset(WavDataset):
    def __init__(self, params: DataConfig, path, val):
        self.params = params
        self.LFR = params.LFR
        self.LFR_m = params.LFR_m
        self.LFR_n = params.LFR_n
        self.val = val
        super().__init__(params, path, val)
        self.data["label"] = self.data["label"].apply(eval)
        self.padding = params.padding

    def __getitem__(self, index):
        d = super().__getitem__(index)
        df = self.minibatch[index]

        labels = list(df["label"])
        texts = list(df["text"])
        labels, labels_len = pad_label(labels, self.padding)
        labels = torch.from_numpy(labels).long()

        d["labels_len"] = labels_len
        d["labels"] = labels
        d["texts"] = texts
        return d

    def __len__(self):
        return self.lengths


