import os
import torch
# import kaldiio as kio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from config import DataConfig
from dataset.transforms import *
from dataset.wav_data import WavDataset
import random


class PhonemeDataset(WavDataset):
    def __init__(self, params: DataConfig, path, val, read_std_embedding=False, embedding_path=None, reverse_labels=False):
        super().__init__(params, path, val)
        self.data["label"] = self.data["label"].apply(eval)
        self.padding = params.padding

        self.read_std_embedding = read_std_embedding
        self.embedding_path = embedding_path
        self.reverse_labels = reverse_labels
        # parsed as int, no need to perform eavl
        # self.data["accent_label"] = self.data["accent_label"].apply(eval)

    def adverse_labels(self, labels):
        new_labels = []
        for label in labels:
            new_label = []
            for l in label:
                r = list(range(41))
                r.pop(l)
                new_label.append(random.choice(r))
            new_labels.append(new_label)
        return new_labels

    def __getitem__(self, index):
        df = self.minibatch[index]
        d = super().__getitem__(index)
        if self.read_std_embedding:
            utt_ids = d["utt_ids"]
            embeddings = []
            for uid in utt_ids:
                e = np.read(F"{self.embedding_path}/{uid}.npy")
                embeddings.append(e)
            embeddings, _ = pad_feature(embeddings, 0)
            d["embeddings"] = embeddings
        labels = list(df["label"])
        if self.reverse_labels:
            labels = self.adverse_labels(labels)
        accent_labels = list(df["accent_label"])
        accent_labels = torch.from_numpy(np.asarray(accent_labels)).long()
        speaker_labels = list(df["speaker_id"])
        speaker_labels = torch.from_numpy(np.asarray(speaker_labels)).long()
        phonemes = list(df["phonemes"])
        labels, labels_len = pad_label(labels, 0)
        labels = torch.from_numpy(labels).long()

        d["labels_len"] = labels_len
        d["labels"] = labels
        d["accent_labels"] = accent_labels
        d["speaker_labels"] = speaker_labels
        d["phonemes"] = phonemes

        return d

    def __len__(self):
        return self.lengths


