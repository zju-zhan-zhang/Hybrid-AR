import os
import torch
# import kaldiio as kio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from config import DataConfig
from dataset.transforms import *


class WavDataset(Dataset):
    def __init__(self, params: DataConfig, path, val):
        self.params = params
        self.val = val
        dataset_file_path = path
        self.data = pd.read_csv(dataset_file_path)
        batch_frames = params.batch_frames
        self.data["feature_lens"] = self.data["feature_lens"].apply(int)
        total_utts = len(self.data)
        self.data = self.data[self.data["feature_lens"] < batch_frames]
        print(F"Total/Filtered utts: {total_utts}/{total_utts - len(self.data)}")
        start = 0
        minibatch = []
        while True:
            total_frames = 0
            end = start
            next_frames = self.data["feature_lens"][end]
            while total_frames + next_frames < batch_frames and end < len(self.data):
                total_frames += next_frames
                end += 1
            # print(total_frames, end-start)
            minibatch.append(self.data[start:end])
            if end == len(self.data):
                break
            start = end
        self.lengths = len(minibatch)
        self.minibatch = minibatch

    def __getitem__(self, index):
        df = self.minibatch[index]
        features = []
        # for f in list(df["feats"]):
        # feature = kio.load_mat(f)
        for f in list(df["feature_path"]):
            feature = np.load(f)
            feature = normalization(feature)
            if not self.val:
                feature = spec_augment(feature)
            features.append(feature)

        features, features_len = pad_feature(features, 0)
        features = torch.from_numpy(features).float()
        utt_ids = list(df["utt_id"])

        return {"features":features,
                "features_len":features_len,
                "utt_ids": utt_ids,
                }

    def __len__(self):
        return self.lengths


