import os
import torch
import kaldiio as kio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from config import DataConfig


def normalization(feature):
    mean = feature.mean()
    std = feature.std()
    return (feature - mean) / std


def apply_cmvn(mat, stats):
    mean = stats[0, :-1] / stats[0, -1]
    variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
    return np.divide(np.subtract(mat, mean), np.sqrt(variance))


def spec_augment(mel_spectrogram, frequency_mask_num=1, time_mask_num=2,
                 frequency_masking_para=5, time_masking_para=15):
    tau = mel_spectrogram.shape[0]
    v = mel_spectrogram.shape[1]

    warped_mel_spectrogram = mel_spectrogram

    # Step 2 : Frequency masking
    if frequency_mask_num > 0:
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = np.random.randint(0, v-f)
            warped_mel_spectrogram[:, f0:f0+f] = 0

    # Step 3 : Time masking
    if time_mask_num > 0:
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = np.random.randint(0, tau-t)
            warped_mel_spectrogram[t0:t0+t, :] = 0

    return warped_mel_spectrogram


class AudioDataset(Dataset):
    def __init__(self, params: DataConfig, path, val):
        self.params = params
        self.val = val
        dataset_file_path = path
        self.data = pd.read_csv(dataset_file_path)
        self.data["label"] = self.data["label"].apply(eval)
        self.data["feature_lens"] = self.data["feature_lens"].apply(int)
        batch_frames = params.batch_frames
        start = 0
        minibatch = []
        while True:
            total_frames = 0
            end = start
            while total_frames < batch_frames and end < len(self.data):
                total_frames += self.data["feature_lens"][end]
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
            # feature = build_LFR_features(feature, self.LFR_m, self.LFR_n)
            feature = normalization(feature)
            if not self.val:
                feature = spec_augment(feature)
            features.append(feature)

        labels = list(df["label"])
        phonemes = list(df["phonemes"])
        features, features_len = pad_feature(features, 0)
        features = torch.from_numpy(features).float()
        labels, labels_len = pad_label(labels, 0)
        labels = torch.from_numpy(labels).long()

        return {"features":features,
                "features_len":features_len,
                "labels_len":labels_len,
                "labels":labels,
                "phonemes":phonemes,
                }

    def __len__(self):
        return self.lengths


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.

    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)

def pad_feature(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    x_len = np.asarray([x.shape[0] for x in xs])
    pad = np.ones(shape=(n_batch, x_len.max(), xs[0].shape[1])) * pad_value
    for i in range(n_batch):
        pad[i, :x_len[i]] = xs[i]
    return pad, torch.from_numpy(x_len).long()


def pad_label(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    x_len = np.asarray([len(x) for x in xs])
    pad = np.ones(shape=(n_batch, x_len.max())) * pad_value
    for i in range(n_batch):
        pad[i, :x_len[i]] = xs[i]
    return pad, torch.from_numpy(x_len).long()


