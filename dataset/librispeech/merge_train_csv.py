import torchaudio
from dataset.transforms import Compose, ComputeMagSpectrogram, LoadAudio
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import numpy as np


if __name__ == "__main__":
    post_fix = "_spm"
    dataset_dir_path = "/home5/zhangzhan/datasets/LirbiSpeechFbanktaKaldi-40/"
    # dataset_dir_path = "/home5/zhangzhan/datasets/LibriSpeech/"
    sets = ["train-clean-100", "train-clean-360", "train-other-500"]
    total_df = None
    for set_name in sets:
        print("*"*50)
        print("Currently processing ", set_name)
        csv_path = F"{dataset_dir_path}/{set_name}{post_fix}.csv"
        df = pd.read_csv(csv_path)
        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df])
    total_df = total_df.sort_values(["feature_lens"])

    csv_path = F"{dataset_dir_path}/train-960{post_fix}.csv"
    total_df.to_csv(csv_path, index=False)

