import torchaudio
from dataset.transforms import Compose, ComputeMagSpectrogram, LoadAudio
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import numpy as np


if __name__ == "__main__":
    dataset_dir_path = "/home5/zhangzhan/datasets/LibriSpeech/"
    sets = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360", "train-other-500"]

    for set_name in sets:
        print("*"*50)
        print("Currently processing ", set_name)
        dataset_path = os.path.join(dataset_dir_path, set_name)
        txt_list = glob(F"{dataset_path}/*/*/*.txt")
        txt_df_list = []
        for file in tqdm(txt_list):
            with open(file, "r") as f:
                text_lines = f.readlines()
            for line in text_lines:
                line = line.strip()
                utt_id, text = line.split(" ", maxsplit=1)
                txt_df_list.append({"utt_id":utt_id, "text":text})
        txt_df = pd.DataFrame(txt_df_list)

        wav_list = glob(F"{dataset_path}/*/*/*.flac")
        df_list = []
        for file in tqdm(wav_list):
            utt_id = os.path.basename(file).replace('.flac', "")
            df_list.append({"utt_id":utt_id, "raw_path":file})
        df = pd.DataFrame(df_list)
        final_df = df.set_index("utt_id").join(txt_df.set_index("utt_id"))

        final_df.to_csv(F"{dataset_dir_path}/{set_name}.csv")