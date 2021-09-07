import torchaudio
from dataset.transforms import Compose, ComputeMagSpectrogram, LoadAudio
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import re


if __name__ == "__main__":
    dataset_dir_path = "/home5/zhangzhan/datasets/AESRC/"
    files = os.listdir(dataset_dir_path)
    accents = []
    for f in files:
        if os.path.isdir(F"{dataset_dir_path}/{f}") and f[0] != '.':
            accents.append(f)
    sub_rules = { " MR\.":" MR_", " MRS\.":" MRS_", " MS\.":"MS_",
                  " ST\.":"STREET", "\s+":" ",
                  "^MR\.": " MR_", "^MRS\.": "^MRS_", "^MS\.": "MS_",}
    total_df = None
    for set_name in accents:
        print("*"*50)
        print("Currently processing ", set_name)
        dataset_path = os.path.join(dataset_dir_path, set_name)
        txt_list = glob(F"{dataset_path}/*/*.txt")
        txt_df_list = []
        for file in tqdm(txt_list):
            with open(file, "r") as f:
                text = f.read()
            text = text.upper()
            if "MR." in text:
                print(text)
            for k, v in sub_rules.items():
                text = re.sub(k, v, text)

            text = re.sub(r'[^\w\s]', '', text)
            utt_id = os.path.basename(file).replace(".txt", "")
            speaker = os.path.dirname(file).split("/")[-1]
            txt_df_list.append({"utt_id":utt_id, "text":text, "accent":set_name, "speaker":speaker})
        txt_df = pd.DataFrame(txt_df_list)

        wav_list = glob(F"{dataset_path}/*/*.wav")
        df_list = []
        for file in tqdm(wav_list):
            utt_id = os.path.basename(file).replace('.wav', "")
            df_list.append({"utt_id":utt_id, "raw_path":file})
        df = pd.DataFrame(df_list)
        final_df = df.set_index("utt_id").join(txt_df.set_index("utt_id"))
        if total_df is None:
            total_df = final_df
        else:
            total_df = pd.concat([total_df, final_df])
    total_df.to_csv(F"{dataset_dir_path}/aesrc_raw.csv")
