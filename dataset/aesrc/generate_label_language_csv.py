import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torchaudio.compliance.kaldi import fbank as compute_fbank
import torchaudio
from utils.text_utils import Mapper


if __name__ == "__main__":
    spm_model = "aesrc_1k.model"
    dataset_dir_path = "/home5/zhangzhan/datasets/AESRC-FbanktaKaldi-40"
    sets = ["aesrc_fbank"]
    with open("language_mapping.txt", "r") as f:
        lines = f.readlines()
    language_mapping = {}
    for i, l in enumerate(lines):
        language_mapping[l.strip()] = i

    mapper = Mapper(spm_model)
    for set_name in sets:
        print("*"*50)
        print("Currently processing ", set_name)
        dataset_path = os.path.join(dataset_dir_path, set_name)

        df = pd.read_csv(F"{dataset_dir_path}/{set_name}.csv")
        df_list = []
        for i in tqdm(range(len(df))):
            current_df = dict(df.iloc[i])
            utt_id = current_df["utt_id"]
            text = current_df["text"]
            label = mapper.encode_sentence(text)
            current_df["label"] = label
            current_df["accent_label"] = language_mapping[current_df["accent"]]
            df_list.append(current_df)
        df = pd.DataFrame(df_list)
    df.head().to_csv("aesrc_sample.csv", index=False)
