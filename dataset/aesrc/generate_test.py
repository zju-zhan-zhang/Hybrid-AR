import torchaudio
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import re


if __name__ == "__main__":
    dataset_dir_path = "/home5/zhangzhan/datasets/AESRC/testwavs/"
    accents = ["wav"]
    total_df = None
    for set_name in accents:
        print("*"*50)
        print("Currently processing ", set_name)
        dataset_path = os.path.join(dataset_dir_path, set_name)
        wav_list = glob(F"{dataset_path}/*.wav")
        df_list = []
        for file in tqdm(wav_list):
            utt_id = os.path.basename(file).replace('.wav', "")
            df_list.append({"utt_id":utt_id, "raw_path":file})
        df = pd.DataFrame(df_list)
        final_df = df
        if total_df is None:
            total_df = final_df
        else:
            total_df = pd.concat([total_df, final_df])
    total_df.to_csv(F"{dataset_dir_path}/aesrc_raw_test.csv")
