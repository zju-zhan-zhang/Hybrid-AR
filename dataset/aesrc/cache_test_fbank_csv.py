import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torchaudio.compliance.kaldi import fbank as compute_fbank
import torchaudio


if __name__ == "__main__":
    dataset_dir_path = "/home5/zhangzhan/datasets/AESRC/testwavs"
    cache_dir_path = "/home5/zhangzhan/datasets/AESRC-FbanktaKaldi-40-test/"
    os.makedirs(cache_dir_path, exist_ok=True)
    os.makedirs(F"{cache_dir_path}/features", exist_ok=True)

    df = pd.read_csv(F"{dataset_dir_path}/aesrc_raw_test.csv")
    df_list = []
    for i in tqdm(range(len(df))):
        current_df = dict(df.iloc[i])
        utt_id = current_df["utt_id"]
        raw_path = current_df["raw_path"]
        raw, sr = torchaudio.load_wav(raw_path)
        fbank = compute_fbank(raw, num_mel_bins=40)
        cache_file_path = F"{cache_dir_path}/features/{utt_id}.npy"
        np.save(cache_file_path, fbank)
        current_df["feature_path"] = cache_file_path
        current_df["feature_lens"] = len(fbank)
        df_list.append(current_df)
    df = pd.DataFrame(df_list)
    df = df.drop("raw_path", axis=1)
    df = df.sort_values(["feature_lens"])
    df.to_csv(F"{cache_dir_path}/aesrc_fbank.csv", index=False)