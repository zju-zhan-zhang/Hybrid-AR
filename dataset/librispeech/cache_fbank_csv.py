import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torchaudio.compliance.kaldi import fbank as compute_fbank
import torchaudio


if __name__ == "__main__":
    dataset_dir_path = "/home5/zhangzhan/datasets/LibriSpeech/"
    cache_dir_path = "/home5/zhangzhan/datasets/LirbiSpeechFbanktaKaldi-40/"
    sets = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360", "train-other-500"]
    for set_name in sets:
        print("*"*50)
        print("Currently processing ", set_name)
        dataset_path = os.path.join(dataset_dir_path, set_name)
        cache_path = os.path.join(cache_dir_path, set_name)
        os.makedirs(cache_path, exist_ok=True)

        df = pd.read_csv(F"{dataset_dir_path}/{set_name}.csv")
        df_list = []
        for i in tqdm(range(len(df))):
            current_df = dict(df.iloc[i])
            utt_id = current_df["utt_id"]
            raw_path = current_df["raw_path"]
            raw, sr = torchaudio.load_wav(raw_path)
            fbank = compute_fbank(raw, num_mel_bins=40)
            cache_file_name = F"{utt_id}.npy"
            cache_file_path = os.path.join(cache_path, cache_file_name)
            np.save(cache_file_path, fbank)
            current_df["feature_path"] = cache_file_path
            current_df["feature_lens"] = len(fbank)
            df_list.append(current_df)
        df = pd.DataFrame(df_list)
        df = df.drop("raw_path", axis=1)
        df = df.sort_values(["feature_lens"])
        df.to_csv(F"{cache_dir_path}/{set_name}.csv", index=False)