import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torchaudio.compliance.kaldi import fbank as compute_fbank
import torchaudio
from utils.text_utils import Mapper


if __name__ == "__main__":
    dataset_dir_path = "/home5/zhangzhan/datasets/LirbiSpeechFbanktaKaldi-40/"
    sets = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360", "train-other-500"]
    mapper = Mapper("../mapping.txt")
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
            phonemes = mapper.g2p_string(text)
            current_df["phonemes"] = phonemes
            current_df["label"] = mapper.encode_phonemes(phonemes)
            df_list.append(current_df)
        df = pd.DataFrame(df_list)
        df.to_csv(F"{dataset_dir_path}/{set_name}_phoneme.csv", index=False)