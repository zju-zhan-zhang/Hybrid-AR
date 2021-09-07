import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from torchaudio.compliance.kaldi import fbank as compute_fbank
import torchaudio
from utils.text_utils import PhonemeMapper as Mapper


if __name__ == "__main__":
    dataset_dir_path = "/home5/zhangzhan/datasets/AESRC-FbanktaKaldi-40/"
    sets = ["aesrc_fbank"]
    mapper = Mapper("../mapping.txt")
    with open("language_mapping.txt", "r") as f:
        lines = f.readlines()
    language_mapping = {}
    for i, l in enumerate(lines):
        language_mapping[l.strip()] = i

    for set_name in sets:
        print("*"*50)
        print("Currently processing ", set_name)
        dataset_path = os.path.join(dataset_dir_path, set_name)

        df = pd.read_csv(F"{dataset_dir_path}/{set_name}.csv")
        speakers = list(df["speaker"])
        speakers = list(set(speakers))
        print(len(speakers))
        speaker_mapping = {}
        for i, s in enumerate(speakers):
            speaker_mapping[s] = i

        df_list = []
        for i in tqdm(range(len(df))):
            current_df = dict(df.iloc[i])
            utt_id = current_df["utt_id"]
            text = current_df["text"]
            phonemes = mapper.g2p_string(text)
            current_df["phonemes"] = phonemes
            current_df["accent_label"] = language_mapping[current_df["accent"]]
            current_df["label"] = mapper.encode_phonemes(phonemes)
            current_df["speaker_id"] = speaker_mapping[current_df["speaker"]]
            df_list.append(current_df)
        df = pd.DataFrame(df_list)
        df.to_csv(F"{dataset_dir_path}/{set_name}_phoneme.csv", index=False)
        df.head().to_csv("aesrc_sample.csv", index=False)
