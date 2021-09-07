import pandas as pd
import json
from tqdm import tqdm
from functools import partial


if __name__ == "__main__":
    with open("cv_speakers.json", "r") as f:
        cv_speakers = json.load(f)
    dataset_dir_path = "/home5/zhangzhan/datasets/AESRC-FbanktaKaldi-40/"
    # csv_prefix = "aesrc_fbank_spm"
    csv_prefix = "aesrc_fbank_phoneme"
    df = pd.read_csv(F"{dataset_dir_path}/{csv_prefix}.csv")
    fake_error = True
    if fake_error:
        df["gt_label"] = df["label"]
        from utils.text_utils import PhonemeReplacer
        # p = 0.4
        p = 1
        replacer = PhonemeReplacer(random_seed=int(p*10))
        replace_func = partial(replacer.replace_labels, p=p)
        df["label"] = df["label"].apply(eval)
        df["label"] = df["label"].apply(replace_func)


    by_speaker = True
    # by_speaker = False
    if by_speaker:
        train_dicts = []
        val_dicts = []
        for i in tqdm(range(len(df))):
            current_df = df.iloc[i]
            accent = current_df["accent"]
            speaker = current_df["speaker"]
            if speaker in cv_speakers[accent]:
                val_dicts.append(dict(current_df))
            else:
                train_dicts.append(dict(current_df))
        train_df = pd.DataFrame(train_dicts)
        val_df = pd.DataFrame(val_dicts)
    else:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=1)

    # postfix = "_speakerR"
    postfix = F"_p{p}"
    train_df.to_csv(F"{dataset_dir_path}/{csv_prefix}_train{postfix}.csv")
    val_df.to_csv(F"{dataset_dir_path}/{csv_prefix}_val{postfix}.csv")
