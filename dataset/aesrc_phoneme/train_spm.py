import sentencepiece as spm
import pandas as pd


if __name__ == "__main__":
    dataset_dir_path = "/home5/zhangzhan/datasets/AESRC/"
    df = pd.read_csv(F"{dataset_dir_path}/aesrc_raw.csv")
    text_lines = list(df["text"])
    with open("transcription.txt", "w") as f:
        for l in text_lines:
            f.write(l+"\n")
    spm.SentencePieceTrainer.train(input='transcription.txt', model_prefix='aesrc', vocab_size=1000, character_coverage=1.0)