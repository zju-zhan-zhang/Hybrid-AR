import sentencepiece as spm
import pandas as pd
from glob import glob


if __name__ == "__main__":
    vocab_size = 31
    dataset_dir_path = "/home5/zhangzhan/datasets/LibriSpeech/"
    csv_files = glob(F"{dataset_dir_path}/*.csv")
    text_lines = []
    for f in csv_files:
        if "960" in f:
            continue
        df = pd.read_csv(f)
        for l in list(df["text"]):
            text_lines.append(l)
    with open("transcription.txt", "w") as f:
        for l in text_lines:
            f.write(l+"\n")
    spm.SentencePieceTrainer.train(input='transcription.txt', model_prefix=F'librispeech_{vocab_size}', vocab_size=vocab_size, character_coverage=1.0)