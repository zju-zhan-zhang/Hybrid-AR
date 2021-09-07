import os


class DataConfig:
    def __init__(self, i=10000):
        # self.train_dataset_path = "/home5/zhangzhan/datasets/LirbiSpeechFbanktaKaldi-40/train-clean-100_phoneme.csv"
        # self.train_dataset_path = "/home3/zhangzhan/datasets/LibriSpeechFbanktaKaldi-40/train-960_spm.csv"
        # self.val_dataset_path = "/home3/zhangzhan/datasets/LibriSpeechFbanktaKaldi-40/dev-clean_spm.csv"
        self.train_dataset_path = "/home5/zhangzhan/datasets/AESRC-FbanktaKaldi-40/aesrc_fbank_phoneme_train.csv"
        self.val_dataset_path = "/home5/zhangzhan/datasets/AESRC-FbanktaKaldi-40/aesrc_fbank_phoneme_val.csv"
        self.test_dataset_path = "/home5/zhangzhan/datasets/AESRC-FbanktaKaldi-40/aesrc_fbank_phoneme_val.csv"

        self.batch_frames = i
        self.padding = 0
        self.embedding_path = F"{os.path.dirname(self.train_dataset_path)}/std_embeddings"
        os.makedirs(self.embedding_path, exist_ok=True)


class TrainConfig:
    def __init__(self):
        self.debug = False

        self.bdim = 256
        self.elayers = 3
        # self.lr = 1e-4
        self.lr = 5e-4
        # self.lr = 1e-3

        self.n_val = 1
        # --------------------- Loss function configs ------------------------ #
        self.loss_function = "ce"
        # self.loss_function = "asoftmax"
        self.m = 0.35
        self.gamma = 2
        # self.loss_function = "focal"
        # --------------------- Merging configs --------------------------------#
        # self.fusion = None
        # self.fusion = "add"
        # self.fusion = "cat"
        self.fusion = "cat_channelattn"
        # --------------------- Other configs --------------------------------#
        self.alpha = 0.1
        self.reverse_label = False







class Config:
    def __init__(self):
        self.data_cfg = DataConfig()
        self.train_cfg = TrainConfig()
