from einops import rearrange
import numpy as np
import os
from model.jasper_multi_output import Jasper
from model.transformer_linear_merging import MyTransformer
from model.channel_attention import ChannelAttention
import torch
import pytorch_lightning as pl
from utils.text_utils import PhonemeMapper, max_decode, cal_cer_batch
from utils.feature_utils import int_to_tensor
from dataset.phoneme_data import PhonemeDataset as AudioDataset
from dataset.wav_data import WavDataset
from config import Config
from torch.utils.data import DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
# from pytorch_lightning.metrics import Accuracy
from torchmetrics import Accuracy
import yaml
import pandas as pd

from model.loss_functions import FocalLoss, AngularPenaltySMLoss


class LitModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.train_cfg.lr
        self.loss = torch.nn.CTCLoss()
        self.mapper = PhonemeMapper("dataset/mapping.txt")
        model_cfg = yaml.load(open("model/jasper-5x3-41.yaml", "r"))
        self.model = Jasper(**model_cfg)
        self.std_model = Jasper(**model_cfg)

        self.alpha = cfg.train_cfg.alpha
        if self.alpha < 0.01:
            self.apply_ctc = False
        else:
            self.apply_ctc = True

        bdim = cfg.train_cfg.bdim
        elayers = cfg.train_cfg.elayers
        self.fusion = cfg.train_cfg.fusion
        if self.fusion is not None:
            self.proj0 = torch.nn.Sequential(
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(), )

            self.proj1 = torch.nn.Sequential(
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(), )
            if self.fusion == "add":
                pass
            elif "cat" in self.fusion:
                self.reshape_layer = torch.nn.Sequential(torch.nn.Linear(1024 * 2, 1024), )
                if "channelattn" in self.fusion:
                    self.channelattn = ChannelAttention(1024 * 2)

        self.loss_function = cfg.train_cfg.loss_function
        self.merging = MyTransformer(idim=1024, odim=8, adim=512, bdim=bdim, elayers=elayers,
                                     loss_function=cfg.train_cfg.loss_function,
                                     m=cfg.train_cfg.m,
                                     gamma=cfg.train_cfg.gamma)

        # self.save_hyperparameters()
        self.acc = Accuracy()
        self.result = []

    def setup(self, stage="test"):
        self.std_model.load_state_dict(torch.load("librispeech_jasper.pth"))
        self.std_model.eval()
        for p in self.std_model.parameters():
            p.requires_grad = False

    def setup(self, stage="fit"):
        self.std_model.load_state_dict(torch.load("librispeech_jasper.pth"))
        for p in self.std_model.parameters():
            p.requires_grad = False

    def forward(self, x_bcl, len_b, y_b):
        train_d_model = True
        if train_d_model:
            d = self.model(x_bcl)
            log_probs_bcl = d["final_output"]
            embedding_bcl = d["encoder_output"]
            embedding_blc = rearrange(embedding_bcl, "b c l -> b l c")
        else:
            with torch.no_grad():
                d = self.model(x_bcl)
                log_probs_bcl = d["final_output"]
                embedding_bcl = d["encoder_output"]
                embedding_blc = rearrange(embedding_bcl, "b c l -> b l c")

        if self.fusion is None:
            merged_embedding_blc = embedding_blc
        else:
            with torch.no_grad():
                std_d = self.std_model(x_bcl)
                std_embedding_bcl = std_d["encoder_output"]
                std_embedding_blc = rearrange(std_embedding_bcl, "b c l -> b l c")
            embedding_blc = self.proj0(embedding_blc)
            std_embedding_blc = self.proj1(std_embedding_blc)
            if self.fusion == "add":
                merged_embedding_blc = embedding_blc + std_embedding_blc
            elif "cat" in self.fusion:
                merged_embedding_blc = torch.cat([embedding_blc, std_embedding_blc], dim=-1)
                if "channelattn" in self.fusion:
                    merged_embedding_bcl = rearrange(merged_embedding_blc, "b l c->b c l")
                    channel_attn = self.channelattn(merged_embedding_bcl)
                    merged_embedding_bcl = channel_attn * merged_embedding_bcl
                    merged_embedding_blc = rearrange(merged_embedding_bcl, "b c l->b l c")
                merged_embedding_blc = self.reshape_layer(merged_embedding_blc)

        mering_d = self.merging(merged_embedding_blc, len_b, y_b)
        pred_bd = mering_d["pred"]
        h_bd = mering_d["hidden"]
        loss = mering_d["loss"]
        d = dict(log_probs_bcl=log_probs_bcl,
                 pred_bd=pred_bd,
                 h_bd=h_bd,
                 embedding_blc=embedding_blc,
                 loss=loss)
        if self.fusion is not None and "channelattn" in self.fusion:
            d["channelattn"] = channel_attn
        return d

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(opt, "max", factor=0.5, min_lr=1e-5, patience=5)
        schedulers = [
            {
                'scheduler': scheduler,
                'monitor': 'val_acc',  # Default: val_loss
                'interval': 'epoch',
                'frequency': self.cfg.train_cfg.n_val
            }
        ]
        # return [opt], schedulers
        return opt

    def run_batch(self, batch):
        features_blc = batch["features"].squeeze(0)
        labels_len = batch["labels_len"].squeeze(0)
        features_len = batch["features_len"].squeeze(0)
        labels = batch["labels"].squeeze(0)
        accent_labels = batch["accent_labels"].squeeze(0)

        features_bcl = rearrange(features_blc, "b l c -> b c l")
        embedding_lens = features_len // 2
        d = self(features_bcl, embedding_lens, accent_labels)
        log_probs_bcl = d["log_probs_bcl"]
        pred_bd = d["pred_bd"]
        h_bd = d["h_bd"]
        embedding_blc = d["embedding_blc"]
        ce_loss = d["loss"]

        bs = log_probs_bcl.shape[0]
        l = log_probs_bcl.shape[2]
        input_lengths = torch.full(size=(bs,), fill_value=l, dtype=torch.long)
        log_probs_lbc = rearrange(log_probs_bcl, "b c l -> l b c")
        if self.apply_ctc:
            ctc_loss = self.loss(log_probs_lbc, labels, input_lengths, labels_len)
            if not ctc_loss < 20000:
                loss = ce_loss
            else:
                loss = self.alpha * ctc_loss + (1 - self.alpha) * ce_loss
        else:
            ctc_loss = 0
            loss = ce_loss
        newd = {"loss": loss, "loss_ctc": ctc_loss, "loss_ce": ce_loss,
                "embedding_blc": embedding_blc,
                "log_probs_bcl": log_probs_bcl, "labels": labels, "pred_bd": pred_bd, "h_bd": h_bd}
        if self.fusion is not None and "channelattn" in self.fusion:
                newd["channelattn"] = d["channelattn"]
        return newd

    def training_step(self, batch, batch_idx):
        d = self.run_batch(batch)
        loss = d["loss"]
        loss_ctc = d["loss_ctc"]
        loss_ce = d["loss_ce"]
        self.log("loss", loss)
        self.log("loss_ctc", loss_ctc, prog_bar=True)
        self.log("loss_ce", loss_ce, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        d = self.run_batch(batch)

        labels = d["labels"]
        loss = d["loss"]
        log_probs_bcl = d["log_probs_bcl"]
        pred_bd = d["pred_bd"]

        p_label = max_decode(log_probs_bcl, dim=1)
        p_text = self.mapper.translate_batch(p_label, return_string=True, ctc_decoding=True)
        gt_text = self.mapper.translate_batch(labels, return_string=True, ctc_decoding=True)
        cer = cal_cer_batch(p_text, gt_text)
        cer = int_to_tensor(cer)

        accent_labels = batch["accent_labels"].squeeze(0)
        # p_accent_labels = max_decode(pred_bd, dim=1)
        # acc = self.calc_acc(p_accent_labels, accent_labels)
        self.acc(torch.softmax(pred_bd, dim=1), accent_labels)
        self.log("val_acc", self.acc)
        self.log("val_loss", loss)
        self.log("cer", cer)

        # Save Librispeech model Embedding if used in testing steps
        newd = dict(
            utt_ids=batch["utt_ids"],
            embedding_blc=d["embedding_blc"],
            pred_bd=pred_bd,
            accent_labels=accent_labels,
        )
        if self.fusion is not None and "channelattn" in self.fusion:
            newd["channelattn"] = d["channelattn"]
        return newd

    def on_test_epoch_start(self):
        os.makedirs("librispeech_model_cache", exist_ok=True)

    def test_step(self, batch, batch_idx):
        d = self.validation_step(batch, batch_idx)
        utt_ids = d["utt_ids"]
        embedding_blcs = d["embedding_blc"]
        accent_labels = d["accent_labels"]
        pred_bd = d["pred_bd"]
        p_accent_labels = max_decode(pred_bd, dim=1)
        # return

        cache_feature = False
        # cache_feature = True

        if self.fusion is not None and "channelattn" in self.fusion:
            channelattn_bc = d["channelattn"]
            for uid, attn_c, gt_label in zip(utt_ids, channelattn_bc, accent_labels):
                pass
                # attn_c = attn_c.cpu().numpy()
                # gt = gt_label.cpu().numpy()
                # save_path = F"attn_cache/{gt}/{uid[0]}_attn.npy"
                # save_path = F"attn_cache/{uid[0]}_attn.npy"
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # np.save(save_path, attn_c)

        for uid, e_lc, gt_label, p, p_label in zip(utt_ids, embedding_blcs, accent_labels, pred_bd, p_accent_labels):
            if cache_feature:
                mean = torch.mean(e_lc, dim=0).unsqueeze(1)
                std = torch.std(e_lc, dim=0).unsqueeze(1)
                e_c = rearrange(torch.cat((mean, std), dim=-1), "D C -> (D C)")
                e_c = e_c.cpu().numpy()
                gt = gt_label.cpu().numpy()
                p = p.cpu().numpy()
                save_path = F"embedding_cache/{gt}/{uid[0]}_emb.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, e_c)
                save_path = F"embedding_cache/{gt}/{uid[0]}_p.npy"
                np.save(save_path, p)
            else:
                d = {"uid": uid[0], "p": p_label.cpu().numpy(), "p_logit": list(p.cpu().numpy()),
                     "gt": gt_label.cpu().numpy()}
                # print(d)
                self.result.append(d)
        return

    def test_epoch_end(self, outputs):
        result_df = pd.DataFrame(self.result)
        print(len(result_df))
        filename = os.path.basename(checkpoint).replace(".ckpt", "")
        version = os.path.dirname(checkpoint).split("/")[-1]
        if version == "checkpoints":
            version = os.path.dirname(checkpoint).split("/")[-2]
        output_dir = F"paper_results/{version}_{filename}"
        print("Outputting to ", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(F"{output_dir}/{self.local_rank}.csv", index=False)

    def train_dataloader(self):
        train_dataset = AudioDataset(cfg.data_cfg, cfg.data_cfg.train_dataset_path, val=False,
                                     reverse_labels=cfg.train_cfg.reverse_label)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
        # if self.cfg.train_cfg.debug:
        #     train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
        # else:
        #     train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=56)
        return train_loader

    def val_dataloader(self):
        val_dataset = AudioDataset(cfg.data_cfg, cfg.data_cfg.val_dataset_path, val=True, reverse_labels=False)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False)
        # if self.cfg.train_cfg.debug:
        #     val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False)
        # else:
        #     val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=False, num_workers=56)
        return val_loader

    def test_dataloader(self):
        test_dataset = AudioDataset(cfg.data_cfg, cfg.data_cfg.test_dataset_path, val=True, reverse_labels=False)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=False)
        # if self.cfg.train_cfg.debug:
        #     test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=False)
        # else:
        #     test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=False, num_workers=56)
        return test_loader


if __name__ == "__main__":
    cfg = Config()
    # n_gpus = 1
    # n_gpus = [0,1,2,3]
    n_gpus = [4]
    # debug = True
    debug = False
    cfg.train_cfg.debug = debug

    train = False
    # train = True

    checkpoint = None

    model = LitModel(cfg)
    # version = "ASR_init"
    if checkpoint is not None:
        pretrained = checkpoint.split("/")[2]
    else:
        pretrained = None
    version = F"{cfg.train_cfg.fusion}"
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, mode="max", monitor="val_acc", verbose=True,
                                          dirpath=F"saved_models/{version}",
                                          filename="{epoch:04d}-{val_acc:.3f}-{cer:.3f}-{d_alpha:.3f}")
    if checkpoint is not None:
        model = model.load_from_checkpoint(checkpoint, cfg=cfg, strict=False)

    lr_logger = LearningRateMonitor(logging_interval='epoch')
    if debug:
        trainer = pl.Trainer(gpus=1)
    else:
        tb_logger = TensorBoardLogger("tb_results", name=version, default_hp_metric=False)
        trainer = pl.Trainer(gpus=n_gpus,
                             callbacks=[lr_logger, checkpoint_callback],
                             gradient_clip_val=5,
                             logger=[tb_logger],
                             precision=16,
                             accelerator="ddp")
    if train:
        trainer.fit(model)
    else:
        trainer.test(model)
