import lightning as L
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from dataUpload import batch_size, ViTDataModule
from model import ViTLightning


wandb_logger = WandbLogger(project='ViT')
wandb_logger.experiment.config["batch_size"] = batch_size

dm = ViTDataModule()

model = ViTLightning({'embed_dim': 576,
                        'num_heads': 12,
                        'img_size': 96,
                        'depth': 8,
                        'patch_size': 4,
                        'in_chans': 3,
                        'num_classes': 10,
                        'drop_rate': 0.1}, lr=3e-4)

trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      devices=1,
                      max_epochs=7,
                      logger = wandb_logger) # type: ignore

trainer.fit(model, dm)

trainer.test(ckpt_path="best")