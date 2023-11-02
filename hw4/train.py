import lightning as L
import wandb
from hw4.dataUpload import CIFAR10DataModule, batch_size
from hw4.model import LitCIFAR10
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='hw4')
wandb_logger.experiment.config["batch_size"] = batch_size

dm = CIFAR10DataModule()

model = LitCIFAR10(*dm.dims, dm.num_classes, hidden_size=256)
trainer = L.Trainer(
    max_epochs=2,
    accelerator="auto",
    devices=1,
    logger = wandb_logger # type: ignore
)
trainer.fit(model)

trainer.test(ckpt_path="best")