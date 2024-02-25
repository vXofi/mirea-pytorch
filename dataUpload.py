import os
import torch
import torchvision
import lightning as L
from torchvision import transforms
from torch.utils.data import random_split


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
batch_size = 8
transform = torchvision.transforms.ToTensor()


class ViTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
    
    def prepare_data(self):
        torchvision.datasets.STL10(root='./data', split = 'train',
                                        download=True, transform=transform)
        torchvision.datasets.STL10(root='./data', split = 'test',
                                       download=True, transform=transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            stl_full = torchvision.datasets.STL10(root='./data', split = 'train', transform=transform)
            self.stl_train, self.stl_val = random_split(stl_full, [0.1, 0.9])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.stl_test = torchvision.datasets.STL10(root='./data', split = 'test', transform=transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.stl_train, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.stl_val, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.stl_test, batch_size=batch_size,
                                          shuffle=False, num_workers=2)