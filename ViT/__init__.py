import os
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import STL10

from ViT import dataUpload
from ViT import model
from ViT import train