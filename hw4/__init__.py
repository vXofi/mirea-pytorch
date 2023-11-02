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
from torchvision.datasets import CIFAR10

from hw4 import dataUpload
from hw4 import model
from hw4 import train