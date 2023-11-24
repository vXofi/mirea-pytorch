import torch
import torchvision
import torchmetrics
import lightning as L
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split
from torchmetrics.functional import accuracy
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from ViT.dataUpload import PATH_DATASETS, batch_size

class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size: int=224, patch_size: int=16, in_chans=3, embed_dim=768):
        super().__init__()
        """
        """

        self.positions = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1,1, embed_dim))

        self.projection = nn.Sequential(
             nn.Conv2d(in_channels=in_chans,
                        out_channels=embed_dim,
                        kernel_size=patch_size,
                        stride=patch_size),
                        Rearrange('b e h w -> b (h w) e'),
        )


    def forward(self, x):
        # проверка на размер изображения
        b, c, h, w = x.shape
        x = self.projection(x).flatten(2).transpose(1, 2)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Linear Layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)


        return x
    
class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=8, qkv_bias=False, attn_drop=0., out_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, out_features = dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x):

        b, n, c = x.shape

        # Attention
        qkv = self.qkv(x).reshape(b, n, 3,
                                  self.num_heads,
                                  c // self.num_heads).permute(2, 0, 3, 1, 4) #?

        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.softmax(dim=-1)
        att = self.attn_drop(att)

        # Out projection
        x = (att @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        x = self.out_drop(x)

        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()

        # Normalization
        self.norm1 = nn.LayerNorm(dim)

        # Attention
        self.attn = Attention(dim, num_heads = num_heads,
                              attn_drop = drop_rate, #same drop rate two times?
                              out_drop = drop_rate)

        # Dropout
        self.dropout = nn.Dropout(drop_rate)

        # Normalization
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        self.mlp = MLP(in_features = dim,
                       hidden_features = int(dim * mlp_ratio),
                       drop=drop_rate)


    def forward(self, x):
        # Attetnion
        x = x + self.dropout(self.attn(self.norm1(x)))

        # MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x

class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 qkv_bias=False, drop_rate=0.,):
        super().__init__()

        # variables
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # Path Embeddings, CLS Token, Position Encoding
        self.patch_embed = PatchEmbedding(img_size = img_size,
                                          patch_size = patch_size,
                                          in_chans = in_chans,
                                          embed_dim = embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer Encoder
        self.pos_drop = nn.Dropout(drop_rate) #sure?
        self.blocks = nn.ModuleList([
            Block(
                dim = embed_dim, num_heads = num_heads,
                mlp_ratio = mlp_ratio, drop_rate = drop_rate)
                for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim) # ?

        # Classifier
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        # Path Embeddings, CLS Token, Position Encoding
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        #cls_tokens = self.cls_token.expand(b, -1, -1)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer Encoder
        for block in self.blocks:
          x = block(x)

        x = self.norm(x)

        # Classifier
        x = self.head(x[:, 0])

        return x

class ViTLightning(L.LightningModule):

    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = ViT(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,5], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")