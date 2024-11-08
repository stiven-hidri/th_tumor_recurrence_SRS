import torch
import torch.nn as nn
from torchvision.models import resnet34
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 512, n_classes: int = 1):       
        super().__init__()
        self.head = nn.Linear(emb_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0]  # Use only the cls token
        x_head = self.head(x)
        return x_head


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 12, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # Using convolutions to replace linear projections
        self.qkv = nn.Conv2d(emb_size, emb_size * 3, kernel_size=1)  # Kernel size 1 to simulate projection
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Conv2d(emb_size, emb_size, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # split keys, queries, and values
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # attention mechanism
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # apply attention weights to values
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Conv2d(emb_size, expansion * emb_size, kernel_size=1),  # 1x1 conv to expand features
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Conv2d(expansion * emb_size, emb_size, kernel_size=1),  # 1x1 conv to reduce features
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 512, drop_p: float = 0., forward_expansion: int = 4, forward_drop_p: float = 0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class DeiT(nn.Sequential):
    def __init__(self, emb_size: int = 512, depth: int = 12, n_classes: int = 1, **kwargs):
        super().__init__(
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


class TransMedModel(nn.Module):
    def __init__(self, patch_size=2):
        super(TransMedModel, self).__init__()
        
        # Define patch size
        self.patch_size = patch_size
        
        # CNN Backbone for feature extraction (e.g., ResNet34)
        self.cnn_backbone = resnet34(pretrained=True)
        self.cnn_backbone.fc = nn.Identity()  # Remove final FC layer for feature output
        
        self.transformer = DeiT()
        
    def forward(self, mr, rtd):
        # Process MR and RTD separately through the CNN
        
        mr_features = self.process_input(mr)
        rtd_features = self.process_input(rtd)
        
        # Combine MR and RTD features along a new modality dimension
        combined_features = torch.cat((mr_features, rtd_features), dim=1)  # (batch_size, 2*num_patches, feature_dim)
        
        # Step 5: Transformer Encoder
        transformer_output = self.transformer(combined_features)  # (batch_size, 2*num_patches, feature_dim)
        
        return transformer_output

    def process_input(self, x):
        
        x = x.unsqueeze(1)
        
        # Step 1: Reshape to combine channel and depth
        x = rearrange(x, 'b c d h w -> b (c d) h w')
        
        # Step 2: Construct 3-channel images from adjacent slices
        x = rearrange(x, 'b (d1 d2) h w -> b d1 3 h w', d2=3)
        
        # Step 3: Create patches of size patch_size x patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=self.patch_size, p2=self.patch_size)
        
        # Flatten patches to pass through CNN
        b, num_patches, c, p1, p2 = x.shape
        x = x.view(b * num_patches, c, p1, p2)
        
        # Step 4: Pass patches through CNN backbone
        features = self.cnn_backbone(x)  # Output: (b*num_patches, feature_dim)
        
        # Reshape to (batch_size, num_patches, feature_dim) for Transformer input
        features = features.view(b, num_patches, -1)
        
        return features
