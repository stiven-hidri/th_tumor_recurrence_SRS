import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet18
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from models.convolutional_backbone import MobileNet, ConvBackbone

from models.mlp_cd import MlpCD

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
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
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                emb_size: int = 512,
                drop_p: float = 0.1,
                forward_expansion: int = 4,
                forward_drop_p: float = 0.1,
                ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 1, emb_size: int = 512, depth_img:int=66, n_modalities:int=2, num_channels:int=3, use_clinical_data:bool=False, out_dim_clincal_features:int=64):
        self.patch_size = patch_size
        super().__init__()
        
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.use_clinical_data = use_clinical_data
        
        self.backbone = resnet18(pretrained=False)
        # self.backbone = ConvBackbone(in_channels=num_channels, out_dim_backbone=emb_size)
        
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, emb_size)  # Adjust final layer
        
        self.final_feat_dim = emb_size + out_dim_clincal_features if use_clinical_data else emb_size
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.final_feat_dim))
        self.positions = nn.Parameter(torch.randn(n_modalities * depth_img * self.patch_size ** 2 // num_channels + 1, self.final_feat_dim))

    def forward(self, input_transformer) -> torch.Tensor:
        # Process MR and RTD through the patch embedding layers
        
        if self.use_clinical_data and type(input_transformer) is tuple:
            multimodal_image, clinical_feat = input_transformer
        else:
            multimodal_image = input_transformer
        
        batch_size = multimodal_image.shape[0]
        
        patch_sequence = rearrange(multimodal_image, 'b n c d h w -> b (n c d) h w')
        
        patch_sequence = rearrange(patch_sequence, 'b (ncd ch3) h w -> b ncd ch3 h w', ch3=3)

        if self.patch_size > 1:
            patch_sequence = rearrange(patch_sequence, 'b d c (h p1) (w p2) -> b (d p1 p2) c h w', p1=self.patch_size, p2=self.patch_size)
        
        patch_sequence = rearrange(patch_sequence, "b l c h w -> (b l) c h w")

        patch_embeddings = self.backbone(patch_sequence)
        
        patch_embeddings = rearrange(patch_embeddings, "(b l) f -> b l f", b=batch_size)
        
        if self.use_clinical_data and type(input_transformer) is tuple:
            clinical_feat = clinical_feat.unsqueeze(1).expand(-1, patch_embeddings.shape[1], -1)
            patch_embeddings = torch.cat((patch_embeddings, clinical_feat), dim=-1)
        
        # Add the class token
        cls_token = repeat(self.cls_token, '() l f -> b l f', b=batch_size)  # Shape: [batch_size, 1, emb_size]
        x = torch.cat((cls_token, patch_embeddings), dim=1)  # Shape: [batch_size, num_patches + 1, emb_size]
        
        # Add positional embeddings
        x += self.positions
        
        return x

class DeiT(nn.Sequential):
    def __init__(self, emb_size: int = 512, depth_img: int = 66, patch_size: int = 2, depth: int = 12, num_heads: int = 8, n_classes: int = 1, use_clinical_data=False, out_dim_clincal_features=64, **kwargs):
        self.final_feat_dim = emb_size + out_dim_clincal_features if use_clinical_data else emb_size
        super().__init__(
            PatchEmbedding(depth_img=depth_img, patch_size=patch_size, emb_size=emb_size, use_clinical_data=use_clinical_data, out_dim_clincal_features=out_dim_clincal_features),
            TransformerEncoder(depth, emb_size=self.final_feat_dim, num_heads=num_heads, **kwargs)
        )

class TransMedModel(nn.Module):
    def __init__(self, patch_size=1, emb_size=512, n_classes=1, use_clinical_data=False, out_dim_clincal_features=64, dropout=.1, depth_attention = 12):
        super(TransMedModel, self).__init__()
        
        # Define patch size
        self.patch_size = patch_size
        self.use_clinical_data = use_clinical_data
        self.final_num_heads = 9 if use_clinical_data else 8
        
        self.transformer = DeiT(emb_size=emb_size, patch_size=patch_size, depth_img=66, depth=depth_attention, num_heads=self.final_num_heads, n_classes=1, drop_p=dropout, use_clinical_data=use_clinical_data, out_dim_clincal_features=out_dim_clincal_features)
        
        if use_clinical_data:
            self.cd_backbone = MlpCD(pretrained=False)
            self.cd_backbone.final_fc = nn.Identity()
            self.head = nn.Linear(emb_size+out_dim_clincal_features, n_classes)
        else:
            self.head = nn.Linear(emb_size, n_classes)
        
    def forward(self, mr, rtd, clinical_data):
        mr = mr[:, None, None, ...]
        rtd = rtd[:, None, None, ...]
        
        multimodal_image = torch.cat((mr, rtd), dim=1)  # batch_size, modalities (2), channel (1), depth, height, width
        
        input_transformer = multimodal_image
        
        # if self.patch_size > 1:
        #     input_transformer = F.pad(input_transformer, (1, 1, 1, 1, 1, 1))

        if self.use_clinical_data:
            clinical_feat = self.cd_backbone(clinical_data)
            input_transformer = (input_transformer, clinical_feat)
        
        transformer_output = self.transformer(input_transformer)[:, 0]
        
        
        # if self.use_clinical_data:
        #     clinical_feat = self.cd_backbone(clinical_data)
        #     transformer_output = torch.cat((transformer_output, clinical_feat), dim=1)
        
        out = self.head(transformer_output)
        
        return out
