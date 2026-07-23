import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class Embed_Patch(nn.Module):
    def __init__(self,in_channels = 3, patch_size = 4, embed_dim = 768, img_size = 224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size)**2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride= patch_size)
    def forward(self,x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(2,1) # ( B, 56*56, 768)
        return x
class TransformerBlockWithPos(nn.Module):
    def __init__(self, num_patches, embed_dim, nhead, mlp_ratio=4.0, depth=1):
        super().__init__()
        
        # 1. Tạo Vector Position riêng cho Block này (Khởi tạo ngẫu nhiên nhỏ tốt hơn số 0)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        # 2. Tạo cấu hình cho Multi-Head Attention và MLP (FeedForward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LayerNorm chuẩn ViT
        )
        
        # 3. Tạo mạng Transformer Encoder với số tầng (depth) tùy ý
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x có shape: (B, num_patches, embed_dim)
        
        # Tự động cộng Vector Position của tầng này vào dữ liệu đầu vào
        x = x + self.pos_embed
        
        # Cho chạy qua Multi-Head Attention và MLP
        x = self.transformer_encoder(x)
        
        return x
class ViT(nn.Module):
    def __init__(self, img_size = 224, patch_size = 4, in_channels = 3, num_classes = 1000,
                 window_size = 7, embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4.0):
        super().__init__()
        self.window_size = window_size
        self.embed = Embed_Patch(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.embed.num_patches
        self.embed_dim = embed_dim
        # Khai báo khối Transformer nhận đầu vào là các CỬA SỔ
        self.transformer_encoder = TransformerBlockWithPos(
            num_patches=window_size * window_size, 
            embed_dim=self.embed_dim, 
            nhead=num_heads, 
            mlp_ratio=mlp_ratio, 
            depth=depth
        )

        # lớp MLP
        self.norm = nn.LayerNorm(self.embed_dim)
        self.mlp_head = nn.Linear(self.embed_dim, num_classes)
    def forward(self, x):
        x = self.embed(x)
        B = x.shape[0]
        N = 3136
        H = W = 56
        window_size = self.window_size
        C = self.embed_dim
        x = x.view(B, 56, 56, 768)
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0,1,3,2,4,5).contiguous()
        x = x.view(-1, window_size * window_size, 768) # ()
        x = self.transformer_encoder(x)
        x = x.view(B, H // window_size, W // window_size, window_size, window_size, C)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.mlp_head(x)
        return x