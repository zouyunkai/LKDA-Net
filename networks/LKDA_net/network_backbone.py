import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from monai.networks.blocks import UnetrUpBlock, UnetOutBlock
from monai.networks.nets import UnetrBasicBlock

class LayerNorm3D(nn.Module):
    """3D通道优先的LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 通道维度在第二维 (N, C, D, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + \
            self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x

class MultiScaleDWConv(nn.Module):
    """多尺度深度卷积模块"""
    def __init__(self, dim, kernels=[5, 7], dilations=[1, 3]):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            padding = (k // 2) * d  # 保持空间尺寸不变
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(dim, dim, kernel_size=k, padding=padding, 
                             dilation=d, groups=dim, bias=False),
                    nn.BatchNorm3d(dim),
                    nn.GELU()
                )
            )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(dim * len(kernels), dim, kernel_size=1),
            LayerNorm3D(dim),
            nn.GELU()
        )

    def forward(self, x):
        features = [conv(x) for conv in self.conv_layers]
        fused = self.fusion(torch.cat(features, dim=1))
        return fused

class LKD_Attention(nn.Module):
    """大核深度卷积注意力模块（详细实现）"""
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        
        # 通道压缩
        self.channel_compression = nn.Sequential(
            nn.Conv3d(dim, dim // reduction_ratio, 1),
            LayerNorm3D(dim // reduction_ratio),
            nn.GELU()
        )
        
        # 多尺度深度卷积
        self.spatial_conv = MultiScaleDWConv(dim // reduction_ratio)
        
        # 通道扩展
        self.channel_expansion = nn.Sequential(
            nn.Conv3d(dim // reduction_ratio, dim, 1),
            LayerNorm3D(dim)
        )
        
        # 空间注意力机制
        self.spatial_att = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim // (reduction_ratio*2), 1),
            nn.GELU(),
            nn.Conv3d(dim // (reduction_ratio*2), dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        
        # 通道压缩
        x_compressed = self.channel_compression(x)
        
        # 空间特征提取
        spatial_feat = self.spatial_conv(x_compressed)
        
        # 通道扩展
        spatial_feat = self.channel_expansion(spatial_feat)
        
        # 通道注意力
        channel_weights = self.channel_att(spatial_feat)
        
        # 空间注意力
        max_pool, _ = torch.max(spatial_feat, dim=1, keepdim=True)
        avg_pool = torch.mean(spatial_feat, dim=1, keepdim=True)
        spatial_weights = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        
        # 注意力融合
        refined_feat = spatial_feat * channel_weights * spatial_weights
        
        return identity + refined_feat

class DWCA(nn.Module):
    """深度卷积增强的倒置瓶颈（详细实现）"""
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        
        self.block = nn.Sequential(
            # 通道扩展
            nn.Conv3d(dim, hidden_dim, 1),
            LayerNorm3D(hidden_dim),
            nn.GELU(),
            
            # 深度卷积
            nn.Conv3d(hidden_dim, hidden_dim, 3, 
                      padding=1, groups=hidden_dim),
            LayerNorm3D(hidden_dim),
            nn.GELU(),
            
            # 通道压缩
            nn.Conv3d(hidden_dim, dim, 1),
            LayerNorm3D(dim)
        )
        
        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x + self.block(x)

class LKDA_Block(nn.Module):
    """完整的LKDA网络块"""
    def __init__(self, dim, drop_path=0., layer_scale=1e-6):
        super().__init__()
        # 注意力模块
        self.norm1 = LayerNorm3D(dim)
        self.attn = LKD_Attention(dim)
        
        # 前馈模块
        self.norm2 = LayerNorm3D(dim)
        self.mlp = DWCA(dim)
        
        # 层缩放和随机深度
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim)) 
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        # 注意力路径
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        # 前馈路径
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class SkipFusion(nn.Module):
    """跳跃连接融合模块（详细实现）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            LayerNorm3D(out_channels),
            nn.GELU()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, enc_feat, dec_feat):
        # 对齐特征维度
        enc_feat = self.proj(enc_feat)
        
        # 生成空间注意力
        combined = enc_feat + dec_feat
        att_map = self.spatial_att(combined)
        
        # 特征融合
        fused_feat = enc_feat * att_map + dec_feat * (1 - att_map)
        return fused_feat

class LKDA_Net(nn.Module):
    """完整的3D医学图像分割网络"""
    def __init__(self, 
                 in_chans=1,
                 out_chans=13,
                 depths=[2, 2, 2, 2],
                 feat_size=[48, 96, 192, 384],
                 drop_path_rate=0.2,
                 layer_scale_init_value=1e-6,
                 spatial_dims=3):
        super().__init__()
        
        # 初始化参数
        self.depths = depths
        self.feat_size = feat_size
        self.num_stages = len(depths)
        
        # 下采样层
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, feat_size[0], kernel_size=4, stride=2, padding=1),
            LayerNorm3D(feat_size[0])
        )
        self.downsample_layers.append(stem)
        
        # 构建下采样路径
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm3D(feat_size[i]),
                nn.Conv3d(feat_size[i], feat_size[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)
        
        # 特征提取阶段
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # 构建各阶段特征提取模块
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[LKDA_Block(
                    dim=feat_size[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale=layer_scale_init_value
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # 解码器上采样模块
        self.decoder = nn.ModuleList([
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[3],
                out_channels=feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name="instance"
            ),
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[2],
                out_channels=feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name="instance"
            ),
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feat_size[1],
                out_channels=feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name="instance"
            )
        ])
        
        # 跳跃连接融合模块
        self.skip_fusions = nn.ModuleList([
            SkipFusion(feat_size[i], feat_size[i]) for i in range(3)
        ])
        
        # 最终输出层
        self.final = UnetOutBlock(spatial_dims, feat_size[0], out_chans)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm3D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 编码器路径
        skips = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i < 3:  # 保存前三个阶段的特征用于跳跃连接
                skips.append(x)
        
        # 解码器路径
        for i in range(len(self.decoder)):
            # 上采样并融合特征
            x = self.decoder[i](x, self.skip_fusions[i](skips[2-i], x))
        
        return self.final(x)