import torch
import torch.nn as nn
import einops
import matplotlib.pyplot as plt
from models.utils.main_blocks import *

def params(module):
    return sum(p.numel() for p in module.parameters())


class UpsampleConv3D(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3, 3),  # 修改为三维卷积核尺寸
                padding=(1, 1, 1),  # 修改为三维的padding
                norm_type=None, 
                activation=False,
                scale=(2, 2, 2),  # 修改为三维的上采样比例
                conv='conv') -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, 
                              mode='trilinear',  # 使用 trilinear 模式进行三维上采样
                              align_corners=True)
        if conv == 'conv':
            self.conv = conv_block_3D(in_features=in_features,  # 假设已有对应的三维conv_block定义
                                    out_features=out_features, 
                                    kernel_size=(1, 1, 1),
                                    padding=(0, 0, 0),
                                    norm_type=norm_type, 
                                    activation=activation)
        elif conv == 'depthwise':
            self.conv = depthwise_conv_block(in_features=in_features,  # 假设已有对应的三维depthwise_conv_block定义
                                    out_features=out_features, 
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    norm_type=norm_type, 
                                    activation=activation)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class depthwise_projection3D(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                groups,
                kernel_size=(1, 1, 1),  # 三维卷积核尺寸
                padding=(0, 0, 0),  # 三维padding
                norm_type=None, 
                activation=False, 
                pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features,  # 使用三维深度可分离卷积块
                                        out_features=out_features, 
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        groups=groups,
                                        pointwise=pointwise, 
                                        norm_type=norm_type,
                                        activation=activation)

    def forward(self, x):
        P = int(x.shape[1] ** (1 / 3))  # 计算立方根，对应三维数据维度调整
        x = einops.rearrange(x, 'B (D H W) C-> B C D H W', D=P, H=P, W=P)  # 调整为三维的维度重排
        x = self.proj(x)
        x = einops.rearrange(x, 'B C D H W -> B (D H W) C')  # 再调整回原始的维度顺序（合并空间维度）
        return x


class conv_projection3D(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.proj = conv_block_3D(in_features=in_features,  # 调用三维卷积块
                                        out_features=out_features, 
                                        kernel_size=(1, 1, 1), 
                                        padding=(0, 0, 0),
                                        norm_type=None,
                                        activation=False)
    def forward(self, x):
        P = int(x.shape[1] ** (1 / 3))  # 计算立方根，对应三维数据维度调整
        x = einops.rearrange(x, 'B (D H W) C-> B C D H W', D=P, H=P, W=P)  # 调整为三维的维度重排
        x = self.proj(x)
        x = einops.rearrange(x, 'B C D H W -> B (D H W) C')  # 再调整回原始的维度顺序（合并空间维度）
        return x


class PatchEmbedding3D(nn.Module):
    def __init__(self, 
                in_features,
                out_features,
                size,  # 这里的size可以理解为三维数据的整体尺寸（比如 D*H*W），需要根据实际传入合适的值
                patch=(2, 2, 2),  # 定义三维的patch尺寸
                proj='conv') -> None:
        super().__init__()
        self.proj = proj
        if self.proj == 'conv':
            self.projection = nn.Conv3d(in_channels=in_features,  # 使用三维卷积进行投影
                                        out_channels=out_features, 
                                        kernel_size=size // patch,  # 按三维patch尺寸进行卷积操作
                                        stride=size // patch, 
                                        padding=(0, 0, 0), 
                                        )
        
    def forward(self, x):
        x = self.projection(x) 
        x = x.flatten(2).transpose(1, 2)  # 合并空间维度并转置，与二维类似但维度不同
        return x


class PoolEmbedding3D(nn.Module):
    def __init__(self,
                pooling,
                patch=(2, 2, 2),  # 三维patch尺寸
                ) -> None:
        super().__init__()
        self.projection = pooling(output_size=patch)

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C D H W -> B (D H W) C')  # 调整维度顺序，合并空间维度
        return x


class Layernorm3D(nn.Module):
    def __init__(self, features, eps=1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(features, eps=eps)                                                  
    def forward(self, x):
        D, H, W = x.shape[2], x.shape[3], x.shape[4]  # 获取三维数据的各维度尺寸
        x = einops.rearrange(x, 'B C D H W -> B (D H W) C')  # 调整维度顺序，合并空间维度
        x = self.norm(x)
        x = einops.rearrange(x, 'B (D H W) C-> B C D H W', D=D, H=H, W=W)  # 再调整回原始的三维维度顺序
        return x       


class ScaleDotProduct3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
                                                    
    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        x12 = torch.einsum('bhdcw, bhdwk -> bhdck', x1, x2) * scale  # 修改 einsum 操作以适配三维数据
        att = self.softmax(x12)
        x123 = torch.einsum('bhdcw, bhdwk -> bhdck', att, x3)  # 修改 einsum 操作以适配三维数据
        return x123