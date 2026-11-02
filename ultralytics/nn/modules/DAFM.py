
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------- 补充依赖模块 ----------------------
class LayerNorm(nn.Module):
    """适配原代码的LayerNorm，支持WithBias类型"""

    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=(LayerNorm_type == "WithBias"))

    def forward(self, x):
        # 输入x: [b, c, h, w]，LayerNorm在通道维度(c)上计算
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b h w c")  # 转为[b, h, w, c]
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")  # 恢复原维度
        return x


class FeedForward(nn.Module):
    """适配原代码的FeedForward，用于DAIMBlock的前馈网络"""

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)  # 中间层通道数

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------- 核心模块 ----------------------
class DualModalAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=6, bias=False):
        super(DualModalAttention, self).__init__()
        # 使用较高维度作为共同维度
        self.common_dim = max(dim1, dim2)
        assert self.common_dim % num_heads == 0, "Common dimension must be divisible by num_heads"

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1))  # 可学习的平衡因子

        # 维度映射：将低维度映射到高维度（共同维度）
        self.map1 = nn.Conv2d(dim1, self.common_dim, kernel_size=1,
                              bias=bias) if dim1 != self.common_dim else nn.Identity()
        self.map2 = nn.Conv2d(dim2, self.common_dim, kernel_size=1,
                              bias=bias) if dim2 != self.common_dim else nn.Identity()

        # 注意力计算（基于共同维度）
        self.qkv1 = nn.Conv2d(self.common_dim, self.common_dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(self.common_dim * 3, self.common_dim * 3,
                                     kernel_size=3, stride=1, padding=1,
                                     groups=self.common_dim * 3, bias=bias)

        self.qkv2 = nn.Conv2d(self.common_dim, self.common_dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(self.common_dim * 3, self.common_dim * 3,
                                     kernel_size=3, stride=1, padding=1,
                                     groups=self.common_dim * 3, bias=bias)

        # 输出映射：将共同维度还原回原始输入维度
        self.project_out1 = nn.Conv2d(self.common_dim, dim1, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(self.common_dim, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = x
        b, c1, h1, w1 = x1.shape
        b, c2, h2, w2 = x2.shape

        # 保存x1原始空间维度用于恢复
        orig_x1_size = (h1, w1)

        # 确保空间维度一致（将x1调整到x2的尺寸，关键修改）
        if (h1, w1) != (h2, w2):
            x1_aligned = F.interpolate(x1, size=(h2, w2), mode='bilinear', align_corners=False)
        else:
            x1_aligned = x1

        # 映射到共同维度（高维度）
        x1_common = self.map1(x1_aligned)
        x2_common = self.map2(x2)

        # 计算QKV
        qkv1 = self.qkv_dwconv1(self.qkv1(x1_common))
        q1, k1, v1 = qkv1.chunk(3, dim=1)

        qkv2 = self.qkv_dwconv2(self.qkv2(x2_common))
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        # 计算注意力增强特征
        enhanced_common_x1 = self.compute_attention(q1, k1, v1, k2, v2, b, h2, w2)  # 使用x2的空间维度
        enhanced_common_x2 = self.compute_attention(q2, k2, v2, k1, v1, b, h2, w2)

        # 映射回原始维度
        out_x1 = self.project_out1(enhanced_common_x1)
        out_x2 = self.project_out2(enhanced_common_x2)

        # 将x1的输出恢复到原始空间维度（如果需要）
        if (h1, w1) != (h2, w2):
            out_x1 = F.interpolate(out_x1, size=orig_x1_size, mode='bilinear', align_corners=False)

        return out_x1, out_x2

    def compute_attention(self, q, k_self, v_self, k_cross, v_cross, b, h, w):
        # 重塑为注意力计算格式
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_self = rearrange(k_self, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_self = rearrange(v_self, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_cross = rearrange(k_cross, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_cross = rearrange(v_cross, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k_self = torch.nn.functional.normalize(k_self, dim=-1)
        k_cross = torch.nn.functional.normalize(k_cross, dim=-1)

        # 自注意力
        self_attn = (q @ k_self.transpose(-2, -1)) * self.temperature
        self_attn = self_attn.softmax(dim=-1)
        self_out = self_attn @ v_self

        # 交叉注意力
        cross_attn = (q @ k_cross.transpose(-2, -1)) * self.temperature
        cross_attn = (-cross_attn).softmax(dim=-1)
        cross_out = cross_attn @ v_cross

        # 融合两种注意力
        beta = torch.sigmoid(self.beta)
        out_combined = beta * self_out + cross_out

        # 重塑回原始空间维度
        out = rearrange(out_combined, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        return out


class DAIMBlock(nn.Module):
    def __init__(self, dim1, dim2, output_dim, num_heads=8, ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type="WithBias"):
        super(DAIMBlock, self).__init__()

        # 各自的Layer normalization
        self.norm1_x1 = LayerNorm(dim1, LayerNorm_type)
        self.norm1_x2 = LayerNorm(dim2, LayerNorm_type)

        # 跨模态注意力（支持不同维度）
        self.cross_attn = DualModalAttention(dim1, dim2, num_heads, bias)

        # 注意力后的Layer normalization
        self.norm2_x1 = LayerNorm(dim1, LayerNorm_type)
        self.norm2_x2 = LayerNorm(dim2, LayerNorm_type)

        # 各自的前馈网络
        self.ffn_x1 = FeedForward(dim1, ffn_expansion_factor, bias)
        self.ffn_x2 = FeedForward(dim2, ffn_expansion_factor, bias)

        # 融合模块 - 用于concat后控制输出维度
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim1 + dim2, dim1 + dim2, kernel_size=3, padding=1, groups=dim1 + dim2, bias=bias),  # 深度卷积
            nn.Conv2d(dim1 + dim2, output_dim, kernel_size=1, bias=bias)  # 点卷积控制输出维度
        )

    def forward(self, x):
        # 保存x1和x2原始形状

        x1, x2 = x
        x1_original_shape = x1.shape
        x2_original_shape = x2.shape

        # 跨模态注意力（内部会将x1对齐到x2的空间维度）
        enhanced_x1, enhanced_x2 = self.cross_attn((self.norm1_x1(x1), self.norm1_x2(x2)))

        # 确保增强后的特征与原始形状一致
        if enhanced_x1.shape != x1_original_shape:
            enhanced_x1 = F.interpolate(enhanced_x1, size=x1_original_shape[2:], mode='bilinear', align_corners=False)
        if enhanced_x2.shape != x2_original_shape:
            enhanced_x2 = F.interpolate(enhanced_x2, size=x2_original_shape[2:], mode='bilinear', align_corners=False)

        x1 = x1 + enhanced_x1
        x2 = x2 + enhanced_x2

        # 前馈网络
        x1 = x1 + self.ffn_x1(self.norm2_x1(x1))
        x2 = x2 + self.ffn_x2(self.norm2_x2(x2))

        # 关键修改：将x1对齐到x2的空间维度（确保输出为64x64）
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)

        # 融合两个模态并控制输出维度
        fused = torch.cat([x1, x2], dim=1)  # 在通道维度拼接
        output = self.fusion_conv(fused)

        return output  # 返回单个输出，空间维度与x2一致


# ---------------------- 测试函数 ----------------------
def main():
    def case(dim1, dim2, output_dim, h1=32, w1=32, h2=32, w2=32):
        """测试特定维度组合的双模态输入"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 2

        # 构造不同维度的输入
        x1 = torch.randn(batch_size, dim1, h1, w1).to(device)
        x2 = torch.randn(batch_size, dim2, h2, w2).to(device)
        print(f"\n测试维度组合: x1={x1.shape}, x2={x2.shape}, 目标输出维度={output_dim}")

        # 测试DAIMBlock（输出空间维度与x2一致）
        daim_block = DAIMBlock(
            dim1=dim1,
            dim2=dim2,
            output_dim=output_dim,
            num_heads=8
        ).to(device)

        output = daim_block(x1, x2)
        expected_shape = (batch_size, output_dim, h2, w2)  # 空间维度与x2保持一致（关键修改）
        print(f"实际输出维度: {output.shape}")
        assert output.shape == expected_shape, \
            f"DAIMBlock 测试失败 (输出维度不匹配: 预期{expected_shape}, 实际{output.shape})"

        print("✅ 测试通过")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    torch.manual_seed(42)  # 固定随机种子

    # 测试空间维度不一致的情况（输出为64x64，与x2一致）
    print("=" * 50)
    print("测试空间维度不一致 (32x32 & 64x64) -> 256 维度（输出64x64）")
    case(128, 256, 256, h1=32, w1=32, h2=64, w2=64)

    # 其他测试案例
    print("\n" + "=" * 50)
    print("测试 128 & 128 -> 256 维度")
    case(128, 128, 256)

    print("\n" + "=" * 50)
    print("测试 128 & 256 -> 256 维度")
    case(128, 256, 256)


if __name__ == "__main__":
    main()