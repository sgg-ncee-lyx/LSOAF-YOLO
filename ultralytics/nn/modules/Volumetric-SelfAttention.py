import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class DecayPos1d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l, l)
        mask = mask.abs()  # (l, l)
        mask = mask * self.decay[:, None, None]  # (num_heads, l, l)
        return mask

    def forward(self, slen):
        mask_c = self.generate_1d_decay(slen)
        retention_rel_pos = mask_c
        return retention_rel_pos


class VolSelfAttention(nn.Module):
    r""" Volumetric Self-Attention (动态适配通道维度)"""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim  # 输入特征维度
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 空间注意力投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 光谱位置先验：使用dim作为embed_dim，不再固定64
        self.realPos = DecayPos1d(dim, num_heads, 2, 4)

        # 通道注意力卷积层
        self.qkv_C = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv_C = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.proj_C = nn.Conv2d(dim, dim, kernel_size=1)

        # 额外注意力模块：输出通道动态匹配dim
        self.Gao_spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_heads, dim // 2, 3, 1, 1),  # 中间通道数为dim//2（动态）
            nn.BatchNorm2d(dim // 2),
            nn.Conv2d(dim // 2, dim, 3, 1, 1),  # 输出通道=dim（与输入特征维度一致）
        )

        self.Gao_channel_attention = nn.Sequential(
            nn.Conv2d(dim // num_heads, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        输入: 4维张量 (B, C, H, W)
        输出: 4维张量 (B, C, H, W)
        """
        # 保存原始4维形状 (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W  # 序列长度 = 空间尺寸乘积

        ###########################
        # 动态生成相对位置偏置
        ###########################
        coords_h = torch.arange(H, device=x.device)
        coords_w = torch.arange(W, device=x.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, H, W)
        coords_flatten = torch.flatten(coords, 1)  # (2, H*W)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += H - 1
        relative_coords[:, :, 1] += W - 1
        relative_coords[:, :, 0] *= 2 * W - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)

        # 动态生成相对位置偏置表
        num_relative_positions = (2 * H - 1) * (2 * W - 1)
        relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, self.num_heads, device=x.device))

        #--------------------------------
        # trunc_normal_(relative_position_bias_table, std=.02)

        # 1. 记下当前 tensor 所在的设备 (cuda:0)
        orig_device = relative_position_bias_table.device

        # 2. 暂时切回 CPU，在 CPU 上执行 trunc_normal_（CPU 不会触发 GPU 动态编译报错）
        cpu_tensor = relative_position_bias_table.to("cpu")
        nn.init.trunc_normal_(cpu_tensor, std=0.02)

        # 3. 将初始化好的数据复制回原始的 GPU 显存中
        relative_position_bias_table.data.copy_(cpu_tensor.to(orig_device))
        #--------------------------------------

        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
            N, N, -1)  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)

        ###########################
        # 4维 -> 3维 (B, H*W, C)
        ###########################
        x_3d = rearrange(x, 'b c h w -> b (h w) c')  # (B, N, C)
        B_, N_, C_ = x_3d.shape  # B_=B, N_=N, C_=C

        ###########################
        # 空间注意力计算
        ###########################
        qkv = self.qkv(x_3d).reshape(B_, N_, 3, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N_, head_dim)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N_, N_)
        attn = attn + relative_position_bias.unsqueeze(0)  # (B_, num_heads, N_, N_)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_, N_) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_, N_)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x1 = (attn @ v).transpose(1, 2).reshape(B_, N_, C_)  # (B_, N_, C_)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        ###########################
        # 光谱注意力计算
        ###########################
        realPos = self.realPos(C_ // self.num_heads)

        x_s = rearrange(x_3d, 'b (h w) c -> b c h w', h=H, w=W)

        qkv_c = self.qkv_dwconv_C(self.qkv_C(x_s))  # (B, 3*C, H, W)
        q_c, k_c, v_c = qkv_c.chunk(3, dim=1)  # 拆分Q/K/V

        q_c = rearrange(q_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_c = rearrange(k_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_c = rearrange(v_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_c = F.normalize(q_c, dim=-1)
        k_c = F.normalize(k_c, dim=-1)
        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature + realPos
        attn_c = attn_c.softmax(dim=-1)

        x2 = (attn_c @ v_c)
        x2 = rearrange(x2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        x2 = self.proj_C(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')  # (B_, N_, C_)

        ###########################
        # 注意力融合（维度已匹配）
        ###########################
        attn_spatial = attn  # (B_, num_heads, N_, N_)
        attn_spatial = self.Gao_spatial_attention(attn_spatial)  # (B_, dim, 1, 1)
        attn_spatial = rearrange(attn_spatial, 'b c 1 1 -> b c')  # (B, dim)
        attn_spatial = attn_spatial.unsqueeze(1).repeat(1, N_, 1)  # (B, N_, dim)

        # 此时attn_spatial与x2的通道数均为dim（C_），可直接相乘
        x4 = attn_spatial * x2  # (B_, N_, C_)

        # 融合所有分支
        x_fused = x1 + x2 + x4  # (B_, N_, C_)

        # 3维 -> 4维 (B, C, H, W)
        output = rearrange(x_fused, 'b (h w) c -> b c h w', h=H, w=W)

        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'






def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



class PSABloc_VolSelfAttention(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()
        self.attn = VolSelfAttention(c, num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x



class C2PSA_VolSelfAttention(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABloc_VolSelfAttention(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))





if __name__ == "__main__":
    # 测试不同输入尺寸和通道数（包括64和128）
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试1：dim=64（原可行 case）
    B, C, H, W = 2, 64, 17, 31
    x1 = torch.randn(B, C, H, W).to(device)
    model1 = VolSelfAttention(dim=C, num_heads=4).to(device)
    output1 = model1(x1)
    print(f"输入1形状: {x1.shape}, 输出1形状: {output1.shape}")
    assert output1.shape == x1.shape, "测试1失败"

    # 测试2：dim=128（原失败 case，现在已修复）
    B, C, H, W = 2, 128, 17, 31
    x2 = torch.randn(B, C, H, W).to(device)
    model2 = VolSelfAttention(dim=C, num_heads=4).to(device)  # 128/4=32，每个头32维
    output2 = model2(x2)
    print(f"输入2形状: {x2.shape}, 输出2形状: {output2.shape}")
    assert output2.shape == x2.shape, "测试2失败"

    # 测试3：dim=256（更大通道数）
    B, C, H, W = 2, 256, 10, 20
    x3 = torch.randn(B, C, H, W).to(device)
    model3 = VolSelfAttention(dim=C, num_heads=8).to(device)  # 256/8=32，每个头32维
    output3 = model3(x3)
    print(f"输入3形状: {x3.shape}, 输出3形状: {output3.shape}")
    assert output3.shape == x3.shape, "测试3失败"

    print("所有测试通过!")