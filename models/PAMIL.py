import numpy as np
import torch
import torch.nn as nn
from math import ceil
from einops import rearrange, reduce
from torch import nn, einsum
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

# helper functions
def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class
class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x


class PositionalScaling2D(nn.Module):
    def __init__(self, dim, num_heads=8, num_freqs=32, hidden_dim=128):
        """
        二维位置感知缩放模块 (修复版)
        :param dim: 特征维度
        :param num_heads: 注意力头数
        :param num_freqs: 频率数量 (默认32)
        :param hidden_dim: 隐藏层维度 (默认128)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_freqs = num_freqs
        self.hidden_dim = hidden_dim
        
        # 为x和y方向分别创建频率矩阵
        freqs_x = self._create_freqs(num_freqs)
        freqs_y = self._create_freqs(num_freqs)
        
        # 注册为缓冲区 (不参与训练)
        self.register_buffer('freqs_x', freqs_x)
        self.register_buffer('freqs_y', freqs_y)
        
        # 位置编码输入维度 = 4 * num_freqs
        pos_dim = 4 * num_freqs
        
        # 位置编码网络
        self.encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )
        
        # 可学习的缩放因子
        self.scale_factor = nn.Parameter(torch.ones(1, num_heads, 1))
        nn.init.normal_(self.scale_factor, mean=1.0, std=0.1)
    
    def _create_freqs(self, num_freqs):
        """创建频率矩阵"""
        freqs = 1.0 / (10000 ** (torch.arange(0, num_freqs, 2, dtype=torch.float) / num_freqs))
        return freqs
    
    def _get_positional_encoding(self, positions):
        """
        生成二维位置编码
        :param positions: 位置张量 [batch_size, seq_len, 2]
        """
        # 确保位置张量是浮点类型
        positions = positions.float()
        
        # 分离x和y坐标
        x = positions[..., 0]
        y = positions[..., 1]
        
        # 计算x方向的正弦和余弦编码
        x_enc = x[..., None] * self.freqs_x
        x_sin = torch.sin(x_enc)
        x_cos = torch.cos(x_enc)
        
        # 计算y方向的正弦和余弦编码
        y_enc = y[..., None] * self.freqs_y
        y_sin = torch.sin(y_enc)
        y_cos = torch.cos(y_enc)
        
        # 合并位置编码
        pos_enc = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)
        
        return pos_enc
    
    def forward(self, positions):
        """
        :param positions: 位置张量 [batch_size, seq_len, 2]
        :return: 位置缩放因子 [batch_size, num_heads, seq_len]
        """
        # 获取位置编码
        pos_enc = self._get_positional_encoding(positions)
        
        # 确保位置编码维度正确
        batch_size, seq_len, _ = pos_enc.shape
        if pos_enc.size(-1) != 4 * self.num_freqs:
            # 自动调整维度
            pos_enc = F.pad(pos_enc, (0, 4 * self.num_freqs - pos_enc.size(-1)))
        
        # 通过神经网络获取缩放因子
        scaling = self.encoder(pos_enc)  # [batch_size, seq_len, num_heads]
        scaling = scaling.permute(0, 2, 1)  # [batch_size, num_heads, seq_len]
        
        # 应用可学习的缩放因子
        scaling = scaling * self.scale_factor
        
        # 使用softplus确保缩放因子为正
        scaling = F.softplus(scaling)
        
        return scaling

    def __init__(self, dim, num_heads=8, num_freqs=32, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_freqs = num_freqs  # 这里 num_freqs 就是 inv_freq 的长度
        self.hidden_dim = hidden_dim

        # 创建 inv_freq，形状为 [num_freqs]
        self.register_buffer('inv_freq', self._build_inv_freq(num_freqs))

        self.input_dim = num_freqs
        self.output_dim = self.input_dim if self.input_dim <= dim // 4 else dim // 4

        # 随机初始化系数矩阵（不训练）
        self.register_buffer('sin_coef', self._init_coef(num_heads, self.input_dim, self.output_dim))
        self.register_buffer('cos_coef', self._init_coef(num_heads, self.input_dim, self.output_dim))

        # Linear encoder（为保持接口一致）
        pos_dim = 4 * num_freqs
        self.encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )

        self.scale_factor = nn.Parameter(torch.ones(1, num_heads, 1))
        nn.init.normal_(self.scale_factor, mean=1.0, std=0.1)

    def _build_inv_freq(self, num_freqs):
        return 1.0 / (10000 ** (torch.arange(0, num_freqs, dtype=torch.float32) / num_freqs))

    def _init_coef(self, n_heads, in_dim, out_dim):
        weight = torch.randn(n_heads, in_dim, out_dim)
        torch.nn.init.xavier_normal_(weight)
        # 加单位矩阵增强稳定性
        eye = torch.eye(in_dim)
        if out_dim > in_dim:
            eye = F.pad(eye, (0, out_dim - in_dim))
        weight += eye[:in_dim, :out_dim]
        return weight

    def _get_fourier_encoding(self, seq_len, device):
        # 生成位置编码
        positions = torch.arange(seq_len, device=device).float()  # [seq_len]
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))  # [seq_len, num_freqs]

        sin, cos = torch.sin(freqs), torch.cos(freqs)  # [seq_len, num_freqs]

        # 应用变换
        sin_proj = torch.einsum("td, hdo -> tho", sin, self.sin_coef)  # [seq_len, num_heads, out_dim]
        cos_proj = torch.einsum("td, hdo -> tho", cos, self.cos_coef)  # [seq_len, num_heads, out_dim]

        # Padding 到 dim // 2
        pad_len = self.dim // 2 - sin_proj.size(-1)
        if pad_len > 0:
            sin_proj = F.pad(sin_proj, (0, pad_len), value=1)
            cos_proj = F.pad(cos_proj, (0, pad_len), value=1)

        # 拼接为 full dim
        sin_proj = torch.cat([sin_proj, sin_proj], dim=-1)
        cos_proj = torch.cat([cos_proj, cos_proj], dim=-1)  # [seq_len, num_heads, dim]

        # 最终 shape: [1, num_heads, seq_len, dim]
        return sin_proj.permute(1, 2, 0), cos_proj.permute(1, 2, 0)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, positions):
        """
        Input: positions [B, seq_len]
        Output: scaling [B, num_heads, seq_len]
        """
        batch_size, seq_len = positions.shape[0], positions.shape[1]
        device = positions.device

        # 生成傅里叶位置编码
        sin_pos, cos_pos = self._get_fourier_encoding(seq_len, device)  # [num_heads, dim, seq_len]
        sin_pos = sin_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)   # [B, num_heads, dim, seq_len]
        cos_pos = cos_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 模拟一个输入向量进行旋转，这里仅用于生成 scaling 向量（不用于注意力）
        dummy = torch.ones(batch_size, self.num_heads, seq_len, self.dim, device=device)
        rotated = dummy * cos_pos.transpose(2, 3) - self.rotate_half(dummy) * sin_pos.transpose(2, 3)  # [B, num_heads, seq_len, dim]

        # 展平维度 [B, seq_len, dim]
        feat = rotated.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)  # [B, seq_len, num_heads * dim]
        feat = feat[..., :4 * self.num_freqs]  # 截断或 padding 到 4*num_freqs

        if feat.size(-1) < 4 * self.num_freqs:
            feat = F.pad(feat, (0, 4 * self.num_freqs - feat.size(-1)))

        scaling = self.encoder(feat)             # [B, seq_len, num_heads]
        scaling = scaling.permute(0, 2, 1)       # [B, num_heads, seq_len]
        scaling = F.softplus(scaling * self.scale_factor)

        return scaling
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class AttentionL(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., pos_scaling=None):
        """
        改进的注意力机制，支持二维位置感知缩放 (修复版)
        :param dim: 特征维度
        :param num_heads: 注意力头数
        :param pos_scaling: 位置缩放模块
        """
        super().__init__()
        self.heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
        
        # 位置缩放模块
        self.pos_scaling = pos_scaling
    
    def forward(self, x, positions=None):
        """
        :param x: 输入特征 [batch_size, seq_len, dim]
        :param positions: 位置信息 [batch_size, seq_len, 2]
        :return: 输出特征 [batch_size, seq_len, dim]
        """
        # 投影到查询/键/值空间
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        b, h, N, d = w.shape
        
        # 计算归一化特征
        w_normed = torch.nn.functional.normalize(w, dim=-2) 
        w_sq = w_normed ** 2

        # 计算位置缩放因子
        if self.pos_scaling is not None and positions is not None:
            pos_scale = self.pos_scaling(positions)
        else:
            pos_scale = 1.0
        
        # 应用位置缩放因子
        w_sum = torch.sum(w_sq, dim=-1)  # [b, h, n]
        scaled_w_sum = w_sum * self.temp * pos_scale
        
        # 计算注意力权重
        Pi = self.attend(scaled_w_sum)  # b * h * n 
        
        # 后续计算
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temp'}

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, pos_embed=None):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = AttentionL(
            dim=dim,
            num_heads=8,
            attn_drop=0.1,
            pos_scaling=pos_embed,
        )

    def forward(self, x, pos = None):
        x = x + self.attn(self.norm(x), pos)
        return x



class PAMIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False):
        super(PAMIL, self).__init__()
        ### 
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            print("dropout: ", dropout)
        self._fc1 = nn.Sequential(*self._fc1)

        # self.pos_layer = PPEG(dim=512)


        self.n_classes = n_classes
        self.pos_embed = PositionalScaling2D(dim=512, num_heads=8)
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512, pos_embed=self.pos_embed)
        self.layer3 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)
        self.survival = survival
        # self.rotary = Rotary2D(dim = 512)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)

        h = x.float()  # [B, n, 1024]
        pos = x[:,:,-2:]
        h = self._fc1(h[:,:,:-2])  

        # print('pos.shape', pos.shape)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 256]
        h = self.layer2(h, pos)
        h = self.layer3(h)

        # ---->predict
        # h = torch.cat((h_1, h_2), dim=1) 
        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)
        results_dict = h
        # ---->predict
        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        # results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return logits, Y_prob, Y_hat, A_raw, results_dict

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.pos_embed = self.pos_embed.to(device)
        self.layer1 = self.layer1.to(device)
        self.attention = self.attention.to(device)
        self.layer2 = self.layer2.to(device)
        self.layer3 = self.layer3.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)



