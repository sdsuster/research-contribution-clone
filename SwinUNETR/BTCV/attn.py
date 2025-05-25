import torch
from collections.abc import Sequence
from torch import nn
from monai.networks.layers import DropPath, trunc_normal_
import math
import einops
# from flash_attn import flash_attn_func
from typing import Tuple


def init_t_xy(end_x: int, end_y: int, zero_center=False):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    
    return t_x, t_y

def init_t_xyz(end_x: int, end_y: int, end_z: int, zero_center=False):
    t = torch.arange(end_x * end_y * end_z, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = ((t // end_x) % end_y).float()  # Compute y-axis
    t_z = (t // (end_x * end_y)).float()  # Compute z-axis
    return t_x, t_y, t_z

def init_random_2d_freqs(head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    theta = theta
    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_random_3d_freqs(head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """
    Initialize frequency parameters for 3D rotary embeddings.

    If head_dim is not divisible by 6, use only the largest multiple of 6.
    Each axis gets three groups of frequency components.
    
    Returns:
      freqs: a tensor of shape [3, num_heads, 3*num_pairs]
      rotary_dim: the number of head dimensions to which rotary embeddings will be applied.
    """
    # Compute the effective rotary dimension (largest multiple of 6 <= head_dim)
    rotary_dim = (head_dim // 6) * 6
    if rotary_dim == 0:
        raise ValueError("head_dim is too small to apply rotary embeddings.")

    # Number of frequency pairs per group (for each axis, we generate three groups)
    num_pairs = rotary_dim // 6  # because 3 groups * num_pairs = rotary_dim/2 (as complex numbers)

    # Create a magnitude vector of length num_pairs
    mag = 1 / (theta ** (torch.arange(num_pairs, dtype=torch.float32) / num_pairs))
    
    freqs_x, freqs_y, freqs_z = [], [], []
    for _ in range(num_heads):
        # Generate axis-specific random angles (or zeros if rotation is disabled)
        angle_x = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        angle_y = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        angle_z = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        
        # For each axis, create three sets of frequency components.
        fx = torch.cat([
            mag * torch.cos(angle_x),
            mag * torch.cos(torch.pi/2 + angle_x),
            mag * torch.cos(torch.pi + angle_x)
        ], dim=-1)
        fy = torch.cat([
            mag * torch.cos(angle_y),
            mag * torch.cos(torch.pi/2 + angle_y),
            mag * torch.cos(torch.pi + angle_y)
        ], dim=-1)
        fz = torch.cat([
            mag * torch.cos(angle_z),
            mag * torch.cos(torch.pi/2 + angle_z),
            mag * torch.cos(torch.pi + angle_z)
        ], dim=-1)
        
        freqs_x.append(fx)
        freqs_y.append(fy)
        freqs_z.append(fz)
    
    freqs_x = torch.stack(freqs_x, dim=0)  # [num_heads, 3*num_pairs]
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs_z = torch.stack(freqs_z, dim=0)
    freqs = torch.stack([freqs_x, freqs_y, freqs_z], dim=0)  # [3, num_heads, 3*num_pairs]
    
    return freqs, rotary_dim

import numpy as np
def compute_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, t_z: torch.Tensor = None):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.amp.autocast('cuda', enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
        if t_z != None:
            freqs_z = (t_z.unsqueeze(-1) @ freqs[2].unsqueeze(-2))
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y + freqs_z)
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-4 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError("freqs_cis shape does not match expected dimensions.")
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast('cuda', enabled=False):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

def init_(tensor: torch.Tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class RopeEmbedding(nn.Module):
    def __init__(self, head_dims, window_size, num_heads, rope_theta=100.0, rope_mixed=True, *args, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.rope_mixed = rope_mixed
        self.head_dims = head_dims


        if len(self.window_size) == 3:
            t_x, t_y, t_z = init_t_xyz(end_x=self.window_size[2], end_y=self.window_size[1], end_z=self.window_size[0])
            self.register_buffer('rope_t_x', t_x)
            self.register_buffer('rope_t_y', t_y)
            self.register_buffer('rope_t_z', t_z)

            
            # Assume head_dim is your full head dimension.
            if self.head_dims % 6 != 0:
                effective_dim = (head_dims // 6) * 6
            else:
                effective_dim = head_dims
            freqs, self.rotary_dim = init_random_3d_freqs(
                head_dim=self.head_dims, num_heads=self.num_heads, theta=rope_theta, 
                rotate=self.rope_mixed
            )
            if effective_dim < head_dims:
                pad_size = head_dims - effective_dim
                pad_shape = list(freqs.shape)
                pad_shape[-1] = pad_size//2
                pad_freqs = torch.zeros(*pad_shape, device=freqs.device, dtype=freqs.dtype)
                # Concatenate along the last dimension
                freqs = torch.cat([freqs, pad_freqs], dim=-1)

            if self.rope_mixed:
                self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
            else:
                self.register_buffer('rope_freqs', freqs)
                freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y, self.rope_t_z)
                self.rope_freqs_cis = freqs_cis

        elif len(self.window_size) == 2:
            t_x, t_y = init_t_xy(end_x=self.window_size[1], end_y=self.window_size[0])
            self.register_buffer('rope_t_x', t_x)
            self.register_buffer('rope_t_y', t_y)
            self.rope_t_z = None

            freqs, self.rotary_dim = init_random_2d_freqs(
                head_dim=self.head_dims, num_heads=self.num_heads, theta=rope_theta, 
                rotate=self.rope_mixed
            )
            if self.rope_mixed:
                self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
            else:
                self.register_buffer('rope_freqs', freqs)
                freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y, self.rope_t_z)
                self.rope_freqs_cis = freqs_cis

    def forward(self, q, k, x_shape, x_device):
        if self.rope_mixed:
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y, self.rope_t_z)
        else:
            freqs_cis = self.rope_freqs_cis.to(x_device)
        # freqs_cis[:, :x_shape[1], :] trick to deal with non equal size of input.
        q, k = apply_rotary_emb(q, k, freqs_cis[:, :x_shape[1], :])

        # print(freqs_cis[0, :, 0], k.max(), k.min())
        return q, k
        

class LinAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_dims: int,
        k: int,
        window_size: Sequence[int],
        share_kv: bool = False,
        flash:bool = True,
        use_rope: bool = True,
        attn_drop: float = 0.0,
        rope_theta:float = 100.,
        rope_mixed: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.window_size = window_size
        if len(self.window_size) == 3:
            self.seq_len = window_size[0] * window_size[1] * window_size[2]
            # self.biases = nn.Parameter(
            #     torch.zeros(
            #         self.seq_len*k,
            #         num_heads,
            #     )
            # )
        elif len(self.window_size) == 2:
            self.seq_len = window_size[0] * window_size[1]
            # self.biases = nn.Parameter(
            #     torch.zeros(self.seq_len*k, num_heads)
            # )


        self.num_heads = num_heads
        self.flash = flash
        self.use_rope = use_rope
        self.head_dims = head_dims
        self.softmax = nn.Softmax(dim=-1)
        if self.use_rope:
            self.rope = RopeEmbedding(head_dims=head_dims, window_size=window_size, num_heads=num_heads, rope_theta=rope_theta, rope_mixed=rope_mixed)



        self.proj_k_len = k
        self.use_lin = k > 0
        self.share_kv = share_kv
        if self.use_lin:
            self.proj_k = nn.Parameter(init_(torch.zeros(self.seq_len, k)))
            if not share_kv:
                self.proj_v = nn.Parameter(init_(torch.zeros(self.seq_len, k)))

        if not self.flash:
            self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, scale, x_shape, x_device):
        b, n, c = x_shape
        # v = v if not self.share_kv else k
        if self.use_rope:
            q, k = self.rope(q, k, x_shape, x_device)
            
        if self.flash:
            q = einops.rearrange(q, 'b h s c -> b s h c')
            if self.use_lin:
                kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
                kv_projs = map(lambda t: t[:n], kv_projs)
                proj_seq_len = lambda args: torch.einsum('bwnd,nk->bwkd', *args)
                k_prime, v_prime = map(proj_seq_len, zip((k, v), kv_projs))

                k_prime = einops.rearrange(k_prime, 'b h s c -> b s h c').contiguous()
                v_prime = einops.rearrange(v_prime, 'b h s c -> b s h c').contiguous()
                attn = flash_attn_func(q.to(torch.bfloat16), k_prime.to(torch.bfloat16), v_prime.to(torch.bfloat16), softmax_scale=scale)
            else:
                k = einops.rearrange(k, 'b h s c -> b s h c')
                v = einops.rearrange(v, 'b h s c -> b s h c')
                attn = flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), softmax_scale=scale)
            # x = einops.rearrange(attn, 'b s h c -> b h s c')
            x = attn.float()
        # if mask is not None:
        #     print(q.shape, mask.shape)
        #     nw = mask.shape[0]
        #     q = nn.functional.relu(q)
        #     q = q.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
        #     q = q.view(-1, self.num_heads, n, n)
        else:
            if self.use_lin:
                kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
                kv_projs = map(lambda t: t[:n], kv_projs)
                proj_seq_len = lambda args: torch.einsum('bwnd,nk->bwkd', *args)
                k, v = map(proj_seq_len, zip((k, v), kv_projs))

            q = q * scale
            attn = torch.einsum('bhnd,bhkd->bhnk', q, k)
            # relative_position_bias = self.relative_position_bias_table[
            #     self.relative_position_index.clone()[:n, :n].reshape(-1)
            # ].reshape(n, n, -1)
            # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            # print(self.relative_position_bias_table.shape)
            # print(n, attn.shape, relative_position_bias, self.relative_position_index.clone().shape)

            # attn = attn + relative_position_bias.unsqueeze(0)
            # attn = attn + self.biases[:n*self.proj_k_len].reshape(-1, n, self.proj_k_len).unsqueeze(0)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(v.dtype)
            x = torch.einsum('bhnk,bhkd->bhnd', attn, v).transpose(1, 2)
            # .transpose(1, 2).reshape(b, n, c)
        return x

class FastAttention(nn.Module):
    pass

class Attention():
    @classmethod
    def construct_lin_attention(cls, att_type:str='linformer', *args, **kwargs):
        assert att_type in ['linformer'], 'linear attention type is not supported.'
        if att_type == 'linformer':
            return LinAttention(*args, **kwargs)
        
        

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WLindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        k: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        share_kv: bool = False,
        use_flash: bool = True,
        use_rope = True,
        linear_attention_type = 'linformer'
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_flash = use_flash
        self.att_core = Attention.construct_lin_attention(att_type=linear_attention_type, head_dims=head_dim, num_heads=num_heads, k=k, window_size=window_size, share_kv=share_kv, flash=use_flash, use_rope = use_rope, attn_drop=attn_drop)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if len(self.window_size) == 3:
            self.seq_len = window_size[0] * window_size[1] * window_size[2]
            # self.biases = nn.Parameter(
            #     torch.zeros(
            #         self.seq_len*k,
            #         num_heads,
            #     )
            # )
        elif len(self.window_size) == 2:
            self.seq_len = window_size[0] * window_size[1]
            # self.biases = nn.Parameter(
            #     torch.zeros(self.seq_len*k, num_heads)
            # )
        # trunc_normal_(self.biases, std=0.02)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


        attn = self.att_core(q=q, k=k, v=v, scale=self.scale, x_shape=(b, n, c), x_device = x.device)

        # x = einops.rearrange(attn, 'b head n c -> b n (head c)')
        x = attn.reshape(b, n, c)
        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x