import dataclasses

import numpy as np
from typing import Any, Dict, Union, Optional
import torch
import math
from copy import deepcopy
from dataclasses import dataclass
from torch import nn
# Turn this on if you run out of memory. it will slow things down tho
DO_GRADIENT_CHECKPOINTING = False
checkpoint_if_enabled = jax.checkpoint if DO_GRADIENT_CHECKPOINTING else lambda x: x

def get_rotary_coordinates(seq_len, dtype=torch.float32, center_origin=True):
    """
    Get rotary coordinates for a single dimension
    :param seq_len: length of sequence (or dimension)
    :param dtype: data type
    :param center_origin: If true then coordinates are from [-seq_len / 2, seq_len / 2].
                          if false then coordinates from    [1, seq_len]
    :return: sequence of length L -- coordinates are from [-L / 2, -L / 2] if center_origin else [1, L]
    """
    if center_origin:
        sl0 = seq_len // 2
        nseq = torch.arange(sl0, dtype=dtype) - float(sl0)
        pseq = 1.0 + torch.arange(seq_len - sl0, dtype=dtype)
        return torch.cat([nseq, pseq], 0)
    return 1.0 + torch.arange(seq_len, dtype=dtype)

def get_rotary_coordinates_2d(h, w, dtype=torch.float32):
    """
    Rotary embeddings for 2d (e.g. an image).
    Scale kinda like we're in a square box and taking a crop. skip zero though
    :param h: How many patches width
    :param w: How many patches height
    :param dtype: dtype
    :return: [h * w, 2] array of coords
    """
    base_scale = 1 / (max(h, w) + 1.0)
    w_coords = base_scale * get_rotary_coordinates(w, dtype=dtype, center_origin=True)
    h_coords = base_scale * get_rotary_coordinates(h, dtype=dtype, center_origin=True)
    return torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), -1).reshape(h * w, 2)

# def multimodal_rotary_coords(h=None, w=None, segment_idx=None, token_idx=None, dtype=jnp.float32,
#                              max_segment=16.0, max_token=1024):
#     """
#     Rotary embeddings for the multimodal transformer
#     :param h: [B, L] h coords (default to 0.0 otherwise)
#     :param w: [B, L] w coords (default to 0.0 otherwise)
#     :param segment_idx: [B, L] segment_idx coords (default to 0.0 otherwise)
#     :param token_idx: [B, L] token_idx coords (default to 0.0 otherwise)
#     :param dtype: final datatype
#     :return: [B, L, 4] rotary coords
#     """
#     bs, ls = zip(*[x.shape for x in [h, w, segment_idx, token_idx] if x is not None])
#     L = ls[0]
#     B = bs[0]
#     assert all([x == L for x in ls])
#     assert all([x == B for x in bs])
#     # if (L > max_token):
#     #     print(f"WARNING: you passed in a sequence of length {L} but max_token is {max_token} so position "
#     #           f"embeddings might be weird. this is only a problem for text sequences though. "
#     #           f"in other words it shouldn't matter for downstream tasks")
#
#     h_vec = jnp.zeros([B, L], dtype=dtype) if (h is None) else h
#     w_vec = jnp.zeros([B, L], dtype=dtype) if (w is None) else w
#     s_vec = jnp.zeros([B, L], dtype=dtype) if (segment_idx is None) else segment_idx / max_segment
#     t_vec = jnp.zeros([B, L], dtype=dtype) if (token_idx is None) else token_idx / max_token
#     return jnp.stack([h_vec, w_vec, s_vec, t_vec], -1)


def construct_rotary_sinusoids(coords, rotary_hsize: int = 32, max_freq=10.0, dtype=None):
    """
    :param coords: [*batch_dims, seq_length, num_dimensions]
    :param rotary_hsize: How many dimensions we will finally use during the rotary embs
    :param max_freq: We will have frequencies that take the entire sequence (in the range of [0, 1]) as the first
                     one, up until take 1/max_freq of the entire sequence, in a logarithmic sequence

    :return: Sinusoids of size [*batch_dims, 2 (cos then sin), seq_len, rotary_hsize]
             they are repeated accordingly
    """
    # Sanity check
    print(coords.shape)
    batch_dims, seq_length, num_dims = coords.shape
    assert rotary_hsize % (num_dims * 2) == 0
    dim_expansion = rotary_hsize // (num_dims * 2)
    assert dim_expansion > 0

    freqs = torch.logspace(0.0, math.log2(max_freq / 2.0), dim_expansion, base=2,
                         dtype=coords.dtype if dtype is None else dtype)
    # for i in range(len(batch_dims) + 2):
    #     freqs = freqs[None]

    radians = coords[..., None] * freqs * np.pi
    radians = radians.reshape(batch_dims, seq_length, num_dims * dim_expansion)
    cos_t = torch.cos(radians)
    sin_t = torch.sin(radians)
    sinusoids = torch.stack([cos_t, sin_t], -3)
    print(sinusoids.shape)
    # Here we're repeating on the final dimension
    # bc later we will go through the first rotary_hsize coordinates and do
    # sin'd part: [-x0, x1, -x2, x3, ....]
    # cos'd part: [x0,  x1,  x2, x3, ....]
    sinusoids = sinusoids.repeat([1, 1, 1, 2])
    return sinusoids
#
#
def apply_rotary(query_key, sinusoids):
    """
    :param query_key: The query, key, or both. [*batch_dims, seq_len, num_heads, size_per_head]
    :param sinusoids:                      [*sin_batch_dims, 2, seq_len, rotary_hsize <= size_per_head // 2]
    :return: query_key with rotary applied
    """

    print('qk_shape', query_key.shape)
    print('sinu_shape', sinusoids.shape)
    sin_batch_dims, two_, seq_len, rotary_hsize = sinusoids.shape
    batch_dims, seq_len, num_heads, size_per_head = query_key.shape

    assert rotary_hsize <= size_per_head
    # for i in range(len(batch_dims) - len(sin_batch_dims)):
    #     sinusoids = sinusoids[None]

    sin = sinusoids[..., 0, :, None, :]
    cos = sinusoids[..., 1, :, None, :]

    qk_rope = query_key[..., :rotary_hsize]
    qk_rotated_two = torch.stack([-qk_rope[..., ::2], qk_rope[..., 1::2]], -1).reshape(qk_rope.shape)

    print(qk_rope.shape)
    print(cos.shape)
    print(qk_rotated_two.shape)
    print(sin.shape)

    qk_rope = qk_rope * cos + qk_rotated_two * sin
    query_key = torch.cat([qk_rope, query_key[..., rotary_hsize:]], -1)
    return query_key

#
# def kernel_init(key, shape, dtype=jnp.float32):
#     """
#     scaling by 0.02 works but for the last linear in the res mlp we often want to scale that down proportional
#     to depth. size is [4H, H]
#
#     hsize to depth:
#     [768 -> 12]   * 1/sqrt(12)
#     [1536 -> 48]  * 1/sqrt(48)
#
#     so i.e.
#     (6144, 1536) -> 0.02 / sqrt(48)
#     (3072, 768) -> 0.02 / sqrt(12)
#
#     roughly an inverse linear relationship. so..
#     scale = alpha / in_size
#     3072 * 0.02 / sqrt(12)  = alpha ~ 18
#
#     because this is multimodal my network is \sqrt{2} deeper, i'll divide by an extra sqrt 2.
#
#     :param key:
#     :param shape:
#     :return:
#     """
#     #
#     if len(shape) == 2:
#         in_size = shape[-2]
#     elif len(shape) == 3:
#         # special case for how jax handled attention shards, etc.
#         # e.g. for (768, 36, 64)   -> dimension is 768
#                 #  (12, 64, 768)   -> dimension is also 768 (first two)
#         in_size = shape[0]
#         out_size = shape[2]
#         if in_size < out_size:
#             in_size *= shape[1]
#     else:
#         print(f"weird shape for kernel init {shape}")
#         in_size = shape[-2]
#
#     stddev = min(18.0 / in_size, 0.02) / np.sqrt(2)
#     return jax.random.truncated_normal(key, -2, 2, shape, dtype) * stddev
# #
def apply_attention(qk, sinusoids, head_size):
    q, k= qk
    query_key = torch.cat([q,k], dim=-2)
    #query_key, value = torch.split(qkv, [2 * num_heads], dim=-2)

    if sinusoids is not None:
        query_key = apply_rotary(query_key, sinusoids)
    print_model(query_key, 'apply_rotary_query_key')
    query, key = query_key.chunk(2, dim=-2)
    print_model(query, 'query aftr rotary')
    print_model(key, 'key aftr rotary')

    #query = query.permute(0, 2, 1, 3) # query : B, S_q, num_head, head_size -> B, num_head, S_q, head_size
    #key = key.permute(0, 2, 1, 3) # key : B, S_k, num_head, head_size -> B, num_head, S_k, head_size

    query = query / math.sqrt(query.shape[-1])
    #att_score = torch.matmul(query, key.transpose(-1, -2))
    att_score = torch.einsum('...qhd,...khd->...hqk', query, key)
    att_probs = nn.functional.softmax(att_score, dim=-1)
    # attention_probs = nn.attention.dot_product_attention_weights(query=query, key=key, bias=attention_bias,
    #                                                              dtype=dtype)
    #
    return att_probs

# apply_attention_ckpt = jax.checkpoint(apply_attention)
#
#
class AttentionLayer(nn.Module):
    """
    Attention layer that is somewhat simpler (since only need to do encoder->encoder attention). but with Rotary
    """
    def __init__(self, size_per_head=64, dtype=torch.float32, hidden_size=768):
        super().__init__()
        self.size_per_head = size_per_head
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_attention_head = self.hidden_size // self.size_per_head
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_head, self.size_per_head)
        x = x.view(*new_x_shape)
        return x #.permute(0, 2, 1, 3)

    def __call__(self, x, sinusoids=None, attention_bias=None):
        """
        :param x: [*batch_dims, seq_len, hidden_size]
        :param attention_bias [batch_dims, 1, seq_len, seq_len]
        :param sinusoids [*batch_dims, seq_len, rotary_hsize <= size_per_head // 2]. This is how we encode position
        :return:
        """
        #print("{}: {}".format(self.name, 'NOT doing rotary ' if sinusoids is None else f'doing rotary: {sinusoids}'), flush=True)
        #print("{}: ~{}doing attnmask~".format(self.name, 'NOT ' if attention_bias is None else ''), flush=True)
        #batch_dims, seq_len, hidden_size = x.shape
        #assert self.hidden_size % self.size_per_head == 0

        x_query = self.transpose_for_scores(self.query_proj(x)) # B S Nd Hs
        x_key = self.transpose_for_scores(self.key_proj(x))  # B S Nd Hs
        x_value = self.transpose_for_scores(self.value_proj(x)) # B S Nd Hs

        print_model(x_query, 'x_query')
        print_model(x_key, 'x_key')
        print_model(x_value, 'x_value')
        # qkv = nn.DenseGeneral(features=(3 * num_heads, self.size_per_head), axis=-1,
        #                       dtype=self.dtype, kernel_init=kernel_init, name='qkv')(x)


        #attn_fn = apply_attention_ckpt if (seq_len > 1024 and self.hidden_size >= 1024) else apply_attention
        att_probs = apply_attention((x_query, x_key), sinusoids, self.size_per_head) # B, nh, S, hd

        ####### NO DROPOUT???
        if attention_bias is not None:
            att_probs = att_probs * attention_bias
        print_model(att_probs, 'att_probs') # B, nh,

        x_value = x_value.permute(0, 2, 1, 3) # B, S, nh, hs => B, nh, S, hs
        print('!!!! att_probs shape', att_probs.shape)
        print('!!!! x_value shape', x_value.shape)
        x = torch.matmul(att_probs, x_value)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.hidden_size,)
        x = x.view(*new_x_shape)

        print(x.shape)
        #### end
        x = self.attn_proj(x)
        # x = nn.DenseGeneral(features=self.hidden_size, axis=(-2, -1), kernel_init=kernel_init,
        #                     dtype=self.dtype, name='attn_proj', use_bias=False)(x)
        return x
#
#
def my_gelu(x):
    return x * torch.sigmoid(1.702 * x)
#
#
class MLPBlock(nn.Module):
    def __init__(self, hidden_size, dtype=torch.float32, expansion_mult=4):
        super(MLPBlock, self).__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.expansion_mult = expansion_mult
        self.intermediate = nn.Linear(self.hidden_size, self.hidden_size * self.expansion_mult)
        self.out = nn.Linear(self.hidden_size * self.expansion_mult, self.hidden_size, bias=False)

    def forward(self, x):
        # x1 = nn.Dense(features=hidden_size * self.expansion_mult, dtype=self.dtype, kernel_init=kernel_init,
        #               name='intermediate')(x)
        x1 = self.intermediate(x)
        x1 = my_gelu(x1)
        #x1 = nn.Dense(features=hidden_size, dtype=self.dtype, kernel_init=kernel_init, name='out', use_bias=False)(x1)
        x1 = self.out(x1)
        return x1
#
#

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, expansion_mult=4, size_per_head=64, dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion_mult = expansion_mult
        self.size_per_head = size_per_head
        self.dtype = dtype
        self.pre_attn_ln = torch.nn.LayerNorm(normalized_shape=self.hidden_size, eps=1e-5, dtype=self.dtype)
        self.pre_mlp_ln = torch.nn.LayerNorm(normalized_shape=self.hidden_size, eps=1e-5, dtype=self.dtype)
        self.attention_layer = AttentionLayer(hidden_size=self.hidden_size, dtype=self.dtype,
                                              size_per_head=self.size_per_head)
        self.mlp_layer = MLPBlock(self.hidden_size, expansion_mult=self.expansion_mult, dtype=self.dtype)


    def forward(self, x, sinusoids=None, attention_bias=None):
        batch_dims, seq_len, hsz = x.shape

        assert hsz == self.hidden_size
        print("Transformer layer my dtype={}; x={}; sinusoids={}".format(
            self.dtype, (x.shape, x.dtype),
            (sinusoids.dtype, sinusoids.shape) if sinusoids is not None else None), flush=True)
        x_ln = self.pre_attn_ln(x)
        print_model(x_ln, 'after_pre_attn_ln')

        x_attn = self.attention_layer(x_ln, sinusoids=sinusoids, attention_bias=attention_bias)
        print_model(x_attn, 'after_attention_layer')

        x += x_attn

        x_ln2 = self.pre_mlp_ln(x)
        print_model(x_ln2, 'after_pre_mlp_ln')

        x_mlp = self.mlp_layer(x_ln2)
        print_model(x_mlp, 'after_mlp_layer')

        x += x_mlp
        return x

class TransformerEncoder(nn.Module):
    """
    1D transformer encoder. You can optionally add a CLS token and we can pool from it for later
    """

    def __init__(self, hidden_size, num_layers, expansion_mult=4, size_per_head=64,
                 dtype=torch.float32, add_cls_token=False, cls_output_size=None, rotary_hsize=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.expansion_mult = expansion_mult
        self.size_per_head = size_per_head
        self.dtype = dtype
        self.add_cls_token = add_cls_token
        self.cls_output_size = cls_output_size
        self.rotary_hsize = rotary_hsize
        #cls_token = self.param('cls', nn.initializers.normal(stddev=0.02), (self.hidden_size,))
        self.cls = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, 1, self.hidden_size)))
        self.cls_proj = nn.Linear(self.hidden_size, self.hidden_size)
        #self.pos_emb = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, seq_len, self.hidden_size)))
        self.pre_ln = torch.nn.LayerNorm(normalized_shape=self.hidden_size, eps=1e-5, dtype=self.dtype)
        self.final_ln = torch.nn.LayerNorm(normalized_shape=self.hidden_size, eps=1e-5, dtype=self.dtype)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(hidden_size=self.hidden_size, expansion_mult=self.expansion_mult,
                              size_per_head=self.size_per_head, dtype=self.dtype) for _ in range(self.num_layers)])

    def forward(self, x, rotary_coords=None, attention_mask=None, is_valid=None):
        """
        :param x: [*batch_dims, L, hidden_size]
        :param rotary_coords: Coords for doing rotary embeddings   [*rotary_batch_dims, L, rotary_axes]

        provide none, or one of the following:
        :param attention_mask: [batch_dims, L, L]. If provided then use this to mask where attention goes
        :param is_valid: [batch_dims, L] Is input X valid
        :return:
        """
        batch_dims, seq_len, hsz = x.shape
        assert hsz == self.hidden_size

        # Add CLS token
        if self.add_cls_token:
            seq_len += 1
            if attention_mask is not None:
                raise ValueError("Attention mask must not be provided if adding CLS token")

            # for i in range(len(batch_dims) + 1):  # For sequence dimension, and batch dimensions
            #     self.cls_token = self.cls_token[None]
            cls_token = torch.tile(self.cls, [batch_dims, 1, 1])
            print_model(cls_token, 'cls_token')

            x = torch.cat([cls_token.to(x.dtype), x], -2)
            if is_valid is not None:
                 is_valid = torch.cat([torch.ones(list(batch_dims) + [1], dtype=torch.bool), is_valid], -1)

            if rotary_coords is not None:
                # CLS token is always 0's
                rotary_coords = torch.cat([torch.zeros_like(rotary_coords[..., :1, :]), rotary_coords], -2)
            print_model(rotary_coords, 'rotary_coords')

        # Optional rotary embeddings
        if rotary_coords is not None:
            print(rotary_coords.shape)
            rotary_batch_dims, seq_len_, rotary_axes = rotary_coords.shape
            assert seq_len_ == seq_len
            assert self.rotary_hsize is not None
            assert self.rotary_hsize <= self.size_per_head
            sinusoids = construct_rotary_sinusoids(rotary_coords, rotary_hsize=self.rotary_hsize)
            print_model(sinusoids, 'sinusoids')

        else:
            sinusoids = None
            print("ROTARY IS NONE SO PROVIDING PE", flush=True)
            # pos_emb = self.param('pe', nn.initializers.normal(stddev=0.02), (seq_len, self.hidden_size))
            # for i in range(len(batch_dims)):
            #     pos_emb = pos_emb[None]
            # x += self.pos_emb

        if (is_valid is not None) and (attention_mask is None):
            print("Constructing the attention mask")
            attention_mask = is_valid[..., None] & is_valid[..., None, :]
        elif (is_valid is not None) and (attention_mask is not None):
            raise ValueError("Provide only one of `is_valid` and `attention_mask` "
                             "as we can use is_valid to construct attention mask")

        if attention_mask is not None:
            # Broadcast attention mask to be over num head`s dimension
            attention_mask = attention_mask[..., None, :, :]
            if attention_mask > 0:
                attention_bias = torch.full(attention_mask.shape, 0.).astype(self.dtype)
            else:
                attention_bias = torch.full(attention_mask.shape, -1e10).astype(self.dtype)
        else:
            attention_bias = None
        print_model(attention_bias, 'attention_bias')

        x = self.pre_ln(x)
        print_model(x, 'after pre_ln')

        for layer_output in self.transformer_layers:
            x = layer_output(x, sinusoids=sinusoids, attention_bias=attention_bias)
        x_ln = self.final_ln(x)
        print_model(x_ln, 'after final_ln')

        info = {}
        if self.add_cls_token:
            cls_vec = x_ln[..., 0, :]
            info['cls'] = self.cls_proj(cls_vec)
            print_model(info['cls'], 'after_cls_proj')
            info['seq'] = x_ln[..., 1:, :]
            print_model(info['seq'], 'info_seq')
        else:
            info['seq'] = x_ln
        return info

def print_model(_param, name):
    start_str = f'---- {name} ----'
    print(start_str)
    print(_param)
    print('=' * len(start_str))


class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, hidden_size=768, size_per_head=64, dtype=torch.float32, num_layers=12,
                 pooling_ratio=2, output_grid_h=12, output_grid_w=20, do_rotary=True):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.size_per_head = size_per_head
        self.num_heads = self.hidden_size // self.size_per_head
        self.dtype = dtype
        self.num_layers = num_layers
        self.pooling_ratio = pooling_ratio
        self.output_grid_h = output_grid_h
        self.output_grid_w = output_grid_w
        self.do_rotary = do_rotary

        self.embedding = nn.Linear(self.hidden_size, self.hidden_size)
        self.transformer = TransformerEncoder(hidden_size=self.hidden_size, dtype=self.dtype,
                                   add_cls_token=True, num_layers=self.num_layers, size_per_head=self.size_per_head)
        self.seq_attnpool = nn.MultiheadAttention(self.hidden_size, self.num_heads, batch_first=True)
        # self.seq_q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # self.seq_k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # self.seq_v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # self.seq_out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        """
        :param x: Patched images of size [B, H*W, P * P * 3]
        :return: A pooled representation of size [B, hidden_size]
                 and a seq representation of size [B, HW // (pooling_ratio ** 2), hidden_size]
        """
        batch_dims, hw, pp3 = x.shape
        assert hw == (self.output_grid_h * self.output_grid_w)
        assert pp3 == (self.patch_size ** 2) * 3

        # i think no need to normalize here because pixel distributions are roughly the same
        x = self.embedding(x)
        print_model(x, 'embedding')

        if self.do_rotary:
            coords = get_rotary_coordinates_2d(self.output_grid_h, self.output_grid_w, dtype=self.dtype)
            print_model(coords, 'coords')
            coords = torch.tile(coords, [batch_dims, 1, 1])
        else:
            coords = None
        t_out = self.transformer(x, rotary_coords=coords)

        # Attention pool
        assert self.output_grid_h % self.pooling_ratio == 0
        assert self.output_grid_w % self.pooling_ratio == 0
        h2 = self.output_grid_h // self.pooling_ratio
        w2 = self.output_grid_w // self.pooling_ratio
        b2 = int(np.prod([batch_dims, h2]))

        seq = torch.reshape(t_out['seq'], [b2, self.pooling_ratio, w2, self.pooling_ratio, self.hidden_size])
        seq = seq.swapaxes(-4, -3)
        seq = seq.reshape([b2 * w2, self.pooling_ratio ** 2, self.hidden_size])

        inputs_q = seq.mean(-2, keepdims=True)
        # seq_q = self.seq_q_proj(inputs_q)
        # seq_k = self.seq_k_proj(seq)
        # seq_v = self.seq_v_proj(seq)

        seq_attn_out, _  = self.seq_attnpool(inputs_q, seq, seq)
        # seq_attn_out = self.seq_out_proj(seq_attn_out)

        seq_attnpool = seq_attn_out.reshape([batch_dims, h2*w2, self.hidden_size])
        # seq_attnpool = nn.MultiHeadDotProductAttention(num_heads=self.hidden_size // self.size_per_head, dtype=self.dtype,
        #                                                deterministic=True,
        #                                                name='seq_attnpool')(inputs_q=inputs_q, inputs_kv=seq)
        # seq_attnpool = seq_attnpool.reshape(list(batch_dims) + [h2 * w2, self.hidden_size])

        t_out['seq_attnpool'] = seq_attnpool
        return t_out
#

class AudioTransformer(nn.Module):
    def __init__(self, patch_size=2, hidden_size=768, dtype=torch.float32, num_layers=12, pooling_ratio=3, do_rotary=True, size_per_head=64):
        self.patch_size: int = patch_size
        self.hidden_size: int = hidden_size
        self.dtype: Any = torch.float32
        self.num_layers: int = num_layers
        self.pooling_ratio: int = pooling_ratio
        self.do_rotary: bool = do_rotary
        self.size_per_head: int = size_per_head

        self.embedding = nn.Conv(features=self.hidden_size, kernel_size=[self.patch_size], strides=[self.patch_size],
                    dtype=self.dtype, kernel_init=kernel_init)

        self.transformer = TransformerEncoder(hidden_size=self.hidden_size, dtype=self.dtype, add_cls_token=True,
                                   num_layers=self.num_layers, size_per_head=self.size_per_head)

    def forward(self, x):
        """
        :param x: Audio sequence of size [B, L, num_mels + 1]
        :return: A pooled representation of size [B, H] and a seq representation of size [B, L // pooling_ratio, H]
        """
        batch_dims, raw_len, num_mels_plus_playback_speed = x.shape
        assert num_mels_plus_playback_speed == 65
        assert raw_len % self.patch_size == 0
        seq_len = raw_len // self.patch_size

        x = self.embedding(x)

        if self.do_rotary:
            coords = get_rotary_coordinates(seq_len, dtype=self.dtype, center_origin=True)[:, None] / seq_len
        else:
            coords = None

        t_out = self.transformer(x, rotary_coords=coords)

        # Attention pool
        assert seq_len % self.pooling_ratio == 0
        l2 = seq_len // self.pooling_ratio
        seq = jnp.reshape(t_out['seq'], [-1, self.pooling_ratio, self.hidden_size])
        seq_attnpool = nn.MultiHeadDotProductAttention(num_heads=self.hidden_size // self.size_per_head,
                                                       dtype=self.dtype,
                                                       deterministic=True,
                                                       name='seq_attnpool')(inputs_q=seq.mean(-2, keepdims=True),
                                                                            inputs_kv=seq)

        seq_attnpool = seq_attnpool.reshape(list(batch_dims) + [l2, self.hidden_size])
        t_out['seq_attnpool'] = seq_attnpool
        return t_out


class SpanTransformer(nn.Module):
    def __init__(self, hidden_size=768, size_per_head=64, dtype=torch.float32, num_layers=3, max_len=16, do_rotary=True):
        super(SpanTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.size_per_head = size_per_head
        self.dtype = dtype
        self.num_layers = num_layers
        self.max_len = max_len
        self.do_rotary = do_rotary

        self.transformer = TransformerEncoder(hidden_size=self.hidden_size, dtype=self.dtype,
                           add_cls_token=True, num_layers=self.num_layers, size_per_head=self.size_per_head)

    def forward(self, x, x_isvalid):
        """
        :param x: Text sequence of size [B, L, H]
        :param x_isvalid: Mask of size [B, L]
        :return: A pooled representation of size [B, H]
        """
        batch_dims, seq_len, hidden_size = x.shape
        assert seq_len < self.max_len
        # don't center bc many things are short
        if self.do_rotary:
            coords = get_rotary_coordinates(seq_len, center_origin=False, dtype=self.dtype)[:, None] / self.max_len
        else:
            coords = None
        t_out = self.transformer(x, is_valid=x_isvalid, rotary_coords=coords)
        return t_out['cls']


class TokenEmbedder(nn.Module):
    def __init__(self, hidden_size, vocab_size=32768, dtype=torch.float32):
        super(TokenEmbedder, self).__init__()
        """
        Independently embed tokens
        """
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dtype = dtype
        _init = nn.initializers.normal(stddev=0.02) if self.hidden_size <= 768 else nn.initializers.xavier_uniform()
        self.everything_embedded = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size, dtype=self.dtype,
                                       embedding_init=_init)


    def forward(self, token_dict):
        """
        :param token_dict: One or multiple tensors of tokens -- embed all at once and keep their shapes
        :return: a dict that's the same size as token_dict
        """
        keys = sorted(token_dict.keys())
        shapes = [token_dict[k].shape for k in keys]
        n_elems = [int(np.prod(y)) for y in shapes]
        x_flat = jnp.concatenate([token_dict[k].reshape(-1) for k in keys], 0)


        embedded = self.everything_embedded(x_flat)

        # this seems to be needed becauase nn.Embed is a thin wrapper around the actual parameter
        # if self.dtype == jnp.bfloat16:
        #     everything_embedded = everything_embedded.astype(jnp.bfloat16)
        embed_split = jnp.split(embedded, np.cumsum(n_elems), axis=0)

        out_dict = {}
        for i, (k_i, s_i, v_i) in enumerate(zip(keys, shapes, embed_split)):
            out_dict[k_i] = v_i.reshape(list(s_i) + [self.hidden_size])
        return out_dict
#
#
def one_hot_pool(do_pool, idx, v, num_segments, real_bsize=None):
    """
    Pools values, this is needed for getting the hidden representations at positions corresponding to mask tokens

    :param do_pool:     [batch_size, L]
    :param idx:         [batch_size, L]. Index into 0...num_segments.
    :param v:           [batch_size, L, h]. The values that will be pooled
    :param num_segments: output size
    :param dtype: dtype
    :return:        [batch_size, num_segments, h] - the pooled values
    """
    B, L, H = v.shape
    assert do_pool.shape == (B, L)
    assert idx.shape == (B, L)

    if real_bsize is not None:
        # Reshape
        l2 = (L * B) // real_bsize
        do_pool = do_pool.reshape((real_bsize, l2))
        idx = idx.reshape((real_bsize, l2))
        v = v.reshape((real_bsize, l2, H))

    if do_pool:
        pointer = idx
    else:
        pointer = torch.full(idx.shape, -1)
    pointer_oh = torch.nn.one_hot(pointer, num_classes=num_segments, dtype=v.dtype)

    attended = torch.einsum('bls,blh->bsh', pointer_oh, v)
    return {'x': attended, 'idx_oh': pointer_oh}
#
#
def unit_normalize(x):
    """
    Normalize `x` to have unit norm over the final dimension
    :param x:
    :return:
    """
    x_f32 = x.to(torch.float32)
    x_norm = x_f32 / torch.sqrt(torch.square(x_f32).sum(-1, keepdims=True) + 1e-5)
    return x_norm.astype(x.dtype)


class MerlotReserve(nn.Module):
    config: Dict = None

    def __init__(self, config):
        super().__init__()
        for k, v in self.config.items():
            setattr(self, k, v)

        self.output_grid_h, self.output_grid_w = self.config['output_grid']
        self.size_per_head = self.config.get('size_per_head', 64)

        self.vision_encoder = VisionTransformer(num_layers=self.config['vit_num_layers'],
                                                patch_size=self.config['vit_patch_size'],
                                                pooling_ratio=self.config['vit_pooling_ratio'],
                                                output_grid_h=self.output_grid_h,
                                                output_grid_w=self.output_grid_w,
                                                hidden_size=self.config['hidden_size'],
                                                dtype=self.dtype,
                                                size_per_head=self.size_per_head,
                                                )
        self.audio_encoder = AudioTransformer(num_layers=self.config['audio_num_layers'],
                                              patch_size=self.config['audio_patch_size'],
                                              pooling_ratio=self.config['audio_seq_length'] // (
                                                  self.config['audio_token_length'] * self.config['audio_patch_size']),
                                              hidden_size=self.config['hidden_size'],
                                              dtype=self.dtype,
                                              size_per_head=self.size_per_head,
                                              )
        self.token_encoder = TokenEmbedder(hidden_size=self.config['hidden_size'],
                                           dtype=self.dtype,
                                           )
        self.span_encoder = SpanTransformer(num_layers=self.config['span_num_layers'], hidden_size=self.hidden_size,
                                            dtype=self.dtype,
                                            size_per_head=self.size_per_head,
                                            )
        self.joint_transformer = TransformerEncoder(hidden_size=self.config['hidden_size'],
                                                    num_layers=self.config['joint_num_layers'],
                                                    add_cls_token=False,
                                                    dtype=self.dtype,
                                                    size_per_head=self.size_per_head,
                                                    )

        self.joint_proj = nn.Dense(features=self.config['hidden_size'], dtype=self.dtype,
                                   kernel_init=kernel_init, name='head')

        self.scale_params = self.param('contrastive_scales', nn.initializers.ones, (3,))

    # @classmethod
    # def from_config(cls, config, **kwargs):
    #     my_config = deepcopy(config)
    #     my_config['model']['data'] = my_config['data']
    #     return cls(config=my_config['model'], **kwargs)

    # def setup(self):
    #     for k, v in self.config.items():
    #         setattr(self, k, v)
    #
    #     self.dtype = torch.float32
    #     print(f"Using dtype {self.dtype}", flush=True)
    #
    #     self.output_grid_h, self.output_grid_w = self.config['output_grid']
    #     self.size_per_head = self.config.get('size_per_head', 64)
    #
    #     self.vision_encoder = VisionTransformer(num_layers=self.config['vit_num_layers'],
    #                                             patch_size=self.config['vit_patch_size'],
    #                                             pooling_ratio=self.config['vit_pooling_ratio'],
    #                                             output_grid_h=self.output_grid_h,
    #                                             output_grid_w=self.output_grid_w,
    #                                             hidden_size=self.config['hidden_size'],
    #                                             dtype=self.dtype,
    #                                             size_per_head=self.size_per_head,
    #                                             )
    #     self.audio_encoder = AudioTransformer(num_layers=self.config['audio_num_layers'],
    #                                           patch_size=self.config['audio_patch_size'],
    #                                           pooling_ratio=self.config['audio_seq_length'] // (
    #                                               self.config['audio_token_length'] * self.config['audio_patch_size']),
    #                                           hidden_size=self.config['hidden_size'],
    #                                           dtype=self.dtype,
    #                                           size_per_head=self.size_per_head,
    #                                           )
    #     self.token_encoder = TokenEmbedder(hidden_size=self.config['hidden_size'],
    #                                        dtype=self.dtype,
    #                                        )
    #     self.span_encoder = SpanTransformer(num_layers=self.config['span_num_layers'], hidden_size=self.hidden_size,
    #                                         dtype=self.dtype,
    #                                         size_per_head=self.size_per_head,
    #                                         )
    #     self.joint_transformer = TransformerEncoder(hidden_size=self.config['hidden_size'],
    #                                                 num_layers=self.config['joint_num_layers'],
    #                                                 add_cls_token=False,
    #                                                 dtype=self.dtype,
    #                                                 size_per_head=self.size_per_head,
    #                                                 )
    #
    #     self.joint_proj = nn.Dense(features=self.config['hidden_size'], dtype=self.dtype,
    #                                kernel_init=kernel_init, name='head')
    #
    #     self.scale_params = self.param('contrastive_scales', nn.initializers.ones, (3,))

    # def init_from_dummy_batch(self, dummy_batch):
    #     def init_model():
    #         k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    #         dummy_batch_jax = {k: jnp.asarray(v[0, 0, None]) for k, v in dummy_batch.items()}
    #         return self.init(k2, dummy_batch_jax)
    #
    #     print("start compiling", flush=True)
    #     params = jax.jit(init_model, backend='cpu')()['params']
    #
    #     # in case anything got initialized to bf16
    #     params = bf16_to_f32(params)
    #     print(clu.parameter_overview.get_parameter_overview(params), flush=True)
    #
    #     return params

    def prepare_multimodal_inputs(self, tokens, token_segment_idx=None, token_embs=None, vision_input=None,
                                  audio_spans=None, audio_pointers=None, padding_len=None, video_src_idx=None):
        """
        Prepare multimodal inputs. Where B is the # of segments, we have:
        :param tokens: [B, seq_len]
        :param token_segment_idx: [B, seq_len] thing of segments for the tokens
        :param token_embs: [B, seq_len, H]. You don't need to provide this, we will construct it if not provided

        :param vision_input: [B, vis_seq_len, H]. Optional

        :param audio_spans: [B, num_audio_seqs, audio_seq_length, H].
        :param audio_pointers: [B, seq_len]. For any token in `tokens` that is equal to AUDIOSPAN, we will select
                              `audio_seq_length` consecutive tokens from the audio span from audio_spans, for that token.
        :param padding_len: Padding len if that needs to be used

        :param video_src_idx: [B, num_segments.] If provided, and token_segment_idx is not None, we will mask
                              attentions such that different videos can't attend to one another
        :return: * Multimodal inputs of size [B, seq_len, H]
                 * Rotary embedding coords of size [B, seq_len, 4] (4 is the dimension we use)
                 * Attention mask of size [B, seq_len, seq_len]
        """
        B, L = tokens.shape
        if token_embs is None:
            print("Constructing token embs in `prepare_multimodal_inputs`", flush=True)
            token_embs = self.token_encoder({'k': tokens})['k']

        if (audio_spans is None) or (audio_pointers is None):
            print("Not including audio input", flush=True)
        else:
            print("adding in audio input!")
            b_, num_audio_seqs, audio_token_length, h_ = audio_spans.shape
            assert b_ == B
            assert self.audio_token_length == audio_token_length

            is_audio_src = (tokens == AUDIOSPAN)
            assert tokens.shape == audio_pointers.shape
            audio_ptr = torch.maximum(audio_pointers, 0)

            # subtle weirdness -- this doesn't check that e.g. if you want span "0" 8 times, you're actually
            # putting that token there 8 times. So you can truncate the sequence, but you have to truncate at the end
            audio_subpos = torch.maximum(torch.cumsum(is_audio_src.astype(jnp.int32), -1) - 1, 0) % self.audio_token_length

            audio_embs = audio_spans[torch.arange(B, dtype=jnp.int32)[:, None], audio_ptr, audio_subpos]
            token_embs = lax.select(jnp.repeat(is_audio_src[..., None], self.hidden_size, axis=-1),
                                    on_true=audio_embs, on_false=token_embs)

        token_idx = torch.tile(1.0 + jnp.arange(L, dtype=self.dtype)[None], [B, 1])
        coords = multimodal_rotary_coords(
            segment_idx=token_segment_idx.astype(self.dtype) if token_segment_idx is not None else None,
            token_idx=token_idx, dtype=self.dtype)

        if vision_input is not None:

            hpool = self.output_grid_h // self.vit_pooling_ratio
            wpool = self.output_grid_w // self.vit_pooling_ratio
            img_coords_pool = get_rotary_coordinates_2d(hpool, wpool, dtype=self.dtype)

            b_, vis_seq_len, h_ = vision_input.shape
            num_pool_segments = vis_seq_len // (hpool * wpool)
            img_coords = jnp.tile(img_coords_pool, [num_pool_segments, 1])
            vis_segment_idx = jnp.arange(num_pool_segments, dtype=jnp.int32).repeat(hpool * wpool)
            print(f"Vision input {hpool}x{wpool}: {vis_seq_len}. num_pool_segments: {num_pool_segments}",
                  flush=True)
            img_coords = jnp.tile(img_coords[None], [B, 1, 1])
            vis_segment_idx = jnp.tile(vis_segment_idx[None], [B, 1])
            img_mm_coords = multimodal_rotary_coords(segment_idx=vis_segment_idx.astype(self.dtype),
                                                   h=img_coords[..., 0], w=img_coords[..., 1], dtype=self.dtype)
            assert img_mm_coords.shape[-2] == vis_seq_len
            coords = jnp.concatenate([coords, img_mm_coords], 1)
            token_embs = jnp.concatenate([token_embs, vision_input], 1)
        else:
            vis_seq_len = 0
            vis_segment_idx = None

        # Attention mask
        is_valid = (tokens != PADDING)
        if vis_seq_len > 0:
            is_valid = jnp.concatenate([is_valid, jnp.ones([B, vis_seq_len], dtype=is_valid.dtype)], 1)

        # Pad everything if needed
        if padding_len is not None:
            extra_len = padding_len - is_valid.shape[1]
            assert extra_len >= 0
            if extra_len > 0:
                print(f"Padding by {extra_len}", flush=True)
                is_valid = jnp.concatenate([is_valid, jnp.zeros([B, extra_len], dtype=is_valid.dtype)], 1)
                coords = jnp.concatenate([coords, jnp.zeros([B, extra_len, 4], dtype=coords.dtype)], 1)
                token_embs = jnp.concatenate(
                    [token_embs, jnp.zeros([B, extra_len, self.hidden_size], dtype=token_embs.dtype)], 1)
        else:
            extra_len = 0

        attn_mask = is_valid[:, None] & is_valid[:, :, None]

        # Deal with the case where we packed multiple things together in a single seqence
        if (video_src_idx is not None) and (token_segment_idx is not None):
            print("Masking using video_src_idx to split up different videos in the same sequence", flush=True)
            batch_indexer = jnp.arange(B, dtype=jnp.int32)[:, None]
            video_src = [video_src_idx[batch_indexer, token_segment_idx]]
            if vis_segment_idx is not None:
                video_src.append(video_src_idx[batch_indexer, vis_segment_idx])
            if extra_len > 0:
                video_src.append(jnp.full([B, extra_len], -1, dtype=jnp.int32))
            video_src = jnp.concatenate(video_src, -1)

            attn_mask &= (video_src[:, None] == video_src[:, :, None])

        return {'x': token_embs, 'rotary_coords': coords, 'attention_mask': attn_mask}

    def __call__(self, batch):
        raise NotImplementedError()

    #####################################################################
    # Below is a preliminary API for zero shot capabilities
    #####################################################################

    def embed_text_spans_only(self, text_spans):
        """
        Use this function to only embed the text span options (for downstream)
        :param text_spans: [B, L] text
        :return: [B, H] matrix of vectors (one per text span option)
        """
        token_embs = self.token_encoder({'text_spans': text_spans})['text_spans']
        return unit_normalize(self.span_encoder(x=token_embs, x_isvalid=text_spans != PADDING))

    def embed_audio_only(self, audio_clips):
        """
        This could be important for Reza investigating audio localization over time?
        :param audio_clips: [num_subsegments, num_hops_per_audio, 65]
        :return: [num_subsegments, H] matrix of vectors (one per audio span option)
        """
        *batch_dims, num_hops_per_audio, num_mels_plus_one = audio_clips.shape
        audio_enc = self.audio_encoder(audio_clips.reshape((-1, self.audio_seq_length, 65)))['cls']
        audio_enc = unit_normalize(audio_enc)
        return audio_enc.reshape(*batch_dims, self.hidden_size)

    def get_imgseq_only(self, imgs):
        """
        Only for precomputing stuff for vision encoder
        :param imgs: [*batch_dims, num_patch_per_img, 768]
        :return: [*batch_dims, num_patch_per_img / 4, 768
        """
        *batch_dims, num_patch_per_img, pp3 = imgs.shape
        imgs_enc = self.vision_encoder(imgs.reshape((-1, num_patch_per_img, pp3)))['seq_attnpool']
        return imgs_enc.reshape(list(batch_dims) + [num_patch_per_img // 4, self.hidden_size])

    def get_audioseq_only(self, audio_clips):
        """
        Only for precomputing stuff for vision encoder
        :param imgs: [*batch_dims, num_patch_per_img, 768]
        :return: [*batch_dims, num_patch_per_img / 4, 768
        """
        return self.audio_encoder(audio_clips.reshape((-1, self.audio_seq_length, 65)))['seq_attnpool']


    def embed_video(self, images, audio_clips, tokens, subseg_idxs):
        """
        This embeds a video, with both images and audio clips.
        NOTE: It's wasted compute if audio_clips is empty (maybe we should have a different function for that)
        :param images: [num_segments, num_patch_per_img, 768] - `prepatchified' images
        :param audio_clips: [num_subsegments, num_hops_per_audio, 65]
        :param tokens: [L] tokens (or the token `AUDIOSPAN' which says we use audio there.)
        :param subseg_idxs: [L] which subsegment we're on, for each token.
        :return: a joint encoding of size [L, H], tokens conditioned on images.
        """
        num_segments, num_patch_per_img, pp3 = images.shape
        assert pp3 == 768

        num_subsegments, num_hops_per_audio, num_mels_plus_one = audio_clips.shape
        assert num_subsegments == 3 * num_segments
        assert num_hops_per_audio == self.audio_seq_length
        assert num_mels_plus_one == 65

        token_length, = tokens.shape
        token_length_, = subseg_idxs.shape
        assert token_length_ == token_length
        ###
        imgs_enc = self.vision_encoder(images.reshape((-1, num_patch_per_img, pp3)))['seq_attnpool']
        imgs_enc = imgs_enc.reshape((num_segments * num_patch_per_img // 4, self.hidden_size))

        # Encode audio to be [num_audio_spans, 6, H]
        audio_enc = self.audio_encoder(audio_clips.reshape((-1, self.audio_seq_length, 65)))['seq_attnpool']

        mm_inputs = self.prepare_multimodal_inputs(
            tokens=tokens[None],
            token_segment_idx=subseg_idxs[None] // 3,
            vision_input=imgs_enc[None],
            audio_pointers=subseg_idxs[None],
            audio_spans=audio_enc[None],
        )
        joint_enc = self.joint_transformer(**mm_inputs)['seq']
        joint_enc = unit_normalize(self.joint_proj(joint_enc[0, :token_length]))
        return joint_enc

    def batch_embed_video(self, images, audio_clips, tokens, subseg_idxs):
        return jax.vmap(self.embed_video)(images, audio_clips, tokens, subseg_idxs)

    def embed_singleimg_with_multiimg_prompt(self, images_prompt, images, tokens, subseg_idxs):
        """
        This embeds a video of images. `img_prompt' is a prefix that is already precomputed.

        :param images_prompt: [num_segments0, num_patch_per_img // 4, hidden_size] - precomputed images that we plug in
        :param images: [num_segments1, num_patch_per_img, 768] - `prepatchified' images
        :param tokens: [L] tokens
        :param subseg_idxs: [L] which subsegment we're on, for each token.
        :return: a joint encoding of size [L, H], tokens conditioned on images.
        """
        ns0 = images_prompt.shape[0]
        ns1, num_patch_per_img, pp3 = images.shape
        assert (ns0 + ns1) <= 8
        imgs_enc = self.vision_encoder(images)['seq_attnpool']
        imgs_enc = jnp.concatenate([images_prompt, imgs_enc], 0)
        imgs_enc = imgs_enc.reshape(((ns0 + ns1) * num_patch_per_img // 4, self.hidden_size))

        token_length, = tokens.shape
        token_length_, = subseg_idxs.shape
        assert token_length_ == token_length

        mm_inputs = self.prepare_multimodal_inputs(
            tokens=tokens[None],
            token_segment_idx=subseg_idxs[None] // 3,
            vision_input=imgs_enc[None],
            audio_pointers=None,
            audio_spans=None,
        )
        joint_enc = self.joint_transformer(**mm_inputs)['seq']
        joint_enc = unit_normalize(self.joint_proj(joint_enc[0, :token_length]))
        return joint_enc

    def embed_preencoded_noaudio(self, images_enc, tokens, subseg_idxs):
        """
        This embeds a video of images. `img_prompt' is a prefix that is already precomputed.

        :param images_enc: [num_segments, num_patch_per_img // 4, hidden_size] - precomputed images that we plug in
        :param tokens: [L] tokens
        :param subseg_idxs: [L] which subsegment we're on, for each token.
        :return: a joint encoding of size [L, H], tokens conditioned on images.
        """
        ns, num_patch_per_img_div_4, hidden_size = images_enc.shape
        images_enc = jnp.reshape(images_enc, [ns * num_patch_per_img_div_4, hidden_size])
        token_length, = tokens.shape
        token_length_, = subseg_idxs.shape
        assert token_length_ == token_length

        mm_inputs = self.prepare_multimodal_inputs(
            tokens=tokens[None],
            token_segment_idx=subseg_idxs[None] // 3,
            vision_input=images_enc[None],
            audio_pointers=None,
            audio_spans=None,
        )
        joint_enc = self.joint_transformer(**mm_inputs)['seq']
        joint_enc = unit_normalize(self.joint_proj(joint_enc[0, :token_length]))
        return joint_enc

    def embed_preencoded_audio(self, images_enc, audio_enc, tokens, subseg_idxs, audio_pointers):
        """
        This embeds a video of images. `img_prompt' is a prefix that is already precomputed.

        :param images_enc: [num_segments, num_patch_per_img // 4, hidden_size] - precomputed images that we plug in
        :param tokens: [L] tokens
        :param subseg_idxs: [L] which subsegment we're on, for each token.
        :return: a joint encoding of size [L, H], tokens conditioned on images.
        """
        token_length, = tokens.shape
        token_length_, = subseg_idxs.shape
        assert token_length_ == token_length

        images_enc = jnp.reshape(images_enc, [-1, self.hidden_size])

        # Encode audio to be [num_audio_spans, 6, H]
        mm_inputs = self.prepare_multimodal_inputs(
            tokens=tokens[None],
            token_segment_idx=subseg_idxs[None] // 3,
            vision_input=images_enc[None],
            audio_pointers=audio_pointers[None],
            audio_spans=audio_enc[None],
        )
        joint_enc = self.joint_transformer(**mm_inputs)['seq']
        joint_enc = unit_normalize(self.joint_proj(joint_enc[0, :token_length]))
        return joint_enc

@dataclass
class PretrainedMerlotReserve:
    encoder: Tokenizer
    params: Dict
    model: MerlotReserve
    _method_cache: Dict = None

    @classmethod
    def from_pretrained(cls, model_name, image_grid_size=(18, 24,), cache_dir=None):
        """
        From a pretrained model
        :param model_name: it has to be `base' or `large'
        :param image_grid_size: Resolution of the images (divided by 16). Valid options are `(18, 24)` for resolution adaptation,
                                `(12, 20)` for pretrained, and `(24, 24)` also for res. adaptation
        :param cache_dir: where to cache it if not None:
        :return:
        """
        from mreserve.checkpoint import load_checkpoint
        import os
        import yaml

        if model_name not in ('base', 'large'):
            raise ValueError("Must provide a model that is `base' or `large'")

        if image_grid_size not in [(18, 32), (12, 20), (24,24)]:
            raise ValueError("Invalid grid size {}".format(image_grid_size))

        param_fn = {
            ('base', (12, 20,)): 'base',
            ('large', (12, 20,)): 'large',
            ('base', (18, 32,)): 'base_resadapt',
            ('large', (18, 32,)): 'large_resadapt',
            ('base', (24, 24,)): 'base_resadapt',
            ('large', (24, 24,)): 'large_resadapt',
        }[model_name, image_grid_size]

        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), '.cache', 'merlotreserve')
        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(cache_dir, param_fn)
        if not os.path.exists(cache_path):
            try:
                from google.cloud import storage
                storage_client = storage.Client()
                bucket = storage_client.bucket('merlotreserve')
                blob = bucket.blob(f'ckpts/{param_fn}')
                print(f"DOWNLOADING! gs://merlotreserve/ckpts/{param_fn}", flush=True)
                blob.download_to_filename(cache_path)
            except:
                import requests
                print(f"DOWNLOADING {param_fn} using requests", flush=True)
                r = requests.get(f'https://storage.googleapis.com/merlotreserve/ckpts/{param_fn}', stream=True)
                with open(cache_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1000):
                        f.write(chunk)
            print("Done downloading")

        params = load_checkpoint(cache_path)['params']

        config_fn = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrain', 'configs', f'{model_name}.yaml')
        with open(config_fn, 'r') as f:
            config = yaml.load(f, yaml.FullLoader)

        config['model']['output_grid'] = image_grid_size

        is_on_tpu = any([x.platform == 'tpu' for x in jax.local_devices()])
        config['model']['use_bfloat16'] = is_on_tpu

        model = MerlotReserve.from_config(config)
        return cls(model=model, params=params, encoder=get_encoder())

    def __getattr__(self, name):
        """
        This is a hack that just calls the inherited model, wrapping the parameters in
        not sure if there's a better way to do things in flax :/
        :param name:
        :return:
        """
        if self._method_cache is None:
            self._method_cache = {}
        if name in self._method_cache:
            return self._method_cache[name]
        elif name in dir(self.model):
            fn = lambda params, *args, **kwargs: self.model.apply({'params': params}, *args, **kwargs, method=getattr(self.model, name))
            fn = jax.jit(fn)
            self._method_cache[name] = partial(fn, self.params)
            return self._method_cache[name]
        else:
            raise ValueError(f"Unknown attribute {name}")

    def get_label_space(self, options):
        """
        :param options: List of options of length B
        :return: a matrix of size [B, H] corresponding to those options
        """
        self.encoder.enable_padding(pad_token='<|PAD|>', length=15)
        answer_table_enc = jnp.array([x.ids[:15] for x in self.encoder.encode_batch(options)])
        self.encoder.no_padding()
        return self.embed_text_spans_only(answer_table_enc)
