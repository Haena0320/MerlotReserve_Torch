from transformers.models.vit.modeling_vit import *
import torch


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
    batch_dims, seq_length, num_dims = coords.shape
    assert rotary_hsize % (num_dims * 2) == 0
    dim_expansion = rotary_hsize // (num_dims * 2)
    assert dim_expansion > 0

    freqs = torch.logspace(0.0, math.log2(max_freq / 2.0), dim_expansion, base=2,
                         dtype=coords.dtype if dtype is None else dtype)
    for i in range(len(batch_dims) + 2):
        freqs = freqs[None]

    radians = coords[..., None] * freqs * np.pi
    radians = radians.reshape(batch_dims, seq_length, num_dims * dim_expansion)
    cos_t = torch.cos(radians)
    sin_t = torch.sin(radians)
    sinusoids = torch.stack([cos_t, sin_t], -3)

    # Here we're repeating on the final dimension
    # bc later we will go through the first rotary_hsize coordinates and do
    # sin'd part: [-x0, x1, -x2, x3, ....]
    # cos'd part: [x0,  x1,  x2, x3, ....]
    sinusoids = torch.repeat(sinusoids, 2, axis=-1)
    return sinusoids

class MRSV_ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.h, self.w = config.output_grid
        self.pp3 = int(config.vit_patch_size ** 2 * 3)
        self.projection = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):#, bool_masked_pos=None, interpolate_pos_encoding=False):
        """
        :param x: Image, [B, Grid_H*Grid_W, P * P * 3]
        :return: x (projected embedidings), coords (rotary position encoding)
        """
        _, hw, pp3 = x.shape
        assert hw == self.hw
        assert pp3 == self.pp3
        embeddings = self.projection(x)

        rotary_coords = get_rotary_coordinates_2d(self.output_grid_h, self.output_grid_w)
        # Add rotary for CLS token

        rotary_bsz, rotary_seq_len, rotary_axes = rotary_coords.shape
        assert hw == rotary_seq_len
        # sinusoids = construct_rotary_sinusoids(rotary_coords, rotary_hsize=self.rotary_hsize)

        return embeddings, rotary_coords


class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, sinusoids, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, sinusoids, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, sinusoids, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, sinusoids, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            sinusoids,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs



class ViTEncoder(nn.Module):
    def __init__(self, config, rotary_hsize=32):
        super().__init__()
        self.config = config
        self.cls = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cls_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.rotary_hsize = rotary_hsize

    def forward(
        self,
        hidden_states,
        rotary_coords,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        bsz, seq_len, hidden = hidden_states.shape
        cls_token = self.cls_token.expand(bsz, -1, -1)
        hidden_states = torch.cat((cls_token, hidden_states), dim=1)
        sinusoids = construct_rotary_sinusoids(rotary_coords, rotary_hsize=self.rotary_hsize)

        for i, layer_module in enumerate(self.layer):

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, sinusoids, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )