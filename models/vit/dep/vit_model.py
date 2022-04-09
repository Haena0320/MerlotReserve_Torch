from transformers import ViTFeatureExtractor, ViTModel
import torch
import numpy as np
import json
from models.vit import vit_utils
from torch import nn

class MrsvVisionTransformer(nn.Module):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__()
        self.config = config
        self.vit_embedding = MRSV_ViTEmbeddings(config)

    def forward(self, pixel_values):

        embedding_output = self.vit_embedding(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

