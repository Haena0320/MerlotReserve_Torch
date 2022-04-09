import torch
from torch import nn
from models.models import MerlotReserve
from models.utils.lowercase_encoder import MASK


class MerlotReserveVCR(MerlotReserve):
    def __init__(self, config):
        super(MerlotReserveVCR, self).__init__(config)

        self.proj = nn.Linear(self.hidden_size, 1, bias=False)
        self.proj.weight = nn.init.normal_(self.proj.weight, std=0.02)
        self._remove_unused_module()

    def _remove_unused_module(self):
        del self.audio_encoder
        del self.span_encoder

    def forward(self, batch):

        batch_size, two_, num_ans_per, token_length = batch['answers'].shape
        answers2d = batch['answers'].reshape(batch_size * 2 * num_ans_per, token_length)

        imgs_out = self.vision_encoder(batch['image'])
        imgs_enc_seq_attnpool = imgs_out['seq_attnpool']
        imgs_enc = imgs_enc_seq_attnpool.repeat(
            [2 * num_ans_per] + ([1] * (len(imgs_enc_seq_attnpool.shape) - 1)))
        #imgs_enc = self.vision_encoder(batch['image'])['seq_attnpool'].repeat(2 * num_ans_per, axis=0)

        mm_inputs = self.prepare_multimodal_inputs(
            tokens=answers2d,
            token_segment_idx=torch.zeros([batch_size * 2 * num_ans_per, token_length], dtype=torch.int32),
            vision_input=imgs_enc,
        )
        joint_encoding = self.joint_transformer(**mm_inputs)['seq']
        joint_encoding = joint_encoding[:, :token_length].reshape(batch_size * 2 * num_ans_per, token_length, self.hidden_size)

        # Pool from the right tokens
        pool_idx = torch.argmax((answers2d == MASK).to(torch.float32), 1)
        pooled_h = joint_encoding[torch.arange(batch_size * 2 * num_ans_per), pool_idx]

        logits = self.proj(pooled_h).reshape([batch_size, 2, num_ans_per])
        return logits