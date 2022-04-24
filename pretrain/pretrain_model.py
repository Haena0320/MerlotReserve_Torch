from mreserve.models import *
from mreserve.utils.lowercase_encoder import AUDIOSPAN, LTOVPOOL, PADDING, MASK, MASKAUDIO
import torch 
import numpy as np

class MerlotReservePretrainer(MerlotReserve):
        
    def forward(self, batch):
        batch_size, num_segments_nvpatch0, pp3 = batch['images'].shape
        nvpatch0 = self.output_grid_w * self.output_grid_h
        num_segments = num_segments_nvpatch0 // nvpatch0
        num_segments_per_group = num_segments // self.num_segment_groups
        # make image embedding 
        imgs_enc = self.vision_encoder(batch['images'].reshape((batch_size * num_segments, nvpatch0, pp3)))
        
        nvpatch1 = nvpatch0 // (self.vit_pooling_ratio ** 2)
        imgs_seq = imgs_enc['seq_attnpool'].reshape(
            [batch_size, self.num_segment_groups, num_segments_per_group * nvpatch1, self.hidden_size])

        if self.config.get('no_vision', False):
            print("\nNO VISION\n\n\n!!!!\n\n\n", flush=True)
            imgs_seq *= 0.0
            
        vis_seq_length = imgs_seq.shape[-2]
        
        # make audio embedding
        # Audio clips are provided as [batch_size, num_segments * num_audio_subsegments * audio_seq_len, num_mels]
        audio_enc = self.audio_encoder(batch['audio_clips'].reshape(
            (batch_size * num_segments * self.num_audio_subsegments, self.audio_seq_length, -1)).float())
        
        # Audio seq is now [batch_size, num_audio_spans, seq_len, H]
        num_audio_spans = num_segments * self.num_audio_subsegments
        audio_seq = audio_enc['seq_attnpool'].reshape(
            [batch_size, num_audio_spans, self.audio_token_length, self.hidden_size])

        audio_cls = audio_enc['cls'].reshape([batch_size, num_audio_spans, self.hidden_size])

        # Flatten text sequence
        for k1 in ['text2audio', 'audio2text']:
            for k2 in ['', '/audio_ptr', '/text_ptr']:
                k = k1 + k2
                batch[k] = batch[k].reshape((-1, self.lang_seq_len))

        for k in ['random_text', 'random_text/text_ptr', 'audio_text_matching', 'audio_text_matching/audio_ptr']:
            batch[k] = batch[k].reshape((-1, self.seq_len))

        batch['text_spans'] = batch['text_spans'].reshape((-1, self.text_span_length))

        # 텍스트 인코딩 
        txt_embs = self.token_encoder(
            {k: batch[k] for k in ['text2audio', 'audio2text', 'audio_text_matching', 'text_spans', 'random_text']})

        batch['video_src_index'] = batch['video_src_index'].reshape(-1, num_segments_per_group) # [batch, 비디오시퀀스(2), 8]

        mm_inputs = {}
        
        ## task 준비 ##
        # joint encoder 의 audio2text task 인풋준비
        num_audio2text_seqs = self.num_audio2text_seqs #1
        mm_inputs['audio2text'] = self.prepare_multimodal_inputs(
            tokens=batch['audio2text'],
            token_segment_idx=(batch['audio2text/audio_ptr'] // self.num_audio_subsegments) % num_segments_per_group,
            token_embs=txt_embs['audio2text'],
            vision_input=torch.tile(imgs_seq, [1, num_audio2text_seqs, 1, 1]).reshape(-1, vis_seq_length, self.hidden_size),
            audio_spans=audio_seq.repeat_interleave(self.num_segment_groups * num_audio2text_seqs, dim=0),
            audio_pointers=batch['audio2text/audio_ptr'],
            padding_len=self.seq_len,
            video_src_idx=self._augment_video_src_idx(torch.tile(batch['video_src_index'].reshape(
                batch_size, self.num_segment_groups, num_segments_per_group), [1, num_audio2text_seqs, 1]).reshape(-1,
                                                                                                              num_segments_per_group)))



        #  audio text span matching  task 인풋 준비
        # image 인풋 없음,  
        mm_inputs['audio_text_matching'] = self.prepare_multimodal_inputs(
            tokens=batch['audio_text_matching'], # masked audio, masked text sequence [batch, 640]
            token_segment_idx=torch.cumsum((batch['audio_text_matching'] == LTOVPOOL).int(), -1),
            token_embs=txt_embs['audio_text_matching'],
            audio_spans=audio_seq,
            audio_pointers=batch['audio_text_matching/audio_ptr'].type(torch.int64),
            padding_len=self.seq_len)

        # text2audio task 인풋 준비 
        num_text2audio_seqs = self.num_text2audio_seqs #1
        mm_inputs['text2audio'] = self.prepare_multimodal_inputs(
            tokens=batch['text2audio'],
            token_segment_idx=(batch['text2audio/audio_ptr'] // self.num_audio_subsegments) % num_segments_per_group,
            token_embs=txt_embs['text2audio'],
            vision_input=torch.tile(imgs_seq, [1, num_text2audio_seqs, 1, 1]).reshape(-1, vis_seq_length,
                                                                                    self.hidden_size),
            audio_pointers=batch['text2audio/audio_ptr'],
            padding_len=self.seq_len,
            video_src_idx=self._augment_video_src_idx(torch.tile(batch['video_src_index'].reshape(
                batch_size, self.num_segment_groups, num_segments_per_group), [1, num_text2audio_seqs, 1]).reshape(-1,
                                                                                                              num_segments_per_group)),
        )
        # fake text 인풋 준비 
        mm_inputs['random_text'] = self.prepare_multimodal_inputs(tokens=batch['random_text'], padding_len=self.seq_len)


        keys = sorted(mm_inputs.keys()) # ['audio2text', 'audio_text_matching', 'random_text', 'text2audio']
        x = torch.concat([mm_inputs[k]['x'] for k in keys], 0)
        coords = torch.concat([mm_inputs[k]['rotary_coords'] for k in keys], 0)
        attnmask = torch.concat([mm_inputs[k]['attention_mask'] for k in keys], 0)
        real_bsizes = [mm_inputs[k]['x'].shape[0] for k in keys] # batch*2, batch, batch, batch*2 [8,4,4,8]

        if not self.config.get('do_rotary', True):
            print("NOT DOING ROTARY", flush=True)
            coords = None

        # 동시에 joint encoder에 입력, 이때 마스킹이 어떻게 되는지?
        joint_enc = self.joint_transformer(x, rotary_coords=coords, attention_mask=attnmask)['seq']
        joint_enc = self.joint_proj(joint_enc)

        mm_outputs = {k: z for k, z in zip(keys, torch.split(joint_enc, tuple(real_bsizes), dim=0))}

        mm_outputs['text2audio'] = mm_outputs['text2audio'][:, :self.lang_seq_len]
        mm_outputs['audio2text'] = mm_outputs['audio2text'][:, :self.lang_seq_len]

        #################################################고쳐야 함 #
        # Get everything needed
        # Vision to Audio
        is_pool = (batch['audio_text_matching'] == LTOVPOOL)
        v2a_cumulative_idx = torch.cumsum(is_pool.int(), -1) - 1
        a2v = one_hot_pool(is_pool,
                           idx=v2a_cumulative_idx.to(self.device),
                           v=mm_outputs['audio_text_matching'].to(self.device),
                           num_segments=num_segments)['x'].reshape((batch_size * num_segments, self.hidden_size))
        ################################################
        # a2v :[batch(4), 16, 768]
        # Text to audio
        ################################################
        t2a_sel = one_hot_pool(
            do_pool=batch['text2audio'] == MASKAUDIO,
            idx=batch['text2audio/audio_ptr'],
            v=mm_outputs['text2audio'].to(self.device),
            num_segments=num_segments * self.num_audio_subsegments,
            real_bsize=batch_size,
        ) # dict {x : pool해야하는 위치의 feature[batch(4), video segments(48), 768], idx_oh (batch, 320 (160*2),48)}
        
        # For text to audio, not all the audio is a "target" so don't include in one part of the loss
        # make audio candidate (for text2audio)
        num_audio_spans_trg = int(num_audio_spans * self.mask_rate) * num_text2audio_seqs #12
        is_selected = t2a_sel['idx_oh'].sum(1) # batch(4), video segments(48)

        idx_sort = torch.argsort(-is_selected, -1)

        best_idxs = idx_sort[:, :num_audio_spans_trg].reshape(batch_size * num_audio_spans_trg) # 총 미니배치 내에서 마스킹한 토큰 id
        batch_indexer = torch.arange(batch_size).repeat(num_audio_spans_trg)
        t2a_sel = t2a_sel['x'][batch_indexer, best_idxs] # 총 미니배치 내에서 마스킹한 토큰(token id ==4)의 feature 값
        a2t_sel = audio_cls[batch_indexer, best_idxs] # audio encoder 에서 출력한 cls features 

        extra_idxs = idx_sort[:, num_audio_spans_trg:].reshape(batch_size * (num_audio_spans - num_audio_spans_trg)) # 미니배치 내에서 마스킹하지 않은 토큰 id들
        batch_indexer = torch.arange(batch_size).repeat(num_audio_spans - num_audio_spans_trg)
        a2t_extra = audio_cls[batch_indexer, extra_idxs]# 총 미니배치 내에서 마스킹하지 않은 토큰(token id != 4) 의 feature 값
        
        
        # make Text candidate(for audio2text)
        ################################################
        num_text_spans = txt_embs['text_spans'].shape[0] // batch_size  #배치내 text spans 갯수 
        t2sp = {}
        for k in ['audio2text', 'text2audio', 'random_text']:
            t2sp[k] = one_hot_pool(
                batch[k] == MASK, #3
                idx=batch[f'{k}/text_ptr'],
                v=mm_outputs[k].to(self.device),
                num_segments=num_text_spans,
                real_bsize=batch_size
            )
            t2sp[k]['count'] = t2sp[k].pop('idx_oh').sum(1)
        t2sp_sel = t2sp['text2audio']['x'] + t2sp['audio2text']['x'] + t2sp['random_text']['x'] 
        t2sp_ct = t2sp['text2audio']['count'] + t2sp['audio2text']['count'] + t2sp['random_text']['count'] # (12, 12, ~)각 태스크에서 정답/오답인 text span
        t2sp_src = torch.stack([torch.zeros_like(t2sp['text2audio']['count']), t2sp['text2audio']['count'],
                              t2sp['audio2text']['count'], t2sp['random_text']['count']], -1).argmax(-1) - 1   # 192, 3개 text span이 'audio2text', 'text2audio', 'random_text'중어디에 해당하는지 마스킹

        
        # Exclude things that have length 0, or that got dropped from the input somehow text span 중  전체 패딩이 아닌경우 찾 
        is_valid = (batch['text_spans'] != PADDING).any(-1).reshape(batch_size, num_text_spans)
        is_valid &= (t2sp_ct > 0.0)
        is_valid = is_valid.type(torch.float32)
        
        
        # Select `num_text_spans_to_include` spans which is less than the number of spans total.
        # Use the `random choice without replacement` hack
        # Choose multimodal spans 4x as often
        prefer_multimodal = np.log(4)
        logits_for_pred = is_valid * 1e6 + prefer_multimodal * (
                    t2sp['text2audio']['count'] + t2sp['audio2text']['count'])
        z = -torch.log(-torch.log(torch.rand([batch_size, num_text_spans]))).cuda()
        is_valid = logits_for_pred + z

        NUM_TO_INCLUDE = self.num_text_spans_to_include
        assert NUM_TO_INCLUDE <= num_text_spans
        print(f"Including {NUM_TO_INCLUDE} / {num_text_spans} text spans per batch (bsize={batch_size}."
              f" but doing it across batches! so some might have more. that's OK though I think", flush=True)
        best_idxs = torch.topk(is_valid.reshape(-1), k=NUM_TO_INCLUDE * batch_size)[1]

        
        # Text candidate 
        t2sp_sel = t2sp_sel.reshape([batch_size * num_text_spans, self.hidden_size])[best_idxs]
        t2sp_src = t2sp_src.reshape([batch_size * num_text_spans])[best_idxs]
        sp2t_sel = self.span_encoder(x=txt_embs['text_spans'][best_idxs],
                                     x_isvalid=batch['text_spans'][best_idxs] != PADDING)
        #################################################################
        
        log_scales = torch.clip(self.scale_params.float(),max=np.log(100.0))
        outputs = {
            'imgs_to_audio': {'x': a2v, 'y': imgs_enc['cls'], 'log_scale': log_scales[0]}, # 모든 audio segment -> 모든 image 매칭 
            'text_to_audio': {'x': t2a_sel, 'y': a2t_sel, 'y_extra': a2t_extra, 'log_scale': log_scales[1]},# 12개 subsegment -> audio 예측
            'stuff_to_span': {'x': t2sp_sel, 'y': sp2t_sel, 'log_scale': log_scales[2], '_sources': t2sp_src}} # audio -> text 

        # before contrastive objective, make normalizeation
        for k in outputs:
            temp_to_use = torch.exp((outputs[k].pop('log_scale') / 2.0) +1e-5)
            for k2 in 'xy':
                outputs[k][k2] = unit_normalize(outputs[k][k2]) * temp_to_use
                k2_extra = f'{k2}_extra'
                if k2_extra in outputs[k]:
                    outputs[k][k2_extra] = unit_normalize(outputs[k][k2_extra]) * temp_to_use

        return outputs
        
    def _augment_video_src_idx(self, video_src_idx):
        """
        Randomly split `video_src_idx` into two portions. basically this means that now we won't have some segments attending
        to other segments. Could be good if we want to often handle short clips of videos
        :param video_src_idx: [B, L] e.g.
          DeviceArray([[1, 1, 1, 1, 1, 1, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)
        :return: [B, L]
        """
        B, L = video_src_idx.shape
        if L == 1:
            print("_augment_video_src_idx: L=1 so cant split", flush=True)
            return video_src_idx

        split_prob = self.config.get('_augment_video_src_idx_prob', 0.1)
        probs = [split_prob / (L - 1) for i in range(L - 1)] + [1 - split_prob]
        print("Augmenting video_src_idx {}x{} with probs {}".format(B, L, probs), flush=True)
        split_from_here = 1 + np.random.choice( a=L, size=B, p=np.array(probs))

        split_mask = split_from_here[:, None] <= np.arange(L)[None]

        # Offset by a big number, say 4L
        video_src_idx = torch.where(torch.tensor(split_mask).to(self.device), video_src_idx + 4 * L, video_src_idx)
        return video_src_idx
    
def loss_fn_given_preds(preds):
    loss_info = {}

    for c_type, c_dict in preds.items():
        numer_logits = (c_dict['x'] * c_dict['y']).sum(-1)
        loss_info[c_type] = 0.0

        if '_sources' in c_dict:
            for k in ['text2audio', 'audio2text', 'random_text']:
                loss_info[f'_{c_type}_from_{k}'] = 0.0
        # For both directions (average the result)
        for k1, k2 in ['xy', 'yx']:
            x = c_dict[k1]
            y = c_dict[k2]

            # Add in extra things that are only valid as targets
            if f'{k2}_extra' in c_dict:
                y = torch.concat([y, c_dict[f'{k2}_extra']])
            #y_allgather = torch.gather(y).reshape(-1, x.shape[-1])

            y_allgather = y
            # print(f"{c_type} {k1}->{k2} dot product sim:  shaped [{x.shape}] -> [{y_allgather.shape}", flush=True)
            denom_logits = torch.einsum('lh,vh->lv', x, y)
            denom_lse = torch.logsumexp(denom_logits.float(), dim=-1)
            denom_lse = (denom_lse - numer_logits).mean() / 2.0

            loss_info[c_type] += torch.clamp(denom_lse, 1e-6)
            if '_sources' in c_dict:
                for i, type_i in enumerate(['text2audio', 'audio2text', 'random_text']):
                    does_match = (c_dict['_sources'] == i).float()
                    loss_match = ((denom_lse - numer_logits) * does_match).sum() / (does_match.sum() + 1e-5)
                    loss_info[f'_{c_type}_from_{type_i}'] += loss_match / 2.0

    loss = sum([v for k, v in loss_info.items() if not k.startswith('_')])
    return loss, loss_info