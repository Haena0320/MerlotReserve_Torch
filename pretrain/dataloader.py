import sys
sys.path.append("/mnt2/user15/merlot_r/merlot_pytorch")
import time
from mreserve.lowercase_encoder import get_encoder, START, END, PADDING, MASK, AUDIOSPAN, LTOVPOOL, MASKAUDIO
import math
import tensorflow as tf
from PIL import Image
from io import BytesIO
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import logging
import regex as re
import numpy as np

import tensorflow_datasets as tfds
import functools
from copy import deepcopy
import random
from collections import defaultdict
import warnings

warnings.filterwarnings(action='ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tf.config.experimental.set_visible_devices([], 'GPU')

logger = tf.get_logger()
encoder = get_encoder()

segment_k = ['image/encoded', 'image/format', 'image/key/sha256','image/height','image/width','spectrogram/encoded',
            'spectrogram/format', 'spectrogram/key/sha256','spectrogram/height','spectrogram/width','spectrogram/magic_number',
          'youtude_id','video_src_index','title','tags','description','meta','playback_speed','start_time','end_time','tok_ids',
            'tok_start_times','tok_end_times','random_text']

#encoded_jpg = segment_list[0]["image/encoded"]
def load_and_resize_img(encoded_jpg, config):
    P = config["vit_patch_size"]
    h1, w1 = config["output_grid"]
    image_to_tensor = transforms.ToTensor()
    img = image_to_tensor(Image.open(BytesIO(encoded_jpg)))
    img, this_image_info = resize_and_pad(img, (h1*P, w1*P),
                                         do_random_scale=config.get('do_random_scale', True),
                                         random_scale_max=config.get('random_scale_max', 1.1),
                                         random_scale_min=config.get('random_scale_min', 1.05),
                                         shrink_both_sides=config.get('shrink_both_sides', True),
                                         do_flip_if_vertical=config.get('do_flip_if_vertical', True),
                                         resize_method="random")
    img = torch.transpose(torch.transpose(img, 0,1), 1, 2)
    img = torch.nn.functional.pixel_shuffle(img[None], int(np.sqrt(P)))
    img = img.reshape(h1*w1, P*P*3)
    return img

def load_audio(encoded_audio, magic_number, playback_speed, config):
    img = torch.tensor(np.array(Image.open(BytesIO(encoded_audio))), dtype=torch.float32)
    img = torch.transpose(img, 0, 1)

    content_len = config["num_audio_subsegments"] * config["audio_seq_length"]
    assert content_len < config["spec_size"]
    paddings = torch.rand([config["num_audio_subsegments"]+1])
    num_pad = config["spec_size"] - content_len
    paddings_int = (num_pad * torch.cumsum(paddings/torch.sum(paddings), dim=0)).int()
    start_idx = paddings_int[:config["num_audio_subsegments"]] + torch.arange(0, config["num_audio_subsegments"]) * config["audio_seq_length"]

    audio_seqs = []
    for i in range(config["num_audio_subsegments"]):
        audio_seqs.append(img[start_idx[i]:(start_idx[i] + config['audio_seq_length'])])

    audio_seqs = torch.stack(audio_seqs)
    audio_seqs = audio_seqs / torch.tensor(magic_number)
    audio_seqs = audio_seqs.view(config["num_audio_subsegments"], config["audio_seq_length"], config["num_mels"])
    playback_speed_f32 = float(playback_speed)
    audio_seqs = torch.cat([audio_seqs, torch.full((config["num_audio_subsegments"], config["audio_seq_length"], 1), playback_speed_f32)], -1)

    fft_window = config['fft_window_size'] / config["sample_rate"]
    fft_to_time_scale = config["fft_hop_length"] / config["sample_rate"]
    audio_start_t = start_idx.float() * fft_to_time_scale - fft_window / 2.0
    audio_end_t = audio_start_t + config['audio_seq_length']*fft_to_time_scale + fft_window 

    return audio_seqs, audio_start_t, audio_end_t

def select_tokens(tokens, padded_seq_len, num_segments):
    L = tokens.shape[0]
    amt_to_truncate = torch.tensor(L - padded_seq_len)
    is_mask = torch.cumsum(tokens[:, 0]==torch.tensor(MASK) | (tokens[:, 0]==torch.tensor([MASKAUDIO])), -1)
    is_audiospan = torch.cumsum(tokens[:, 0] ==AUDIOSPAN, -1)

    lhs_amt = torch.sum((is_mask==0)&(is_audiospan==0)) # 왼쪽에서 처음 마스킹 되는 곳
    rhs_amt = torch.sum(is_mask==is_mask[-1]) -1  # 오른쪽에서 마지막 마스킹 되는 곳

    # truncate from both sides
    trunc_start = torch.minimum(amt_to_truncate // 2, lhs_amt)
    trunc_end = torch.minimum(amt_to_truncate-trunc_start, rhs_amt)
    trunc_start = torch.minimum(amt_to_truncate-trunc_end, lhs_amt)

    tokens0= tokens[trunc_start:(L-trunc_end)]

    keep_logits = 1e7 * (torch.eq(tokens0[:, 0], torch.tensor(MASK)) & (tokens0[:, 0]!= torch.tensor(AUDIOSPAN)))

    segment_to_score = torch.randn([num_segments], dtype=torch.float32)*1e5
    keep_logits += torch.gather(segment_to_score, -1, tokens0[:, 1])
    
    _, idx2 = torch.sort(random_categorical_without_replacement(keep_logits, padded_seq_len))
    tokens1 = tokens0[idx2]
    if tokens0.shape[0]>padded_seq_len:
        print(tokens1)
        return tokens1
    else:
        return tokens0
    

def _one_hot(idx, N):
    m = idx.shape[0]
    return torch.eq(torch.arange(0, N)[:, None], idx[None]).any(1)


def shift_ragged_tokens_at_positions(tokens_ragged, positions, right_to_left=True):
    N = tokens_ragged.nrows()
    row_lengths = tokens_ragged.row_lengths()

    postiions = positions.int()
    pos_onehot = _one_hot(positions, N)
    pos_onehot = torch.logical_and(pos_onehot, torch.greater(row_lengths, 0))
    amt_to_take = pos_onehot.int()
    
    if right_to_left:
        amt_to_take = amt_to_take[1:]
        sub1 = torch.concat([torch.tensor([0]), -amt_to_take], 0)
        add1 = torch.concat([amt_to_take, torch.tensor([0])], 0)
        
    else:
        amt_to_take = amt_to_take[:-1]
        sub1 = torch.concat([-amt_to_take, torch.tensor([0])], 0)
        add1 = torch.concat([torch.tensor([0]), amt_to_take], 0) 
    row_lengths = row_lengths + sub1 + add1
    
    return RaggedTensor.from_row_lengths(tokens_ragged, row_lengths)

def random_do_both_directions(f):
    # Decorator to do right than left, then left than right, or the other way around
    def _f(x, **kwargs):
        x_rtl0 = f(x, **kwargs, right_to_left=True)
        x_rtl1 = f(x_rtl0, **kwargs, right_to_left=False)

        x_ltr0 = f(x, **kwargs, right_to_left=False)
        x_ltr1 = f(x_ltr0, **kwargs, right_to_left=True)
        if sample_bernoulli(0.5):
            return x_rtl1
        else:
            return x_ltr1
    return _f
        
@random_do_both_directions
def reassign_empty_tokens(tokens_ragged, *, mask_idx, right_to_left:bool=True):
    N = tokens_ragged.nrows() # 48
    mask_idx_onehot = _one_hot(mask_idx, N) #[t,t,t,f,f,f,f,f,t,f]
    row_lengths = tokens_ragged.row_lengths()
    needs_tokens = torch.logical_and(mask_idx_onehot, torch.eq(row_lengths, torch.tensor([0])))
    can_donate = torch.logical_and(torch.logical_not(mask_idx_onehot), torch.greater_equal(row_lengths, 2))
    
    if right_to_left:
        positions = torch.where(torch.logical_and(can_donate[1:], needs_tokens[:-1]))[0] + 1
        return shift_ragged_tokens_at_positions(tokens_ragged, positions)
    else:
        positions = torch.where(torch.logical_and(can_donate[:-1], needs_tokens[1:]))[0]
        return shift_ragged_tokens_at_positions(tokens_ragged, positions)
    

@random_do_both_directions
def increase_textmask(tokens_ragged, *, mask_idx, tok_centroids_vals, audio_start_end, right_to_left, delta_thresh=0.1):
    nrows_real = tokens_ragged.nrows()
    value_rowids = tokens_ragged.value_rowids()
    tok_centroids_expanded = ragged_from_rowids(tok_centroids_vals.numpy(), list(value_rowids+1), nrows=nrows_real+2)
    nmask = mask_idx.shape[0] #12
    row_lenths = [len(t) for t in tok_centroids_expanded]
    if right_to_left:
        t_out_right = [rt[i] for i in (mask_idx+2).numpy()]
        t_out_right =[np.min(i) if len(i) > 0 else 10000.0 for i in t_out_right]

        audio_boundary_r = torch.gather(torch.tensor(audio_start_end)[:, 1], -1,mask_idx)
        delta_r = (torch.tensor(t_out_right) - torch.tensor(audio_boundary_r))


        take_from_right = torch.less(delta_r, delta_thresh)
        right_is_masked = torch.any(torch.eq(mask_idx[:, None]+1, mask_idx[None]), -1)
        take_from_right = torch.logical_and(take_from_right, torch.logical_not(right_is_masked))
        take_from_right = torch.logical_and(take_from_right, torch.less(mask_idx+1, nrows_real))

        take_from_right_idx = torch.gather(mask_idx +1, -1,torch.where(take_from_right)[0])
        return shift_ragged_tokens_at_positions(tokens_ragged, take_from_right_idx, right_to_left=True)
    
    else:
        t_out_left = [rt[i] for i in mask_idx.numpy()]
        t_out_left =[np.max(i) if len(i) > 0 else -10000.0 for i in t_out_left]

        audio_boundary_l = torch.gather(torch.tensor(audio_start_end)[:, 0], 0,mask_idx)
        delta_l = (torch.tensor(audio_boundary_l)-torch.tensor(t_out_left))

        take_from_left = torch.less(delta_l, delta_thresh)
        left_is_masked = torch.any(torch.eq(mask_idx[:, None]-1, mask_idx[None]), -1)

        take_from_left = torch.logical_and(take_from_left, torch.logical_not(left_is_masked))
        take_from_left = torch.logical_and(take_from_left, torch.greater(mask_idx, 0))

        take_from_left_idx = torch.gather(mask_idx -1, -1,torch.where(take_from_left)[0])
        return shift_ragged_tokens_at_positions(tokens_ragged, take_from_left_idx, right_to_left=False)
    
    
        
def mask_tokens(tokens_ragged, mask_idx, do_audio_span=None, audio_token_length=6, text_span_start_counter=0,
                num_groups=1, padded_seq_len=None, do_audio_mask=False):
    
    N = tokens_ragged.nrows()
    mask_idx, _ = torch.sort(mask_idx, 0)
    mask_idx = mask_idx.squeeze()
    text_spans = [tokens_ragged.__getitem__(idx) for idx in mask_idx]
    mask_idx_onehot = _one_hot(mask_idx, N)
    
    if do_audio_span is not None: # audio -> text masking 
        do_audio_span = torch.logical_and(do_audio_span, torch.logical_not(mask_idx_onehot))
        audio_span_full = torch.full([N, audio_token_length], AUDIOSPAN)
        tokens_ragged = RaggedTensor([torch.tensor(audio_span_full[i]) if mask else tokens_ragged.__getitem__(i) for i, mask in enumerate(do_audio_span)])

    mask_tok = torch.full([N, 1], MASK)
    if do_audio_mask: 
        mask_tok = torch.concat([mask_tok, torch.full([N, 1], MASKAUDIO)], 1)
    tokens_ragged = RaggedTensor([torch.tensor(mask_tok[i]) if mask else tokens_ragged.__getitem__(i) for i, mask in enumerate(mask_idx_onehot)])
    text_ptr = torch.cumsum(mask_idx_onehot, -1, dtype=torch.int32)-1  + text_span_start_counter
    text_prt = torch.where(mask_idx_onehot, text_ptr, torch.full([N], -1).int()) # 48개 시퀀스 중 마스킹 된 시퀀스

    grp_size = N // num_groups

    output_grouped = []
    for i in range(num_groups):
        tokens_ragged_i = RaggedTensor(tokens_ragged[i*grp_size:(i+1)*grp_size])
        idxs_i = tokens_ragged_i.row_lengths()
        audio_ptr_i = torch.tensor(idxs_i)+ i*grp_size # audio span sequence (2개중 1개)
        text_ptr_i = text_prt[i*grp_size:(i+1)*grp_size] # text span sequence (2개중 1개, 길이 : 24)
        text_ptr_i = torch.gather(text_ptr_i, -1, torch.tensor(idxs_i))# masking text span , 길이 : 93

        output_i = torch.stack([tokens_ragged_i.values().int(), audio_ptr_i, text_ptr_i], -1)
        if padded_seq_len is not None:
            is_over_budget = output_i.shape[0] > padded_seq_len
            if is_over_budget:
                output_i = select_tokens(output_i, padded_seq_len, num_segments=N)
            else:
                output_i = pad_tokens_to_fixed_size(output_i, padded_seq_len)
        output_grouped.append(output_i)
    return text_spans, output_grouped # label, input_sample 



def convert_rawtext_into_fake_segments(tokens, desired_len, span_budget, use_v1_stats=False):
    """
    :param tokens: Tokens that we will mask. I'm only going to mask alphanumeric characters
    :param desired_len: desired length of the tokens
    :param mask_rate: How much to mask
    :return A ragged list of tokens
    """
    # # I got this empirically to minimize KL divergence between lengths of this and audio-to-text and text-to-audio
    if use_v1_stats:
        weights = [0.0210583 , 0.03984984, 0.06506665, 0.09467365, 0.12138153,
           0.13305461, 0.12973022, 0.11296043, 0.09024, 0.06730134,
           0.04789645, 0.03232633, 0.02123288, 0.01397406, 0.00925371]
        print(f"rawtext stats v1 -- should be for yttemporal 180m , weight {weights}", flush=True)

    else:
        weights = [0.03233136, 0.05236081, 0.08763368, 0.11757072, 0.13737426,
           0.13717706, 0.12541218, 0.10262764, 0.0771088 , 0.05364242,
           0.0342899 , 0.0203823 , 0.01177542, 0.00664939, 0.00366406]
        print(f"rawtext stats v2 -- should be for ytmega, weight {weights}", flush=True)


    ev = sum(i * w_i for i, w_i in enumerate(weights)) + 1
    logger.info("mask weights ev={:.3f}, weights={}".format(ev, weights))
    # k masked tokens that cover an expected length of k * e
    # L - k non masked tokens
    # mask rate is then ek/(L-k+ek)
    # some algebra and then
    #####################

    # I'm going to be conservative here bc I don't want to have too many tokens
    L = desired_len + int((ev * 0.85 - 1) * span_budget)
    L = torch.minimum(torch.tensor(L), torch.tensor(tokens.shape[0]))
    sample = torch.distributions.categorical.Categorical(torch.log(torch.tensor([weights])))
    segm_lens = torch.tensor([sample.sample() for  i in range(L)]) + 1

    # Truncate to the suggested length
    segm_lens_keep = torch.less_equal(torch.cumsum(segm_lens, dim=-1, dtype=torch.int32), torch.tensor(L))
    segm_lens = segm_lens[torch.where(segm_lens_keep)[0]]

    # Randomly truncate tokens if it's really long
    l_sel = torch.sum(segm_lens)
    wiggle_room = tokens.shape[0] - l_sel
    random_offset = (torch.rand(size=[])*torch.maximum(wiggle_room, torch.tensor(1))).int()
    tokens_ragged = RaggedTensor.from_row_lengths(tokens[random_offset:(random_offset + l_sel)].unsqueeze(0), list(segm_lens.numpy()))

    extra_lhs = tokens[:random_offset]
    extra_rhs = tokens[(random_offset+l_sel):]
    return tokens_ragged, extra_lhs, extra_rhs

def filter_out_tokens_not_in_youtube(spans_i, token_is_valid_tf=None):
    if token_is_valid_tf is None:
        token_is_valid_tf = torch.tensor(TOKEN_IS_VALID).bool()
    new_span_idx = torch.cat([torch.stack([token_is_valid_tf[v] for v in values]) for values in spans_i])
    new_span_idx= torch.where(new_span_idx)[0]
    spans_i_values= torch.cat(spans_i)[new_span_idx]
    spans_i_valuerowids = flatten([[i]*len(span) for i, span in enumerate(spans_i)])
    spans_i = RaggedTensor.from_value_rowids(spans_i_values, spans_i_valuerowids, nrows=len(spans_i))
    return spans_i.__getvalues__()

def dataset_parser(record, config):
    # load data
    num_segments = config['num_segments']

    segment_list = [{k: (dataset.pop(f'c{i:02d}/{k}') if f'c{i:02d}/{k}' in dataset else 1) for k in segment_k2f} for i in range(num_segments)]
    features = {}
    
    load_single_img = functools.partial(load_and_resize_img, config=config)
    features['images'] = torch.stack([load_single_img(x['image/encoded']) for x in segment_list])

    if config.get("disable_imgs_dataloader", False):
        print("Disabling audio from the dataloader level!!!", flush=True)
        features["audio_clips"] *=0.0

    magic_numbers = torch.tensor([x["spectrogram/magic_number"] for x in segment_list])
    encodeds = [x["spectrogram/encoded"] for x in segment_list]
    playback_speeds = torch.tensor([x["playback_speed"] for x in segment_list])

    load_single_audio = functools.partial(load_audio, config=config)

    audio_clip = []
    audio_start = []
    audio_end = []
    for x in segment_list:
        audio_clip_, audio_s, audio_e = load_single_audio(x["spectrogram/encoded"], x["spectrogram/magic_number"],x["playback_speed"]) 
        audio_start.append(audio_s)
        audio_end.append(audio_e)
        audio_clip.append(audio_clip_)
    features["audio_clips"] = torch.stack(audio_clip)
    audio_start = torch.stack(audio_start)
    audio_end = torch.stack(audio_end)

    if config.get('disable_audio_dataloader', False):
        print("Disabling audio from the dataloader level!!!", flush=True)
        features['audio_clips'] *= 0.0

    ######################################################

    num_audio_spans = num_segments * config['num_audio_subsegments']
    num_audio_spans_trg = int(num_audio_spans * config['mask_rate'])
    num_text2audio_seqs = config['num_text2audio_seqs']
    num_audio2text_seqs = config['num_audio2text_seqs']

    segment_idx = []
    tok_centroids_all = []
    audio_start_end_all = []
    t_start = 0.0


    for i, segment_i in enumerate(segment_list):
        # Partition the tokens into the audio segments
        tok_centroids = (torch.tensor(segment_i['tok_start_times']) + torch.tensor(segment_i['tok_end_times']))/2
        audio_centroids = (audio_start[i] + audio_end[i]) / 2.0
        tok_to_audio = torch.abs(tok_centroids[:, None] - audio_centroids[None])
        assignment = torch.argmin(tok_to_audio, 1).int()
        assignment = cumulative_maximum_int(assignment)
        segment_idx.append(assignment + torch.tensor(i * config['num_audio_subsegments']))

        # Keep track of timesteps -- this is in case mulitple things are in the batch
        tok_centroids_all.append(tok_centroids + t_start)
        audio_start_end_all.append(torch.stack([audio_start[i], audio_end[i]], -1) + t_start)

        t_start += (torch.tensor(segment_i['end_time']) - torch.tensor(segment_i['start_time']))

    segment_idx = torch.concat(segment_idx, 0)
    values = flatten([x['tok_ids'] for x in segment_list])
    tokens_ragged = RaggedTensor.from_value_rowids(values, segment_idx, num_audio_spans)
    tok_centroids_vals = torch.concat(tok_centroids_all, 0)
    audio_start_end = torch.concat(audio_start_end_all, 0)

    audio_spans_trg_idx = uniform_random_select(n=num_audio_spans, num_samples=num_audio_spans_trg * \
                                                (num_text2audio_seqs + num_audio2text_seqs), sort_idx=False)


    text_to_audio_idx = audio_spans_trg_idx[:num_audio_spans_trg*num_text2audio_seqs]
    text_to_audio_idx = text_to_audio_idx.view(num_text2audio_seqs,num_audio_spans_trg)

    audio_to_text_idx = audio_spans_trg_idx[num_audio_spans_trg*num_text2audio_seqs:]
    audio_to_text_idx = audio_to_text_idx.view(num_audio2text_seqs,num_audio_spans_trg)

    spans_all = []
    tokens_all = []

    for i in range(num_text2audio_seqs):
        tokens_ragged_i = reassign_empty_tokens(tokens_ragged, mask_idx=text_to_audio_idx[i])

        tokens_ragged_i = increase_textmask(tokens_ragged_i, mask_idx=text_to_audio_idx[i],
                                           tok_centroids_vals=tok_centroids_vals,
                                           audio_start_end=audio_start_end,
                                           delta_thresh=0.18)

        spans, output_groups = mask_tokens(tokens_ragged_i, mask_idx=text_to_audio_idx[i],
                                          text_span_start_counter=i*num_audio_spans_trg,
                                          num_groups=config["num_segment_groups"],
                                          padded_seq_len=config["lang_seq_len"],
                                          do_audio_mask=True)

        spans_all.append(spans)
        tokens_all.extend(output_groups)

    features["text2audio"] = torch.stack(tokens_all, 0)
    #######################################################
    # Now do audio -> text. will this be easier? hope so!
    audio_tokens_all = []
    for i in range(num_audio2text_seqs): # 48
        audio_span_trg_idx = audio_to_text_idx[i]

        one_hot_mask = _one_hot(audio_span_trg_idx, N=num_audio_spans)
        one_hot_mask_exp = torch.concat([torch.tensor([False]), one_hot_mask, torch.tensor([False])], 0) # 50
        should_textify = torch.logical_or(one_hot_mask_exp[2:], one_hot_mask_exp[:-2]) # 48

        should_textify = torch.logical_and(should_textify, torch.logical_not(one_hot_mask))
        should_textify = torch.logical_and(should_textify, sample_bernoullis(config.get('convert_extra_span_to_text_prob', 0.8),
                                                                            N=num_audio_spans))

        spans, output_groups = mask_tokens(tokens_ragged, mask_idx=audio_span_trg_idx,
                                           do_audio_span=torch.logical_not(should_textify),
                                           audio_token_length=config['audio_token_length'],
                                           padded_seq_len=config['lang_seq_len'],
                                           text_span_start_counter=(i + num_text2audio_seqs) * num_audio_spans_trg,
                                           num_groups=config['num_segment_groups'])
        spans_all.append(spans)
        audio_tokens_all.extend(output_groups)

    features['audio2text'] = torch.stack(audio_tokens_all, 0)
    #####################################
    # For the audio -> image part

    max_text_seq_len = config.get('max_text_seq_len', config['seq_len']) #640
    use_audio_tokens = sample_bernoulli(config.get('use_audio_token_prob', 1.0))
    matching_toks = []

    for i, segment_i in enumerate(segment_list): # 전체 비디오 정보에서 텍스트, 오디오 토큰 시퀀스 정보 생성
        matching_toks.append(torch.stack([torch.tensor(LTOVPOOL), torch.tensor(i * config['num_audio_subsegments']), torch.tensor(-1)])[None])

        audio_subseg = []
        for j in range(config['num_audio_subsegments']): # 3
            new_subseg = torch.stack([torch.tensor(AUDIOSPAN),torch.tensor( j + i * config['num_audio_subsegments']), torch.tensor(-1)])[None]
            audio_subseg.append(torch.tile(new_subseg, [config['audio_token_length'], 1]))
        audio_subseg = torch.concat(audio_subseg, 0)
        text_subseg = torch.stack([
            torch.tensor(segment_i["tok_ids"]),
            torch.zeros_like(torch.tensor(segment_i["tok_ids"]))+i*config["num_audio_subsegments"],
            torch.zeros_like(torch.tensor(segment_i["tok_ids"]))-1], 1)
        if use_audio_tokens:
            matching_toks.append(audio_subseg)
        else:
            matching_toks.append(text_subseg)

    matching_toks=torch.concat(matching_toks, 0)

    aux_info = torch.concat([
        torch.tensor([START]), torch.tensor(encoder.encode("title:").ids), torch.tensor(segment_list[0]["title"]),
        torch.tensor([START]), torch.tensor(encoder.encode("description:").ids), torch.tensor(segment_list[0]["description"]),
        torch.tensor([START])+ torch.tensor(encoder.encode('tags:').ids), torch.tensor(segment_list[0]["tags"]), torch.tensor([END])
    ], 0)

    aux_info = torch.stack([aux_info, torch.zeros_like(aux_info)-1, torch.zeros_like(aux_info)-1], 1)

    extra_space_for_desc = torch.maximum(torch.tensor(max_text_seq_len - matching_toks.shape[0]), torch.tensor(0))
    aux_info = aux_info[:extra_space_for_desc]
    matching_toks = torch.concat([aux_info, matching_toks], 0) # 시퀀스에 대한 매칭 토큰, 오디오 시퀀스정보 

    features["audio_text_matching"] = pad_tokens_to_fixed_size(matching_toks,config["seq_len"])


    # is_valid = re.compile(r"^[ A-Za-z0-9\-$%&'+,./:?@\[\]_’]*$")
    is_valid = re.compile(r"^[ A-Za-z0-9']*$")
    TOKEN_IS_VALID = [(i > 10) and bool(is_valid.match(encoder.decode([i]))) for i in range(encoder.get_vocab_size())]
    bad_tokens = [149, 4858, 9504, 15162, 22312, 22433, 32156]
    for i in bad_tokens:
        TOKEN_IS_VALID[i] = False


    ####################### Random text-> constrastive learning 때 사용할 text 시퀀스들

    num_text_seqs_in_record = config['num_text_seqs_in_record']
    random_text = torch.stack([torch.tensor(x['random_text']) for i, x in enumerate(segment_list) if i < config['num_text_seqs_in_record']])


    assert config['num_text_seqs'] <= num_text_seqs_in_record
    random_inds = uniform_random_select(num_text_seqs_in_record, config['num_text_seqs'])
    random_text = random_text[random_inds]
    random_text_l = []
    counter = num_audio_spans_trg * (num_audio2text_seqs + num_text2audio_seqs)

    token_is_valid_tf = torch.tensor(TOKEN_IS_VALID).bool()

    ## rawtxt to fake segment 
    for i in range(config['num_text_seqs']):
        # span_budget = int(desired_len / (ev / mask_rate - ev + 1))
        _ev = 5.5
        if 'text_span_budget' in config:
            span_budget = config['text_span_budget']
        else:
            span_budget = int(max_text_seq_len / (_ev / config['mask_rate'] - _ev + 1.0)) 
        print(f"Using span budget of {span_budget}", flush=True)
        tokens_ragged_i, extra_lhs, extra_rhs = convert_rawtext_into_fake_segments(random_text[i],
                                                                                   desired_len=max_text_seq_len,
                                                                                   span_budget=span_budget,
                                                                                   use_v1_stats='ytt180m' in config['train_fns'])

        want_to_mask =[torch.stack([token_is_valid_tf[v] for v in values]) for values in tokens_ragged_i.__getvalues__()]
        mask_w = 0.2 + 0.8 * torch.tensor([True if span.sum() ==len(span) else False for span in want_to_mask]).float()
        do_mask_i = random_categorical_without_replacement(logits=torch.log(mask_w), num_samples=span_budget)
        do_mask_i = torch.sort(do_mask_i)[0]
        spans_i, tokens_i = mask_tokens(tokens_ragged_i, do_mask_i, text_span_start_counter=counter, num_groups=1)

        # Add in extra LHS and extra RHS if under max len (패딩, 추가로 Rawtext 넣어줌)
        tokens_i = tokens_i[0]
        amt_needed = torch.maximum(torch.tensor(max_text_seq_len - tokens_i.shape[0]), torch.tensor(0))
        extra_lhs_len = extra_lhs.shape[0]
        amt_lhs = torch.minimum(torch.tensor(extra_lhs_len), amt_needed // 2)

        extra_lhs = torch.stack([extra_lhs[(extra_lhs_len - amt_lhs):], torch.zeros([amt_lhs], dtype=torch.int32), torch.zeros([amt_lhs], dtype=torch.int32)-1], 1)

        extra_rhs_len = extra_rhs.shape[0]
        amt_rhs = torch.minimum(torch.tensor(extra_rhs_len), (amt_needed+1) // 2)
        extra_rhs = torch.stack([extra_rhs[:amt_rhs], tokens_i[-1, 1] + torch.ones([amt_rhs], dtype=torch.int32), torch.zeros([amt_rhs], dtype=torch.int32)-1], 1)
        tokens_i = torch.concat([extra_lhs, tokens_i, extra_rhs], 0)

        # OK now we pad to the length of the joint transformer
        tokens_i = pad_tokens_to_fixed_size(tokens_i, padded_seq_len=config['seq_len']) # 640, 3

        # Filter out tokens not seen in YouTube

        spans_i = filter_out_tokens_not_in_youtube(spans_i, token_is_valid_tf=token_is_valid_tf)

        counter += span_budget
        random_text_l.append(tokens_i)
        spans_all.append(spans_i)

    if config['num_text_seqs'] > 0:
        features['random_text'] = torch.stack(random_text_l, 0)

    # Video src idx per segment
    features['video_src_index'] = torch.stack([x['video_src_index'] for x in segment_list])
    features['meta'] = segment_list[0]['meta']
    features['youtube_id'] = segment_list[0]['youtube_id']

    if config.get('encode_meta', False):
        features['youtube_id'] = encode_string(features['youtube_id'], 11)
        features['meta'] = encode_string(features['meta'], 256)
    
    return features


class MerlotDataset(Dataset):
    def __init__(self, config, fns, num_devices=None, is_training=True):
        self.merged_config = deepcopy(config['data'])
        self.merged_config.update(config['model'])
        for fn in fns:
            self.dataset.extend(torch.load(fn))
        print(f"!! total {len(self.dataset)} data is loaded !!")
    def __getitem__(self, idx):
        return functools.partial(data_parser(self.dataset[idx]), config=self.merged_config)
    
    def __len__(self):
        return len(self.dataset)
    
class MerlotLoader(DataLoader):
    @classmethod
    def from_dataset(cls, fns, batch_size=4, num_workers=3, num_gpus=1, is_training=True, **kwargs):
        
        loader = cls(
        dataset=data, 
        batch_size=batch_size*num_gpus,
        shuffle=data.is_train,
        num_workers=num_workers,
        collate_fn = lambda x:handle_batch(x, to_gpu=False),
        drop_last=is_training,
        pin_memory=False,
        **kwargs)
        return loader

def make_dataset_fn(config, fns=None, batch_size=4,num_devices=1, is_training=True):
    data = MerlotDataset(config, fns, num_devices=num_devices)
    loader = MerlotLoader.from_dataset(fns, batch_size, num_workers, num_gpus, is_training)
    return loader
    

def input_fn_builder(config, make_dataset_fn):

    num_hosts = torch.get_num_threads()
    num_devices = torch.cuda.device_count()
    batch_size = config["device"]["batch_size"] // num_hosts
    
    random.seed(time.time())
    np.random.seed(time.time())
    torch.manual_seed(time.time())
    
    matching_fns = []
    for i in range(config["data"]["num_train_files"]):
        if i % num_hosts == current_host:
            matching_fns.append(config["data"]["train_fns"].format(i))
    assert (len(matching_fns) > 0)
    
    def _multi_iterator0():
        n_fns_per_cycle = min(config["device"].get('n_fns_per_cycle', 32), len(matching_fns))
        while (len(matching_fns) % n_fns_per_cycle) != 0:
            print(f"!!!Truncating n_fns_per_cycle {n_fns_per_cycle} -> {n_fns_per_cycle -1} so it fits", flush=True)
            n_fns_per_cycle -= 1
        n_epochs = 0
        while True:
            fns_shuff = [x for x in matching_fns]
            random.shuffle(fns_shuff)
            print(f"Now on epoch {n_epochs}")
            for s, e in batch_index_iterator(len(fns_shuff), batch_size=n_fns_per_cycle, skip_end=True):
                print(f"Resetting iterator, epoch={n_epochs}, batch of fns={s}:{e}/{len(fns_shuff)}", flush=True)
                try:
                    dataset = make_dataset_fn(config, fns=fns_shuff[s:e], batch_size=batch_size,
                                             num_devices=num_devices, is_training=True)
                except Exception as e:
                    print(str(e), flush=True)
                    time.sleep(5)
            n_epochs += 1
    return _multi_iterator0()
