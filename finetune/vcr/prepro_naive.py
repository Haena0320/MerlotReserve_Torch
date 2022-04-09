import os
import sys

sys.path.append(os.getcwd())
from finetune.common_data_utils import *

import argparse

from PIL import Image, ImageDraw, ImageFont
import json
import random
import numpy as np
from tempfile import TemporaryDirectory
from copy import deepcopy
import regex as re
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import colorsys
from sacremoses import MosesDetokenizer
import hashlib
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    '--mode',
    default=None,
    type=str,
    help='Ignore it'
)
parser.add_argument(
    '--random_seed',
    default=1337,
    type=int,
    help='which seed to use'
)
parser.add_argument(
    '--raw_data_dir',
    default='/mnt3/user8/vcr/data/raw_data',
    type=str,
)
parser.add_argument(
    '--out_data_dir',
    type=str,
    default='/mnt3/user8/torch-mrsv/data/finetune/vcr',

)
parser.add_argument(
    '--split',
    default='test',
    type=str
)
parser.add_argument(
    '--fold',
    default=0,
    type=int,
    help='which fold we are on'
)
parser.add_argument(
    '--num_folds',
    default=256,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)

args = parser.parse_args()
random.seed(args.random_seed)
detokenizer = MosesDetokenizer(lang='en')

def draw_boxes_on_image(image, metadata, tokenl_to_names, flip_lr=False):
    """
    Draw boxes on the image
    :param image:
    :param metadata:
    :param tokenl_to_names:
    :return:
    """
    #####################################################
    # last draw boxes on images
    image_copy = deepcopy(image)
    scale_factor = image.size[0] / metadata['width']

    boxes_to_draw = sorted(set([z for x in tokenl_to_names.keys() for z in x]))
    font_i = ImageFont.truetype(font='/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', size=17)

    for i in boxes_to_draw:
        name_i = tokenl_to_names[tuple([i])]
        box_i = np.array(metadata['boxes'][i][:4]) * scale_factor
        color_hash = int(hashlib.sha256(name_i.encode('utf-8')).hexdigest(), 16)

        # Hue between [0,1],
        hue = (color_hash % 1024) / 1024
        sat = (color_hash % 1023) / 1023

        # luminosity around [0.5, 1.0] for border
        l_start = 0.4
        l_offset = ((color_hash % 1025) / 1025)
        lum = l_offset * (1.0 - l_start) + l_start
        txt_lum = l_offset * 0.1

        color_i = tuple((np.array(colorsys.hls_to_rgb(hue, lum, sat)) * 255.0).astype(np.int32).tolist())
        txt_colori = tuple((np.array(colorsys.hls_to_rgb(hue, txt_lum, sat)) * 255.0).astype(np.int32).tolist())

        x1, y1, x2, y2 = box_i.tolist()
        if flip_lr:
            x1_tmp = image_copy.width - x2
            x2 = image_copy.width - x1
            x1 = x1_tmp

        shape = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

        draw = ImageDraw.Draw(image_copy, mode='RGBA')
        # draw.line(shape, fill=color_i, width=3)
        draw.rectangle([(x1, y1), (x2, y2)], fill=color_i + (32,), outline=color_i + (255,), width=2)
        txt_w, txt_h = font_i.getsize(name_i)

    return image_copy


def iterate_through_examples():
    if args.split not in ('train', 'val', 'test'):
        raise ValueError("unk split")
    with open(os.path.join(args.raw_data_dir, args.split + '.jsonl'), 'r') as f:
        for idx, l in enumerate(f):
            if idx % args.num_folds != args.fold:
                continue
            item = json.loads(l)

            with open(os.path.join(args.raw_data_dir, 'vcr1images', item['metadata_fn']), 'r') as f:
                metadata = json.load(f)

            image = Image.open(os.path.join(args.raw_data_dir, 'vcr1images', item['img_fn']))
            image = resize_image(image, shorter_size_trg=450, longer_size_max=800)

            ######################################################################
            # Tie tokens with names
            # the metadata file has the names only, ie
            # ['person', 'person', 'person', 'car']

            # questions refer to this through an index, ie
            # 2 for person3
            tokenl_to_names = {}
            type_to_ids_globally = defaultdict(list)
            object_count_idx = []

            for obj_id, obj_type in enumerate(metadata['names']):
                object_count_idx.append(len(type_to_ids_globally[obj_type]))
                type_to_ids_globally[obj_type].append(obj_id)

            for obj_type, obj_ids in type_to_ids_globally.items():
                if len(obj_ids) == 1:
                    # Replace all mentions with that single object ('car') with "the X"
                    tokenl_to_names[tuple(obj_ids)] = f'the {obj_type}'

            def get_name_from_idx(k):
                """
                If k has a length of 1: we're done
                otherwise, recurse and join
                :param k: A tuple of indices
                :return:
                """
                if k in tokenl_to_names:
                    return tokenl_to_names[k]

                if len(k) == 1:
                    obj_type = metadata['names'][k[0]]
                    obj_idx = object_count_idx[k[0]]
                    name = '{} {}'.format(obj_type.capitalize(), obj_idx+1)
                    tokenl_to_names[k] = name
                    return name

                names = [get_name_from_idx(tuple([k_sub])) for k_sub in k]

                if len(names) <= 2:
                    names = ' and '.join(names)
                else:
                    # who gives a fuck about an oxford comma
                    names = ' '.join(names[:-2]) + ' ' + ' and '.join(names[-2:])

                tokenl_to_names[k] = names
                return names

            def fix_token(tok):
                """
                Fix token that's either a list (of object detections) or a word
                :param tok:
                :return:
                """
                if not isinstance(tok, list):
                    # just in case someone said `Answer:'. unlikely...
                    if tok != 'Answer:':
                        return tok.replace(':', ' ')
                    return tok
                return get_name_from_idx(tuple(tok)[:2])

            def fix_tokenl(token_list):
                out = detokenizer.detokenize([fix_token(tok) for tok in token_list])
                out = re.sub(" n't", "n't", out)
                out = re.sub("n' t", "n't", out)

                # remove space before some punctuation
                out = re.sub(r'\s([\',\.\?])', r'\1', out)
                # fix shit like this: `he' s writing.`"
                out = re.sub(r'\b\'\ss', "'s", out)

                # kill some punctuation
                out = re.sub(r'\-\;', ' ', out)

                # remove extra spaces
                out = re.sub(r'\s+', ' ', out.strip())
                return out

            qa_query = fix_tokenl(item['question'])
            qa_choices = [fix_tokenl(choice) for choice in item['answer_choices']]
            qar_choices = [fix_tokenl(choice) for choice in item['rationale_choices']]

            img_boxes = draw_boxes_on_image(image, metadata, tokenl_to_names)
            if 'answer_label' in item:
                qar_query = '{} Answer: {}'.format(qa_query, qa_choices[item['answer_label']])

                # flip boxes
                everything = ' '.join(qa_choices + qar_choices + [qa_query])
                if ('right' in everything) or ('left' in everything):
                    img_lr = img_boxes
                else:
                    img_lr = draw_boxes_on_image(image, metadata, tokenl_to_names, flip_lr=True)

                yield {'qa_query': qa_query, 'qa_choices': qa_choices, 'qa_label': item['answer_label'],
                       'qar_query': qar_query, 'qar_choices': qar_choices, 'qar_label': item['rationale_label'],
                       'id': str(item['annot_id']), 'image': img_boxes, 'image_fliplr': img_lr,
                       }
            else:
                # Test set
                assert len(item['answer_choices']) == 4
                for i, qa_choice_i in enumerate(qa_choices):
                    qar_query = '{} Answer: {}'.format(qa_query, qa_choice_i)
                    yield {'qa_query': qa_query, 'qa_choices': qa_choices, 'qa_label': 0,
                           'qar_query': qar_query, 'qar_choices': qar_choices, 'qar_label': 0,
                           'id': '{}-qar-conditioned_on_a{}'.format(item['annot_id'], i), 'image': img_boxes
                           }

num_written = 0
max_len = 0

from copy import deepcopy
for j, ex in tqdm(enumerate(iterate_through_examples())):
    tmp_image = deepcopy(ex['image'])
    ex['image'] = pil_image_to_jpgstring(ex['image'])
    ex['image_fliplr'] = pil_image_to_jpgstring(ex.get('image_fliplr', tmp_image))
    #ex_num = ex['id'].split('-')[-1]
    #ex_num = (8 - len(ex_num)) * '0' + ex_num
    torch.save(ex, os.path.join(args.out_data_dir, f"{args.split}-{ex['id']}.torch-pkl"))


print(f'Finished writing {num_written} questions; max len = {max_len}', flush=True)

