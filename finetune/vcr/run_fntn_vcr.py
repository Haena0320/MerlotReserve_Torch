import sys
import os
import argparse
import yaml
from datetime import datetime
import torch
import json

from utils.utils import *
from finetune.vcr.models.vcr_model import MerlotReserveVCR
from finetune.vcr.utils.vcr_data_loader import get_vcr_dataloaders


ROOT_DIR = '/mnt/user8/workspace/torch-mrsv'
CONFIG_DIR = os.path.join(ROOT_DIR, 'finetune', 'vcr', 'config')
OUTPUT_DIR = '/mnt3/user8/torch-mrsv/log/finetune/vcr'
DATASET_DIR = '/mnt3/user8/torch-mrsv/data/finetune/vcr'

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='v01.yml')
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--model_name', type=str, default='mrsv')
parser.add_argument('--model_size', type=str, default='base')
parser.add_argument('--output_name', type=str, default='vTEST')

# -- Hyperparameters
parser.add_argument('--epoch_num', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--grad_accum', type=int, default=1)


args = parser.parse_args()

# -- Path Setting
config_path = os.path.join(CONFIG_DIR, f'{args.model_name}_{args.model_size}',args.config)
out_path = os.path.join(OUTPUT_DIR, f'{args.model_name}_{args.model_size}', args.output_name)
dataset_path = DATASET_DIR
logging_dir = out_path + '/logging.log'
tb_dir = out_path + '/tb'
eval_dir = out_path + '/eval'
ckpt_dir = out_path + '/ckpt'

if not os.path.exists(out_path):
    os.makedirs(out_path)
    os.mkdir(tb_dir)
    os.mkdir(eval_dir)
    os.mkdir(ckpt_dir)

# -- File Loading
config = load_yaml(config_path)
config['model']['dtype'] = torch.float32

# -- Model Loading
model = MerlotReserveVCR(config['model'])
# LOAD PARAMETERS 해야함.

# -- Save Settings
with open(os.path.join(out_path, 'config.json',), 'w') as f:
    config['model']['dtype'] = str(config['model']['dtype'])
    json.dump(config, f)
with open(os.path.join(out_path, 'argparse.json'), 'w') as f:
    json.dump(args.__dict__, f)

# -- Load Dataloader
data_loader_settings = {'dataset_path' : dataset_path, 'batch_size' : args.batch_size}

loaders = get_vcr_dataloaders(data_loader_settings, config)
train_loader, valid_loader, test_loader = loaders

from tqdm import tqdm
for i in tqdm(train_loader):
    z = model(i)
    break
    i = i


for k in i:
    print(i[k].dtype)
# -- 여기 아래로는 원래 지워야 함.
data_loader_settings = {'dataset_path' : dataset_path, 'batch_size' : args.batch_size}
loaders = get_vcr_dataloaders(data_loader_settings, config)

train_loader, val_loader, test_loader = loaders

dataset_path = data_loader_settings['dataset_path']
batch_size = data_loader_settings['batch_size']

example_paths = os.listdir(dataset_path)
train_paths = list()
valid_paths = list()
test_paths = list()

for path in example_paths:
    path = os.path.join(dataset_path, path)
    if 'train' in path:
        train_paths.append(path)
    elif 'val' in path:
        valid_paths.append(path)
    elif 'test' in path:
        test_paths.append(path)
    else:
        print(path)
        raise NotImplementedError
    
encoder = get_encoder()
train_dataset = get_vcr_dataset(train_paths, encoder, 'train')

tensors = train_dataset[0]

tensors.keys()

for key in tensors.keys():
    print(key, tensors[key])#.shape)

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True,
                         num_workers=1, collate_fn=collate_pad_vcr_data)

max_seq = 0
from tqdm import tqdm
for idx, batch in tqdm(enumerate(train_data_loader)):
    for key in batch.keys():
        print(key, batch[key].shape)
### Dataset ----------
# input : dict_keys(['qa_query', 'qa_choices', 'qa_label', 'qar_query', 'qar_choices', 'qar_label', 'id', 'image', 'image_fliplr'])
# output : dict_keys(['answers', 'image', 'labels'])

dataset_path = data_loader_settings['dataset_path']
batch_size = data_loader_settings['batch_size']

example_paths = os.listdir(dataset_path)
train_paths = list()
valid_paths = list()
test_paths = list()

for path in example_paths:
    path = os.path.join(dataset_path, path)
    if 'train' in path:
        train_paths.append(path)
    elif 'val' in path:
        valid_paths.append(path)
    elif 'test' in path:
        test_paths.append(path)
    else:
        print(path)
        raise NotImplementedError

# Data Loader Building


from models.utils.lowercase_encoder import get_encoder, MASK

encoder = get_encoder()
train_dataset = get_vcr_dataset(train_paths, encoder, 'train')
val_dataset = get_vcr_dataset(valid_paths, encoder, 'val')
test_dataset = get_vcr_dataset(test_paths, encoder, 'test')

from 


from finetune.vcr.utils.vcr_data_loader import VCRDataset
from models.utils.lowercase_encoder import get_encoder, MASK
encoder = get_encoder()

dataset = VCRDataset(train_paths)
item = dataset[0]
idxed_item = dict()

#k2f = {'image':None,'image_fliplr':None, 'id':None}

for prefix in ['qa', 'qar']:
    idxed_item[f'{prefix}_query'] = encoder.encode(item[f'{prefix}_query']).ids
    for i, choice_i in enumerate(encoder.encode_batch(item[f'{prefix}_choices'])):
        idxed_item[f'{prefix}_choice_{i}'] = choice_i.ids
    idxed_item[f'{prefix}_label'] = item[f'{prefix}_label']

from finetune.common_data_utils import load_and_resize_img
idxed_item['image'] = load_and_resize_img(item['image'], config['model'])
sep_tokens = {'qa':encoder.encode('answer: ').ids, 'qar':encoder.encode('rationale: ').ids}

answers = list()
for prefix in ['qa', 'qar']:
    query = idxed_item[f'{prefix}_query']
    for i in range(config['data']['num_answers']):
        option_i = query + sep_tokens[prefix] + idxed_item[f'{prefix}_choice_{i}']
        option_i = option_i[:config['data']['lang_seq_len']-1] + [MASK]
        option_i = option_i + [0] * (config['data']['lang_seq_len'] - len(option_i))  # Must be optimized
        answers.append(option_i)



for epoch in range(1, args.epoch_num+1):
    print(f'--- Epoch {epoch} ---')
    for items in train_loader:
        break
    break



