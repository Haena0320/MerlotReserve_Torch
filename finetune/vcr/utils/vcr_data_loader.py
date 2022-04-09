import os
import torch
from torch.utils.data import Dataset, DataLoader

from models.utils.lowercase_encoder import get_encoder, MASK
from finetune.common_data_utils import load_and_resize_img

def collate_pad_vcr_data(batch):

    max_seq = 0

    for example in batch:
        max_seq = max(max_seq, example['max_ans_seq'])

    final_batch = {'image':list(), 'answers':list(), 'labels':list()}
    for example in batch:
        num_qas, num_candidate, seq = example['answers'].shape
        more_pad = max_seq - seq
        example['answers'] = torch.cat( [example['answers'], torch.zeros(num_qas, num_candidate, more_pad)], dim=-1)

        final_batch['image'].append(example['image'])
        final_batch['answers'].append(example['answers'])
        final_batch['labels'].append(example['labels'])

    final_batch = {
        'image':torch.stack(final_batch['image'], dim=0),
        'answers':torch.stack(final_batch['answers'], dim=0),
        'labels':torch.stack(final_batch['labels'], dim=0)
    }

    return final_batch

class VCRDataset(Dataset):
    def __init__(self, paths, encoder, config):
        super(VCRDataset, self).__init__()
        self.paths = paths
        self.encoder = encoder
        self.config = config

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        example_path = self.paths[idx]
        example = torch.load(example_path)
        idxed_item = dict()

        for prefix in ['qa', 'qar']:
            idxed_item[f'{prefix}_query'] = self.encoder.encode(example[f'{prefix}_query']).ids
            for i, choice_i in enumerate(self.encoder.encode_batch(example[f'{prefix}_choices'])):
                idxed_item[f'{prefix}_choice_{i}'] = choice_i.ids
            idxed_item[f'{prefix}_label'] = example[f'{prefix}_label']

        idxed_item['image'] = load_and_resize_img(example['image'], self.config['model'])
        sep_tokens = {'qa': self.encoder.encode('answer: ').ids, 'qar': self.encoder.encode('rationale: ').ids}

        answers = list()
        max_answer_len = 0
        for prefix in ['qa', 'qar']:
            query = idxed_item[f'{prefix}_query']
            for i in range(self.config['data']['num_answers']):
                option_i = query + sep_tokens[prefix] + idxed_item[f'{prefix}_choice_{i}']
                option_i = option_i[:self.config['data']['lang_seq_len'] - 1] + [MASK]
                answers.append(option_i)
                max_answer_len = max(max_answer_len, len(option_i))

        for idx, option_i in enumerate(answers):
            answers[idx] = option_i + [0] * (max_answer_len - len(option_i))  # Must be optimized

        image = idxed_item['image']
        answers = torch.tensor(answers).reshape(2, 4, -1)
        labels = torch.tensor([idxed_item['qa_label'], idxed_item['qar_label']])

        return {'image':image, 'answers':answers, 'labels':labels, 'max_ans_seq':max_answer_len}

def get_vcr_dataset(paths, encoder, data_type, config):
    vcr_dataset = VCRDataset(paths, encoder, config)
    print(f'Length of {data_type} samples : {len(paths)}')
    return vcr_dataset

def get_vcr_dataloaders(settings, config):
    dataset_path = settings['dataset_path']
    batch_size = settings['batch_size']

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
    encoder = get_encoder()
    train_dataset = get_vcr_dataset(train_paths, encoder, 'train', config)
    val_dataset = get_vcr_dataset(valid_paths, encoder, 'val', config)
    test_dataset = get_vcr_dataset(test_paths, encoder, 'test', config)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True,
                                   collate_fn=collate_pad_vcr_data,  num_workers=8)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False,
                                   collate_fn=collate_pad_vcr_data,  num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False,
                                   num_workers=1)

    return train_data_loader, val_data_loader, test_data_loader