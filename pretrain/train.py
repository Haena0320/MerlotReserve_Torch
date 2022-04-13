import os, sys
sys.path.append("/mnt2/user15/merlot_r/merlot_pytorch")
import yaml
from datetime import datetime
from pretrain.dataloader import input_builder
from pretrain.data_utils import batch_index_iterator
from pretrain.pretrain_model import *
#from pretrain.utils import *

import pytz
import argparse
import numpy as np
import functools
import time
import random
import multiprocessing
import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from tqdm import tqdm
import re
import shutil

WORK_DIR = "/mnt2/user15/merlot_r/merlot_pytorch/pretrain"

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)


def random_control(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_processing(num_threads=2):
    num_all_hosts = torch.get_num_threads()
    set_num_hosts = min(num_threads, num_all_hosts)
    torch.set_num_threads(set_num_hosts)
    num_gpus = torch.cuda.device_count()
    print("num_gpus : {}".format(num_gpus))
    num_cpus = multiprocessing.cpu_count()
    if num_cpus == 0:
        raise ValueError('you need gpus ! ')
    return num_gpus, num_cpus, set_num_hosts


def set_opt(model, optim_config):
    # return model, optimizer, scheduler

    no_decay = ["bias", "BatchNorm.weight", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': optim_config["weight_decay_rate"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    adam_betas = (0.9, 0.98)

    optimizer = AdamW(optimizer_grouped_parameters, lr=optim_config["learning_rate"], betas=adam_betas,
                      weight_decay=optim_config["weight_decay_rate"])

    pct_start_ = optim_config["num_warmup_steps"] / optim_config["num_train_steps"]
    scheduler = OneCycleLR(optimizer, max_lr=0.2, pct_start=0.05, total_steps=optim_config["num_train_steps"],
                           epochs=10, anneal_strategy='cos')
    return model, optimizer, scheduler


def find_checkpoint(save_dir):
    checkpoint_files = os.listdir(save_dir)
    have_checkpoint = (save_dir is not None and any("model_state_epoch_" in x for x in checkpoint_files))
    if not have_checkpoint:
        print("there is no checkpoint ! please train model")
        return None

    model_checkpoints = [x for x in checkpoint_files if "model_state_epoch" in x]
    found_epochs = [re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1) for x in
                    model_checkpoints]  # [0,1,2,3,4,...]
    int_epochs = []
    for epoch in found_epochs:
        pieces = epoch.split(".")
        if len(pieces) == 1:
            int_epochs.append([int(pieces[0]), 0])
        else:
            int_epochs.append([int(pieces[0]), int(pieces[1])])
    last_epoch = sorted(int_epochs, reverse=True)[0]
    if last_epoch[1] == 0:
        epoch_to_load = str(last_epoch[0])
    else:
        epoch_to_load = "{0}.{1}".format(last_epoch[0], last_epoch[1])

    model_path = os.path.join(save_dir, "model_state_epoch_{}.th".format(epoch_to_load))
    training_state_path = os.path.join(save_dir, "training_state_epoch_{}.th".format(epoch_to_load))
    return str(model_path), str(training_state_path)


def restore_checkpoint(model, optimizer, lr_scheduler, ckpt_folder):
    checkpoint = find_checkpoint(ckpt_folder)
    if checkpoint is None:
        return 0, []

    model_path, training_state_path = checkpoint
    model_state = torch.load(model_path, map_location='cuda:0')
    training_state = torch.load(training_state_path, map_location='cuda:0')

    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    optimizer.load_state_dict(training_state["optimizer"])

    if lr_scheduler is not None and "lr_scheduler" in training_state:
        lr_scheduler.load_state_dict(training_state['lr_scheduler'])

    epoch_to_return = training_state['epoch'] + 1
    return epoch_to_return


def save_checkpoint(epoch, model, optimizer, scheduler, ckpt_folder):
    if ckpt_folder is not None:
        model_path = os.path.join(ckpt_folder, "model_state_epoch_{}.th".format(epoch))
        model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
        torch.save(model_state, model_path)

    training_state = {"epoch": epoch,
                      "optimizer": optimizer.state_dict(),
                      "lr_scheduler": scheduler.state_dict()}

    training_path = os.path.join(ckpt_folder, "training_state_epoch_{}.th".format(epoch))
    torch.save(training_state, training_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model!')
    parser.add_argument(
        '-config_file',
        help='Where the config.yaml is located',
        type=str,
        default="configs/mini.yaml"
    )
    parser.add_argument(
        '-output_dir',
        help='Override output directory (otherwise we do whats in the config file and add timestamp).',
        dest='output_dir',
        default='',
        type=str,
    )

    parser.add_argument(
        '-disable_wandb',
        help='dont log this result on weights and biases',
        dest='disable_wandb',
        action='store_true')

    parser.add_argument(
        '-seed',
        help='control random',
        dest='seed',
        default=0,
        type=int)

    parser.add_argument(
        '-thread',
        help='set thread num',
        dest='thread',
        default=2)
    args = parser.parse_args()

    random_control(args.seed)
    NUM_GPUS,NUM_CPUS, NUM_HOSTS = set_processing(args.thread)

    print(f"Loading from {args.config_file}", flush=True)
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

        seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
        seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")

        if NUM_GPUS > 0:
            config['data']['num_train_files'] = 1
            config['device']['output_dir'] = args.output_dir
            config['model']['use_bfloat16'] = False
            config['device']['batch_size'] = 6
            config['optimizer']['num_train_steps_override'] = 1000
        else:
            print("oh no, you need gpus !!!!!")

    config['_path'] = args.config_file

    # model
    model = MerlotReservePretrainer.from_config(config, logger=None, device=0)

    model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()

    model, optimizer, scheduler = set_opt(model, config["optimizer"])

    save_folder = os.path.join(WORK_DIR, config['device']["output_dir"])
    log_folder = os.path.join(save_folder, "log")

    if os.path.exists(save_folder):
        print("Found folder! restoring", flush=True)
        start_epoch = restore_checkpoint(model, optimizer, scheduler, save_folder)
    else:
        print("Making directories")
        os.mkdir(save_folder)
        os.mkdir(log_folder)
        start_epoch = 0
        shutil.copy2(args.params, args.folder)

    writer = SummaryWriter(str(log_folder))

    # train
    batch_size = config["device"]["batch_size"] // NUM_HOSTS
    matching_fns = []

    for i in range(200):
        matching_fns.append(f"/mnt2/user15/merlot_r/merlot_reserve_backup/data/tfrecordes/train{i:05d}of17262.tfrecord")
    assert (len(matching_fns) > 0)
    print(matching_fns)
    n_fns_per_cycle = min(config["device"].get('n_fns_per_cycle', 32), len(matching_fns))
    while (len(matching_fns) % n_fns_per_cycle) != 0:
        print(f"!!!Truncating n_fns_per_cycle {n_fns_per_cycle} -> {n_fns_per_cycle - 1} so it fits", flush=True)
        n_fns_per_cycle -= 1

    n_epochs = 0

    time_elapsed = []
    while True:
        fns_shuff = [x for x in matching_fns]
        random.shuffle(fns_shuff)
        print(f"Now on epoch {n_epochs}")
        for s, e in batch_index_iterator(len(fns_shuff), batch_size=n_fns_per_cycle, skip_end=True):
            print(f"Resetting iterator, epoch={n_epochs}, batch of fns={s}:{e}/{len(fns_shuff)}", flush=True)
            try:
                dataloader = input_builder(config, fns=fns_shuff[s:e], num_workers=NUM_HOSTS, batch_size=batch_size,
                                           num_devices=1, is_training=True)
                ###########################################
                for step, batch in enumerate(tqdm(dataloader)):
                    st = time.time()
                    batch_ = {k: v.cuda() for k, v in batch.items()}
                    outputs = model(batch_)
                    loss, loss_info = loss_fn_given_preds(outputs)
                    writer.add_scalar("loss_all/train", loss.item(), step)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    optimizer.zero_grad()

                    writer.add_scalar("losses/imgs_to_audio", loss_info["imgs_to_audio"].item(), step)
                    writer.add_scalar("losses/text_to_audio", loss_info["text_to_audio"].item(), step)
                    writer.add_scalar("losses/stuff_to_span", loss_info["stuff_to_span"].item(), step)

                    time_elapsed.append(time.time() - st)
                    if len(time_elapsed) > 100:
                        tsum = sum(time_elapsed)
                        print("Completed 100 batches in {:.3f}sec".format(tsum), flush=True)
                        time_elapsed = []

            ###########################################
            except Exception as e:
                print(str(e), flush=True)
                time.sleep(5)
        save_checkpoint(n_epochs, model, optimizer, scheduler, save_folder)
        print(f"Saved epoch {n_epochs}", flush=True)
        n_epochs += 1
