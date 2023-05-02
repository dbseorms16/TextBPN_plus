import os
import gc
import time
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import SynthText, TotalText, Ctw1500Text, Icdar15Text, LsvtTextJson,\
    Mlt2017Text, TD500Text, ArtTextJson, Mlt2019Text, Ctw1500Text_New, TotalText_New, ArtText, CustomText
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import Augmentation
from cfglib.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from cfglib.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary
from util.shedule import FixLR
import torch.distributed as dist
# multiprocessing.set_start_method("spawn", force=True)
import collections
from parallel import DataParallelModel, DataParallelCriterion
lr = None
train_step = 0


def save_model(model, epoch, lr, optimzer):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'TextBPN_{}_{}.pth'.format(cfg.net, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict()
        # 'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    state_dict = torch.load(model_path)
    
    new_dict = collections.OrderedDict()
    
    if cfg.mgpu:
        for k in state_dict['model'].keys():
            new_k = 'module.' + k
            new_dict[new_k] = state_dict['model'][k]
    
    model.load_state_dict(new_dict)


def _parse_data(inputs):
    input_dict = {}
    inputs = list(map(lambda x: to_device(x), inputs))
    input_dict['img'] = inputs[0]
    input_dict['train_mask'] = inputs[1]
    input_dict['tr_mask'] = inputs[2]
    input_dict['distance_field'] = inputs[3]
    input_dict['direction_field'] = inputs[4]
    input_dict['weight_matrix'] = inputs[5]
    input_dict['gt_points'] = inputs[6]
    input_dict['proposal_points'] = inputs[7]
    input_dict['ignore_tags'] = inputs[8]

    return input_dict

from itertools import chain

def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    # scheduler.step()

    for i, inputs in enumerate(train_loader):
        data_time.update(time.time() - end)
        train_step += 1
        
        input_dict = _parse_data(inputs)
        output_dict = model(input_dict)
        
        # loss_dict = criterion(input_dict, output_dict, eps=epoch+1)
        loss_dict = DataParallelCriterion(criterion(input_dict, output_dict, eps=epoch+1))
        loss = loss_dict["total_loss"]
        # backward
        try:
            optimizer.zero_grad()
            loss.backward()
            
        except:
            print("loss gg")
            continue
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0) and epoch % 8 == 0:
            visualize_network_output(output_dict, input_dict, mode='train')

        if i % cfg.display_freq == 0:
            gc.collect()
            print_inform = "({:d} / {:d}) ".format(i, len(train_loader))
            for (k, v) in loss_dict.items():
                print_inform += " {}: {:.4f} ".format(k, v.item())
            print(print_inform)

    if cfg.exp_name == 'Synthtext' or cfg.exp_name == 'ALL':
        if epoch % cfg.save_freq == 0:
            save_model(model, epoch, scheduler.get_lr(), optimizer)
    elif cfg.exp_name == 'MLT2019' or cfg.exp_name == 'ArT' or cfg.exp_name == 'MLT2017':
        if epoch < 50 and cfg.max_epoch >= 200:
            if epoch % (2*cfg.save_freq) == 0:
                save_model(model, epoch, scheduler.get_lr(), optimizer)
        else:
            if epoch % cfg.save_freq == 0:
                save_model(model, epoch, scheduler.get_lr(), optimizer)
    # else:
        # if epoch % 5 == 0:
    save_model(model, epoch, scheduler.get_lr(), optimizer)

    print('Detail=No_bp, gt=4 , Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()), 'Training Loss: {}'.format(losses.avg))


def main():

    global lr
    if cfg.exp_name == 'Totaltext':
        trainset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Synthtext':
        trainset = SynthText(
            data_root='data/SynthText',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Ctw1500':
        trainset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Icdar2015':
        trainset = Icdar15Text(
            data_root='data/Icdar2015',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    elif cfg.exp_name == 'MLT2017':
        trainset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'TD500':
        trainset = TD500Text(
            data_root='data/TD500',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'ArT':
        trainset = ArtTextJson(
            data_root='data/ArT',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'MLT2019':
        trainset = Mlt2019Text(
            data_root='data/MLT-2019',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Custom':
        trainset = CustomText(
            data_root='data/Custom_data',
            is_training=True,
            load_memory=cfg.load_memory,
            cfg=cfg,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
        
    elif cfg.exp_name == 'ALL':
        trainset_art = ArtTextJson(
            data_root='data/ArT',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds))

        trainset_mlt19 = Mlt2019Text(
            data_root='data/MLT-2019',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds))

        trainset_lsvt = LsvtTextJson(
            data_root="/home/prir1005/pubdata/LSVT",
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds))
        trainset = ConcatDataset([trainset_lsvt, trainset_mlt19, trainset_art])
        valset = None

    else:
        print("dataset name is not correct")

    
    if cfg.mgpu:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                    num_workers=cfg.num_workers,
                                    pin_memory=True, sampler=train_sampler)  # generator=torch.Generator(device=cfg.device)
    else:
        train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                    shuffle=True, num_workers=cfg.num_workers,
                                    pin_memory=True)  # generator=torch.Generator(device=cfg.device)
    # Model
    model = TextNet(backbone=cfg.net, is_training=True)
    
    criterion = TextLoss()
    if cfg.mgpu:
        # model = nn.DataParallel(model, device_ids=[int(x) for x in args.gpu.split(',')]).cuda()
        model = DataParallelModel(model)

        # model = model.module.to(cfg.device)
    else:
        model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True
    if cfg.resume:
        load_model(model, cfg.resume)

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == 'Synthtext':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == 'Synthtext':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
        scheduler.step()
        # if epoch <= 300:
        #     continue
        train(model, train_loader, criterion, scheduler, optimizer, epoch)

    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2022)
    torch.manual_seed(2022)
    
    dist_url = 'env://'
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
        
    dist.init_process_group("nccl", init_method=dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    dist.barrier()
    
    if dist.get_rank()  == 0:
        print(f'RANK {rank}, WORLD_SIZE {world_size}, LOCAL_RANK {local_rank}')
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    # print_config(cfg)

    # print('Start training TextBPN.')
    # print('Loading from {}'.format(cfg.resume))
    # main
    main()

