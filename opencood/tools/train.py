                      # -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.inference import inference
from opencood.utils.benchmark import compute_fps, compute_gflops
from tabulate import tabulate

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--finetune_from', default='',
                        help='start finetune_from checkpoint')
    parser.add_argument('--freeze', action='store_true',
                        help="freeze the param of pretrained model")
    parser.add_argument('--benchmark', action='store_true',)
    parser.add_argument('--metric',  type=str, default="old",
                        help='eval epoch')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(
        hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)
        sampler_test = DistributedSampler(opencood_validate_dataset,
                                          shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=hypes['train_params']['num_workers'],
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                shuffle=False,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
        test_loader = DataLoader(opencood_validate_dataset,
                                 batch_size=1,
                                 sampler=sampler_test,
                                 shuffle=False,
                                 num_workers=16,
                                 collate_fn=opencood_train_dataset.collate_batch_test,
                                 drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=hypes['train_params']['num_workers'],
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)
        test_loader = DataLoader(opencood_validate_dataset,
                                 batch_size=1,
                                 num_workers=16,
                                 collate_fn=opencood_train_dataset.collate_batch_test,
                                 shuffle=False,
                                 pin_memory=False,
                                 drop_last=False)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for fine tune:
    if opt.finetune_from:
        pretrained_model_dict = torch.load(opt.finetune_from, map_location='cpu')
        diff_keys = {k:v for k, v in pretrained_model_dict.items() if k not in model.state_dict()}
        if diff_keys:
            print(f"!!! PreTrained model has keys: {diff_keys.keys()}, \
                which are not in the model you have created!!!")
        diff_keys = {k:v for k, v in model.state_dict().items() if k not in pretrained_model_dict.keys()}
        if diff_keys:
            print(f"!!! Created model has keys: {diff_keys.keys()}, \
                which are not in the model you have trained!!!")
        model.load_state_dict(pretrained_model_dict, strict=False)
        
        ## whether freeze the pretrained model
        if opt.freeze:
            for name, value in model.named_parameters():
                if name in pretrained_model_dict:
                    value.requires_grad = False

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.benchmark:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # gflops = compute_gflops(model, opencood_validate_dataset, approximated=False)
        fps = compute_fps(model, opencood_validate_dataset, num_iters=20, batch_size=1)
        # bfps = compute_fps(model, opencood_validate_dataset, num_iters=20, batch_size=1)
        tab_keys = ["#Params(M)", "FPS"]
        tab_vals = [n_params / 10 ** 6, fps]
        table = tabulate([tab_vals], headers=tab_keys, tablefmt="pipe",
                         floatfmt=".3f", stralign="center", numalign="center")
        print("===== Benchmark (Crude Approx.) =====\n" + table)
        return

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)#True
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup[[]]
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    # def print_strides(model):
    #     for name, param in model.named_parameters():
    #         if param.grad is not None:
    #             print(f"Parameter: {name}, Parameter strides: {param.stride()}, Gradient strides: {param.grad.stride()}")
    #         else:
    #             print(f"Parameter: {name}, Parameter strides: {param.stride()}, Gradient is None")

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        # pbar2 = None

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)
            # tzh batch_data['ego'] 现在是[],包含多帧，做了后续处理

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            cur_data = batch_data['ego'][-1]
            
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(
                    ouput_dict, cur_data['label_dict'], (cur_data['label_dict_all'], cur_data['record_len']))
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(
                        ouput_dict, cur_data['label_dict'], (cur_data['label_dict_all'], cur_data['record_len']))
            
            if i % 10 == 0:
                criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
                pbar2.update(10)

            if not opt.half:
                final_loss.backward()
                # print_strides(model)
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    cur_data = batch_data['ego'][-1]
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict, cur_data['label_dict'], (cur_data['label_dict_all'], cur_data['record_len']))
                    # final_loss = criterion(ouput_dict, cur_data['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['test_freq'] == 0:
            # import ipdb;ipdb.set_trace()
            inference(model, opencood_validate_dataset, test_loader, device,
                      model_dir=saved_path, writer=writer, epoch=epoch)
    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
