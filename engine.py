# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from timm.utils import accuracy
from timm.optim import create_optimizer

import clip
import utils
import random

def create_text(class_names):
    task_level_info = 'A photo of'
    if isinstance(class_names,str):
        class_names = [class_names] 
    for i, name in enumerate(class_names):
        if i == 0:
            icre_info = ' '+str(name)
        else:
            icre_info = ' '+'or'+' '+str(name)
        task_level_info += icre_info
    task_level_info = task_level_info + "."
    return task_level_info

def create_feature(clip_model, text, device):
    text = clip.tokenize(text).to(device)
    return clip_model.encode_text(text)

def selecte_class_neg_sample(class_names, task_id):
    assert task_id > 0
    valid_class_names = class_names[:task_id]

    rdm_task_idx = random.randint(0,len(valid_class_names)-1)
    rdm_class_idx = random.randint(0,len(valid_class_names[0])-1)
    return class_names[rdm_task_idx][rdm_class_idx]

def prompt_visualization(count_n, vis_task_id, png_name:str=None):
    assert vis_task_id is not None, 'task ids are not inputed' 
    assert len(count_n) == len(vis_task_id), 'the dimensions are not equal'
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    task_n = len(vis_task_id)
    color = [plt.cm.tab20(2*i) for i in range(task_n)]
    
    xs = np.arange(10)
    zs = [i for i in range(task_n)]
    for id, (c, z) in enumerate(zip(color, zs)):
        ys = count_n[id]
        cs = [c] * len(xs)
        ax.bar(xs, ys, zs=z, zdir='y', color=cs)

    png_name = ('png_name'+'.png') if png_name else 'visual.png'
    plt.savefig(png_name)
    # plt.show()
   
def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, clip_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, class_names=None, args = None,
                    task_level_features=None):

    model.train(set_training_mode)
    original_model.eval()
    clip_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('basic_Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if task_id > 0:
        metric_logger.add_meter('class_Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('task_Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    task_level_pos_feature, task_level_neg_feature = None, None
    if args.use_text_encoder and original_model is not None:
        if task_level_features is not None:
            task_level_pos_feature = task_level_features[-1]
            task_level_neg_feature = task_level_features[:-1]
        # pos_class_names = class_names[task_id]
        
        # task_level_pos_name = create_text(class_names=pos_class_names)
        # task_level_pos_feature = create_feature(clip_model, task_level_pos_name, model.device)
        
        # task_level_neg_feature = None
        # if task_id > 0:
        #     task_level_neg_feature = []
        #     random_neg_id = random.randint(0, task_id-1) 
        #     task_level_neg_name = create_text(class_names=class_names[random_neg_id])
        #     task_level_neg_feature = create_feature(clip_model, task_level_neg_name, model.device)
        #     # task_level_neg_feature = clip_model.encode_text(clip.tokenize(task_neg_name).to(model.device))
    prompt_id = []
    for input_list in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input_list[0].to(device, non_blocking=True)
        target = input_list[1].to(device, non_blocking=True)
        
        name = input_list[2] if args.use_text_encoder else None

        class_level_pos_feature, class_level_neg_feature = None, None
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
                
                if task_id > 0 and name is not None and class_names is not None:
                    class_pos_name = list(map(create_text, name))

                    class_level_pos_feature = torch.zeros((len(name),cls_features.shape[1]))
                    for i, pos_name in enumerate(class_pos_name):
                        class_level_pos_feature[i] = create_feature(clip_model, pos_name, model.device)

                    neg_class = selecte_class_neg_sample(class_names,task_id)
                    class_level_neg_name = create_text(neg_class)
                    # print('class_level_neg_name', class_level_neg_name)
                    class_level_neg_feature = create_feature(clip_model, class_level_neg_name,model.device)    
            else:
                cls_features = None

        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode, 
                       task_level_pos_text=task_level_pos_feature, task_level_neg_text=task_level_neg_feature,
                       class_level_pos_text=class_level_pos_feature, class_level_neg_text=class_level_neg_feature)
        logits = output['logits']
        prompt_id.extend(output['prompt_idx'].flatten().tolist())
        
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            basic_loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        task_loss, class_loss = torch.zeros_like(loss), torch.zeros_like(loss)
        total_loss = basic_loss
        if task_id > 0:
            if 'class_loss' in output or 'task_loss' in output:
                task_loss = output['task_loss']
                class_loss = output['class_loss']
                total_loss = basic_loss + task_loss + class_loss
        else:
            total_loss = basic_loss
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True) 
        # for name, parms in model.named_parameters():	
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		#     ' -->grad_value:',parms.grad)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(basic_Loss=basic_loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        if task_id > 0:
            metric_logger.update(task_Loss=task_loss.item())
            metric_logger.update(class_Loss=class_loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    metric_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metric_dict['prompt_ids'] = np.bincount(prompt_id,minlength=10)
    return metric_dict


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()
    with torch.no_grad():
        prompt_ids = []
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']
            prompt_ids.extend(output['prompt_idx'].flatten().tolist())

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    metric_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metric_dict['prompt_ids'] = np.bincount(prompt_ids,minlength=10)
    return metric_dict


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    acc_for_each_task = []
    total_id = []
    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
        acc_for_each_task.append(test_stats['Acc@1'])
        
        total_id.append(test_stats['prompt_ids'])
        test_stats['prompt_ids'] = str(test_stats['prompt_ids'])
        
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    test_stats['acc_all'] = acc_for_each_task
    test_stats['avg_res'] = str(result_str)
    
    prompt_visualization(total_id, range(task_id+1),'test_'+str(task_id))

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, clip_model,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, class_names=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    prompt_his = []
    task_level_texts = []
    for task_id in range(args.num_tasks):
       # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        task_level_feature = None
        if args.use_text_encoder:
            with torch.no_grad():
                cur_task_classes = class_names[task_id]
                text = "A photo of "+cur_task_classes[0]
                for i in range(1,len(cur_task_classes)):
                    text = text + " or " +cur_task_classes[i]
                task_level_texts.append(text+".")
            task_level_tokens = clip.tokenize(task_level_texts).cuda()
            task_level_feature = clip_model.encode_text(task_level_tokens)
        #         class_text = ["A photo of {}.".format(cl) for cl in cl]
        for epoch in range(args.epochs):         
            train_stats = train_one_epoch(model=model, original_model=original_model, clip_model=clip_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, class_names=class_names, args=args,task_level_features=task_level_feature)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)
        
        prompt_his.append(train_stats['prompt_ids'])
        train_stats['prompt_ids'] = str(train_stats['prompt_ids'])
        
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
                
    prompt_visualization(prompt_his, range(10),'train_'+str(10))