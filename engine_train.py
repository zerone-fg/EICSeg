import sys
from typing import Iterable
import torch
import lr_sched as lr_sched
import misc as misc
import wandb
from torch import autograd
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
from example_Data.wbc import WBCDataset
from example_Data.stare import StareDataset
from example_Data.spine import SPINEDataset
from example_Data.acdc import ACDCDataset
import itertools
from collections import defaultdict
from tqdm.auto import tqdm
from eval.wbc import inference_multi_our as inf1


def get_loss_scale_for_deepspeed(optimizer):
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm


@torch.no_grad()
def test_pt(model):
    model = model.eval()

    d_support = WBCDataset('JTSC', split='support', label=None, size=(448, 448))
    d_test = WBCDataset('JTSC', split='test', label=None, size=(448, 448))

    n_support = 32
    n_predictions = 200
    support_images, support_labels, _= zip(*itertools.islice(d_support, n_support))

    support_images = torch.stack(support_images).to('cpu')
    support_labels = torch.stack(support_labels).to('cpu')

    results = defaultdict(list)

    idxs = np.random.permutation(len(d_test))[:n_predictions]

    for i in tqdm(idxs):
        image, label, _= d_test[i]
        vals = inf1(model, image, label, support_images, support_labels, 'cuda')

        for k, v in vals.items():
            results[k].append(v)
    
    scores = results.pop('score')
    avg_score = np.mean(scores)
    return avg_score


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    model_without_ddp: torch.nn.Module,
                    best_score=0.79,
                    log_writer=None,
                    global_rank=None,
                    args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    wandb_images = []
    


    for data_iter_step, (samples, targets, references, ref_masks) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)  #####  2, 3, 448, 448
        targets = targets.to(device, non_blocking=True)  #####  2, 5, 448, 448
        references = references.to(device, non_blocking=True) ##### 2, 5, 3, 448, 448
        ref_masks = ref_masks.to(device, non_blocking=True)   ##### 2, 5, 5, 448, 448

        with autograd.detect_anomaly():
            loss = model(samples, targets, references, ref_masks)
            loss_value = loss.item()

            if loss_scaler is None:
                loss /= accum_iter
                loss.backward()

                if (data_iter_step + 1) % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                loss /= accum_iter
                if (data_iter_step + 1) % accum_iter == 0:
                    optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
        if data_iter_step % 200 == 0:
            import os
            from pathlib import Path
            score = test_pt_1(model_without_ddp)
            model = model.train()

            if score > best_score:
                best_score = score
                misc.save_model(
                                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                    loss_scaler=loss_scaler, epoch=data_iter_step, best_score=best_score)
            
            
            if data_iter_step != 0:
                try:
                    os.remove(Path(args.output_dir) / ('checkpoint-%s.pth' % (data_iter_step - 200)))
                except:
                    continue

    if global_rank == 0 and args.log_wandb and len(wandb_images) > 0:
        wandb.log({"Training examples": wandb_images})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, best_score



@torch.no_grad()
def evaluate_pt(data_loader, model, device, epoch=None, global_rank=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    wandb_images = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[1]
        references = batch[2]
        ref_masks = batch[3]
    
        loss = model(samples, targets, references, ref_masks)

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print('Val loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if global_rank == 0 and args.log_wandb:
        wandb.log({**{f'test_{k}': v for k, v in out.items()},'epoch': epoch})
        if len(wandb_images) > 0:
            wandb.log({"Testing examples": wandb_images[::2][:20]})
    return out
