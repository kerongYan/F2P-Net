#!/usr/bin/env python3
"""
Simplified training script for F2P_Net.
Usage examples:
    python train.py --dataset MT --shot_num 64
    python train.py --dataset NEU_Seg --shot_num 64
    python train.py --dataset KolektorSDD2 --shot_num 64
    python train.py --dataset DAGM2007 --shot_num 64
"""

import argparse
import os
import random
import warnings
from functools import partial
from os.path import join
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ignore all warnings
warnings.filterwarnings("ignore")

# project utilities / datasets / models
from misc import get_idle_gpu, get_idle_port, set_randomness
from F2P_Net.datasets.neu_seg import NEU_SegDataset
from F2P_Net.datasets.mt import MTDataset
from F2P_Net.datasets.kolektorsdd2 import KolektorSDD2_Dataset
from F2P_Net.datasets.dagm2007 import DAGM2007_Dataset
from F2P_Net.datasets.transforms import HorizontalFlip, VerticalFlip
from F2P_Net.utils.eval_metrics import performances
from F2P_Net.models.modeling import F2P_Net
from F2P_Net.utils.evaluators import StreamSegMetrics

# Mapping of dataset name to dataset class
DATASET_MAP = {
    'NEU_Seg': NEU_SegDataset,
    'MT': MTDataset,
    'KolektorSDD2': KolektorSDD2_Dataset,
    'DAGM2007': DAGM2007_Dataset,
}


def parse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_dir', default='/home/lenovo/yankerong/F2P_Net/exp', type=str,
        help="Directory to save best checkpoint files."
    )
    parser.add_argument(
        '--data_dir', default='/home/lenovo/yankerong/F2P_Net/data', type=str,
        help="Directory that contains datasets."
    )
    parser.add_argument(
        '--num_workers', default=None, type=int,
        help="num_workers for dataloaders. Default: 1 for 1-shot, 4 for others."
    )
    parser.add_argument(
        '--train_bs', default=None, type=int,
        help="Training dataloader batch size. Default: 1 for 1-shot, 4 for larger shots."
    )
    parser.add_argument(
        '--val_bs', default=None, type=int,
        help="Validation dataloader batch size. Default: 1 for 1-shot, 4 for larger shots."
    )
    parser.add_argument(
        '--dataset', default='NEU_Seg', type=str, choices=list(DATASET_MAP.keys()),
        help="Target dataset name."
    )
    parser.add_argument(
        '--shot_num', default=64, type=int, choices=[1, 4, 16, 32, 64, None],
        help="Shot number: 1/4/16/32/64. Use None or omit for full-shot."
    )
    parser.add_argument(
        '--sam_type', default='vit_l', type=str, choices=['vit_b', 'vit_l', 'vit_h'],
        help='Backbone SAM type.'
    )
    parser.add_argument(
        "--class_name", default="NEU_Seg", choices=[
            "NEU_Seg", "MT", "KolektorSDD2", "DAGM2007",
        ], help="Class/dataset name used by evaluation."
    )
    return parser.parse_args()


def save_best_model(model, save_path, current_metric, best_metric, metric_name):
    """Save model when a new best metric is found."""
    if current_metric > best_metric:
        print(f"New best {metric_name}: {current_metric:.4f} (previous: {best_metric:.4f})")
        best_metric = current_metric
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}.")
    return best_metric


def calculate_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """Dice loss for logits (applies sigmoid inside)."""
    assert inputs.size(0) == targets.size(0)
    inputs = inputs.sigmoid()
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


class FocalLoss(nn.Module):
    """Binary focal loss wrapper accepting logits."""

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def pixel_dice_loss1(pred, target):
    """Simple dice loss for probabilities."""
    smooth = 1e-6
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def focal_loss_fns(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss variant using probabilities (not logits)."""
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool = True):
    """Initialize RNG for DataLoader workers for reproducibility."""
    seed = base_seed if same_worker_seed else base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def batch_to_cuda(batch: dict, device: torch.device):
    """Move selected tensors inside a batch to the given device with appropriate dtypes."""
    for key, val in batch.items():
        if key in ['images', 'gt_masks', 'point_coords', 'box_coords', 'noisy_object_masks', 'object_masks']:
            batch[key] = [item.to(device=device, dtype=torch.float32) if item is not None else None for item in val]
        elif key in ['point_labels']:
            batch[key] = [item.to(device=device, dtype=torch.long) if item is not None else None for item in val]
    return batch


def ensure_4d_masks(masks: List[torch.Tensor]):
    """
    Ensure list of mask tensors are 4D: [B, C, H, W].
    This modifies the list elements in-place.
    """
    for i in range(len(masks)):
        m = masks[i]
        if len(m.shape) == 2:
            masks[i] = m[None, None, ...]
        elif len(m.shape) == 3:
            # if shape [B, H, W] -> [B, 1, H, W], if [C, H, W] -> [1, C, H, W]
            if m.shape[0] == 1 or m.shape[0] == 2:
                # assume [B, H, W] -> add channel dim
                if m.ndim == 3:
                    masks[i] = m[:, None, ...]
                else:
                    masks[i] = m[None, None, ...]
            else:
                masks[i] = m[:, None, ...]
        elif len(m.shape) != 4:
            raise RuntimeError("Mask tensor must be 2D/3D/4D. Found shape: %s" % (str(m.shape),))


def main_worker(worker_id, worker_args):
    """Main training worker for single or multi-GPU runs."""
    set_randomness()
    worker_id = int(worker_id)

    gpu_num = len(worker_args.used_gpu)
    world_size = int(os.environ.get('WORLD_SIZE', gpu_num))
    base_rank = int(os.environ.get('RANK', 0))
    # compute local rank for this process
    local_rank = base_rank * gpu_num + worker_id

    if gpu_num > 1:
        dist.init_process_group(backend='nccl', init_method=worker_args.dist_url,
                                world_size=world_size, rank=local_rank)

    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)

    # select dataset class
    if worker_args.dataset not in DATASET_MAP:
        raise ValueError(f"Invalid dataset name: {worker_args.dataset}")
    dataset_class = DATASET_MAP[worker_args.dataset]

    transforms = [VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]
    dataset_dir = join(worker_args.data_dir, worker_args.dataset)
    print("Loading the data required for training...")
    train_dataset = dataset_class(data_dir=dataset_dir, train_flag=True,
                                  shot_num=worker_args.shot_num, transforms=transforms)
    print(f"train_dataset: {len(train_dataset)}")
    val_dataset = dataset_class(data_dir=dataset_dir, train_flag=False)

    # determine number of worker threads for dataloaders
    default_train_workers = 1 if worker_args.shot_num == 1 else 4
    train_workers = worker_args.num_workers if worker_args.num_workers is not None else default_train_workers
    val_workers = worker_args.num_workers if worker_args.num_workers is not None else 2

    # sampler for distributed training
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if dist.is_initialized() else None

    train_batch_size = worker_args.train_bs if worker_args.train_bs is not None else 1
    val_batch_size = worker_args.val_bs if worker_args.val_bs is not None else 1

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=(sampler is None),
        num_workers=train_workers,
        sampler=sampler,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_workers,
        drop_last=False,
        collate_fn=val_dataset.collate_fn
    )

    model = F2P_Net(model_type=worker_args.sam_type).to(device=device)

    if dist.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # using the local device id for DDP device mapping
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[worker_id], output_device=worker_id,
                                                          find_unused_parameters=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=2000, eta_min=1e-5)

    max_epoch_num = 100
    best_miou = 0.0

    class_names = ['Background', 'Foreground']
    iou_eval = StreamSegMetrics(class_names=class_names)

    exp_path = join(worker_args.exp_dir,
                    f'{worker_args.dataset}_{worker_args.sam_type}_{worker_args.shot_num if worker_args.shot_num else "full"}shot')
    os.makedirs(exp_path, exist_ok=True)

    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    accumulation_steps = 22 if worker_args.shot_num is None else 1
    model.train()

    for epoch in range(1, max_epoch_num + 1):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        train_pbar = tqdm(total=len(train_dataloader), desc=f'train epoch {epoch}', leave=False) if local_rank == 0 else None

        optimizer.zero_grad()
        for train_step, batch in enumerate(train_dataloader):
            batch = batch_to_cuda(batch, device)

            # forward pass: returns mask predictions, image-level predictions and pixel-level predictions
            masks_pred, images_pre, pixels_pre = model(
                imgs=batch['images'],
                point_coords=batch['point_coords'],
                point_labels=batch['point_labels'],
                box_coords=batch['box_coords'],
                noisy_masks=batch['noisy_object_masks'],
            )

            labels = batch['label']  # list of image-level labels
            masks_gt = batch['object_masks']

            # ensure masks have shape [B, C, H, W]
            ensure_4d_masks(masks_pred)
            ensure_4d_masks(masks_gt)
            ensure_4d_masks(pixels_pre)

            bce_loss_list, dice_loss_list, focal_loss_list, anomaly_loss_list = [], [], [], []

            for i in range(len(masks_pred)):
                pred = masks_pred[i]
                label = masks_gt[i]
                label_bin = torch.where(torch.gt(label, 0.), 1., 0.)

                b_loss = F.binary_cross_entropy_with_logits(pred, label_bin.float())
                d_loss = calculate_dice_loss(pred, label_bin)
                f_loss = focal_loss_fn(pred, label_bin.float())

                bce_loss_list.append(b_loss)
                dice_loss_list.append(d_loss)
                focal_loss_list.append(f_loss)

                # image-level loss: images_pre[i] vs labels[i]
                label_tensor = torch.tensor(labels[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(images_pre[i].device)
                image_level_loss = nn.BCELoss()(images_pre[i], label_tensor)
                anomaly_loss_list.append(image_level_loss)

            # pixel-level losses (pixels_pre are probabilities)
            pixel_bce_loss_list, pixel_dice_loss_list, pixel_focal_loss_list = [], [], []
            for i in range(len(pixels_pre)):
                pix = pixels_pre[i]            # probability map tensor
                gt_mask = masks_gt[i]         # corresponding ground-truth mask
                # if gt_mask has batch dim >1, keep first
                gt_mask_proc = gt_mask[0:1, :] if gt_mask.shape[0] >= 1 else gt_mask
                pixel_bce_loss_list.append(F.binary_cross_entropy(pix, gt_mask_proc.float()))
                pixel_dice_loss_list.append(pixel_dice_loss1(pix, gt_mask_proc.float()))
                pixel_focal_loss_list.append(focal_loss_fns(pix, gt_mask_proc.float()))

            # aggregate losses
            bce_loss = sum(bce_loss_list) / len(bce_loss_list)
            dice_loss = sum(dice_loss_list) / len(dice_loss_list)
            focal_loss = sum(focal_loss_list) / len(focal_loss_list)
            anomaly_loss = sum(anomaly_loss_list) / len(anomaly_loss_list)
            pixel_bce_loss = sum(pixel_bce_loss_list) / len(pixel_bce_loss_list)
            pixel_dice_loss = sum(pixel_dice_loss_list) / len(pixel_dice_loss_list)
            pixel_focal_loss = sum(pixel_focal_loss_list) / len(pixel_focal_loss_list)

            total_loss = (1.0 * bce_loss + 1.0 * dice_loss + 5.0 * focal_loss +
                          anomaly_loss + 1.0 * pixel_bce_loss + 1.0 * pixel_dice_loss + 5.0 * pixel_focal_loss)

            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                bce_loss=bce_loss.clone().detach(),
                dice_loss=dice_loss.clone().detach(),
                focal_loss=focal_loss.clone().detach(),
                anomaly_loss=anomaly_loss.clone().detach(),
                pixel_bce_loss=pixel_bce_loss.clone().detach(),
                pixel_dice_loss=pixel_dice_loss.clone().detach(),
                pixel_focal_loss=pixel_focal_loss.clone().detach()
            )

            # gradient accumulation
            total_loss = total_loss / accumulation_steps
            total_loss.backward()

            if (train_step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # reduce losses across processes for logging if distributed
            if dist.is_initialized():
                for key in list(loss_dict.keys()):
                    if hasattr(loss_dict[key], 'detach'):
                        loss_dict[key] = loss_dict[key].detach()
                    dist.reduce(loss_dict[key], dst=0, op=dist.ReduceOp.SUM)
                    loss_dict[key] /= dist.get_world_size()

            if train_pbar:
                str_step_info = (
                    "Epoch: {epoch}/{epochs}. Loss: {total_loss:.4f}(total), {bce_loss:.4f}(bce), "
                    "{dice_loss:.4f}(dice), {focal_loss:.4f}(focal), {anomaly_loss:.4f}(anomaly), "
                    "{pixel_bce_loss:.4f}(pixel_bce), {pixel_dice_loss:.4f}(pixel_dice), {pixel_focal_loss:.4f}(pixel_focal)"
                ).format(
                    epoch=epoch, epochs=max_epoch_num,
                    total_loss=loss_dict['total_loss'], bce_loss=loss_dict['bce_loss'],
                    dice_loss=loss_dict['dice_loss'], focal_loss=loss_dict['focal_loss'],
                    anomaly_loss=loss_dict['anomaly_loss'], pixel_bce_loss=loss_dict['pixel_bce_loss'],
                    pixel_dice_loss=loss_dict['pixel_dice_loss'], pixel_focal_loss=loss_dict['pixel_focal_loss']
                )
                train_pbar.set_postfix_str(str_step_info)
                train_pbar.update(1)

        # validation loop (run every 10 epochs and some mid-range epochs as original)
        if epoch % 10 == 0 or (18 <= epoch < 30 and epoch % 2 == 0):
            model.eval()
            valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False) if local_rank == 0 else None

            all_anomaly_scores = []
            all_masks_preds = []
            all_masks_gts = []
            all_images_pre = []
            all_labels_eval = []
            all_masks_pred_list = []
            all_masks_gt_list = []

            val_model = model.module if hasattr(model, 'module') else model

            with torch.no_grad():
                for val_step, batch in enumerate(val_dataloader):
                    batch = batch_to_cuda(batch, device)
                    # set inference image and run inference (API per original)
                    val_model.set_infer_img(img=batch['images'])
                    masks_pred, images_pre, pixels_pre = val_model.infer(box_coords=batch['box_coords'])

                    masks_gt = batch['gt_masks']
                    labels = batch['label']
                    indices = batch.get('index_name', None)

                    ensure_4d_masks(masks_pred)
                    ensure_4d_masks(masks_gt)
                    ensure_4d_masks(pixels_pre)

                    # store predictions and ground truth for evaluation
                    all_masks_pred_list.extend([mask.squeeze().cpu() for mask in masks_pred])
                    all_masks_gt_list.extend([mask.squeeze().cpu() for mask in masks_gt])

                    # update IoU evaluator
                    iou_eval.update(masks_gt, masks_pred, batch.get('index_name'))

                    all_masks_preds.append(masks_pred[0])
                    all_anomaly_scores.append(pixels_pre[0])
                    all_labels_eval.append(labels[0])
                    all_masks_gts.append(masks_gt[0])
                    all_images_pre.append(images_pre[0])

                    if valid_pbar:
                        valid_pbar.update(1)
                        valid_pbar.set_postfix_str(f"Epoch: {epoch}/{max_epoch_num}.")

            miou = iou_eval.compute()[0]['Mean Foreground IoU']
            iou_eval.reset()

            if miou > best_miou:
                best_model_filename = "best_model_mIoU.pth"
                torch.save(
                    model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                    join(exp_path, best_model_filename)
                )
                best_miou = miou
                print(f'Best mIoU has been updated to {best_miou:.2%}!')

                if epoch >= 10:
                    # compute Image-AUROC and Pixel-AUROC and other metrics using helper
                    ret_metrics = performances(worker_args.class_name, all_anomaly_scores, all_masks_preds,
                                               all_labels_eval, all_masks_gts, all_images_pre,
                                               all_masks_gt_list, all_masks_pred_list)

                    class_name = worker_args.class_name
                    pixel_auroc = ret_metrics.get(f"{class_name}_pixel_auc")
                    recall = ret_metrics.get(f"{class_name}_recall")
                    precision = ret_metrics.get(f"{class_name}_precision")
                    f1_score = ret_metrics.get(f"{class_name}_f1_score")
                    mAP = ret_metrics.get(f"{class_name}_mAP")

                    print(f"Epoch {epoch + 1}: mAP={mAP:.4f}, Recall={recall:.4f}, "
                          f"Pixel-AUROC={pixel_auroc:.4f}, F1 Score={f1_score:.4f}, mIoU={miou:.2%}, Precision={precision:.4f}")

            if valid_pbar:
                valid_pbar.close()
            model.train()


if __name__ == '__main__':
    # specify visible GPUs here (can be changed by user)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

    args = parse()
    # determine used GPUs either from env or helper
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        used_gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        used_gpu = get_idle_gpu(gpu_num=4)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, used_gpu))

    args.used_gpu, args.gpu_num = used_gpu, len(used_gpu)
    print(f"Total available GPUs: {torch.cuda.device_count()}")

    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)
    else:
        # initialize multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            try:
                mp.set_start_method('forkserver')
                print("Spawn start method not available; using forkserver instead.")
            except RuntimeError as e:
                raise RuntimeError(
                    "Neither spawn nor forkserver methods available for multiprocessing. Error: %s" % str(e)
                )

        # use localhost dist url and a random free port
        args.dist_url = "tcp://127.0.0.1" + f':{get_idle_port()}'
        mp.spawn(main_worker, nprocs=args.gpu_num, args=(args,))
