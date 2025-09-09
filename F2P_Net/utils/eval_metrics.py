import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score

def compute_image_auroc(all_images_pre, all_labels, chunk_size=100):
 
    scores_list = []
    labels_list = []
    total = len(all_images_pre)
    for i in range(0, total, chunk_size):
        batch_images = all_images_pre[i:i+chunk_size]
        batch_labels = all_labels[i:i+chunk_size]
        
        batch_scores = np.array([
            tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor 
            for tensor in batch_images
        ], dtype=np.float32)
        batch_labels_np = np.array([
            tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor 
            for tensor in batch_labels
        ], dtype=np.uint8)
        scores_list.append(batch_scores.flatten())
        labels_list.append(batch_labels_np.flatten())
    
    all_scores = np.concatenate(scores_list)
    all_labels_np = np.concatenate(labels_list)
    return roc_auc_score(all_labels_np, all_scores)

def compute_pixel_auroc(preds, masks_gt, chunk_size=100):

    score_chunks = []
    label_chunks = []
    total = len(preds)
    for i in range(0, total, chunk_size):
        chunk_scores = []
        chunk_labels = []
        for p, m in zip(preds[i:i+chunk_size], masks_gt[i:i+chunk_size]):
            
            p_np = p.squeeze(0).detach().cpu().numpy().astype(np.float32).flatten()
            
            if isinstance(m, torch.Tensor):
                m_np = m.detach().cpu().numpy().astype(np.uint8).flatten()
            else:
                m_np = np.array(m, dtype=np.uint8).flatten()
            chunk_scores.append(p_np)
            chunk_labels.append(m_np)
        score_chunks.append(np.concatenate(chunk_scores))
        label_chunks.append(np.concatenate(chunk_labels))
    
    all_scores = np.concatenate(score_chunks)
    all_labels = np.concatenate(label_chunks)
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    return auc(fpr, tpr)

def calculate_ap_at_thresholds(all_masks_pred, all_masks_gt, thresholds):
   
    aps = []
    for thresh in thresholds:
        tp, fp, fn = 0, 0, 0
        for pred, gt in zip(all_masks_pred, all_masks_gt):
            
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu()
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu()
            pred_bin = (pred >= thresh).int()
            tp += ((pred_bin == 1) & (gt == 1)).sum().item()
            fp += ((pred_bin == 1) & (gt == 0)).sum().item()
            fn += ((pred_bin == 0) & (gt == 1)).sum().item()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        aps.append(precision * recall)
    
    return aps, np.mean(aps)

def performances(class_name, preds, all_masks_preds, all_labels, masks_gt, all_images_pre, all_masks_gt, all_masks_pred):
    ret_metrics = {}


    pixel_auroc = compute_pixel_auroc(preds, masks_gt, chunk_size=100)
    
  
    total_tp, total_fp, total_fn = 0, 0, 0
    for pred, gt in zip(all_masks_pred, all_masks_gt):
        
        pred = pred.int() if isinstance(pred, torch.Tensor) else pred.astype(np.int32)
        gt = gt.int() if isinstance(gt, torch.Tensor) else gt.astype(np.int32)
        total_tp += ((pred == 1) & (gt == 1)).sum().item()
        total_fp += ((pred == 1) & (gt == 0)).sum().item()
        total_fn += ((pred == 0) & (gt == 1)).sum().item()
    
    recall = total_tp / (total_tp + total_fn + 1e-10)
    precision = total_tp / (total_tp + total_fp + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)


    _, mAP = calculate_ap_at_thresholds(all_masks_pred, all_masks_gt, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])


    metrics_map = {
        # "image_auc": image_auroc,
        "pixel_auc": pixel_auroc,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "mAP": mAP
    }
    for k, v in metrics_map.items():
        ret_metrics[f"{class_name}_{k}"] = v
    
    return ret_metrics
# plotting imports
import matplotlib
import os
matplotlib.use('Agg')  # no display
import matplotlib.pyplot as plt

# try to import sklearn for PR curve calculation; if missing we'll skip PR plots
try:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
def save_plots(exp_path, train_losses, val_losses, miou_list, pixel_pr=None, dpi=1000):
    os.makedirs(exp_path, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_path, 'loss_curve.png'), dpi=dpi)
    plt.close()

    # IoU curve
    plt.figure()
    plt.plot(epochs, miou_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('mIoU per Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(exp_path, 'miou_curve.png'), dpi=dpi)
    plt.close()

    # PR curves if provided (image_pr and pixel_pr are tuples (prec,rec,ap))

    if pixel_pr is not None:
        prec, rec, ap = pixel_pr
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Pixel-level PR curve (AP={ap:.4f})')
        plt.grid(True)
        plt.savefig(os.path.join(exp_path, 'pr_curve_pixel.png'), dpi=dpi)
        plt.close()
