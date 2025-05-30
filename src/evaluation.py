import numpy as np
from PIL import Image

def load_gt(gt_path):
    gt = Image.open(gt_path)
    gt = np.array(gt)
    red_mask = (gt[:, :] == np.array([255, 0, 0])).all(axis=-1)
    blue_mask = (gt[:, :] == np.array([0, 0, 255])).all(axis=-1)
    return red_mask | blue_mask

def evaluate_segmentation(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Prediction and ground truth masks must have the same shape.\nCurrent shapes: "
                         f"pred_mask: {pred_mask.shape}, gt_mask: {gt_mask.shape}")

    assert pred_mask.dtype == bool and gt_mask.dtype == bool, \
        "Both prediction and ground truth masks must be boolean arrays."

    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score