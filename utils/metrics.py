import torch
import numpy as np


def calculate_shape_IoU(pred, seg_label, category_id, mapping):
    pred = torch.max(pred.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy()
    seg_label = torch.max(seg_label.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy()
    category_id = torch.max(category_id[:, :, 0], dim=1)[1].detach().cpu().numpy()
    categories = []
    shape_ious = []
    for shape_id in range(category_id.shape[0]):
        hash_code = mapping[str(category_id[shape_id])]
        parts_id = mapping[hash_code]['parts_id']
        categories.append(mapping[hash_code]['category'])
        part_ious = []
        for part in parts_id:
            I = np.sum(np.logical_and(pred[shape_id] == part, seg_label[shape_id] == part))
            U = np.sum(np.logical_or(pred[shape_id] == part, seg_label[shape_id] == part))
            if U == 0:
                iou = 1  # If the union of ground truth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious, categories


def calculate_category_IoU(shape_ious, categories, mapping):
    IoU = {}
    category_IoU = {}
    for hash_code in list(mapping.keys()):
        if len(hash_code) == 8:
            IoU[mapping[hash_code]['category']] = []
    for category, shape_iou in zip(categories, shape_ious):
        IoU[category].append(shape_iou)
    for category in list(IoU.keys()):
        category_IoU[f'{category}_IoU'] = sum(IoU[category]) / len(IoU[category])
    return category_IoU
