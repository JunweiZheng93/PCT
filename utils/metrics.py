import torch
import numpy as np


def calculate_shape_IoU(pred, seg_label, category_id, mapping):
    pred = torch.max(pred.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy()
    seg_label = torch.max(seg_label.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy()
    category_id = torch.max(category_id[:, :, 0], dim=1)[1].detach().cpu().numpy()
    shape_ious = []
    for shape_id in range(category_id.shape[0]):
        hash_code = mapping[str(category_id[shape_id])]
        parts_id = mapping[hash_code]['parts_id']
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
    return shape_ious
