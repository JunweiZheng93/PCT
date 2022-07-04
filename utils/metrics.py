import numpy as np
import torch


def calculate_shape_IoU(pred, seg_label, category_id, mapping):
    category_id_to_hash_code_mapping = {}
    for hash_code in list(mapping.keys()):
        category_id_to_hash_code_mapping[str(mapping[hash_code]['category_id'])] = hash_code
    shape_ious = []
    for shape_id in range(category_id.shape[0]):
        hash_code = category_id_to_hash_code_mapping[str(category_id[shape_id])]
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


# def calculate_shape_IoU(pred, seg_label, category_id, mapping):
#     category_id_to_hash_code_mapping = {}
#     for hash_code in list(mapping.keys()):
#         category_id_to_hash_code_mapping[str(mapping[hash_code]['category_id'])] = hash_code
#     categories = []
#     shape_ious = []
#     for shape_id in range(category_id.shape[0]):
#         hash_code = category_id_to_hash_code_mapping[str(category_id[shape_id])]
#         parts_id = mapping[hash_code]['parts_id']
#         categories.append(mapping[hash_code]['category'])
#         part_ious = []
#         for part in parts_id:
#             I = np.sum(np.logical_and(pred[shape_id] == part, seg_label[shape_id] == part))
#             U = np.sum(np.logical_or(pred[shape_id] == part, seg_label[shape_id] == part))
#             if U == 0:
#                 iou = 1  # If the union of ground truth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / float(U)
#             part_ious.append(iou)
#         shape_ious.append(np.mean(part_ious))
#     return shape_ious, categories


# def calculate_shape_IoU(pred, seg_label, category_id, num_parts, start_index):
#
#     pred = torch.max(pred.permute(0, 2, 1), dim=2)[1].detach()
#     seg_label = torch.max(seg_label.permute(0, 2, 1), dim=2)[1].detach()
#     category_id = torch.max(category_id[:, :, 0], dim=1)[1].detach()
#     parts = num_parts[category_id]
#     start_ids = start_index[category_id]
#     shape_ious = []
#     for i, (part, start_id) in enumerate(zip(parts, start_ids)):
#         part_ious = []
#         for j in torch.arange(part):
#             part_id = start_id + j
#             I = torch.sum(torch.logical_and(pred[i] == part_id, seg_label[i] == part_id))
#             U = torch.sum(torch.logical_or(pred[i] == part_id, seg_label[i] == part_id))
#             if U == 0:
#                 iou = 1.  # If the union of ground truth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / U.to(torch.float32)
#             part_ious.append(iou)
#         shape_ious.append(torch.mean(torch.Tensor(part_ious)))
#     return torch.mean(torch.Tensor(shape_ious).cuda())


def calculate_category_IoU(shape_ious, categories, mapping):
    collections = {}
    category_IoU = {}
    for hash_code in list(mapping.keys()):
        collections[mapping[hash_code]['category']] = []
    for category, shape_iou in zip(categories, shape_ious):
        collections[category].append(shape_iou)
    for category in list(collections.keys()):
        category_IoU[category] = sum(collections[category]) / len(collections[category])  # TODO: zero division bug
    return category_IoU
