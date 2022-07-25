import os
import shutil
import numpy as np
import pkbar
import math
from plyfile import PlyData, PlyElement


def visualize_shapenet_predictions(config, samples, preds, seg_labels, cls_label, shape_ious):
    base_path = f'./artifacts/{config.wandb.name}/vis_pred'
    if os.path.exists(f'artifacts/{config.wandb.name}/vis_pred'):
        shutil.rmtree(base_path)
        # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select predictions
    samples_tmp = []
    preds_tmp = []
    seg_gts_tmp = []
    categories_tmp = []
    ious_tmp = []
    for cat_id in range(16):
        samples_tmp.append(samples[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        preds_tmp.append(preds[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        seg_gts_tmp.append(seg_labels[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        categories_tmp.append(np.asarray(categories)[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_preds.num_vis])
    samples = np.concatenate(samples_tmp)
    preds = np.concatenate(preds_tmp)
    seg_labels = np.concatenate(seg_gts_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized prediction files, please wait...', target=len(samples))
    for i, (sample, pred, seg_gt, category, iou) in enumerate(zip(samples, preds, seg_labels, categories, shape_ious)):
        xyzRGB = []
        xyzRGB_gt = []
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        pred_saved_path = f'{cat_path}/{category}_{i}_pred_{math.floor(iou * 1e5)}.ply'
        gt_saved_path = f'{cat_path}/{category}_{i}_gt.ply'
        for xyz, p, gt in zip(sample, pred, seg_gt):
            xyzRGB_tmp = []
            xyzRGB_gt_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend(config.datasets.cmap[str(p)])
            xyzRGB.append(tuple(xyzRGB_tmp))
            xyzRGB_gt_tmp.extend(list(xyz))
            xyzRGB_gt_tmp.extend(config.datasets.cmap[str(gt)])
            xyzRGB_gt.append(tuple(xyzRGB_gt_tmp))
        vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        PlyData([vertex]).write(pred_saved_path)
        vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        PlyData([vertex]).write(gt_saved_path)
        pbar.update(i)
    print(f'Done! All ply files are saved in {base_path}')
