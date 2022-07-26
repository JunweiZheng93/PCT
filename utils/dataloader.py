import ssl
import shutil
import wget
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import json
import numpy as np
import h5py
import glob
from utils import data_augmentation


# ================================================================================
# Yi650M shapenet dataloader

def download_shapenet_Yi650M(url, saved_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    print('Downloading dataset, please wait...')
    wget.download(url=url, out=saved_path)
    print()
    return str(Path(saved_path, url.split('/')[-1]).resolve())


def unpack_shapenet_Yi650M(file, unpacked_path):
    print('Unpacking dataset, please wait...')
    shutil.unpack_archive(file, unpacked_path)
    os.remove(file)
    return str(Path(unpacked_path, os.path.splitext(os.path.basename(file))[0]).resolve())


class ShapeNet_Yi650M(torch.utils.data.Dataset):
    def __init__(self, root, json_path, mapping, selected_points, seed):
        self.root = root
        self.mapping = mapping
        self.selected_points = selected_points
        self.seed = seed
        self.samples = []
        for each_path in json_path:
            with open(each_path, 'r') as f:
                self.samples.extend(json.load(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        category_hash, pcd_hash = sample.split('/')[1:]

        # get point cloud seg label
        parts_id = self.mapping[category_hash]['parts_id']
        seg_label_path = os.path.join(self.root, category_hash, 'points_label', f'{pcd_hash}.seg')
        seg_label = np.loadtxt(seg_label_path).astype('float32')
        # get a fixed number of points from every point cloud
        np.random.seed(self.seed)
        if self.selected_points <= len(seg_label):
            choice = np.random.choice(len(seg_label), self.selected_points, replace=False)
        else:
            choice = np.random.choice(len(seg_label), self.selected_points, replace=True)
        seg_label = seg_label[choice]
        # shuffle points within one point cloud
        indices = list(range(seg_label.shape[0]))
        np.random.shuffle(indices)
        seg_label = seg_label[indices]
        # match parts id and convert seg label to one hot
        diff = min(parts_id) - 1
        seg_label = seg_label + diff
        seg_label = F.one_hot(torch.Tensor(seg_label).long(), 50).to(torch.float32).permute(1, 0)

        # get category one hot
        category_id = self.mapping[category_hash]['category_id']
        category_id = F.one_hot(torch.Tensor([category_id]).long(), 16).to(torch.float32).permute(1, 0)

        # get point cloud
        pcd_path = os.path.join(self.root, category_hash, 'points', f'{pcd_hash}.pts')
        pcd = np.loadtxt(pcd_path)
        pcd = pcd[choice]
        pcd = pcd[indices]
        pcd = torch.Tensor(pcd).to(torch.float32)
        pcd = pcd.permute(1, 0)

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_id.shape == (16, 1)
        return pcd, seg_label, category_id


def get_shapenet_dataset_Yi650M(url, saved_path, unpacked_path, mapping, selected_points=1024, seed=0):

    # check if dataset already exists
    path = Path(unpacked_path, 'shapenetcore_partanno_segmentation_benchmark_v0')
    if not path.exists():
        file = download_shapenet_Yi650M(url, saved_path)
        dataset_path = unpack_shapenet_Yi650M(file, unpacked_path)
    else:
        dataset_path = str(path.resolve())

    # get datasets json files
    train_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_train_file_list.json')
    validation_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_val_file_list.json')
    test_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_test_file_list.json')

    # get datasets
    train_set = ShapeNet_Yi650M(dataset_path, [train_json], mapping, selected_points, seed)
    validation_set = ShapeNet_Yi650M(dataset_path, [validation_json], mapping, selected_points, seed)
    trainval_set = ShapeNet_Yi650M(dataset_path, [train_json, validation_json], mapping, selected_points, seed)
    test_set = ShapeNet_Yi650M(dataset_path, [test_json], mapping, selected_points, seed)

    return train_set, validation_set, trainval_set, test_set


# ================================================================================
# AnTao350M shapenet dataloader

def download_shapenet_AnTao350M(url, saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    if not os.path.exists(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')):
        zipfile = os.path.basename(url)
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


class ShapeNet_AnTao350M(torch.utils.data.Dataset):
    def __init__(self, saved_path, partition, selected_points, augmentation, jitter, std, clip, rotate, which_axis,
                 angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                 x_scale_range, y_scale_range, z_scale_range):
        self.selected_points = selected_points
        self.augmentation = augmentation
        if augmentation:
            self.augmentation_list = []
            if jitter:
                self.augmentation_list.append([data_augmentation.jitter, [std, clip]])
            if rotate:
                self.augmentation_list.append([data_augmentation.rotate, [which_axis, angle_range]])
            if translate:
                self.augmentation_list.append([data_augmentation.translate, [x_translate_range, y_translate_range, z_translate_range]])
            if anisotropic_scale:
                self.augmentation_list.append([data_augmentation.anisotropic_scale, [x_scale_range, y_scale_range, z_scale_range]])
            if not jitter and not rotate and not translate and not anisotropic_scale:
                raise ValueError('At least one kind of data augmentation should be applied!')
        self.all_pcd = []
        self.all_cls_label = []
        self.all_seg_label = []
        if partition == 'trainval':
            file = glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
                   + glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
        else:
            file = glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*%s*.h5' % partition))
        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            pcd = f['data'][:].astype('float32')
            cls_label = f['label'][:].astype('int64')
            seg_label = f['pid'][:].astype('int64')
            f.close()
            self.all_pcd.append(pcd)
            self.all_cls_label.append(cls_label)
            self.all_seg_label.append(seg_label)
        self.all_pcd = np.concatenate(self.all_pcd, axis=0)
        self.all_cls_label = np.concatenate(self.all_cls_label, axis=0)
        self.all_seg_label = np.concatenate(self.all_seg_label, axis=0)

    def __len__(self):
        return self.all_cls_label.shape[0]

    def __getitem__(self, index):
        # get category one hot
        category_id = self.all_cls_label[index, 0]
        category_onehot = F.one_hot(torch.Tensor([category_id]).long(), 16).to(torch.float32).permute(1, 0)

        # get point cloud seg label
        seg_label = self.all_seg_label[index].astype('float32')
        # get a fixed number of points from every point cloud
        seg_label = seg_label[:self.selected_points]
        # shuffle points within one point cloud
        indices = list(range(seg_label.shape[0]))
        np.random.shuffle(indices)
        seg_label = seg_label[indices]
        # match parts id and convert seg label to one hot
        seg_label = F.one_hot(torch.Tensor(seg_label).long(), 50).to(torch.float32).permute(1, 0)

        # get point cloud
        pcd = self.all_pcd[index]
        pcd = pcd[:self.selected_points]
        pcd = pcd[indices]
        if self.augmentation:
            choice = np.random.choice(len(self.augmentation_list))
            augmentation, params = self.augmentation_list[choice]
            pcd = augmentation(pcd, *params)
        pcd = torch.Tensor(pcd).to(torch.float32)
        pcd = pcd.permute(1, 0)

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_id.shape == (16, 1)
        return pcd, seg_label, category_onehot


def get_shapenet_dataset_AnTao350M(url, saved_path, selected_points, augmentation, jitter, std, clip, rotate, which_axis,
                 angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                 x_scale_range, y_scale_range, z_scale_range):
    # download dataset
    download_shapenet_AnTao350M(url, saved_path)
    # get dataset
    train_set = ShapeNet_AnTao350M(saved_path, 'train', selected_points, augmentation, jitter, std, clip, rotate, which_axis,
                                   angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                   x_scale_range, y_scale_range, z_scale_range)
    validation_set = ShapeNet_AnTao350M(saved_path, 'val', selected_points, False, jitter, std, clip, rotate, which_axis,
                                        angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                        x_scale_range, y_scale_range, z_scale_range)
    trainval_set = ShapeNet_AnTao350M(saved_path, 'trainval', selected_points, augmentation, jitter, std, clip, rotate, which_axis,
                                      angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                      x_scale_range, y_scale_range, z_scale_range)
    test_set = ShapeNet_AnTao350M(saved_path, 'test', selected_points, False, jitter, std, clip, rotate, which_axis,
                                  angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                  x_scale_range, y_scale_range, z_scale_range)
    return train_set, validation_set, trainval_set, test_set


# ================================================================================
# modelnet dataloader

def get_modelnet_dataloader():
    pass


class ModelNet(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


# ================================================================================
# s3dis dataloader

def get_s3dis_dataloader():
    pass


class S3DIS(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
