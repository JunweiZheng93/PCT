import ssl
import shutil
import wget
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import json
import numpy as np


def download_dataset(url, saved_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    print('Downloading dataset, please wait...')
    wget.download(url=url, out=saved_path)
    print()
    return str(Path(saved_path, url.split('/')[-1]).resolve())


def unpack_dataset(file, unpacked_path):
    print('Unpacking dataset, please wait...')
    shutil.unpack_archive(file, unpacked_path)
    os.remove(file)
    return str(Path(unpacked_path, os.path.splitext(os.path.basename(file))[0]).resolve())


def get_shapenet_dataloader(url, saved_path, unpacked_path, mapping, selected_points=2500, seed=0, batch_size=64,
                            shuffle=True, num_workers=1, drop_last=False, prefetch=64):
    # check if dataset already exists
    path = Path(unpacked_path, 'shapenetcore_partanno_segmentation_benchmark_v0')
    if not path.exists():
        file = download_dataset(url, saved_path)
        dataset_path = unpack_dataset(file, unpacked_path)
    else:
        dataset_path = str(path.resolve())

    # get datasets json files
    train_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_train_file_list.json')
    validation_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_val_file_list.json')
    test_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_test_file_list.json')

    # get datasets
    train_set = ShapeNet(dataset_path, train_json, mapping, selected_points, seed)
    validation_set = ShapeNet(dataset_path, validation_json, mapping, selected_points, seed)
    test_set = ShapeNet(dataset_path, test_json, mapping, selected_points, seed)

    # get dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle, num_workers=num_workers,
                                               drop_last=drop_last, prefetch_factor=prefetch)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size, shuffle, num_workers=num_workers,
                                                    drop_last=drop_last, prefetch_factor=prefetch)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle, num_workers=num_workers,
                                              drop_last=drop_last, prefetch_factor=prefetch)
    return train_loader, validation_loader, test_loader


def get_modelnet_dataloader():
    pass


def get_s3dis_dataloader():
    pass


class ShapeNet(torch.utils.data.Dataset):
    def __init__(self, root, json_path, mapping, selected_points, seed):
        self.root = root
        self.mapping = mapping
        self.selected_points = selected_points
        self.seed = seed
        with open(json_path, 'r') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        category_hash, pcd_hash = sample.split('/')[1:]

        # get category one hot
        category_id = self.mapping[category_hash]['category_id']
        category_id = F.one_hot(torch.Tensor([category_id]).long(), 16).to(torch.float32)

        # get point cloud seg label
        parts_id = self.mapping[category_hash]['parts_id']
        seg_label_path = os.path.join(self.root, category_hash, 'points_label', f'{pcd_hash}.seg')
        seg_label = np.loadtxt(seg_label_path).astype('float32')
        # get a fixed number of points from every point cloud
        np.random.seed(self.seed)
        choice = np.random.choice(len(seg_label), self.selected_points, replace=True)  # TODO: check min points of pcd in ShapeNet, replace=False?
        seg_label = seg_label[choice]
        # match parts id and convert seg label to one hot
        diff = max(parts_id) - np.max(seg_label)
        seg_label = seg_label + diff
        seg_label = F.one_hot(torch.Tensor(seg_label).long(), 50).to(torch.float32)

        # get point cloud
        pcd_path = os.path.join(self.root, category_hash, 'points', f'{pcd_hash}.pts')
        pcd = torch.Tensor(np.loadtxt(pcd_path)[choice]).to(torch.float32)

        # pcd.shape == (N, 3)    seg_label.shape == (N, 50)    category_id.shape == (16, )
        return pcd, seg_label, category_id


class ModelNet(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class S3DIS(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
