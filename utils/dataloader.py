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
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    print('Downloading dataset, please wait...')
    wget.download(url=url, out=saved_path)
    print()
    return str(Path(saved_path, url.split('/')[-1]).resolve())


def unpack_dataset(file, unpacked_path):
    print('Unpacking dataset, please wait...')
    shutil.unpack_archive(file, unpacked_path)
    os.remove(file)
    return str(Path(unpacked_path, os.path.splitext(os.path.basename(file))[0]).resolve())


# def download_shapenet(url, saved_path):
#     if not os.path.exists(saved_path):
#         os.makedirs(saved_path)
#     if not os.path.exists(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')):
#         zipfile = os.path.basename(url)
#         os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
#         os.system('mv %s %s' % ('hdf5_data', os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')))
#         os.system('rm %s' % (zipfile))


def get_shapenet_dataloader(url, saved_path, unpacked_path, mapping, selected_points=1024, seed=0, batch_size=32,
                            shuffle=True, num_workers=1, prefetch=64, pin_memory=True):
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
                                               drop_last=True, prefetch_factor=prefetch, pin_memory=pin_memory)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size, shuffle, num_workers=num_workers,
                                                    drop_last=True, prefetch_factor=prefetch, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle, num_workers=num_workers,
                                              drop_last=True, prefetch_factor=prefetch, pin_memory=pin_memory)
    return train_loader, validation_loader, test_loader


# def load_shapenet(url, saved_path, partition):
#     download_shapenet(url, saved_path)
#     all_data = []
#     all_label = []
#     all_seg = []
#     if partition == 'trainval':
#         file = glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
#                + glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
#     else:
#         file = glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
#     for h5_name in file:
#         f = h5py.File(h5_name, 'r+')
#         data = f['data'][:].astype('float32')
#         label = f['label'][:].astype('float32')
#         seg = f['pid'][:].astype('float32')
#         f.close()
#         all_data.append(data)
#         all_label.append(label)
#         all_seg.append(seg)
#     all_data = np.concatenate(all_data, axis=0)
#     all_label = np.concatenate(all_label, axis=0)
#     all_seg = np.concatenate(all_seg, axis=0)
#     return all_data, all_label, all_seg


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

        # get point cloud seg label
        parts_id = self.mapping[category_hash]['parts_id']
        seg_label_path = os.path.join(self.root, category_hash, 'points_label', f'{pcd_hash}.seg')
        seg_label = np.loadtxt(seg_label_path).astype('float32')
        # shuffle points within one point cloud
        indices = list(range(seg_label.shape[0]))
        np.random.shuffle(indices)
        seg_label = seg_label[indices]
        # get a fixed number of points from every point cloud
        np.random.seed(self.seed)
        if self.selected_points <= len(seg_label):
            choice = np.random.choice(len(seg_label), self.selected_points, replace=False)
        else:
            choice = np.random.choice(len(seg_label), self.selected_points, replace=True)
        seg_label = seg_label[choice]
        # match parts id and convert seg label to one hot
        diff = max(parts_id) - np.max(seg_label)
        seg_label = seg_label + diff
        seg_label = F.one_hot(torch.Tensor(seg_label).long(), 50).to(torch.float32).permute(1, 0)

        # get category one hot
        category_id = self.mapping[category_hash]['category_id']
        category_id = F.one_hot(torch.Tensor([category_id]).long(), 16).to(torch.float32).permute(1, 0)

        # get point cloud
        pcd_path = os.path.join(self.root, category_hash, 'points', f'{pcd_hash}.pts')
        pcd = np.loadtxt(pcd_path)
        pcd = pcd[indices]
        pcd = pcd[choice]
        pcd = torch.Tensor(pcd).to(torch.float32)
        pcd = pcd.permute(1, 0)

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_id.shape == (16, 1)
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
