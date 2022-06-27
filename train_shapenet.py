from utils import dataloader
from models import shapenet_model
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import pkbar
import warnings
import wandb
from utils import metrics
import os
import shutil
import copy
import numpy as np


@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main(config):

    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f'Working directory is not the same as project root. Exit.')

    # overwrite the default config with user config
    if config.usr_config:
        usr_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, usr_config)

    if config.wandb.enable:
        # initialize wandb
        wandb.login(key=config.wandb.api_key)
        del config.wandb.api_key
        config_dict = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config_dict, name=config.wandb.name)

    # gpu setting
    device = torch.device(f'cuda:{config.train.which_gpu[0]}' if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=UserWarning)

    # get datasets
    train, validation, _ = dataloader.get_shapenet_dataloader(config.datasets.url, config.datasets.saved_path, config.datasets.unpack_path, config.datasets.mapping, config.datasets.selected_points, config.datasets.seed,
                                                              config.train.dataloader.batch_size, config.train.dataloader.shuffle, config.train.dataloader.num_workers, config.train.dataloader.prefetch, config.train.pin_memory)

    # get model
    my_model = shapenet_model.ShapeNetModel(config.point2neighbor_block.enable, config.point2neighbor_block.use_embedding, config.point2neighbor_block.embedding_channels_in,
                                            config.point2neighbor_block.embedding_channels_out, config.point2neighbor_block.point2neighbor.K,
                                            config.point2neighbor_block.point2neighbor.xyz_or_feature, config.point2neighbor_block.point2neighbor.feature_or_diff,
                                            config.point2neighbor_block.point2neighbor.qkv_channels, config.point2neighbor_block.point2neighbor.ff_conv1_channels_in,
                                            config.point2neighbor_block.point2neighbor.ff_conv1_channels_out, config.point2neighbor_block.point2neighbor.ff_conv2_channels_in,
                                            config.point2neighbor_block.point2neighbor.ff_conv2_channels_out, config.edgeconv_block.K, config.edgeconv_block.xyz_or_feature,
                                            config.edgeconv_block.feature_or_diff, config.edgeconv_block.conv1_channel_in, config.edgeconv_block.conv1_channel_out,
                                            config.edgeconv_block.conv2_channel_in, config.edgeconv_block.conv2_channel_out, config.point2point_block.enable,
                                            config.point2point_block.use_embedding, config.point2point_block.embedding_channels_in, config.point2point_block.embedding_channels_out,
                                            config.point2point_block.point2point.qkv_channels, config.point2point_block.point2point.ff_conv1_channels_in,
                                            config.point2point_block.point2point.ff_conv1_channels_out, config.point2point_block.point2point.ff_conv2_channels_in,
                                            config.point2point_block.point2point.ff_conv2_channels_out, config.conv_block.channels_in, config.conv_block.channels_out)
    my_model.to(device)
    if torch.cuda.is_available():
        my_model = torch.nn.DataParallel(my_model, device_ids=config.train.which_gpu)

    # get optimizer
    if config.train.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(my_model.parameters(), lr=config.train.lr, weight_decay=1e-4)
    elif config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(my_model.parameters(), lr=config.train.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise ValueError('Not implemented!')

    # get lr scheduler
    if config.train.lr_scheduler.enable:
        if config.train.lr_scheduler.which == 'stepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train.lr_scheduler.stepLR.decay_step, gamma=config.train.lr_scheduler.stepLR.gamma)
        elif config.train.lr_scheduler.which == 'expLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_scheduler.expLR.gamma)
        elif config.train.lr_scheduler.which == 'cosLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.lr_scheduler.cosLR.T_max, eta_min=config.train.lr_scheduler.cosLR.eta_min)
        else:
            raise ValueError('Not implemented!')

    # get loss function
    if config.train.label_smoothing:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.train.epsilon)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(config.train.epochs):
        # start training
        my_model.train()
        kbar = pkbar.Kbar(target=len(train), epoch=epoch, num_epochs=config.train.epochs, always_stateful=True)
        train_loss_list = []
        pred_list = []
        seg_label_list = []
        cls_label_list = []
        for i, (samples, seg_labels, cls_label) in enumerate(train):
            samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
            preds = my_model(samples, cls_label)
            train_loss = loss_fn(preds, seg_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_list.append(train_loss.detach().cpu().numpy())
            pred_list.append(torch.max(preds.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
            seg_label_list.append(torch.max(seg_labels.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
            cls_label_list.append(torch.max(cls_label[:, :, 0], dim=1)[1].detach().cpu().numpy())
            kbar.update(i)
        current_lr = optimizer.param_groups[0]['lr']
        if config.train.lr_scheduler.enable:
            scheduler.step()

        # calculate metrics
        preds = np.concatenate(pred_list, axis=0)
        seg_labels = np.concatenate(seg_label_list, axis=0)
        cls_label = np.concatenate(cls_label_list, axis=0)
        shape_ious, categories = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
        train_category_iou = metrics.calculate_category_IoU(shape_ious, categories, config.datasets.mapping)
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_miou = sum(shape_ious) / len(shape_ious)

        # log results
        metric_dict = {'train': {'shapenet': {'lr': current_lr, 'loss': train_loss, 'mIoU': train_miou}}}
        metric_dict['train']['shapenet'].update(train_category_iou)
        if config.wandb.enable and (epoch+1) % config.train.validation_freq:
            wandb.log(metric_dict, commit=True)
        elif config.wandb.enable and not (epoch+1) % config.train.validation_freq:
            wandb.log(metric_dict, commit=False)

        # start validation
        if not (epoch+1) % config.train.validation_freq:
            my_model.eval()
            val_loss_list = []
            pred_list = []
            seg_label_list = []
            cls_label_list = []
            with torch.no_grad():
                for samples, seg_labels, cls_label in validation:
                    samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
                    preds = my_model(samples, cls_label)
                    val_loss = loss_fn(preds, seg_labels)
                    val_loss_list.append(val_loss.detach().cpu().numpy())
                    pred_list.append(torch.max(preds.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                    seg_label_list.append(torch.max(seg_labels.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                    cls_label_list.append(torch.max(cls_label[:, :, 0], dim=1)[1].detach().cpu().numpy())

            # calculate metrics
            preds = np.concatenate(pred_list, axis=0)
            seg_labels = np.concatenate(seg_label_list, axis=0)
            cls_label = np.concatenate(cls_label_list, axis=0)
            shape_ious, categories = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
            val_category_iou = metrics.calculate_category_IoU(shape_ious, categories, config.datasets.mapping)
            val_loss = sum(val_loss_list) / len(val_loss_list)
            val_miou = sum(shape_ious) / len(shape_ious)

            # log results
            metric_dict = {'validation': {'shapenet': {'loss': val_loss, 'mIoU': val_miou}}}
            metric_dict['validation']['shapenet'].update(val_category_iou)
            kbar.update(i+1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_mIoU', train_miou), ('val_loss', val_loss), ('val_mIoU', val_miou)])
            if config.wandb.enable:
                wandb.log(metric_dict, commit=True)
                # save model
                if epoch >= 1 and val_miou > val_miou_old:
                    state_dict = my_model.state_dict()
                    for key in list(state_dict.keys()):
                        if key.startswith('module'):  # the keys will start with 'module' when using gpu
                            state_dict[key[7:]] = state_dict.pop(key)
                    torch.save(state_dict, './wandb/latest-run/files/checkpoint.pt')
                val_miou_old = copy.deepcopy(val_miou)
        else:
            kbar.update(i+1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_mIoU', train_miou)])

    # save artifacts to wandb server
    if config.wandb.enable:
        artifacts = wandb.Artifact(config.wandb.name, type='runs')
        # add configuration file
        OmegaConf.save(config=config, f='./usr_config.yaml', resolve=False)
        artifacts.add_file('./usr_config.yaml')
        os.remove('./usr_config.yaml')
        # add model architecture
        artifacts.add_file('./models/shapenet_model.py')
        # add model weights
        artifacts.add_file('./wandb/latest-run/files/checkpoint.pt')
        # log artifacts
        run.log_artifact(artifacts)
        wandb.finish(quiet=True)
        shutil.rmtree('./wandb/')


if __name__ == '__main__':
    main()
