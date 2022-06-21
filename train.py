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

    # initialize wandb
    if config.wandb.enable:
        config_dict = OmegaConf.to_container(config, resolve=True)
        del config_dict['wandb'], config_dict['usr_config']
        wandb.login(key=config.wandb.api_key)
        wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config_dict, name=config.wandb.name)

    # gpu setting
    device = torch.device(f'cuda:{config.train.which_gpu[0]}' if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=UserWarning)

    # get datasets
    train, validation, test = dataloader.get_shapenet_dataloader(config.datasets.url, config.datasets.saved_path, config.datasets.unpack_path, config.datasets.mapping, config.datasets.selected_points, config.datasets.seed,
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
    if config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(my_model.parameters(), lr=config.train.lr)
    elif config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(my_model.parameters(), lr=config.train.lr)
    else:
        raise ValueError(f'only adam or sgd is valid currently, got {config.train.optimizer}')

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

    for epoch in range(config.train.epochs):
        # start training
        my_model.train()
        kbar = pkbar.Kbar(target=len(train), epoch=epoch, num_epochs=config.train.epochs, always_stateful=True)
        train_loss_list = []
        shape_ious = []
        categories = []
        for i, (samples, seg_labels, cls_label) in enumerate(train):
            samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
            preds, train_loss = my_model(samples, cls_label, seg_labels)
            train_loss = torch.sum(train_loss) / config.train.dataloader.batch_size
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_list.append(train_loss.detach())
            shape_iou, category = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
            shape_ious.extend(shape_iou)
            categories.extend(category)
            kbar.update(i)
        current_lr = optimizer.param_groups[0]['lr']
        if config.train.lr_scheduler.enable:
            scheduler.step()
        # calculate metrics
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_miou = sum(shape_ious) / len(shape_ious)
        train_category_iou = metrics.calculate_category_IoU(shape_ious, categories, config.datasets.mapping)
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
            shape_ious = []
            categories = []
            for samples, seg_labels, cls_label in validation:
                samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
                preds, val_loss = my_model(samples, cls_label, seg_labels)
                val_loss = torch.sum(val_loss) / config.train.dataloader.batch_size
                val_loss_list.append(val_loss.detach())
                shape_iou, category = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
                shape_ious.extend(shape_iou)
                categories.extend(category)
            # calculate metrics
            val_loss = sum(val_loss_list) / len(val_loss_list)
            val_miou = sum(shape_ious) / len(shape_ious)
            val_category_iou = metrics.calculate_category_IoU(shape_ious, categories, config.datasets.mapping)
            metric_dict = {'validation': {'shapenet': {'loss': val_loss, 'mIoU': val_miou}}}
            metric_dict['validation']['shapenet'].update(val_category_iou)
            kbar.update(i+1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_mIoU', train_miou), ('val_loss', val_loss), ('val_mIoU', val_miou)])
            if config.wandb.enable:
                wandb.log(metric_dict, commit=True)
        else:
            kbar.update(i+1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_mIoU', train_miou)])
    wandb.finish(quiet=True)


if __name__ == '__main__':
    main()
