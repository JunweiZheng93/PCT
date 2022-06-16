from utils import dataloader
from models import shapenet_model
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import pkbar
import warnings


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
    for name, param in my_model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    # get optimizer
    if config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(my_model.parameters(), lr=config.train.lr)
    elif config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(my_model.parameters(), lr=config.train.lr)
    else:
        raise ValueError(f'only adam or sgd is valid currently, got {config.train.optimizer}')

    # start training
    for epoch in range(config.train.epochs):
        my_model.train()
        kbar = pkbar.Kbar(target=len(train), epoch=epoch, num_epochs=config.train.epochs, always_stateful=True)
        loss_list = []
        for i, (samples, seg_labels, cls_label) in enumerate(train):
            samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
            train_loss = torch.sum(my_model(samples, cls_label, seg_labels)) / config.train.dataloader.batch_size
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            loss_list.append(train_loss.detach())
            acc_loss = sum(loss_list) / len(loss_list)
            kbar.update(i+1, values=[("loss", acc_loss)])

        # start validation
        if not (epoch+1) % config.train.validation_freq:
            my_model.eval()
            val_loss_list = []
            for i, (samples, seg_labels, cls_label) in enumerate(validation):
                samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
                val_loss = torch.sum(my_model(samples, cls_label, seg_labels)) / config.train.dataloader.batch_size
                val_loss_list.append(val_loss.detach())
                val_acc_loss = sum(val_loss_list) / len(val_loss_list)
            kbar.add(0, values=[("val_loss", val_acc_loss)])


if __name__ == '__main__':
    main()
