import wandb
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import warnings
import os
import importlib


@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main(config):

    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f'Working directory is not the same as project root. Exit.')

    # get test configurations
    if config.usr_config:
        test_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, test_config)

    # download artifacts
    if config.wandb.enable:
        wandb.login(key=config.wandb.api_key)
        api = wandb.Api()
        artifact = api.artifact(f'{config.wandb.entity}/{config.wandb.project}/{config.wandb.name}:latest')
        local_path = f'./artifacts/{config.wandb.name}'
        artifact.download(root=local_path)
        os.system(f'touch ./artifacts/__init__.py')
        os.system(f'touch {local_path}/__init__.py')
    else:
        raise ValueError('W&B is not enabled!')

    # import modules saved by previous run
    dataloader = importlib.import_module(f"artifacts.{config.wandb.name}.dataloader")
    shapenet_model = importlib.import_module(f"artifacts.{config.wandb.name}.shapenet_model")
    metrics = importlib.import_module(f"artifacts.{config.wandb.name}.metrics")

    # overwrite the default config with previous run config
    run_config = OmegaConf.load(f'{local_path}/usr_config.yaml')
    for key in list(test_config.keys()):
        del run_config[key]
    config = OmegaConf.merge(config, run_config)

    # gpu setting
    device = torch.device(f'cuda:{config.test.which_gpu[0]}' if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=UserWarning)

    # get datasets
    _, _, test = dataloader.get_shapenet_dataloader(config.datasets.url, config.datasets.saved_path, config.datasets.unpack_path, config.datasets.mapping, config.datasets.selected_points, config.datasets.seed,
                                                    config.test.dataloader.batch_size, config.test.dataloader.shuffle, config.test.dataloader.num_workers, config.test.dataloader.prefetch, config.test.pin_memory,
                                                    config.test.dataloader.smoothing, config.test.dataloader.epsilon)

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
    my_model.load_state_dict(torch.load(f'{local_path}/checkpoint.pt'))
    my_model.to(device)
    my_model.eval()
    if torch.cuda.is_available():
        my_model = torch.nn.DataParallel(my_model, device_ids=config.test.which_gpu)

    # start test
    loss_list = []
    shape_ious = []
    categories = []
    print('Start testing, please wait...')
    for samples, seg_labels, cls_label in test:
        samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
        preds, loss = my_model(samples, cls_label, seg_labels)
        loss = torch.sum(loss) / config.test.dataloader.batch_size
        loss_list.append(loss.detach())
        shape_iou, category = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
        shape_ious.extend(shape_iou)
        categories.extend(category)

    # calculate metrics
    loss = sum(loss_list) / len(loss_list)
    miou = sum(shape_ious) / len(shape_ious)
    category_iou = metrics.calculate_category_IoU(shape_ious, categories, config.datasets.mapping)
    print('='*60)
    print(f'loss: {loss}')
    print(f'mIoU: {miou}')
    for category in list(category_iou.keys()):
        print(f'{category}: {category_iou[category]}')
    print('='*60)


if __name__ == '__main__':
    main()
