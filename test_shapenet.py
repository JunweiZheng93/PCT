import wandb
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import os
import numpy as np
from utils import dataloader, metrics
from models import shapenet_model
import torch.multiprocessing as mp
import torch.distributed as dist
import pkbar
from utils.visualization import visualize_shapenet_predictions


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

    # overwrite the default config with previous run config
    run_config = OmegaConf.load(f'{local_path}/usr_config.yaml')
    config = OmegaConf.merge(config, run_config)

    # multiprocessing for ddp
    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.test.ddp.which_gpu).replace(' ', '').replace('[', '').replace(']', '')
        mp.spawn(test, args=(config,), nprocs=config.test.ddp.nproc_this_node, join=True)
    else:
        raise ValueError('Please use GPU for testing!')


def test(local_rank, config):

    rank = config.test.ddp.rank_starts_from + local_rank

    # process initialization
    os.environ['MASTER_ADDR'] = str(config.test.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(config.test.ddp.master_port)
    os.environ['WORLD_SIZE'] = str(config.test.ddp.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # gpu setting
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)  # which gpu is used by current process
    print(f'[init] pid: {os.getpid()} - global rank: {rank} - local rank: {local_rank} - cuda: {config.test.ddp.which_gpu[local_rank]}')

    # get datasets
    if config.datasets.dataset_name == 'shapenet_Yi650M':
        _, _, _, test_set = dataloader.get_shapenet_dataset_Yi650M(config.datasets.url, config.datasets.saved_path, config.datasets.unpack_path, config.datasets.mapping, config.datasets.selected_points, config.datasets.seed)
    elif config.datasets.dataset_name == 'shapenet_AnTao350M':
        _, _, _, test_set = dataloader.get_shapenet_dataset_AnTao350M(config.datasets.url, config.datasets.saved_path, config.datasets.selected_points)
    else:
        raise ValueError('Not implemented!')

    # get sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    test_loader = torch.utils.data.DataLoader(test_set, config.test.dataloader.batch_size_per_gpu, num_workers=config.test.dataloader.num_workers, drop_last=True, prefetch_factor=config.test.dataloader.prefetch, pin_memory=config.test.dataloader.pin_memory, sampler=test_sampler)

    # get model
    my_model = shapenet_model.ShapeNetModel(config.embedding.which_embedding, config.embedding.linear.emb_in, config.embedding.linear.emb_out,
                                            config.embedding.neighbor2point_embedding.K, config.embedding.neighbor2point_embedding.neighbor_selection_method, config.embedding.neighbor2point_embedding.group_type,
                                            config.embedding.neighbor2point_embedding.q_in, config.embedding.neighbor2point_embedding.q_out,
                                            config.embedding.neighbor2point_embedding.k_in, config.embedding.neighbor2point_embedding.k_out,
                                            config.embedding.neighbor2point_embedding.v_in, config.embedding.neighbor2point_embedding.v_out, config.embedding.neighbor2point_embedding.num_heads,
                                            config.embedding.point2point_embedding.q_in, config.embedding.point2point_embedding.q_out,
                                            config.embedding.point2point_embedding.k_in, config.embedding.point2point_embedding.k_out,
                                            config.embedding.point2point_embedding.v_in, config.embedding.point2point_embedding.v_out, config.embedding.point2point_embedding.num_heads,
                                            config.embedding.edgeconv_embedding.K, config.embedding.edgeconv_embedding.neighbor_selection_method, config.embedding.edgeconv_embedding.group_type,
                                            config.embedding.edgeconv_embedding.conv1_in, config.embedding.edgeconv_embedding.conv1_out,
                                            config.embedding.edgeconv_embedding.conv2_in, config.embedding.edgeconv_embedding.conv2_out, config.embedding.edgeconv_embedding.pooling,
                                            config.neighbor2point_block.enable, config.neighbor2point_block.neighbor2point.scale, config.neighbor2point_block.neighbor2point.shared_ca,
                                            config.neighbor2point_block.neighbor2point.concat_ms_inputs, config.neighbor2point_block.neighbor2point.mlp_or_ca, config.neighbor2point_block.neighbor2point.K,
                                            config.neighbor2point_block.neighbor2point.neighbor_selection_method, config.neighbor2point_block.neighbor2point.group_type,
                                            config.neighbor2point_block.neighbor2point.q_in, config.neighbor2point_block.neighbor2point.q_out, config.neighbor2point_block.neighbor2point.k_in,
                                            config.neighbor2point_block.neighbor2point.k_out, config.neighbor2point_block.neighbor2point.v_in,
                                            config.neighbor2point_block.neighbor2point.v_out, config.neighbor2point_block.neighbor2point.num_heads, config.neighbor2point_block.neighbor2point.ff_conv1_channels_in,
                                            config.neighbor2point_block.neighbor2point.ff_conv1_channels_out, config.neighbor2point_block.neighbor2point.ff_conv2_channels_in,
                                            config.neighbor2point_block.neighbor2point.ff_conv2_channels_out, config.edgeconv_block.K, config.edgeconv_block.neighbor_selection_method,
                                            config.edgeconv_block.group_type, config.edgeconv_block.conv1_channel_in, config.edgeconv_block.conv1_channel_out,
                                            config.edgeconv_block.conv2_channel_in, config.edgeconv_block.conv2_channel_out, config.edgeconv_block.pooling, config.point2point_block.enable,
                                            config.point2point_block.use_embedding, config.point2point_block.embedding_channels_in, config.point2point_block.embedding_channels_out,
                                            config.point2point_block.point2point.qkv_channels, config.point2point_block.point2point.ff_conv1_channels_in,
                                            config.point2point_block.point2point.ff_conv1_channels_out, config.point2point_block.point2point.ff_conv2_channels_in,
                                            config.point2point_block.point2point.ff_conv2_channels_out, config.conv_block.channels_in, config.conv_block.channels_out)
    my_model.eval()
    my_model = my_model.to(device)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model)
    map_location = {'cuda:0': f'cuda:{local_rank}'}
    my_model.load_state_dict(torch.load(f'./artifacts/{config.wandb.name}/checkpoint.pt', map_location=map_location))

    # get loss function
    if config.test.label_smoothing:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.test.epsilon)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    # start test
    loss_list = []
    pred_list = []
    seg_label_list = []
    cls_label_list = []
    sample_list = []
    with torch.no_grad():
        if rank == 0:
            print(f'Print Results: {config.test.print_results} - Visualize Predictions: {config.test.visualize_preds.enable} - Visualize Neighbors: {config.test.visualize_neighbors.enable}')
            pbar = pkbar.Pbar(name='Start testing, please wait...', target=len(test_loader))
        for i, (samples, seg_labels, cls_label) in enumerate(test_loader):
            samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
            preds = my_model(samples, cls_label)
            loss = loss_fn(preds, seg_labels)

            # collect the result among all gpus
            pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            seg_label_gather_list = [torch.empty_like(seg_labels).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            cls_label_gather_list = [torch.empty_like(cls_label).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            sample_gather_list = [torch.empty_like(samples).to(device) for _ in range(config.test.ddp.nproc_this_node)]
            torch.distributed.all_gather(pred_gather_list, preds)
            torch.distributed.all_gather(seg_label_gather_list, seg_labels)
            torch.distributed.all_gather(cls_label_gather_list, cls_label)
            torch.distributed.all_gather(sample_gather_list, samples)
            torch.distributed.all_reduce(loss)
            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                seg_labels = torch.concat(seg_label_gather_list, dim=0)
                seg_label_list.append(torch.max(seg_labels.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                cls_label = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(torch.max(cls_label[:, :, 0], dim=1)[1].detach().cpu().numpy())
                samples = torch.concat(sample_gather_list, dim=0)
                sample_list.append(samples.permute(0, 2, 1).detach().cpu().numpy())
                loss /= config.test.ddp.nproc_this_node
                loss_list.append(loss.detach().cpu().numpy())
                pbar.update(i)

    if rank == 0:
        preds = np.concatenate(pred_list, axis=0)
        seg_labels = np.concatenate(seg_label_list, axis=0)
        cls_label = np.concatenate(cls_label_list, axis=0)
        samples = np.concatenate(sample_list, axis=0)

        # calculate metrics
        shape_ious = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
        category_iou = metrics.calculate_category_IoU(shape_ious, cls_label, config.datasets.mapping)
        miou = sum(shape_ious) / len(shape_ious)
        loss = sum(loss_list) / len(loss_list)
        if config.test.print_results:
            print(f'loss: {loss}')
            print(f'mIoU: {miou}')
            for category in list(category_iou.keys()):
                print(f'{category}: {category_iou[category]}')

        # generating visualized prediction files
        if config.test.visualize_preds.enable:
            visualize_shapenet_predictions(config, samples, preds, seg_labels, cls_label, shape_ious)


if __name__ == '__main__':
    main()
