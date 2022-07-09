from utils import dataloader
from models import shapenet_model
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import pkbar
import wandb
from utils import metrics
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda import amp
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

    # multiprocessing for ddp
    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.ddp.which_gpu).replace(' ', '').replace('[', '').replace(']', '')
        mp.spawn(train, args=(config,), nprocs=config.train.ddp.nproc_this_node, join=True)
    else:
        exit('It is almost impossible to train this model using CPU. Please use GPU! Exit.')


def train(local_rank, config):  # the first arg must be local rank for the sake of using mp.spawn(...)

    rank = config.train.ddp.rank_starts_from + local_rank

    if config.wandb.enable and rank == 0:
        # initialize wandb
        wandb.login(key=config.wandb.api_key)
        del config.wandb.api_key, config.test
        config_dict = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config_dict, name=config.wandb.name)
        # cache source code for saving
        OmegaConf.save(config=config, f=f'/tmp/{run.id}_usr_config.yaml', resolve=False)
        os.system(f'cp ./models/shapenet_model.py /tmp/{run.id}_shapenet_model.py')
        os.system(f'cp ./utils/dataloader.py /tmp/{run.id}_dataloader.py')
        os.system(f'cp ./utils/metrics.py /tmp/{run.id}_metrics.py')
        os.system(f'cp ./utils/ops.py /tmp/{run.id}_ops.py')
        os.system(f'cp ./train_shapenet.py /tmp/{run.id}_train_shapenet.py')
        os.system(f'cp ./test_shapenet.py /tmp/{run.id}_test_shapenet.py')

    # process initialization
    os.environ['MASTER_ADDR'] = str(config.train.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(config.train.ddp.master_port)
    os.environ['WORLD_SIZE'] = str(config.train.ddp.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # gpu setting
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)  # which gpu is used by current process
    print(f'[init] pid: {os.getpid()} - global rank: {rank} - local rank: {local_rank} - cuda: {config.train.ddp.which_gpu[local_rank]}')

    # create a scaler for amp
    scaler = amp.GradScaler()

    # get dataset
    if config.datasets.dataset_name == 'shapenet_Yi650M':
        train_set, validation_set, trainval_set, test_set = dataloader.get_shapenet_dataset_Yi650M(config.datasets.url, config.datasets.saved_path, config.datasets.unpack_path, config.datasets.mapping, config.datasets.selected_points, config.datasets.seed)
    elif config.datasets.dataset_name == 'shapenet_AnTao350M':
        train_set, validation_set, trainval_set, test_set = dataloader.get_shapenet_dataset_AnTao350M(config.datasets.url, config.datasets.saved_path, config.datasets.selected_points)
    else:
        raise ValueError('Not implemented!')

    # get sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_set)
    trainval_sampler = torch.utils.data.distributed.DistributedSampler(trainval_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    train_loader = torch.utils.data.DataLoader(train_set, config.train.dataloader.batch_size_per_gpu, num_workers=config.train.dataloader.num_workers, drop_last=True, prefetch_factor=config.train.dataloader.prefetch, pin_memory=config.train.dataloader.pin_memory, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(validation_set, config.train.dataloader.batch_size_per_gpu, num_workers=config.train.dataloader.num_workers, drop_last=True, prefetch_factor=config.train.dataloader.prefetch, pin_memory=config.train.dataloader.pin_memory, sampler=validation_sampler)
    trainval_loader = torch.utils.data.DataLoader(trainval_set, config.train.dataloader.batch_size_per_gpu, num_workers=config.train.dataloader.num_workers, drop_last=True, prefetch_factor=config.train.dataloader.prefetch, pin_memory=config.train.dataloader.pin_memory, sampler=trainval_sampler)
    test_loader = torch.utils.data.DataLoader(test_set, config.train.dataloader.batch_size_per_gpu, num_workers=config.train.dataloader.num_workers, drop_last=True, prefetch_factor=config.train.dataloader.prefetch, pin_memory=config.train.dataloader.pin_memory, sampler=test_sampler)

    # if combine train and validation
    if config.train.dataloader.combine_trainval:
        train_sampler = trainval_sampler
        train_loader = trainval_loader
        validation_loader = test_loader

    # get model
    my_model = shapenet_model.ShapeNetModel(config.embedding.which_embedding, config.embedding.linear.emb_in, config.embedding.linear.emb_out,
                                            config.embedding.point2neighbor_embedding.K, config.embedding.point2neighbor_embedding.neighbor_selection_method, config.embedding.point2neighbor_embedding.group_type,
                                            config.embedding.point2neighbor_embedding.q_in, config.embedding.point2neighbor_embedding.q_out,
                                            config.embedding.point2neighbor_embedding.k_in, config.embedding.point2neighbor_embedding.k_out,
                                            config.embedding.point2neighbor_embedding.v_in, config.embedding.point2neighbor_embedding.v_out, config.embedding.point2neighbor_embedding.num_heads,
                                            config.embedding.point2point_embedding.q_in, config.embedding.point2point_embedding.q_out,
                                            config.embedding.point2point_embedding.k_in, config.embedding.point2point_embedding.k_out,
                                            config.embedding.point2point_embedding.v_in, config.embedding.point2point_embedding.v_out, config.embedding.point2point_embedding.num_heads,
                                            config.embedding.edgeconv_embedding.K, config.embedding.edgeconv_embedding.neighbor_selection_method, config.embedding.edgeconv_embedding.group_type,
                                            config.embedding.edgeconv_embedding.conv1_in, config.embedding.edgeconv_embedding.conv1_out,
                                            config.embedding.edgeconv_embedding.conv2_in, config.embedding.edgeconv_embedding.conv2_out, config.embedding.edgeconv_embedding.pooling,
                                            config.point2neighbor_block.enable, config.point2neighbor_block.point2neighbor.scale, config.point2neighbor_block.point2neighbor.shared_ca, config.point2neighbor_block.point2neighbor.K,
                                            config.point2neighbor_block.point2neighbor.neighbor_selection_method, config.point2neighbor_block.point2neighbor.group_type,
                                            config.point2neighbor_block.point2neighbor.q_in, config.point2neighbor_block.point2neighbor.q_out, config.point2neighbor_block.point2neighbor.k_in,
                                            config.point2neighbor_block.point2neighbor.k_out, config.point2neighbor_block.point2neighbor.v_in,
                                            config.point2neighbor_block.point2neighbor.v_out, config.point2neighbor_block.point2neighbor.num_heads, config.point2neighbor_block.point2neighbor.ff_conv1_channels_in,
                                            config.point2neighbor_block.point2neighbor.ff_conv1_channels_out, config.point2neighbor_block.point2neighbor.ff_conv2_channels_in,
                                            config.point2neighbor_block.point2neighbor.ff_conv2_channels_out, config.edgeconv_block.K, config.edgeconv_block.neighbor_selection_method,
                                            config.edgeconv_block.group_type, config.edgeconv_block.conv1_channel_in, config.edgeconv_block.conv1_channel_out,
                                            config.edgeconv_block.conv2_channel_in, config.edgeconv_block.conv2_channel_out, config.edgeconv_block.pooling, config.point2point_block.enable,
                                            config.point2point_block.use_embedding, config.point2point_block.embedding_channels_in, config.point2point_block.embedding_channels_out,
                                            config.point2point_block.point2point.qkv_channels, config.point2point_block.point2point.ff_conv1_channels_in,
                                            config.point2point_block.point2point.ff_conv1_channels_out, config.point2point_block.point2point.ff_conv2_channels_in,
                                            config.point2point_block.point2point.ff_conv2_channels_out, config.conv_block.channels_in, config.conv_block.channels_out)

    # synchronize bn among gpus
    if config.train.ddp.syn_bn:  #TODO: test performance
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)

    # get ddp model
    my_model = my_model.to(device)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model)

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

    val_miou_list = [0]
    # start training
    for epoch in range(config.train.epochs):
        my_model.train()
        train_sampler.set_epoch(epoch)
        train_loss_list = []
        pred_list = []
        seg_label_list = []
        cls_label_list = []
        if rank == 0:
            kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=config.train.epochs, always_stateful=True)
        for i, (samples, seg_labels, cls_label) in enumerate(train_loader):
            optimizer.zero_grad()
            samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
            if config.train.amp:
                with amp.autocast():
                    preds = my_model(samples, cls_label)
                    train_loss = loss_fn(preds, seg_labels)
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = my_model(samples, cls_label)
                train_loss = loss_fn(preds, seg_labels)
                train_loss.backward()
                optimizer.step()

            # collect the result from all gpus
            pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.train.ddp.nproc_this_node)]
            seg_label_gather_list = [torch.empty_like(seg_labels).to(device) for _ in range(config.train.ddp.nproc_this_node)]
            cls_label_gather_list = [torch.empty_like(cls_label).to(device) for _ in range(config.train.ddp.nproc_this_node)]
            torch.distributed.all_gather(pred_gather_list, preds)
            torch.distributed.all_gather(seg_label_gather_list, seg_labels)
            torch.distributed.all_gather(cls_label_gather_list, cls_label)
            torch.distributed.all_reduce(train_loss)
            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                seg_labels = torch.concat(seg_label_gather_list, dim=0)
                seg_label_list.append(torch.max(seg_labels.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                cls_label = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(torch.max(cls_label[:, :, 0], dim=1)[1].detach().cpu().numpy())
                train_loss /= config.train.ddp.nproc_this_node
                train_loss_list.append(train_loss.detach().cpu().numpy())
                kbar.update(i)

        # decay lr
        current_lr = optimizer.param_groups[0]['lr']
        if config.train.lr_scheduler.enable:
            scheduler.step()

        # calculate metrics
        if rank == 0:
            preds = np.concatenate(pred_list, axis=0)
            seg_labels = np.concatenate(seg_label_list, axis=0)
            cls_label = np.concatenate(cls_label_list, axis=0)
            shape_ious = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
            train_miou = sum(shape_ious) / len(shape_ious)
            train_loss = sum(train_loss_list) / len(train_loss_list)

        # log results
        if rank == 0:
            metric_dict = {'shapenet_train': {'lr': current_lr, 'loss': train_loss, 'mIoU': train_miou}}
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
                for samples, seg_labels, cls_label in validation_loader:
                    samples, seg_labels, cls_label = samples.to(device), seg_labels.to(device), cls_label.to(device)
                    preds = my_model(samples, cls_label)
                    val_loss = loss_fn(preds, seg_labels)

                    # collect the result among all gpus
                    pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.train.ddp.nproc_this_node)]
                    seg_label_gather_list = [torch.empty_like(seg_labels).to(device) for _ in range(config.train.ddp.nproc_this_node)]
                    cls_label_gather_list = [torch.empty_like(cls_label).to(device) for _ in range(config.train.ddp.nproc_this_node)]
                    torch.distributed.all_gather(pred_gather_list, preds)
                    torch.distributed.all_gather(seg_label_gather_list, seg_labels)
                    torch.distributed.all_gather(cls_label_gather_list, cls_label)
                    torch.distributed.all_reduce(val_loss)
                    if rank == 0:
                        preds = torch.concat(pred_gather_list, dim=0)
                        pred_list.append(torch.max(preds.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                        seg_labels = torch.concat(seg_label_gather_list, dim=0)
                        seg_label_list.append(torch.max(seg_labels.permute(0, 2, 1), dim=2)[1].detach().cpu().numpy())
                        cls_label = torch.concat(cls_label_gather_list, dim=0)
                        cls_label_list.append(torch.max(cls_label[:, :, 0], dim=1)[1].detach().cpu().numpy())
                        val_loss /= config.train.ddp.nproc_this_node
                        val_loss_list.append(val_loss.detach().cpu().numpy())

            # calculate metrics
            if rank == 0:
                preds = np.concatenate(pred_list, axis=0)
                seg_labels = np.concatenate(seg_label_list, axis=0)
                cls_label = np.concatenate(cls_label_list, axis=0)
                shape_ious = metrics.calculate_shape_IoU(preds, seg_labels, cls_label, config.datasets.mapping)
                val_miou = sum(shape_ious) / len(shape_ious)
                val_loss = sum(val_loss_list) / len(val_loss_list)

            # log results
            if rank == 0:
                metric_dict = {'shapenet_val': {'loss': val_loss, 'mIoU': val_miou}}
                kbar.update(i+1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_mIoU', train_miou), ('val_loss', val_loss), ('val_mIoU', val_miou)])
                if config.wandb.enable:
                    wandb.log(metric_dict, commit=True)
                    # save model
                    if val_miou >= max(val_miou_list):
                        state_dict = my_model.state_dict()
                        torch.save(state_dict, f'/tmp/{run.id}_checkpoint.pt')
                    val_miou_list.append(val_miou)
        else:
            if rank == 0:
                kbar.update(i+1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_mIoU', train_miou)])

    # save artifacts to wandb server
    if config.wandb.enable and rank == 0:
        artifacts = wandb.Artifact(config.wandb.name, type='runs')
        artifacts.add_file(f'/tmp/{run.id}_usr_config.yaml', name='usr_config.yaml')
        artifacts.add_file(f'/tmp/{run.id}_shapenet_model.py', name='shapenet_model.py')
        artifacts.add_file(f'/tmp/{run.id}_dataloader.py', name='dataloader.py')
        artifacts.add_file(f'/tmp/{run.id}_metrics.py', name='metrics.py')
        artifacts.add_file(f'/tmp/{run.id}_ops.py', name='ops.py')
        artifacts.add_file(f'/tmp/{run.id}_train_shapenet.py', name='train_shapenet.py')
        artifacts.add_file(f'/tmp/{run.id}_test_shapenet.py', name='test_shapenet.py')
        artifacts.add_file(f'/tmp/{run.id}_checkpoint.pt', name='checkpoint.pt')
        run.log_artifact(artifacts)
        wandb.finish(quiet=True)


if __name__ == '__main__':
    main()
