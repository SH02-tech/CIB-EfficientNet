import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer, TrainerXMI
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_train', module_data)
    valid_data_loader = config.init_obj('data_loader_val', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    if config['arch'].get('pretrained', False):
        efficient_weights = filter(
            lambda p: p.requires_grad and not any(n.startswith('classifier') for n, param in model.named_parameters() if param is p),
            model.parameters()
        )
        classifier_weights = filter(
            lambda p: p.requires_grad and any(n.startswith('classifier') for n, param in model.named_parameters() if param is p),
            model.parameters()
        )

        dict_weights_optimization = [
            {'params': efficient_weights, 'lr': config['optimizer']['args']['lr'] * config['arch']['pretrained']['reduce_factor']},
            {'params': classifier_weights, 'lr': config['optimizer']['args']['lr']}
        ]

        optimizer = config.init_obj('optimizer', torch.optim, dict_weights_optimization)
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    if config['trainer'].get('loss_type', '') == 'multiloss':
        trainer = TrainerXMI(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)
    else:
        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--rf', '--reduce_factor'], type=float, target='arch;pretrained;reduce_factor'),
        CustomArgs(['--bs_train', '--batch_size_train'], type=int, target='data_loader_train;args;batch_size'),
        CustomArgs(['--bs_val', '--batch_size_val'], type=int, target='data_loader_val;args;batch_size'),
        CustomArgs(['--bs_test', '--batch_size_test'], type=int, target='data_loader_test;args;batch_size'),
        CustomArgs(['--w_entropy', '--weight_entropy'], type=float, target='loss;args;w_entropy'),
        CustomArgs(['--w_mi', '--weight_mi'], type=float, target='loss;args;w_mi'),
        CustomArgs(['--w_cov', '--weight_cov'], type=float, target='loss;args;w_cov'),
        CustomArgs(['--w_ortho', '--weight_ortho'], type=float, target='loss;args;w_ortho'),
        CustomArgs(['--w_l1', '--weight_l1'], type=float, target='loss;args;w_l1'),
        CustomArgs(['--w_l2', '--weight_l2'], type=float, target='loss;args;w_l2')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
