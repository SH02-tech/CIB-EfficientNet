import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.evaluation import evaluate
import os

def main(config, output_file=None):
    logger = config.get_logger('test')

    # if output file not specified, use the default path inside config directory
    if output_file is None:
        base_path = os.path.dirname(config.resume)
        output_file = os.path.join(base_path, "test_dict.pth")

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_test', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = config.init_obj('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # total_loss = 0.0
    # individual_losses = {}
    # total_metrics = torch.zeros(len(metric_fns))

    # with torch.no_grad():
    #     for i, (data, target) in enumerate(tqdm(data_loader)):
    #         data, target = data.to(device), target.to(device)

    #         if config['arch']['type'] == 'xMIEfficientNet':
    #             output, mi_layer_weights, features  = model(data, output_features=True)
    #         else:
    #             output = model(data)

    #         #
    #         # save sample images, or do something with output here
    #         #

    #         # computing loss, metrics on test set

    #         if config['arch']['type'] == 'xMIEfficientNet':
    #             loss, info_dict = loss_fn(output, target, mi_layer_weights, features)

    #             for key, value in info_dict.items():
    #                 if key not in individual_losses:
    #                     individual_losses[key] = 0.0
    #                 individual_losses[key] += value.item()
    #         else:
    #             loss = loss_fn(output, target)

    #         batch_size = data.shape[0]
    #         total_loss += loss.item() * batch_size
    #         for i, metric in enumerate(metric_fns):
    #             total_metrics[i] += metric(output, target) * batch_size

    eval_dict = evaluate(model, data_loader, device, verbose=True, num_shapes_per_class=5)

    # save eval dict to file
    torch.save(eval_dict, output_file)

    total_loss = eval_dict['loss']
    accuracy = eval_dict['accuracy']

    print("Datos guardados en:", output_file)
    print("Total Loss:", total_loss)
    print("Accuracy:", accuracy)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--output_file', default=None, type=str,
                      help='output file to save results (default: None)')

    parsed = args.parse_args()

    config = ConfigParser.from_args(args)
    main(config, parsed.output_file)
