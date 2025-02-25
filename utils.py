from datetime import datetime
import json
import logging
import os
import pytz
import shutil
import sys
from tqdm import tqdm
from models.components.helpers import *
from scipy.ndimage import gaussian_filter1d

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
matplotlib.rcParams['savefig.dpi'] = 300  # Uncomment for higher plot resolutions

logger = logging.getLogger('TS.utils')


class Params:
    """
    Class that loads hyperparameters from a json file as a dictionary (also support nested dicts).
    Example:
    params = Params(json_path)

    # access key-value pairs
    params.learning_rate
    params['learning_rate']

    # change the value of learning_rate in params
    params.learning_rate = 0.5
    params['learning_rate'] = 0.5

    # print params
    print(params)

    # combine two json files
    params.update(Params(json_path2))
    """

    def __init__(self, json_path=None):
        if json_path is not None and os.path.isfile(json_path):
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        else:
            self.__dict__ = {}

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path=None, params=None):
        """Loads parameters from json file"""
        if json_path is not None:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        elif params is not None:
            self.__dict__.update(vars(params))
        else:
            raise Exception('One of json_path and params must be provided in Params.update()!')

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return getattr(self, str(key))

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __str__(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4, ensure_ascii=False)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    """
    _logger = logging.getLogger('VM')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%m/%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)
            self.setStream(tqdm)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))
    # handler = logging.StreamHandler(stream=sys.stdout)
    # _logger.addHandler(handler)

    # https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python?noredirect=1&lq=1
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            _logger.info('=*=*=*= Keyboard interrupt =*=*=*=')
            return

        _logger.error("Exception --->", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, global_step, checkpoint, is_best=False):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        global_step: (int) number of updates performed
        checkpoint: (string) folder where parameters are to be saved
        is_best: (boolean)
    """
    if is_best:
        filepath = os.path.join(checkpoint, f'best.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'latest.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved to {filepath}')


def load_checkpoint(file_dir, restore_file, model, optimizer=None, loss=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint = os.path.join(f'runs/{file_dir}', restore_file + '.pth.tar')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    else:
        logger.info(f'Restoring parameters from {checkpoint}')
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    if loss:
        loss = np.load(os.path.join(file_dir, restore_file + '_loss.npy'))
        return checkpoint['global_step'], loss
    else:
        return checkpoint['global_step']


def plot_all_epoch(variable1, variable2, save_name, plot_title, location='./figures/', plot_start=0):
    num_samples = variable1.shape[0]
    if num_samples > plot_start:
        x = np.arange(start=plot_start, stop=num_samples)
        f = plt.figure()
        plt.title(plot_title)
        ax1 = plt.gca()
        line1, = ax1.plot(x, variable1[plot_start:num_samples])
        ax2 = ax1.twinx()
        line2, = ax2.plot(x, variable2[plot_start:num_samples], c='r')
        plt.legend((line1, line2), ("Test metrics", "Validation metrics"))
        ax1.set_ylabel("Test")
        ax2.set_ylabel("Validation")
        f.savefig(os.path.join(location, save_name + '_summary.png'))
        plt.close()


def name_with_datetime():
    now = datetime.now(tz=pytz.utc)
    now = now.astimezone(pytz.timezone('US/Pacific'))
    return now.strftime("%Y-%m-%d_%H:%M:%S")


def cum_by_axis1(input_x):
    cum_input = np.zeros(input_x.shape)
    for i in range(cum_input.shape[1]):
        cum_input[:, i] = np.sum(input_x[:, :(i + 1)], axis=1)
    return cum_input


def plot_all_loss(loss_summary, save_name, plot_title, location='./figures/'):
    gaussian_window_size = 3
    loss_cum = cum_by_axis1(loss_summary)
    loss_cum = gaussian_filter1d(loss_cum, gaussian_window_size, axis=0)
    num_loss = loss_cum.shape[1]
    color_list = ['b', 'r', 'g', 'm', 'y']
    loss_list = ['loss_1', 'loss_2', 'loss_3', 'loss_4', 'loss_5']
    f = plt.figure()
    plt.title(plot_title)
    num_batches = loss_cum.shape[0]
    if num_batches > 10000:
        pack_size = num_batches // 10000
        x = np.arange(num_batches)[0:num_batches:pack_size]
        plt.fill_between(x, 0, loss_cum[0:num_batches:pack_size, 0], color='b', alpha=0.2, label='loss_1')
        for i in range(num_loss - 1):
            plt.fill_between(x, loss_cum[0:num_batches:pack_size, i], loss_cum[0:num_batches:pack_size, i + 1],
                             color=color_list[i + 1], alpha=0.2, label=loss_list[i + 1])
    else:
        x = np.arange(num_batches)
        plt.fill_between(x, 0, loss_cum[:, 0], color='b', alpha=0.2, label='loss_1')
        for i in range(num_loss - 1):
            plt.fill_between(x, loss_cum[:, i], loss_cum[:, i + 1], color=color_list[i + 1], alpha=0.2,
                             label=loss_list[i + 1])
    plt.yscale('log')
    plt.legend()
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()


def calc_metrics_all(predict_all, gt_all, params):
    summary_metric = dict()
    summary_metric['mae'] = accuracy_MAE(predict_all, gt_all, params.missing_value)
    summary_metric['mse'] = accuracy_MSE(predict_all, gt_all, params.missing_value)
    return summary_metric


# for plots
def batch_metrics(sample_params, labels, missing_value=0):
    metric = dict()
    metric['p50'] = accuracy_ROU_(0.5, sample_params[:, :, 0], labels, missing_value)
    metric['p10'] = accuracy_ROU_(0.1, sample_params[:, :, 1], labels, missing_value)
    metric['p90'] = accuracy_ROU_(0.9, sample_params[:, :, 2], labels, missing_value)
    return metric


def final_metrics_list_to_int(summary_metric):
    final_metric = dict()
    for metric, value_array in summary_metric.items():
        for i in range(value_array.shape[0]):
            final_metric[metric + ' ' + str(i)] = value_array[i]
    return final_metric


def model_list():
    """
    List all available models found under ./model.
    """
    files = os.listdir('./model')
    files = [name.replace('.py', '') for name in files if name.endswith('.py')]
    return files
