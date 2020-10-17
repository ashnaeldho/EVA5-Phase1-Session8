import os
import torch
import collections
import configparser
from torch import nn
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as f
from torchsummary import summary
import CONSTANTS as constants

config = configparser.RawConfigParser()
current_path = os.getcwd()

if not os.path.isfile(os.path.join(current_path, "config.cfg")):
    raise FileNotFoundError('\n Configuration file "config.cfg" not found in the root directory of the project.')

config.read(os.path.join(current_path, "config.cfg"))


def get_config_details():
    """
    Reads the configuration file and sets the attribute for this function the first time so subsequent calls avoid re-reading config file

    :return: 2 level dict, where outer key is section name and outer value is a dict,
             inner key is parameter name corresponding to outer key and value is the corresponding value for inner key.
    """
    if not hasattr(get_config_details, 'config_dict'):
        get_config_details.config_dict = collections.defaultdict(dict)

        for section in config.sections():
            get_config_details.config_dict[section] = dict(config.items(section))

    return get_config_details.config_dict


def get_combos_and_trackers():
    """Makes a tracker for every combination given in config file.
    :returns a tuple of list of all combinations in string and tracker dict with key as combination name and value as
            dictionaries with 'misclassified', 'train_losses','test_losses','train_accuracy', 'test_accuracy'
            for each combo key"""

    d = {
        'misclassified': [],
        'train_losses': [],
        'test_losses': [],
        'train_accuracy': [],
        'test_accuracy': []
    }

    all_combo_list = get_config_details()[constants.MODEL_CONFIG][constants.COMBOS].split(',')
    tracker = {}
    for item in all_combo_list:
        tracker[item] = deepcopy(d)
    del d
    return all_combo_list, tracker


def check_gpu_availability(seed=101):
    """
    Checks if a GPU is available. Uses :param seed to set seed value with torch if GPU is available.
    :param seed: the seeding value
    :return: cuda flag, type: bool
    """
    cuda = torch.cuda.is_available()
    if cuda:
        print('\n CUDA is available')
        torch.cuda.manual_seed(seed)
    else:
        print("\n No GPU")
    return cuda


def get_device():
    """ :returns the device type available"""
    return 'cuda' if check_gpu_availability() else 'cpu'


class GhostBatchNorm(nn.BatchNorm2d):
    """ from https://github.com/apple/ml-cifar-10-faster/blob/master/utils.py
        Implements Ghost Batch Normalization"""

    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return f.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return f.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


def print_summary(*, model: torch.nn.Module, input_size: tuple):
    """Will utilize torchsummary to print summary of the model"""
    if not (isinstance(model, torch.nn.Module) and (isinstance(input_size, tuple) and len(input_size) == 3)):
        raise Exception("\nCheck the model passed or the input size(must be a tuple of c, h, w)")

    print(summary(model=model, input_size=input_size))


def get_optimizer(*, model):
    """
    Checks for the given Optimizer type, learning rate, momentum and weight decay in configuration and returns the optim
    :param model: the model for which the optimizer will be used
    :returns the optimizer after going through the configuration file for essential parameters for the given optimizer to use
    """

    lr = 0.01
    momentum = wd = 0.0
    nesterov = False
    optim_dict = config[constants.OPTIMIZER]
    regul_dict = config[constants.REGULARIZATION]

    if constants.LR in optim_dict.keys():
        lr = float(optim_dict[constants.LR])
    if constants.L2 in regul_dict.keys():
        wd = float(regul_dict[constants.L2])
    if constants.MOMENTUM in optim_dict.keys():
        momentum = float(optim_dict[constants.MOMENTUM])
    if constants.NESTEROV in optim_dict.keys():
        nesterov = True

    if constants.SGD in optim_dict[constants.OPTIM_TYPE].split()[0].lower():
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)

    if constants.ADAM in optim_dict[constants.OPTIM_TYPE].split()[0].lower():
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def get_scheduler(*, optimizer):
    """
    Gets the scheduler type and other parameters from config and returns the corresponding scheduler
    :param optimizer: the optimizer on which scheduler will run
    :return: scheduler, type either torch.optim.lr_scheduler or None in case no values are given in config
    """
    schdlr_dict = config[constants.SCHEDULER]
    step, gamma = 5, 0.001
    if len(schdlr_dict.keys()) > 0:
        if constants.SCHEDULER_TYPE in schdlr_dict.keys():
            scheduler = schdlr_dict[constants.SCHEDULER_TYPE]

        if constants.STEP in schdlr_dict.keys():
            step = int(schdlr_dict[constants.STEP])

        if constants.GAMMA in schdlr_dict.keys():
            gamma = float(schdlr_dict[constants.GAMMA])

        if constants.STEP_LR in scheduler.lower():
            return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step, gamma=gamma)

        if scheduler.lower() == constants.MULTI_STEP_LR and constants.MILESTONES in schdlr_dict.keys():
            return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=list(map(int, schdlr_dict[
                                                            constants.MILESTONES].strip().split(','))),
                                                        gamma=gamma)

    else:
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1)


def plot(*, title, x_label, y_label, tracker, category):
    """
    Plots and saves the plot with given params. Uses tracker for getting the data and category is for the target.
    :param title: string describing the title of the plot
    :param x_label: str, xlabel
    :param y_label: str, ylabel
    :param tracker: dict, containing data, where outer key is the type and inner key is the category and inner v holds the values
    :param category: str, one of the inner keys for tracker
    :return: None
    """

    for type_, d in tracker.items():
        for k, v in d.items():
            if k.lower() == category:
                x = [*range(len(v))]
                plt.plot(x, v, label=type_)
                break

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(title + ".png", bbox_inches='tight')


def plot_misclassified(*, tracker, main_category):
    for type_, d in tracker.items():
        for k, v in d.items():
            if type_.lower() == main_category and k.lower() == constants.MISCLASSIFIED:
                fig = plt.figure(figsize=(10, 10))
                for i in range(25):
                    sub = fig.add_subplot(5, 5, i + 1)
                    plt.imshow(v[i][0].cpu().numpy().squeeze(), cmap='gray', interpolation='none')
                    sub.set_title(
                        "Pred={}, Act={}".format(str(v[i][1].data.cpu().numpy()[0]), str(v[i][2].data.cpu().numpy())))
                plt.tight_layout()
                plt.show()
                plt.savefig(main_category + "_misclassified.png")


def get_dataloader_args():
    dataloader_args = dict(shuffle=bool(config[constants.MODEL_CONFIG][constants.SHUFFLE]),
                           batch_size=int(config[constants.MODEL_CONFIG][constants.BATCH_SIZE]),
                           num_workers=int(config[constants.MODEL_CONFIG][constants.WORKERS]),
                           pin_memory=bool(
                               config[constants.MODEL_CONFIG][constants.PIN_MEMORY])) if check_gpu_availability() \
        else dict(shuffle=bool(config[constants.MODEL_CONFIG][constants.SHUFFLE]),
                  batch_size=int(config[constants.MODEL_CONFIG][constants.BATCH_SIZE]))
    return dataloader_args


def get_all_models_summary():
    """ Collects all models from file `models.py` and uses torchsummary to print the summary"""
    import inspect
    from models import model_dummy_file
    for i in [m[0] for m in inspect.getmembers(model_dummy_file, inspect.isclass) if 'Net' in m[0]]:
        print(f'\nModel name: {i}')
        print_summary(model=model_dummy_file.str_to_class(i)().to(get_device()), input_size=(1, 28, 28))


def show_model_summary(title=None, *, model, input_size):
    """
    Calls `summary` method from torchsummary for the passed model and input size
    :param: title: title to show before printing summary
    :param model: the model to show the summary, detailed layers and parameters
    :param input_size: the input data size
    """
    if title:
        print(title)
    print(summary(model=model, input_size=input_size))


def get_dataset_name(*, session):
    return {"s6": "mnist", "s7": "cifar10"}.get(session.lower(), "mnist")


def get_input_size(*, dataset):
    return {"mnist": (1, 28, 28),
            "cifar10": (3, 32, 32)}.get(dataset.lower(), "mnist")
