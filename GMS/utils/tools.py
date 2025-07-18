import torch
import os
import random
import numpy as np
import logging


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_reproducer(seed=2333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def load_checkpoint(model: torch.nn.Module, path: str, vae_model: None, vae_model_load = False) -> torch.nn.Module:
    if os.path.isfile(path):
        logging.info("=> loading checkpoint '{}'".format(path))
        
        # remap everthing onto CPU  [CHANGED] --> Added weights_only=True to prevent warnings
        state = torch.load(path, weights_only = True, map_location=lambda storage, location: storage)

        # load weights
        model.load_state_dict(state['model'])
        logging.info("Loaded LMM model from {}".format(path))

        if vae_model_load:
            if 'vae_model' in state:
                vae_model.load_state_dict(state['vae_model'])
                logging.info("Loaded VAE model from {}".format(path))
                return model, vae_model
            else:
                logging.warning("VAE model not found in checkpoint, returning only model and default vae_model")
                return model, vae_model

    else:
        model = None
        logging.info("=> no checkpoint found at '{}'".format(path))
    return model


def save_checkpoint(model: torch.nn.Module, save_name: str, path: str, vae_model = None, vae_model_save = False) -> None:
    model_savepath = os.path.join(path, 'checkpoints')

    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)

    file_name = os.path.join(model_savepath, save_name)

    if vae_model_save:
        torch.save({
            'model': model.state_dict(),
            'vae_model': vae_model.state_dict()
        }, file_name)
        logging.info("save model and vae_model to {}".format(file_name))

    else:
        torch.save({
            'model': model.state_dict(),
        },file_name)

    logging.info("save model to {}".format(file_name))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_learning_rate(optimizer, initial_lr, epoch, reduce_epoch, decay=0.5):
    lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info('Change Learning Rate to {}'.format(lr))
    return lr


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def print_options(configs):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in configs.items():
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    logging.info(message)

    # save to the disk
    file_name = os.path.join(configs['log_path'], '{}_configs.txt'.format(configs['phase']))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
