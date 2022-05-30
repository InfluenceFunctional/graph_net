from torch_models import *
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime but useful for debugging
import numpy as np


def get_grad_norm(model):
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(params) == 0:
        norm = 0
    else:
        norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).cpu() for p in params]), 2.0).item()

    return norm


def set_lr(schedulers, optimizer, config, err_tr, hit_max_lr):
    if config.lr_schedule:
        lr = optimizer.param_groups[0]['lr']
        if lr > 1e-4:
            schedulers[0].step(torch.mean(torch.stack(err_tr)))  # plateau scheduler

        if not hit_max_lr:
            if lr <= config.max_lr:
                schedulers[1].step()
            else:
                hit_max_lr = True
        elif hit_max_lr:
            if lr > config.learning_rate:
                schedulers[2].step()  # start reducing lr
    lr = optimizer.param_groups[0]['lr']
    print("Learning rate is {:.5f}".format(lr))
    return optimizer


def checkConvergence(config, record):
    """
    check if we are converged
    condition: test loss has increased or levelled out over the last several epochs
    :return: convergence flag
    """

    converged = False
    if type(record) == list:
        record = np.asarray(record)

    if len(record) > (config.history + 2):
        if all(record[-config.history:] > np.amin(record)):
            converged = True
            print("Model converged, target diverging")

        criteria = np.var(record[-config.history:]) / np.abs(np.average(record[-config.history:]))
        print('Convergence criteria at {:.3f}'.format(np.log10(criteria)))
        if criteria < config.convergence_eps:
            converged = True
            print("Model converged, target stabilized")

    return converged


def save_model(model, optimizer):
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/model_ckpt')


def load_model(config, model, optimizer):
    '''
    Check if a checkpoint exists for this model - if so, load it
    :return:
    '''
    checkpoint = torch.load('ckpts/model_ckpt')

    if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config.device == 'cuda':
        # model = nn.DataParallel(self.model) # enables multi-GPU training
        print("Using ", torch.cuda.device_count(), " GPUs")
        model.to(torch.device("cuda:0"))
        for state in optimizer.state.values():  # move optimizer to GPU
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    return model, optimizer
