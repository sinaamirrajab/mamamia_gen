from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR, \
    CosineAnnealingWarmRestarts, ConstantLR, LambdaLR

import torch
import numpy as np

def findLRGamma(startingLR, endingLR, num_epochs):
    '''
    Gamma getter. Based on a minimum and maximum learning rate, calculates the Gamma
    necessary to go from minimum to maximum in num_epochs.
    :param startingLR: First Learning Rate.
    :param endingLR: Final Learning Rate.
    :param num_epochs: Number of epochs.
    :return: gamma value
    '''
    gamma = np.e ** (np.log(endingLR / startingLR) / num_epochs)
    return gamma

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def getLearningRate(optimizer, warmup_lr, base_lr, n_epochs_warmup,
                    n_epochs_shift, n_epochs, lr_scheduler_type='linear',
                    end_lr=0.0000001, sanity_test=True):

    if n_epochs_shift > n_epochs:
        raise ValueError("Argument n_epochs cannot be smaller than n_epochs_shift!")

    lr_scheduler_type = lr_scheduler_type.lower()
    gamma_shift = findLRGamma(warmup_lr, base_lr, n_epochs_shift - n_epochs_warmup)

    if lr_scheduler_type == 'constant':
        lambda1 = lambda epoch: int(0 <= epoch < n_epochs_warmup) * (1.0) \
                                + int(n_epochs_warmup <= epoch < n_epochs_shift) * (
                                            gamma_shift ** (epoch - n_epochs_warmup)) \
                                + int(n_epochs_shift <= epoch) * (1 / base_lr)

    elif lr_scheduler_type == 'exponential':
        gamma_decay = findLRGamma(base_lr, end_lr, n_epochs - n_epochs_shift)
        lambda1 = lambda epoch: int(0 <= epoch < n_epochs_warmup) * (1.0) \
                                + int(n_epochs_warmup <= epoch < n_epochs_shift) * (gamma_shift ** (epoch - n_epochs_warmup)) \
                                + int(n_epochs_shift <= epoch) * ((gamma_decay) ** (epoch - n_epochs_shift))

    elif lr_scheduler_type == 'linear':
        lambda1 = lambda epoch: int(0 <= epoch < n_epochs_warmup) * (1.0) \
                                + int(n_epochs_warmup <= epoch < n_epochs_shift) * (1 - (epoch - n_epochs_warmup) / (n_epochs_shift - n_epochs_warmup)) \
                                + int(n_epochs_shift <= epoch) * (1 - (epoch - n_epochs_shift) / (n_epochs - n_epochs_shift))

    elif lr_scheduler_type == 'cosine':
        lambda1 = lambda epoch: int(0 <= epoch < n_epochs_warmup) * (1.0) \
                                + int(n_epochs_warmup <= epoch < n_epochs_shift) * (gamma_shift ** (epoch - n_epochs_warmup)) \
                                + int(n_epochs_shift <= epoch) * (0.5 * (1 + np.cos(np.pi * (epoch - n_epochs_shift) / (n_epochs - n_epochs_shift))))

    else:
        raise ValueError("Learning rate type not supported. Enter: 'constant', 'linear', 'cosine', or 'exponential'."
                         f" '{lr_scheduler_type}' was found.")

    scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])

    # Sanity check to ensure no NaNs in the learning rate values
    if sanity_test:
        dummy_net = torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3)  # Dummy model
        dummy_optimizer = torch.optim.Adam(dummy_net.parameters(), lr=warmup_lr)  # Dummy optimizer
        dummy_scheduler = LambdaLR(dummy_optimizer, lr_lambda=[lambda1])

        for e in range(n_epochs):
            dummy_optimizer.step()
            dummy_scheduler.step()
            if np.isnan(get_lr(dummy_optimizer)):
                raise ValueError("Learning rate schedule results in NaN. "
                                 "Adjust the warmup period or decrease the total number of epochs.")

    return scheduler
