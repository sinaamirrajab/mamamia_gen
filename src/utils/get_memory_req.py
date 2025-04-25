import torch
import numpy as np

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size + buffer_size

def get_activation_size(model, input_size):
    def forward_hook(module, input, output):
        if isinstance(output, (tuple, list)):
            for out in output:
                activation_size.append(out.nelement() * out.element_size())
        else:
            activation_size.append(output.nelement() * output.element_size())

    activation_size = []
    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(forward_hook))

    with torch.no_grad():
        model(torch.randn(*input_size))

    for hook in hooks:
        hook.remove()

    return sum(activation_size)

def estimate_memory_usage(model, input_size, batch_size):
    param_size = get_model_size(model)
    activation_size = get_activation_size(model, input_size)
    total_size = param_size + activation_size * batch_size
    return total_size