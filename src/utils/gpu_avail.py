import torch
import os
# print("Torch version:", torch.__version__)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

def get_available_gpus():
    """
    Returns the number of available GPUs.

    Returns:
        int: The number of available GPUs. Returns 0 if no GPUs are available.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 0

gpu_count = get_available_gpus()
print("Number of available GPUs:", gpu_count)
