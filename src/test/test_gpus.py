

import torch

# Check if PyTorch is installed
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
   
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
   
    # List available GPUs
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Run a test tensor operation on the GPU
    device = torch.device("cuda")  # Use the default GPU (cuda:0)
    test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Test tensor created on GPU: {test_tensor}")

    # Perform a simple operation
    result = test_tensor * 2
    print(f"Test operation result on GPU: {result}")

else:
    print("CUDA is not available. Please check your installation.")


