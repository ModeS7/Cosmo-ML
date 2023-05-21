# Code here is taken from CosmoML repository in CosmoAI-AES organisation
# It was writen by Hans Georg Schaathun

import torch

def cudaDiagnostic():
   print(f"CUDA Availability:           {torch.cuda.is_available()}")
   print(f"CUDA version:                {torch.version.cuda}")
   cuda_id = torch.cuda.current_device()
   print(f"ID of current CUDA device:   {cuda_id}")
   print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
