# Code here is taken from CosmoML repository in CosmoAI-AES organisation
# It was writen by Hans Georg Schaathun and then edited by Modestas Sukarevicius

#! /usrbin/env python3

"""
The MLSystem class provides defaults for all the components of
a machine learning system.  The default implementation uses the CPU.
Subclasses should override key functions for more advanced systems.
"""

import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
from MLSystem import MLSystem, getArgs

# from Dataset import *
import cudaaux

class CudaModel(MLSystem):
   def __init__(self,model='squeezenet_pretrained',criterion=None,optimizer=None,nepoch=2, learning_rate=0.0001):
        super().__init__(model,criterion,optimizer,nepoch, learning_rate)

        if not torch.cuda.is_available():
            raise Exception( "CUDA is not available" )

        self.device = torch.device( "cuda" )
        self.model = self.model.to(self.device)
        if args.weights:
            self.model.load_state_dict(torch.load(args.weights))


if __name__ == "__main__":
    args = getArgs()

    print( "CudaModel (CosmoML) test script." )
    cudaaux.cudaDiagnostic()
    print( "Configuring ... " )

    ml = CudaModel()
    ml.systemTest(args)

