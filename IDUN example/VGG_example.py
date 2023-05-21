
import torch
from MLSystem import MLSystem, getArgs
import cudaaux

class CudaModel(MLSystem):
   def __init__(self,model='vgg',criterion=None,optimizer=None,nepoch=1,learning_rate=0.0001):
        super().__init__(model,criterion,optimizer,nepoch,learning_rate)

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
