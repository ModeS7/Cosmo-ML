Code here is taken from CosmoML repository in CosmoAI-AES organisation
It was writen by Hans Georg Schaathun and then edited by Modestas Sukarevicius.

# Design

The `MLSystem` class is designed to do a vanilla setup running on the CPU.
The `CudaModel` class inherits from `MLSystem`, making sufficient overrides
to run on a GPU.

# Setup and running

## Local machine

Step 1 - Install cuda toolkit from Nvidia if you don't have it already. 
It can be found here: https://developer.nvidia.com/cuda-toolkit

Step 2 - Copy the repository, setup virtual python environment for this project. 
Then install all the needed libraries. 
```sh
git clone <Cosmo-ML url>
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
Step 3 - Selecting network and few hyperparameters: 
open CudaModel.py -> write a network to test
model='<wanted neural network>'(all neural network options are in MLSystem.py). 
At the same time number of epochs and learning rate can be chosen.

Step 4 - Choosing the rest of hyperparameters: open MLSystem.py here batch size, 
optimizer, criteration(loss function) is chosen.

Step 5 - To start testing command like this with arguments should be used: 
```sh
python CudaModel.py -t "<csv file for training data set>" -i "<csv file for testing data set>" 
-T "<directory for training data set images>" -I "<directory for testing data set images>" 
-p "<csv file for predicted values on test set at the end of training>" 
-o "<csv file for total loss data of every epoch>"
```

## IDUN

Setup for IDUN is the same as for local machine.

Since IDUN can run multiple test at the same time, for every test a CudaModel.py file
replacement is created like VGG_example.py that goes with slurm file VGG.slurm.

To start testing on IDUN command like this should be used: 
```sh
sbatch VGG.slurm
```
To check the status of the tests:
```sh
squeue --user=<username>
```
To cancel the test:
```sh
scancel --name=<job name>
```