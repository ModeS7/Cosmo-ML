# CosmoAI

*Machine learning tools for lens mass reconstruction*

The **Simulation Tool** has been renamed as 
[CosmoSim](https://github.com/CosmoAI-AES/CosmoSim).
This repo will continue the related machine learning work.

On [Virtual Environments in Python](https://stackoverflow.com/questions/14684968/how-to-export-virtualenv)

# Design

The `MLSystem` class is designed to do a vanilla setup running on the CPU.
The `CudaModel` class inherits from `MLSystem`, making sufficient overrides
to run on a GPU.

# Problems

1.  Python often crashes in the `loss.backward()` call, at different stages
    of the iteration.  It is possible that this is a memory allocation problem
    depending on other activity on the box.  It seems to happen more often with
    concurrent interactive use.

# Setup at IDUN

Tested 4 May 2023, the following setup sequence seems to work at IDUN.
We have previously not been able to use the PyTorch module and therefore
install via pip.

```sh
module load CMake/3.22.1-GCCcore-11.2.0 
module load SciPy-bundle/2021.10-foss-2021b
pip3 install pip --upgrade
pip3 install torch
```

It is recommended to keep a python virtual environment, e.g.

```sh
python3 -m venv ~/CosmoML.venv
. ~/CosmoML.venv/bin/activate
pip3 install pip --upgrade
pip3 install torch
```
