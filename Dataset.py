"""
The Dataset class manages loading and access to a dataset
in the CosmoAI model.
"""

import torch
import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset 
import pandas as pd


class CosmoDataset(Dataset):
    """CosmoAI  dataset."""
    _columns = ["x", "y", "einsteinR", "sigma", "sigma2", "theta"]

    def getSlice(self):
        return self._columns

    def getDim(self):
        return len(self.getSlice())

    def __init__(self, csvfile, imgdir=".", imgdirtest='.', columns=None):
        """
        Args:
            csvfile (string): Path to the csv file with annotations.
            imgdir (string): Directory with all the images.
        """
        self.frame = pd.read_csv(csvfile)
        self.imgdir = imgdir
        self.imgdirtest = imgdirtest
        if columns != None:
            self._columns = columns

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fn = os.path.join(self.imgdir, self.imgdirtest,
                          self.frame.iloc[idx, 1])
        image = io.imread(fn)[np.newaxis, :, :].astype(np.float32) / 255
        image = torch.from_numpy(image)
        targets = self.frame.loc[idx, self.getSlice()]
        targets = np.array(targets).astype(np.float32)
        targets = torch.from_numpy(targets)
        index = int(self.frame.loc[idx, "index"])

        return (image, targets, index)

class CosmoDataset1(CosmoDataset):
    _columns = ["x", "y", "einsteinR", "sigma", "sigma2", "theta"]
class CosmoDataset2(CosmoDataset):
    _columns = ["alpha[1][0]","alpha[1][2]","beta[1][2]","alpha[2][1]","beta[2][1]","alpha[2][3]"]
