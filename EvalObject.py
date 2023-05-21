# Code here is taken from CosmoML repository in CosmoAI-AES organisation
# It was writen by Hans Georg Schaathun and then edited by Modestas Sukarevicius

#! /usr/bin/env python

import pandas as pd
from skimage import io
import torch
import numpy as np

class EvalObject:
    def __init__(self,ev):
         """The `EvalObject` is instantiated with a list of dict objects,
         where each dict object is the output from `MLSystem.trainOne()`."""
         tr = [x["training"] for x in ev]
         tt = [x["test"] for x in ev]
         self.loss = torch.tensor([x["loss"] for x in ev])
         self.loss.reshape((len(self.loss), 1))
         self.testMSELoss = torch.stack([x["TotalMSE"] for x in tt])
         self.testMAELoss = torch.stack([x["TotalMAE"] for x in tt])
         self.testErrorMean = torch.stack([x["ErrorMean"] for x in tt])
         self.testErrorStDev = torch.stack([x["ErrorStDev"] for x in tt])
         self.testAbsMean = torch.stack([x["AbsMean"] for x in tt])
         self.testAbsStDev = torch.stack([x["AbsStDev"] for x in tt])

         self.trainingMSELoss = torch.stack([x["TotalMSE"] for x in tr])
         self.trainingMAELoss = torch.stack([x["TotalMAE"] for x in tr])
         self.trainingErrorMean = torch.stack([x["ErrorMean"] for x in tr])
         self.trainingErrorStDev = torch.stack([x["ErrorStDev"] for x in tr])
         self.trainingAbsMean = torch.stack([x["AbsMean"] for x in tr])
         self.trainingAbsStDev = torch.stack([x["AbsStDev"] for x in tr])
         self._mat = None
         self._headers = ["Loss"]
    def getMatrix(self):
        "Return the evaluation statistics as a numpy `array`."
        if self._mat == None:
           loss = self.loss.numpy()
           s = loss.shape
           if len(s) == 1:
               loss = loss.reshape((s[0], 1))
               trainingMSELoss = self.trainingMSELoss.cpu().numpy().reshape((s[0], 1))
               trainingMAELoss = self.trainingMAELoss.cpu().numpy().reshape((s[0], 1))
               testMSELoss = self.testMSELoss.cpu().numpy().reshape((s[0], 1))
               testMAELoss = self.testMAELoss.cpu().numpy().reshape((s[0], 1))
           ls = [loss]
           ls.append(trainingMSELoss)
           ls.append(trainingMAELoss)
           ls.append(testMSELoss)
           ls.append(testMAELoss)
           ls.append(self.testErrorMean.cpu().numpy())
           ls.append(self.testErrorStDev.cpu().numpy())
           ls.append(self.testAbsMean.cpu().numpy())
           ls.append(self.testAbsStDev.cpu().numpy())
           ls.append(self.trainingErrorMean.cpu().numpy())
           ls.append(self.trainingErrorStDev.cpu().numpy())
           ls.append(self.trainingAbsMean.cpu().numpy())
           ls.append(self.trainingAbsStDev.cpu().numpy())
           self._mat = np.hstack(ls)
        return self._mat
    def setHeaders(self,h):
        """Set the headers for CSV output.
        The input should be the same list of labels that are
        used by `CosmoDataset` to define the columns."""
        ls = (["tLoss"] +
           ["tMSELoss (Training Set)"] +
           ["tMAELoss (Training Set)"] +
           ["tMSELoss (Test Set)"] +
           ["tMAELoss (Test Set)"] +
           [x + " (Error Mean - Test Set)" for x in h] +
           [x + " (Error StDev - Test Set)" for x in h] +
           [x + " (AbsError Mean - Test Set)" for x in h] +
           [x + " (AbsError StDev - Test Set)" for x in h] +
           [x + " (Error Mean - Training Set)" for x in h] +
           [x + " (Error StDev - Training Set)" for x in h] +
           [x + " (AbsError Mean - Training Set)" for x in h] +
           [x + " (AbsError StDev - Training Set)" for x in h])
        self._headers = dict(enumerate(ls))
    def writecsv(self,fn="eval.csv"):
        "Write the evaluation results to the given CSV file."
        m = self.getMatrix()
        csv = pd.DataFrame(m)
        csv.rename(columns=self._headers, inplace=True)
        csv.to_csv(fn)


class PredObject:
    def __init__(self,pred):
         """The `PredObject` is instantiated with the output from
         `MLSystem.getPred()`."""
         self._mat = pred.cpu().numpy()
         self._headers = {0: "Index"}
    def getMatrix(self):
        "Return the evaluation statistics as a numpy `array`."
        return self._mat
    def setHeaders(self,h):
        """Set the headers for CSV output.
        The input should be the same list of labels that are
        used by `CosmoDataset` to define the columns."""
        ls = (["Index"] +
           [x + " (Ground Truth)" for x in h] +
           [x + " (Predicted)" for x in h])
        self._headers = dict(enumerate(ls))
    def writecsv(self,fn="pred.csv"):
        "Write the evaluation results to the given CSV file."
        m = self.getMatrix()
        csv = pd.DataFrame(m)
        csv.rename(columns=self._headers, inplace=True)
        csv.to_csv(fn)

