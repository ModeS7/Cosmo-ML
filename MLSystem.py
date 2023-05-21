# Code here is taken from CosmoML repository in CosmoAI-AES organisation
# It was writen by Hans Georg Schaathun and then edited by Modestas Sukarevicius

#! /usrbin/env python3

"""
The MLSystem class provides defaults for all the components of
a machine learning system.  The default implementation uses the CPU.
Subclasses should override key functions for more advanced systems.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset


import time
# from tqdm import tqdm

from Dataset import *
from EvalObject import EvalObject, PredObject
from Networks.Inception3 import Inception3
from Networks.AlexNet import AlexNet
from Networks.ResNet import resnet18, resnet34, resnet50, \
    resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, \
    resnext101_64x4d, wide_resnet50_2, wide_resnet101_2
from Networks.VGG import vgg11, vgg13, vgg16, \
    vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from Networks.DenseNet import densenet121, \
    densenet161, densenet169, densenet201
from Networks.EfficientNet import efficientnet_b0, \
     efficientnet_b2, efficientnet_b3, \
    efficientnet_b4, efficientnet_b5, efficientnet_b6, \
    efficientnet_b7, efficientnet_v2_l, efficientnet_v2_m, \
    efficientnet_v2_s, efficientnet_b3_5, efficientnet_b4_5
from Networks.ConvNeXt import convnext_tiny, \
    convnext_small, convnext_base, convnext_large
#from Networks.NASNet import NASNetAMobile , NASNetALarge
from Networks.MnasNet import mnasnet1_0, mnasnet3_8, mnasnet6_0
from Networks.SqueezeNet import squeezenet1_0, squeezenet1_1
from Networks.Vision_Transformer import vit_b_16, vit_b_32, \
    vit_l_16, vit_l_32, vit_h_14
from Networks.Swin_Transformer import swin_t, swin_s, swin_b, swin_v2_b, \
    swin_v2_t, swin_v2_s
from torchvision.models.inception import BasicConv2d
import torchvision.models as models
import argparse
import torchvision.models.vision_transformer as vit

class MLSystem:
    def __init__(self, model=None, criterion=None, optimizer=None, nepoch=2, learning_rate=0.01):
        """Construct a Machine Learning system for CosmoSim data.

        :param model: a pyTorch model instance; default `Inception3()`
        :param criterion: a pyTorch loss function; default `MSELoss()`
        :param optimizer: a pyTorch optimiser; default `Adam()`
        """
        self.num_epochs = nepoch
        self.batch_size = 10
        self.learning_rate = learning_rate
        self.device = None
        self.nparams = len(CosmoDataset1._columns)
        self.epochstrained = 0
        self.incep = False


        # Initialize your network, loss function, and optimizer
        if model == None:
            self.model = AlexNet(num_outputs=self.nparams, extra_layers=False)
            # self.model = squeezenet1_1(num_outputs=self.nparams, extra_layers=True)
            # self.model = Inception3(num_outputs=self.nparams, extra_layers=True)
            # self.model = resnet152(num_outputs=self.nparams, extra_layers=True)   # 101_32x8d, 101_64x4d wide_101_2 does not run on less than 8gb vram
            # self.model = vgg19(num_outputs=self.nparams)  # 19_bn does not run on less than 8gb vram
            # self.model = efficientnet_b3_5(num_outputs=self.nparams, extra_layers=True)   # b4 and v2_m does not run on less than 8gb vram
            # self.model = densenet201(num_outputs=self.nparams, extra_layers=True)   # all run on 8gb vram, potentialy can enable memory saving with checkpointing
            # self.model = convnext_tiny(num_outputs=self.nparams)
            # self.model = mnasnet3_8(num_outputs=self.nparams, extra_layers=True)
            # self.model = NASNetAMobile(num_outputs=self.nparams) # does not work, I have a dream that one day it will
            # self.model = vit_b_16(num_outputs=self.nparams)
            # self.model = swin_v2_s(num_outputs=self.nparams)

        if model == 'alexnet_pretrained':
            alexnet = models.alexnet(pretrained=True)
            alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
            alexnet.classifier[6] = nn.Linear(4096, self.nparams)
            self.model = alexnet
        if model == 'squeezenet_pretrained':
            squeezenet = models.squeezenet1_1(pretrained=True)
            squeezenet.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2)
            squeezenet.classifier[1] = nn.Conv2d(512, self.nparams, kernel_size=1)
            self.model = squeezenet
        if model == 'squeezenet_pretrained_extra':
            squeezenet = models.squeezenet1_1(pretrained=True)
            squeezenet.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2)
            final_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            squeezenet.final_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            squeezenet.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(1024 * 6 * 6, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.nparams)
            )
            self.model = squeezenet
        if model == 'inception_pretrained_extra':
            inception = models.inception_v3(pretrained=True, transform_input=False)
            inception.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
            inception.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.nparams))
            self.model = inception
            self.incep = True
        if model == 'inception_pretrained':
            inception = models.inception_v3(pretrained=True, transform_input=False)
            inception.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
            inception.fc = nn.Linear(2048, self.nparams)
            self.model = inception
            self.incep = True
        if model == 'vit_pretrained':
            model = vit.vit_b_16(pretrained=True)
            model.patch_embed.proj = torch.nn.Conv2d(1, model.patch_embed.proj.out_channels, kernel_size=16, stride=16)
            model.head = torch.nn.Linear(model.head.in_features, self.nparams)
            self.model = model
        if model == 'resnet_pretrained':
            resnet = models.resnet152(pretrained=True)
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.fc = nn.Sequential(
                nn.Linear(resnet.fc.in_features, self.nparams))
            self.model = resnet
        if model == 'densenet_pretrained':
            densenet = models.densenet201(pretrained=True)
            densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            densenet.classifier = nn.Sequential(
                nn.Linear(densenet.classifier.in_features, self.nparams),)
            self.model = densenet
        if model == 'vgg_pretrained':
            vgg = models.vgg19(pretrained=True)
            vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            num_features = vgg.classifier[-1].in_features
            vgg.classifier[-1] = torch.nn.Linear(num_features, self.nparams)
            self.model = vgg
        if model == 'efficientnet_pretrained':
            model = models.efficientnet_b0(pretrained=True)
            weight = model._conv_stem.weight
            new_weight = torch.nn.Parameter(weight[:, :1, :, :])
            model._conv_stem = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            model._conv_stem.weight = new_weight
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.nparams)
            self.model = model

        if model == 'alexnet':
            self.model = AlexNet(num_outputs=self.nparams, extra_layers=False)
        if model == 'squeezenet':
            self.model = squeezenet1_1(num_outputs=self.nparams, extra_layers=True)
        if model == 'inception':
            self.model = Inception3(num_outputs=self.nparams, extra_layers=True)
        if model == 'inception_vanila':
            self.model = Inception3(num_outputs=self.nparams, extra_layers=False)
        if model == 'resnet':
            self.model = resnet152(num_outputs=self.nparams, extra_layers=True)
        if model == 'vgg':
            self.model = vgg19(num_outputs=self.nparams)
        if model == 'efficientnet':
            self.model = efficientnet_b4_5(num_outputs=self.nparams, extra_layers=True)
        if model == 'efficientnet_vanilla':
            self.model = efficientnet_b7(num_outputs=self.nparams, extra_layers=False)
        if model == 'efficientnet_v2':
            self.model = efficientnet_v2_l(num_outputs=self.nparams, extra_layers=True)
        if model == 'densenet':
            self.model = densenet201(num_outputs=self.nparams, extra_layers=True)
        if model == 'convnext':
            self.model = convnext_tiny(num_outputs=self.nparams)
        if model == 'mnasnet':
            self.model = mnasnet3_8(num_outputs=self.nparams, extra_layers=True)
        if model == 'vit':
            self.model = vit_b_16(num_outputs=self.nparams)
        if model == 'swin':
            self.model = swin_v2_s(num_outputs=self.nparams)


        if criterion == None:
            # The default criterion is Mean Squared Error
            #self.criterion = nn.MSELoss()
            self.criterion = nn.L1Loss()
            # criterations for evaluation
            self.criterionMSE = nn.MSELoss()
            self.criterionMAE = nn.L1Loss()
        if optimizer == None:
            # self.optimizer = torch.optim.SGD(self.model.parameters(),
            #                                lr=self.learning_rate, momentum=0.9)
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate)
            # self.optimizer = torch.optim.RMSprop(self.model.parameters(),
            #                                    lr=self.learning_rate)

    def loadtrainingdata(self, fn="train.csv", dataset=None,
                         dataFraction=0.1, dataFractionTest=0.1):
        """Load the dataset for training.
        The parameter may be either a filename `fn` which would
        be loaded into a `CosmoDataset`, or `dataset` which
        should be a pre-defined `CosmoDataset` object (of any subclass).
        dataFraction decided the fraction of total images to be
        used in training.
        dataFractionTest decides the fraction of total images to
        be used in testing of training data.
        """
        if dataset:
            self.train_dataset = dataset
        else:
            self.train_dataset = CosmoDataset1(fn)
        self.ntrain = len(self.train_dataset)
        self.train_datasetTest = Subset(self.train_dataset,
                                        range(int(self.ntrain * dataFractionTest)))
        self.trainloaderTest = DataLoader(dataset=self.train_datasetTest,
                                          batch_size=self.batch_size, shuffle=True)
        self.ntrainTest = len(self.train_datasetTest)

        self.train_dataset = Subset(self.train_dataset,
                                    range(int(self.ntrain * dataFraction)))
        self.ntrain = len(self.train_dataset)

        self.trainloader = DataLoader(dataset=self.train_dataset,
                                      batch_size=self.batch_size, shuffle=True)
        self.img_size = self.train_dataset[0][0].shape


    def loadtestdata(self, fn="test.csv", dataset=None):
        """Load the dataset for testing.
        The parameter may be either a filename `fn` which would
        be loaded into a `CosmoDataset1`, or a pre-defined
        `CosmoDataset` object (of any subclass).
        dataFraction decided the fraction of total
        images to be used in training.
        """
        if dataset:
            self.test_dataset = dataset
        else:
            self.test_dataset = CosmoDataset1(fn)
        self.testloader = DataLoader(dataset=self.test_dataset,
                                     batch_size=self.batch_size)
        self.ntest = len(self.test_dataset)


    def printparam(self):
        """Print statistics of the training scenario to stdout.
        This includes epoch number, dataset sizes, and image size."""
        print(f'num_epochs: {self.num_epochs}, '
              + f'batch size: {self.batch_size}, lr: {self.learning_rate}')
        print(f'image size: {self.img_size}')
        print(f'train samples: {self.ntrain}({self.ntrainTest}) '
              f'test samples: {self.ntest}\n')

    def trainOne(self, verbose=True):
        """Train the network for one epoch.
        Return the total loss.
        If `verbose` is `True` the training loss is printed on stdout
        for each minibatch.
        """
        tloss = 0.0
        epoch = self.epochstrained
        for i, (images, params, index) in enumerate(self.trainloader):
            if self.device:
                images = images.to(self.device)
                params = params.to(self.device)
            self.optimizer.zero_grad()

            # Forward + Backward + Optimiser
            output = self.model(images)
            if epoch == 0 and self.incep:
                output = output[0]
            loss = self.criterion(output, params)
            loss.backward()
            self.optimizer.step()

            tloss += loss.item() * len(images)
            if verbose:
                print(f"Batch no. {epoch + 1}-{i + 1}: loss = {loss.item()}; "
                      f"tloss = {tloss}")
        self.epochstrained += 1
        return tloss

    def train(self, nepoch=None, test=False, verbose=True):
        """Train the network.

        The return value is a list of training losses per epoch if
        `test` is `False`.  If `test` is `True`, the return value
        is an `EvalObject` containing comprehensive perfomance
        statistics.

        :param nepoct: Number of epochs; if None the object default is used.
        :param test: If True, the model is tested after each epoch
        :param verbose: If True, training loss is written for each epoch
        """
        timer = time.time()
        lossAcrossEpochs = []
        if nepoch == None:
            nepoch = self.num_epochs
        else:
            nepoch += self.epochstrained
        startidx = self.epochstrained
        try:
            # This is based on Listing 3-4.
            self.model.train()
            for epoch in range(startidx, nepoch):

                tloss = self.trainOne(verbose=verbose)
                if verbose:
                    print(f"Epoch {epoch + 1}: Loss = {tloss}")
                if test:
                    ob = {"loss": tloss,
                          "training": self.getLoss(trainingset=True),
                          "test": self.getLoss()
                          }
                    if verbose: print(ob)
                    lossAcrossEpochs.append(ob)
                else:
                    lossAcrossEpochs.append(tloss)

        except KeyboardInterrupt:
            print("Training aborted by keyboard interrupt.")

        if test:
            lossAcrossEpochs = EvalObject(lossAcrossEpochs)
            lossAcrossEpochs.setHeaders(self.test_dataset.getSlice())
        return lossAcrossEpochs

    def getLoss(self, printDetails=True, trainingset=False):
        """
        Test the model and report various performance heuristics.

        :param printDetails: if True, heuristics are printed on stdout
        :param trainingset: if True, the test is made on the traininset
        :returns: A `dict` with various heuristics
        """
        total_loss, total_MSE_loss, total_MAE_loss = 0, 0, 0

        self.model.eval()
        errors = []
        if trainingset:
            dataloader = self.trainloaderTest
        else:
            dataloader = self.testloader
        with torch.no_grad():
            for (images, params, index) in dataloader:
                if self.device:
                    images = images.to(self.device)
                    params = params.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, params)
                MSE_loss = self.criterionMSE(output, params)
                MAE_loss = self.criterionMAE(output, params)
                total_loss += loss * len(images)
                total_MSE_loss += MSE_loss * len(images)
                total_MAE_loss += MAE_loss * len(images)
                for i, param in enumerate(params):
                    error = output[i] - param
                    mse = error ** 2
                    mse = mse.sum().item() / self.nparams
                    errors.append(error)
                    if printDetails:
                        niceoutput = [round(n, 3) for n in output[i].tolist()]
                        niceparam = [round(n, 3) for n in param.tolist()]
                        if i < 1: # Print only the first image in the batch
                            print(f"{f'{round(mse, 4)} Correct: {niceparam}' : <40}"
                                  f"{f'Output: {niceoutput}' : ^40}")
        # The average is wrong if the last batch has fewer images
        errorMat = torch.stack(errors)
        errorAbs = errorMat.abs()
        mean = errorAbs.mean(axis=0)
        stdev = errorAbs.std(axis=0, unbiased=True)
        mean2 = errorMat.mean(axis=0)
        stdev2 = errorMat.std(axis=0, unbiased=True)
        if printDetails:
            print("Mean absolute error", mean)
            print("Standard deviation", stdev)
            print("Mean signed error", mean2)
            print("Standard deviation", stdev2)
            print("Total MSE loss", total_MSE_loss)
            print("Total MAE loss", total_MAE_loss)
            print(f"Loss/sample = {total_loss / self.ntest}")
        return {"TotalMSE": total_MSE_loss,
                "TotalMAE": total_MAE_loss,
                "ErrorMean": mean2,
                "ErrorStDev": stdev2,
                "AbsMean": mean,
                "AbsStDev": stdev
                }

    def getPred(self, trainingset=False):
        """
        Test the model and return the prediction results.

        :param trainingset: if True, the test is made on the traininset
        :returns: A tensor containing index, ground truth, and prediction.
        """
        self.model.eval()
        tl = []
        if trainingset:
            dataloader = self.trainloader
        else:
            dataloader = self.testloader
        i = 0
        with torch.no_grad():
            for (images, params, index) in dataloader:
                if self.device:
                    images = images.to(self.device)
                    params = params.to(self.device)
                output = self.model(images)
                idxtensor = index.reshape((len(index), 1))
                print("batch number:",i)
                i += 1
                tl.append(torch.cat([
                    idxtensor, params.cpu(), output.cpu()], axis=1))
        return torch.cat(tl)

    def savemodel(self, fn="save-model"):
        """
        Save the pyTorch model to file.
        Note that the `MLSystem` object is not stored; only the
        actual trained model.
        """
        torch.save(self.model.state_dict(), fn)

    def systemTest(self, args):
        """Test the entire system with training and testing.
        The input is an object returned by getArgs(), and thus
        representing CLI arguments.
        """
        if args.imagedir:
            imgdir = args.imagedir
        else:
            imgdir = "./"
        if args.imagedirtest:
            imgdirtest = args.imagedirtest
        else:
            imgdirtest = "./"

        if args.amp6:
            ob = CosmoDataset2(csvfile=args.train, imgdir=imgdir)
            self.loadtrainingdata(dataset=ob)
            ob = CosmoDataset2(csvfile=args.test, imgdirtest=imgdirtest)
            self.loadtestdata(dataset=ob)
        else:
            ob = CosmoDataset1(csvfile=args.train, imgdir=imgdir)
            self.loadtrainingdata(dataset=ob)
            ob = CosmoDataset1(csvfile=args.test, imgdirtest=imgdirtest)
            self.loadtestdata(dataset=ob)



        self.printparam()
        print("Training ...")
        if args.epochs:
            nepochs = int(args.epochs)
        else:
            nepochs = None
        if args.evalfile:
            res = self.train(nepoch=nepochs, test=True)
            res.writecsv(args.evalfile)
        else:
            self.train(nepoch=nepochs, test=False)

        if args.msavefile:
            self.savemodel(args.msavefile)
        else:
            self.savemodel("model.pt")

        if args.evalfile == False:
            print("Post-training test ...")
            loss = self.getLoss()
            print('Loss Results:', loss)

        if args.predictionfile:
            pred = self.getPred()
            ob = PredObject(pred)
            ob.setHeaders(self.test_dataset._columns)
            ob.writecsv(args.predictionfile)


def getArgs():
    parser = argparse.ArgumentParser(
        prog='CosmoML test script (cuda version)',
        description='Train and test a regression model',
        epilog='')

    parser.add_argument('-t', '--train',
                        help="Training data set")
    parser.add_argument('-T', '--test',
                        help="Testing data set")
    parser.add_argument('-i', '--imagedir',
                        help="Image directory")
    parser.add_argument('-I', '--imagedirtest',
                        help='Test image directory')
    parser.add_argument('-o', '--evalfile',
                        help="Filename for evaluation output")
    parser.add_argument('-p', '--predictionfile',
                        help="Filename for prediction output")
    parser.add_argument('-e', '--epochs',
                        help="Testing data set")
    parser.add_argument('-W', '--weights',
                        help="File with pre trained weights")
    parser.add_argument('-s', '--msavefile',
                        help="File for trained weights, end in .pt")
    parser.add_argument('-a', '--amp6', action='store_true',
                        help="Estimate amplitude with 6 parameters")

    return parser.parse_args()

if __name__ == "__main__":
    print("MLSystem test script.\nConfiguring ... ")

    args = getArgs()
    ml = MLSystem()
    ml.systemTest(args)