'''
Created on 03.12.2018

@author: el-sharkawy
'''
from metricsML import DataConvertor
from metricsML import MachineLearning
from metricsML import Normalizator
import numpy as np

### File Operations ###
def convertTwoCSVFiles(folder="data/", goodfile="all_good_output.csv", badfile="all_error_output.csv"):
    DataConvertor.readFromTwoFiles(folder, goodfile, badfile)
    
def convertOneCSVFile(folder="data/", file="Sample.csv"):
    DataConvertor.readFromSingleFile(folder, file)
    
def loadNumpyArray(file):
    return np.load(file)
    
### Machine learning ###
def normalization(normalization, train_data, test_data, validation_data=None):
    Normalizator.normalization(normalization, train_data, test_data, validation_data)

def binaryClassification(train_data, train_labels,test_data,test_labels,nEpochs, lrate, layerSize, columns, rp=0.01):
    return MachineLearning.binaryClassification(train_data, train_labels,test_data,test_labels,nEpochs, lrate, layerSize, rp, columns=columns)

# binaryClassification=MachineLearning.binaryClassification