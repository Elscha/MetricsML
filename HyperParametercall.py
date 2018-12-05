import metricsML.MetricsML as ml
import numpy as np           # Reading CSV ans saving vectors as binary files
from metricsML.NormalizationType import NormalizationType
import sys # See: https://stackoverflow.com/a/39784510

# import metricsML.* as ml
def printResults(message, loss, accuracy, columns, layer, epoch, lr, rp):
    print(message)
    print("  Loss:                     " + str(loss))
    print("  Accuracy:                 " + str(accuracy))
    print("  Columns:                  " + str(columns))
    print("  Layer:                    " + str(layer))
    print("  Epochs:                   " + str(epoch))
    print("  Learning Rate:            " + str(lr))
    print("  Regularization Parameter: " + str(rp))

def hyperParameterOptimization(train_data_file, train_labels_file, test_data_file, test_labels_file, startIndex, endIndex):
    print("Reading input data")
    # Training data (80%)
    train_data=np.load(train_data_file)
    train_labels=np.load(train_labels_file)
    # Evaluation data (10%)
    test_data=np.load(test_data_file)
    test_labels=np.load(test_labels_file)
    
    print("Data read")
    ml.normalization(NormalizationType.PERCENTAGE, train_data, test_data)
    print("Data normalized")
    
#     layers                   = [[8], [8, 16], [8, 16, 32]]
    layers                   = [[8]]
#     epochs                   = [4, 10, 20]
    epochs                   = [4]
#     learningRates            = [0.001, 0.01]
    learningRates            = [0.001]
#     regularizationParameters = [0.05, 0.1, 0.15, 0.2]
    regularizationParameters = [0.05, 0.1]
    
    best_loss     = 1000000
    best_accuracy = 0
    
    for c in range(startIndex, endIndex):
        columns = [c]
        for layer in layers:
            for epoch in epochs:
                print(str(columns) + " columns in " + str(layer) + " layers in " + str(epoch) + " epochs")
                for lr in learningRates:
                    for rp in regularizationParameters:
                        tlos, tacc = ml.binaryClassification(train_data,train_labels, test_data,test_labels, epoch, lr, layer, columns, rp)
                        if (tacc > best_accuracy):
                            best_loss = tlos
                            best_accuracy = tacc
                            best_columns = columns
                            best_layer = layer
                            best_epoch = epoch
                            best_lr = lr
                            best_rp = rp
                            printResults("New Intermediate Result:", best_loss, best_accuracy, best_columns, best_layer, best_epoch, best_lr, best_rp)
    printResults("Final Results:", best_loss, best_accuracy, best_columns, best_layer, best_epoch, best_lr, best_rp)

hyperParameterOptimization("data/train_data.npy", "data/train_labels.npy", "data/test_data.npy", "data/test_labels.npy", int(sys.argv[1]), int(sys.argv[2]))


# readFromSingleFile()

# layers                   = [[8], [8, 16], [8, 16, 32], [16, 32], [16, 32, 64], [32, 64]]
# epochs                   = [4, 8, 10, 20, 50, 60]
# learningRates            = [0.001, 0.01, 0.1, 0.2]
# columnsList              = [[0], [1], [2]]
# layers                   = [[8], [8, 16], [8, 16, 32]]
# epochs                   = [4, 10, 20]
# learningRates            = [0.001, 0.01]
# regularizationParameters = [0.05, 0.1, 0.15, 0.2]
# 
# best_loss     = 1000000
# best_accuracy = 0
# # Training data (80%)
# train_data=np.load("data/train_data.npy")
# train_labels=np.load("data/train_labels.npy")
# # Evaluation data (10%)
# test_data=np.load("data/test_data.npy")
# test_labels=np.load("data/test_labels.npy")
# 
# # for columns in train_data.shape[1]
# for c in range(0, train_data.shape[1]):
#     columns = [c]
#     for layer in layers:
#         for epoch in epochs:
#             print(str(columns) + " columns in " + str(layer) + " layers in " + str(epoch) + " epochs")
#             for lr in learningRates:
#                 for rp in regularizationParameters:
#                     tlos, tacc = ml.binaryClassification(train_data,train_labels, test_data,test_labels, epoch, lr, layer, columns, rp)
#                     if (tacc > best_accuracy):
#                         best_loss = tlos
#                         best_accuracy = tacc
#                         best_columns = columns
#                         best_layer = layer
#                         best_epoch = epoch
#                         best_lr = lr
#                         best_rp = rp
#                         printResults("New Intermediate Result:", best_loss, best_accuracy, best_columns, best_layer, best_epoch, best_lr, best_rp)
# #                         saveModel(model)
# printResults("Final Results:", best_loss, best_accuracy, best_columns, best_layer, best_epoch, best_lr, best_rp)