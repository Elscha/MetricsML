import numpy as np           # Reading CSV ans saving vectors as binary files
import tensorflow as tf      # Tensorflow
from tensorflow import keras # Simplified Tensorflow Framework
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import math

def plotWeights(model):
    for layer in model.get_weights():
        fig, ax = plt.subplots()
        im = ax.imshow(layer)
        plt.show()

def saveModel(model, folder="data/"):
    print("In folder " + folder)
    # Saves complete model with
    # Architecture
    # Weights
    # Training configuration (Loss, Optimizer)
    # State of Optimizer
    model.save_weights(folder + 'model.h5')
    # Saves only Architecture of Model
    np.save(folder+"modelW.npy",model.get_weights())
    print(model.get_weights())
    file = open(folder + 'model.json', "w")
    file.write(model.to_json())
    print("Finished with saving")
    file.close();

def binaryClassification(train_data, train_labels,test_data,test_labels, nEpochs, lrate, layerSize, rp=0.01, columns=None):
    ### Read input data    ###
#     # Training data (80%)
#     train_data=np.load(folder + "train_data.npy")
#     train_labels=np.load(folder + "train_labels.npy")
#     # Evaluation data (10%)
#     test_data=np.load(folder + "test_data.npy")
#     test_labels=np.load(folder + "test_labels.npy")
    
    if (columns is not None):
        train_data = train_data[:, columns]
        test_data  = test_data[:, columns]    
    
    ### Define Neuronal Network
    layers=[keras.layers.Dense(i, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(rp)) for i in layerSize]
    layers.append(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model = keras.Sequential(layers)
    model.compile(optimizer = tf.train.AdamOptimizer(),
                  lr        = lrate, 
                  loss      = 'binary_crossentropy',
                  metrics   = ['accuracy'])
    ### Execute model
#     history =  model.fit(train_data, train_labels, epochs=nEpochs, verbose=1, validation_data=[test_data,test_labels]) #--> Use this to grep & plot this per Epochs (last line)
    history = model.fit(train_data, train_labels, epochs=nEpochs, verbose=0)
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
    
    if (math.isnan(history.history['loss'])):
        print("Loss was Not a number")
    
    plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
#     saveModel(model);
    plotWeights(model)
    return test_loss, test_acc