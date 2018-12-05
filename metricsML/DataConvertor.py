import numpy as np           # Reading CSV ans saving vectors as binary files
import pandas as pd          # For reading CSV files
# np.set_printoptions(threshold=np.nan) # Print arrays completely

def readFromTwoFiles(folder="data/", goodfile="all_good_output.csv", badfile="all_error_output.csv"):
    ### Positive and negative data ###
    sampleP_data=np.asarray(pd.read_csv(folder + goodfile))
    sampleP_data=sampleP_data[:, 3:sampleP_data.shape[1]]
    print(goodfile + " -> " + str(sampleP_data.shape))
    
    sampleN_data=np.asarray(pd.read_csv(folder + badfile))
    sampleN_data=sampleN_data[:, 8:sampleN_data.shape[1]]
    print(badfile + " ->  " + str(sampleN_data.shape))
    
    ### Labels ###
    sampleP_labels=np.ones(sampleP_data.shape[0])
    sampleN_labels=np.zeros(sampleN_data.shape[0])
    np.random.shuffle(sampleP_data)
    np.random.shuffle(sampleN_data)
    __splitnSaveData(sampleP_data, sampleN_data, sampleP_labels, sampleN_labels, "data/")

def readFromSingleFile(folder="data/", file="Sample.csv"):
    sample_data=np.asarray(pd.read_csv(folder + file))
    nRows=len(sample_data)
    
    ### Positive and negative data ###
    sampleP_data=sample_data[int(nRows/2):]
    sampleN_data=sample_data[:int(nRows/2)]
    ### Labels ###
    sampleP_labels=np.ones(int(nRows/2))
    sampleN_labels=np.zeros(int(nRows/2))
    np.random.shuffle(sampleP_data)
    np.random.shuffle(sampleN_data)
    __splitnSaveData(sampleP_data, sampleN_data, sampleP_labels, sampleN_labels, folder)

### Separate input data and save them ###
def __splitnSaveData(sampleP_data, sampleN_data, sampleP_labels, sampleN_labels, folder="data/"):
    nRows=min(sampleP_data.shape[0], sampleN_data.shape[0])
    print("Selecting " + str(nRows) + " rows.")
    
    # Training data (80%)
    pos80p=int(nRows*0.8)
    pos90p=int(nRows*0.9)
    pos100p=nRows
    train_data=np.concatenate([sampleP_data[:pos80p],sampleN_data[:pos80p]])
    train_labels=np.concatenate([sampleP_labels[:pos80p],sampleN_labels[:pos80p]])
    print("Train data -> " + str(train_data.shape))
    print("Train labels -> " + str(train_labels.shape))
    # Evaluation data (10%)
    test_data=np.concatenate([sampleP_data[pos80p:pos90p],sampleN_data[pos80p:pos90p]])
    test_labels=np.concatenate([sampleP_labels[pos80p:pos90p],sampleN_labels[pos80p:pos90p]])
    print("Test data -> " + str(test_data.shape))
    print("Test labels -> " + str(test_labels.shape))
    # Hyperparameter optimization (10%)
    validation_data=np.concatenate([sampleP_data[pos90p:pos100p],sampleN_data[pos90p:pos100p]])
    validation_labels=np.concatenate([sampleP_labels[pos90p:pos100p],sampleN_labels[pos90p:pos100p]])
    print("Validation data -> " + str(validation_data.shape))
    print("Validation labels -> " + str(validation_labels.shape))
     
    ### Save data in binary format ###
    np.save(folder + "train_data.npy", train_data);
    np.save(folder + "train_labels.npy", train_labels);
    np.save(folder + "test_data.npy", test_data);
    np.save(folder + "test_labels.npy", test_labels);
    np.save(folder + "validation_data.npy", validation_data);
    np.save(folder + "validation_labels.npy", validation_labels);