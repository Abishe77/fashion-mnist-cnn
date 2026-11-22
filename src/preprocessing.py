
# importing the fashion MNIST data
from tensorflow.keras.datasets import fashion_mnist
#for splitting we use sklearn function
from sklearn.model_selection import train_test_split
import numpy as np
#for json and storage of pp metrics
import json
from pathlib import Path

def load_and_preprocess():
    #loading the dataset
    '''its like
    (60k grayscale images , one label per image), this is train
    (10k grayscale images , one label per image) this is test
    the training set will be split into train and validation sets later
    '''
    (X_train_full , y_train_full) , (X_test , y_test) = fashion_mnist.load_data()

    #it is split here into validation and training sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2, # reserves 20% of 60000 images to validation
        random_state=42, #the seed we use for split so that it can be reproduced
        stratify=y_train_full #maintains the proportion
    )

    '''expected shapes we get is
    for x_train = 48000 x 28 x 28 (60k * 80 /100)
    for x_val = 12000 x 28 x 28  (60k * 20 /100)
    '''

    '''now scaling the images , since the max I value is
    255 if we divide every image pixel value by 255
    we get every pixel values in [0,1]
    '''

    X_train = X_train.astype(np.float32) / 255
    X_val = X_val.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255

    #calculating training set mean and variance
    mean = np.mean(X_train)
    variance = np.var(X_train)
    epsilon = 1e-7 #10^-7 for standardisation

    #Now normalising train , validation and test sets using training mean and var
    X_train = (X_train - mean) / (np.sqrt(variance + epsilon))
    X_val = (X_val - mean) / (np.sqrt(variance + epsilon))
    X_test = (X_test - mean) / (np.sqrt(variance + epsilon))

    '''currently our train image shape is (48000 , 28 , 28)
    but CNN expects four dimension so we use
    newaxis alias to compensate for that
    it will become four dimensions and the
    fourth dimension means the channel
    for grayscale channel = 1, for RGB its 3
    '''
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    #the actual json
    preprocessing_stats = {
        "mean" : float(mean),
        "variance" : float(variance),
        "epsilon"  : float(epsilon),
        "scale" : 255.0
    }

    #dumping the json into artifacts
    stats_path = Path(__file__).resolve().parent.parent / "artifacts" / "preprocessing.json"

    with open(stats_path , "w") as file: #initiating the path in write mode
        json.dump(preprocessing_stats , file , indent=4)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

    #test scripts
    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    print("Training mean:", np.mean(X_train))
    print("Training variance:", np.var(X_train))

    print("NaN values:", np.isnan(X_train).any())
    print("Infinite values:", np.isinf(X_train).any())
