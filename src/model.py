#importing the modules required for CNN
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#func for creating the model

def build_model():
    model = Sequential([
        Input(shape = (28,28,1)),
        Conv2D(filters = 32 , kernel_size= 3 , activation= "relu" , padding ="same"),
        #32 feature maps , kernel of size 3x3 , same padding preseves dim with stride 1
        # so (28 x 28 x 1) becomes (28 x 28 x 32)
        MaxPooling2D(pool_size = 2),
        #takes convo2d and changes to (14 x 14 x 32)
        Conv2D(filters = 64 , kernel_size= 3 , activation= "relu", padding= "same"),
        MaxPooling2D(pool_size = 2),
        #flattening so that NN can understand it as vectors
        Flatten(),
        Dense(units = 128 , activation = "relu"),
        Dropout(rate=0.3), #to reduce overfitting , randomly sets 30% of IP activations to 0
        Dense(units = 10 , activation = "softmax")
        #softmax returns probablistic distribution for all 10 class we make
        #adding all values gives 1
    ])

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()