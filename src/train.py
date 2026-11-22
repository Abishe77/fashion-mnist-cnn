#importing the modules
from preprocessing import load_and_preprocess
from model import build_model
from tensorflow.keras.optimizers import Adam
#for storing the model weights with min validation loss
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path #to store the model under artifacts as keras file


def train():
    X_train , X_val , X_test , y_train , y_val , y_test = load_and_preprocess()
    model = build_model()

    model.compile(
        optimizer = Adam(learning_rate = 0.001), # the optimiser we use for updation of weights and biases
        loss = "sparse_categorical_crossentropy", #the loss function
        metrics = ["accuracy"] #the score we are interested in
    )

    artifact_dir = Path(__file__).resolve().parent.parent / "artifacts" 

    early_stopping = EarlyStopping(
        monitor="val_loss" ,
        patience= 3,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
    filepath=  artifact_dir / "fashion_cnn.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min"
    )

    #training the actual model
    history = model.fit(
        X_train,
        y_train,
        #the above is image and corresponding labels
        validation_data = (X_val,y_val),
        #the above evaluates performance on unseen val examples
        epochs = 15,
        #15 training loops or passes
        batch_size = 64,
        #processes upto 64 image per gradient update
        callbacks = [early_stopping ,checkpoint]
    )

if __name__ == "__main__":
    train()

