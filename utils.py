import pandas as pd
import numpy as np

from sklearn.preprocessing import Normalizer

from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization

from tensorflow.keras import regularizers


from dataclasses import dataclass

@dataclass
class FLconfig:
    """
    Configuration class for federated learning.
    """

    # Learning rate for the model
    LEARNING_RATE: float = 0.001

    # Number of local epochs for each round of federated learning
    local_epochs: int = 3

    # Number of rounds in the federated learning process
    num_rounds: int = 10

    # Batch size used in each round of federated learning
    batch_size: int = 1000

    # Directory path for storing experiment-related files
    specified_dir: str = "./experiments"

    # Minimum number of clients required for evaluation
    min_evaluate_clients: int = 4

    # Minimum number of clients required for model fitting
    min_fit_clients: int = 4

    # Minimum number of available clients required for the federated learning process
    min_available_clients: int = 4

    # Categorical columns to be used in the dataset
    categorical_columns_for_dataset = [
        "proto", "service", "state", "service", "sttl",
        "dttl", "swin", "dwin", "trans_depth", "ct_ftp_cmd",
        "is_ftp_login", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
        "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
        "ct_dst_src_ltm", "ct_state_ttl", "ct_flw_http_mthd",
    ]



def preprocess_dataset(path_to_csv):
    """
    Preprocesses the dataset from a CSV file.

    Parameters:
    - path_to_csv: Path to the CSV file containing the dataset.

    Returns:
    - X: Preprocessed input features.
    - Y: Labels.
    """

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(path_to_csv)

    # Assign the DataFrame to the traindata variable
    traindata = df

    # Drop the "attack_cat" column, as it has a value only when the label is not 0
    traindata = traindata.drop(["attack_cat"], axis=1)

    # Split the input features (X) and labels (Y)
    X = traindata.iloc[:, 1:43]
    Y = traindata.iloc[:, 43:]

    # Extract categorical columns from X
    cat_X = X[FLconfig.categorical_columns_for_dataset]

    # Drop the categorical columns from X
    X = X.drop(FLconfig.categorical_columns_for_dataset, axis=1)
    X_columns_without_cat = X.columns

    # Normalize the numerical features using a Normalizer
    scaler = Normalizer().fit(X)
    X = scaler.transform(X)
    X = pd.DataFrame(X, columns=X_columns_without_cat)

    # Concatenate the normalized numerical features (X) with the categorical features (cat_X)
    X = pd.concat([X, cat_X], axis=1)

    return X, Y


def lr_schedule_func(rnd):
    """
    Learning rate schedule function for federated learning.

    Parameters:
    - rnd: Round number.

    Returns:
    - lr: Learning rate.
    """
    initial_learning_rate = 0.001  # Initial learning rate value
    decay_rate = 0.5  # Rate at which the learning rate decays
    decay_steps = 2  # Number of steps after which the learning rate decays

    lr = initial_learning_rate * decay_rate ** (rnd // decay_steps)  # Calculating the learning rate

    return lr

def lr_schedule_func_dummy(epoch, lr):
    """
    A dummy learning rate schedule function that returns the same learning rate value it receives.

    Parameters:
    - epoch: Current epoch number.
    - lr: Current learning rate.

    Returns:
    - lr: Unchanged learning rate.
    """
    return lr


def get_model():
    """
    Returns a compiled model.

    Returns:
    - model: Compiled model.
    """

    # Create a sequential model
    model = Sequential()

    # Add a fully connected layer with 1024 units and ReLU activation
    model.add(Dense(1024, input_dim=43, activation='relu'))

    # Apply batch normalization to the previous layer's output
    model.add(BatchNormalization())

    # Apply dropout regularization with a rate of 0.01
    model.add(Dropout(0.01))

    # Add another fully connected layer with 768 units, hyperbolic tangent (tanh) activation, and L2 regularization
    model.add(Dense(768, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))

    # Apply batch normalization
    model.add(BatchNormalization())

    # Apply dropout regularization
    model.add(Dropout(0.01))

    # Add another fully connected layer with 512 units, tanh activation, and L2 regularization
    model.add(Dense(512, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))

    # Apply batch normalization
    model.add(BatchNormalization())

    # Apply dropout regularization
    model.add(Dropout(0.01))

    # Add another fully connected layer with 256 units, tanh activation, and L2 regularization
    model.add(Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))

    # Apply batch normalization
    model.add(BatchNormalization())

    # Apply dropout regularization
    model.add(Dropout(0.01))

    # Add another fully connected layer with 128 units, tanh activation, and L2 regularization
    model.add(Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))

    # Apply batch normalization
    model.add(BatchNormalization())

    # Apply dropout regularization
    model.add(Dropout(0.01))

    # Add a final fully connected layer with 1 unit for binary classification
    model.add(Dense(1))

    # Apply sigmoid activation to obtain probabilities
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model





