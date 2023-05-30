# -*- coding: utf-8 -*-


import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils import FLconfig, preprocess_dataset, ModelCheckpoint, lr_schedule_func_dummy
import argparse, os
import numpy as np
import pandas as pd
import flwr as fl
from utils import get_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam



# Define Flower client
class M103Client(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, path_to_csv, client_no):

        """
        initialize the flower client
        """

        # Set the path to the CSV file
        self.path_to_csv = path_to_csv

        # Set the model
        self.model = model

        # Set the training data
        self.X_train, self.y_train = X_train, y_train

        # Set the validation data
        self.X_val, self.y_val = X_val, y_val

        # Set the client number
        self.client_no = client_no

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")
    
    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # weights file name
        
        # Update local model weights
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        epochs: int = config["local_epochs"]
        round_no = config['round_no']
        experiment_dir = config['experiment_dir']
        LEARNING_RATE = config['LEARNING_RATE']
        batch_size: int = FLconfig.batch_size

        
        # create client directory
        client_directory = os.path.join(experiment_dir,f"client_{self.client_no}")
        # create logs path
        logs_path = os.path.join(client_directory,"logs.csv")
        # create weights directory path for client
        weights_directory_for_client = os.path.join(client_directory,"weights")
        # create a csv logger
        csv_logger = tf.keras.callbacks.CSVLogger(logs_path, separator=',', append=True)
        # create a learning reate scheduler
        lr_scheduler = LearningRateScheduler(lr_schedule_func_dummy)
        # get the name of csv provided
        weights_file_name = os.path.abspath(path_to_csv).split(os.sep)[-1].split('.')[0]
        # create a callback for model checkpoints
        checkpoint_callback = ModelCheckpoint(weights_directory_for_client,save_best_only=True,monitor = "loss")
        # Train the model using hyperparameters from config
        self.model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=LEARNING_RATE),metrics=['accuracy'])
        
        history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size,
            epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[checkpoint_callback, csv_logger, lr_scheduler]
        )

        
        

        # get updated model parameters
        parameters_prime = self.model.get_weights()
        # get the number of examples in the training set
        num_examples_train = len(self.X_train)

        # create dictionary to be returned to server
        results = {
            "loss": fl.common.ndarray_to_bytes(np.asarray(history.history["loss"])),
            "accuracy": fl.common.ndarray_to_bytes(np.asarray(history.history["accuracy"])),
            "client_num": self.client_no,
        }

        return parameters_prime,num_examples_train,results


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)
    
        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val)
        num_examples_test = len(self.X_val)
        return loss, num_examples_test, {"accuracy": accuracy}

def main(path_to_csv,client_no) -> None:
    # Get the compiled model
    model = get_model()

    # Preprocess the dataset and get train-test-validation split
    X, y = preprocess_dataset(path_to_csv)

    # Split the data into train and validation sets
    train_end_val_start = int(len(X) * .85)
    X_train, y_train = X[:train_end_val_start], y[:train_end_val_start]
    X_val, y_val = X[train_end_val_start:], y[train_end_val_start:]

    # Create a Flower client object and start the Flower client
    client = M103Client(model, X_train, y_train, X_val, y_val, path_to_csv, client_no)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8000",
        client=client,
    )


if __name__ == "__main__":

    # Define the argument parser to receive arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--PATH_TO_CSV", help="PATH OF CSV")  # Argument for the path to the CSV file
    parser.add_argument("-c", "--CLIENT_NO", help="The number assigned to client")  # Argument for the client number
    args = parser.parse_args()

    # Get the arguments provided via the command line
    path_to_csv = args.PATH_TO_CSV  # Path to the CSV file
    client_no = args.CLIENT_NO  # Client number

    # Get the absolute path of the CSV file
    path_to_csv = os.path.abspath(path_to_csv)

    # Check if the path to the CSV file is valid
    if path_to_csv is None or not path_to_csv.endswith('.csv') or not os.path.exists(path_to_csv):
        raise Exception("PATH_TO_CSV variable must be provided via command line and the path must exist and end with .csv")

    # Call the main function with the provided arguments
    main(path_to_csv=path_to_csv, client_no=client_no)
