import os, time, argparse
import flwr as fl
import pandas as pd
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils import get_model,FLconfig,preprocess_dataset, lr_schedule_func

def get_eval_fn(model):

    """Return an evaluation function for server-side evaluation."""

    print("-----------------------------------------")
    print('Evaluation at server side has started....!!!')
    print("-----------------------------------------")


    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters_ndarrays, evaluate_config):

        """
        Responsible for setting the weights and saving them in server directory
        """
        
        model.set_weights(parameters_ndarrays)  # Update model with the latest parameters

        # check if logs file exists ; it will exist if round > 1
        if os.path.isfile(os.path.join(model.current_experiment_dir,f'logs.csv')):
            df = pd.read_csv(os.path.join(model.current_experiment_dir,f'logs.csv'))
        else:
            df = pd.DataFrame(columns=["round","test_loss","test_accuracy"])
        # evaluate the model on test set
        res = model.evaluate(model.X,model.y)
        # get loss and acc from evaluation results
        loss, acc = res
        # create a dictionary for metrics
        metrics_dict = {

                "round":[server_round],
                "test_loss":loss,
                "test_accuracy":acc
            }
        # create a new dataframe from it 
        # concatenate this dataframe with read dataframe or empty dataframe
        df = pd.concat([df,pd.DataFrame(metrics_dict)])
        # save the dataframe
        df.to_csv(os.path.join(model.current_experiment_dir,f'logs.csv'),index=False)
        # save the weights of model into the server directory of current experiment
        model.save_weights(os.path.join(model.server_weights,f"ROUND_{model.round_no}.h5"))
        # increment the round_no
        model.round_no += 1
        # random values of evaluation requited by flwr framework
        return float(0.1),{}
        

    return evaluate


def get_model_init_weights(model):
    """
    Returns the initial weights of model
    """
    # get the initial weights of model
    weights = model.get_weights()

    return weights

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """

    global model
    # val_steps = 5 if rnd < 4 else 10
    return {
        "experiment_dir": model.current_experiment_dir,
        "round_no":rnd,
        }


def fit_config(rnd: int):
    """
    Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """

    global model
    
    '''Adding learning rate decay over rounds'''
    # make the config which will be sent to client for training

    lr = lr_schedule_func(rnd)

    server_config = {
        "experiment_dir": model.current_experiment_dir, 
        "round_no":rnd, 
        "LEARNING_RATE": lr,
        "local_epochs": FLconfig.local_epochs
        } 
        
    return server_config

class FedAvg_with_saving_logs(fl.server.strategy.FedAvg):

    """
    Responsible for writing logs in the server directory for current experiment, 
    aggregating weights is done by super class provided by flwr framewor
    """

    def aggregate_fit(self,server_round,results,failures):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        return aggregated_parameters, aggregated_metrics

def main():
    # set path to the test data csv
    path_to_csv = "./data/centralized_test_data.csv"
    # making a global model variable
    global model

    '''get compiled model'''
    model = get_model()

    '''adding round_no attribute to model'''
    model.round_no = 0
    specified_dir = FLconfig.specified_dir
    if os.path.exists(specified_dir) is False:
        os.makedirs(specified_dir)

    '''get current experiment_no and setting up directory'''
    experiment_names = os.listdir(specified_dir)
    current_experiment_name = len(experiment_names)+1
    model.current_experiment_dir = os.path.join(specified_dir,f"experiment_{current_experiment_name}")
    model.server_weights = os.path.join(model.current_experiment_dir,"server_weights")
    if not os.path.exists(model.current_experiment_dir):
        os.makedirs(model.current_experiment_dir)
    if not os.path.exists(model.server_weights):
        os.makedirs(model.server_weights)

    # get the preprocessed test set dataset
    X,y = preprocess_dataset(path_to_csv)
    # add test set to model as variables
    model.X,model.y = X,y

    # make custom strategy for model which will average weights of each client and set weights on new model 
    strategy = FedAvg_with_saving_logs(
        fraction_fit=1.,
        fraction_evaluate=1.,
        min_evaluate_clients=FLconfig.min_evaluate_clients,
        min_fit_clients=FLconfig.min_fit_clients,
        min_available_clients=FLconfig.min_available_clients,
        evaluate_fn=get_eval_fn(model),
        initial_parameters =fl.common.ndarrays_to_parameters(get_model_init_weights(model)),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )
    # start the server
    server_config = fl.server.ServerConfig(num_rounds=FLconfig.num_rounds)
    fl.server.start_server(server_address="127.0.0.1:8000", config=server_config, strategy=strategy)
    


if __name__ == "__main__":
    

    main()
