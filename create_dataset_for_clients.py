import pandas as pd
import os
from utils import FLconfig
# downloaded csv for train and test set interchanged, test-set has 175341 entries and train-set has 82,332
path_to_train_set = "./data/from_unsw_website/UNSW_NB15_training-set.csv"
path_to_test_set = "./data/from_unsw_website/UNSW_NB15_testing-set.csv"
number_of_clients = 4
path_to_client_dataset = "./data/client_dataset"
if not os.path.isdir(path_to_client_dataset):
    os.makedirs(path_to_client_dataset,exist_ok=True)

def partition_dataframe(df,number_of_clients):
    shuffled_df = df.sample(frac=1, random_state=42)  # Shuffle the DataFrame randomly

    # Reset the index of the shuffled DataFrame
    df = shuffled_df.reset_index(drop=True)
    # Calculate the size of each partition
    partition_size = (len(df) // number_of_clients)+1

    # Partition the dataframe
    for idx,i in enumerate(range(0, len(df), partition_size)):
        partition = df[i:i+partition_size]
        partition.to_csv(os.path.join(path_to_client_dataset,f"client_{idx}.csv"),index=False)




def create_categorical_dict_and_change_values_of_dataframe(df_train_test):

    # create dictionary of binary mapping of each column
    categorical_columns_dict = {i:{} for i in FLconfig.categorical_columns_for_dataset}
    for cat_col in categorical_columns_dict.keys():
        unique_entries_for_col = df_train_test[cat_col].unique()

        categorical_columns_dict[cat_col] = {k: v for k, v in zip(unique_entries_for_col, [i for i in range(len(unique_entries_for_col))])}


    # for train and test dataframe, map the values of each column in dataframe with dictionary and replace
    for cat_col in FLconfig.categorical_columns_for_dataset:
        df[cat_col] = df[cat_col].replace(categorical_columns_dict[cat_col])

    for cat_col in FLconfig.categorical_columns_for_dataset:
        df_test[cat_col] = df_test[cat_col].replace(categorical_columns_dict[cat_col])

    # write the csv to data directory
    df.to_csv(f"./data/centralized_data.csv",index=False)
    df_test.to_csv(f"./data/centralized_test_data.csv",index=False)

df = pd.read_csv(path_to_train_set)
df_test = pd.read_csv(path_to_test_set)
df_train_test = pd.concat([df,df_test]).copy()

create_categorical_dict_and_change_values_of_dataframe(df_train_test)

partition_dataframe(df,number_of_clients)










