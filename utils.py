from typing import Union

import pandas as pd
import numpy as np
from pathlib import Path


def read_data(name_train, name_test, data_dir: Union[str, Path] = "data",
              timestamp_col_name: str = "timestamp") -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object

    Args:

        data_dir: str or Path object specifying the path to the directory 
                  containing the data

        target_col_name: str, the name of the column containing the target variable

        timestamp_col_name: str, the name of the column or named index 
                            containing the timestamps
    """

    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob("*.csv"))

    for csv_file in csv_files:
        if name_train in csv_file.stem:
            data_path_train = csv_file
        if name_test in csv_file.stem:
            data_path_test = csv_file
    #print("Reading file in {}".format(data_path))

    train_data = pd.read_csv(
        data_path_train,
        parse_dates=[timestamp_col_name],
        index_col=[timestamp_col_name],
        infer_datetime_format=True,
        low_memory=False
    )
    test_data = pd.read_csv(
        data_path_test,
        parse_dates=[timestamp_col_name],
        index_col=[timestamp_col_name],
        infer_datetime_format=True,
        low_memory=False
    )

    # Make sure all "n/e" values have been removed from df.
    if is_ne_in_df(train_data):
        raise ValueError(
            "data frame contains 'n/e' values. These must be handled")
    if is_ne_in_df(test_data):
        raise ValueError(
            "data frame contains 'n/e' values. These must be handled")

    train_data = to_numeric_and_downcast_data(train_data)
    test_data = to_numeric_and_downcast_data(test_data)

    # Make sure data is in ascending order by timestamp
    #data.sort_values(by=[timestamp_col_name], inplace=True)

    return train_data, test_data, data_path_train.stem, data_path_test.stem


def is_ne_in_df(df: pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """

    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns

    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')

    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df


def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences. 

    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences. 

    Args:
        num_obs (int): Number of observations (time steps) in the entire 
                       dataset for which indices must be generated, e.g. 
                       len(data)

        window_size (int): The desired length of each sub-sequence. Should be
                           (input_sequence_length + target_sequence_length)
                           E.g. if you want the model to consider the past 100
                           time steps in order to predict the future 50 
                           time steps, window_size = 100+50 = 150

        step_size (int): Size of each step as the data sequence is traversed 
                         by the moving window.
                         If 1, the first sub-sequence will be [0:window_size], 
                         and the next will be [1:window_size].

    Return:
        indices: a list of tuples
    """

    stop_position = len(data)-1  # 1- because of 0 indexing

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0

    subseq_last_idx = window_size

    indices = []

    while subseq_last_idx <= stop_position:

        indices.append((subseq_first_idx, subseq_last_idx))

        subseq_first_idx += step_size

        subseq_last_idx += step_size

    return indices


def prepare_elevation_profile(jump, df, column_to_be_shifted):
    jump = int(jump)
    new_column = np.zeros(len(df))
    new_column[jump:] = df.iloc[:-jump, column_to_be_shifted].to_numpy()
    return new_column
