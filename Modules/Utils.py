import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from colorama import Fore
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import exc, create_engine

POSTGRESQL_URL = 'postgresql://augeias:augeias@83.212.19.17:5432/augeias'


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def split_sequences(features, targets, n_steps_in, n_steps_out, n_sliding_steps, window_type):
    # https://www.kaggle.com/code/iamleonie/time-series-forecasting-building-intuition/notebook#Fundamentals
    """
    Edited from
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

    Args:
    * features: univariate or multivariate input sequences
    * targets: univariate or multivariate output sequences
    * n_steps_in: length of input sequence for sliding window.
    * n_steps_out: length of output sequence
    * n_sliding_steps: window step size
    * window_type: 'sliding' or 'expanding'
    """
    X, y = list(), list()

    for i in range(0, len(features), n_sliding_steps):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # print(end_ix, out_end_ix, len(features))

        # check if we are beyond the sequences
        if out_end_ix > len(features):
            break

        # gather input and output parts of the pattern
        if window_type == 'sliding':  # Sliding window
            seq_x, seq_y = features[i:end_ix, :], targets[end_ix:out_end_ix]
        else:  # expanding window
            seq_x, seq_y = features[0:end_ix, :], targets[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def daypart(hour):
    if hour in [2, 3, 4, 5]:
        return "dawn"
    elif hour in [6, 7, 8, 9]:
        return "morning"
    elif hour in [10, 11, 12, 13]:
        return "noon"
    elif hour in [14, 15, 16, 17]:
        return "afternoon"
    elif hour in [18, 19, 20, 21]:
        return "evening"
    else:
        return "midnight"

def fill_missing(data:pd.DataFrame):
    data.index = pd.to_datetime(data.index)

    min_timestamp = data.index.min()
    max_timestamp = data.index.max()
    data = data.resample('1h').asfreq().reindex(pd.date_range(min_timestamp, max_timestamp, freq='1h')).fillna(
        method='ffill')
    return data
    date_string = _out.index
def create_multivariate_data(data: pd.DataFrame):
    _out = data.copy()
    _out = fill_missing(_out)
    hour = pd.to_datetime(_out.index, utc=True).hour
    month = pd.to_datetime(_out.index, utc=True).month
    _out['hour'] = hour.values
    # _out['month'] = month.values
    # _out['daypart'] = hour.values
    # label_encoder = LabelEncoder()
    # _out['daypart'] = label_encoder.fit_transform(_out['daypart'].apply(daypart))
    #
    # _out['t-1'] = _out['mape_temp'].shift(1)
    # _out.dropna(inplace=True)
    # a = pd.date_range(_out.index.min(), _out.index.max(), freq='H').difference(_out.index)
    # print(_out)
    # _out = _out.diff(2)
    # print(_out)
    # _out.dropna(inplace=True)
    return _out


def create_prediction_data(data: pd.DataFrame, scaler, n_steps_in, features):
    _out = data.copy()
    hour = pd.to_datetime(_out.index, utc=True).hour
    _out['hour'] = hour.values
    _out = _out.to_numpy().reshape(1, _out.shape[0] * _out.shape[1])
    # _out = scaler.transform(_out)
    _out = _out.reshape(n_steps_in, features)
    _out = _out.reshape(1, _out.shape[0], _out.shape[1])
    return _out


def print_line(line_length: int = 100):
    print(f"{Fore.WHITE}-" * line_length)


def print_versions():
    print_line()
    print(f"{Fore.CYAN}Tensorflow Version: {tf.__version__}")
    print(f"{Fore.CYAN}Pandas Version: {pd.__version__}")
    print(f"{Fore.CYAN}Numpy Version: {np.__version__}")
    print(f"{Fore.CYAN}System Version: {sys.version}")
    print_line()


def get_data(table_name: str, col_name: str, source: str = None):
    if source is None:
        data = pd.read_csv(f"Data/{table_name}.csv", index_col="timestamp", usecols=["timestamp", col_name])
        return data
    elif source == 'database':
        engine = create_engine(POSTGRESQL_URL)
        print(f"getting data from {table_name}")
        sql = f"""SELECT * FROM {table_name} order by timestamp """
        try:
            data = pd.read_sql(sql, con=engine, index_col="timestamp")
            data.to_csv(f"Data/{table_name}.csv")
            return data[[col_name]]
        except exc.SQLAlchemyError as e:
            print(e)
