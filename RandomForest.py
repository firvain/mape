# Importing the libraries
import numpy as np  # for array operations
import pandas as pd
import matplotlib.pyplot as plt  # for data visualization
import matplotlib as mpl
from colorama import Fore
from numpy import asarray

from Modules.Utils import get_data, set_seed, fill_missing

# scikit-learn modules
from sklearn.model_selection import train_test_split  # for splitting the data
from sklearn.metrics import mean_squared_error, mean_absolute_error  # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor  # for building the model

set_seed(42)
mpl.rcParams['figure.figsize'] = (11.69, 8.27)
mpl.rcParams['axes.grid'] = False


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit a random forest model and make a one-step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)

    # seed history with training dataset
    history = [x for x in train]

    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = random_forest_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print(f'>expected={testy}, predicted={yhat}')
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


# load the dataset
col_name = 'mape_temp'
table_name = "accuweather_addvantage_mape"
n_in = 24
n_out = 24
series = get_data(source="database", table_name=table_name, col_name=col_name)

# series = fill_missing(series)
values = series.values[:-n_out]


train = series_to_supervised(values, n_in=n_in)

trainX, trainy = train[:, :-1], train[:, -1]

# fit model
model = RandomForestRegressor(n_estimators=1000, random_state=0)
model.fit(trainX, trainy)
predictions = []


def spl(n_in, n_out, values):
    row = values[-n_in:]
    for i in range(0, n_out):
        row = row[-n_in:].flatten()
        # print(row)
        yhat = model.predict(asarray([row]))
        # print('Input: %s, Predicted: %.10f' % (row, yhat[0]))
        row = np.append(row, yhat[0])
        # print(row)
        predictions.append(yhat[0])


spl(n_in=n_in, n_out=n_out, values=values)

MAE = mean_absolute_error(series.values[-n_out:].flatten(), predictions)
plt.xticks(rotation=90)

plt.plot(series.iloc[-n_out:].index.to_numpy(), series.values[-n_out:].flatten(), label='Ground Truth', color='green')
plt.plot(series.iloc[-n_out:].index.to_numpy(), predictions, color='blue', label='Predictions')
plt.title(f'Variable: {col_name} - Dataset:{table_name} - MAE: {MAE}')
plt.legend()
plt.tight_layout()
plt.show()
