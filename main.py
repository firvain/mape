import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from colorama import Fore, init
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sqlalchemy import exc
from keras.models import load_model
from sklearn.metrics import mean_squared_error

from Modules.Utils import set_seed, print_versions, get_data, create_multivariate_data, print_line, split_sequences, \
    create_prediction_data, fill_missing
"""
Overfitting if: training loss << validation loss
Underfitting if: training loss >> validation loss
Just right if training loss ~ validation loss
"""
mpl.rcParams['figure.figsize'] = (17, 5)
mpl.rcParams['axes.grid'] = False

init(autoreset=True)

plot_history = True
plot_raw = False

if __name__ == '__main__':
    set_seed(42)
    print_versions()

    col_name = 'mape_temp'
    table_name = "openweather_addvantage_mape"
    raw_data = get_data(source="database", table_name=table_name, col_name=col_name)

    print(f'{Fore.BLUE}raw_data.shape: ', raw_data.shape)

    if plot_raw:
        plt.plot(raw_data)
        plt.show()

    multi_data = create_multivariate_data(raw_data)
    multi_data = multi_data.iloc[:-48]

    print(f'{Fore.YELLOW}multi_data.shape: ', multi_data.shape)
    features = multi_data.shape[1]
    print(f"no of features: {features}")

    print_line()

    n_steps_in = 1
    n_steps_out = 24

    sets = int(multi_data.shape[0] / n_steps_in)
    print(f"{Fore.MAGENTA}sets: {sets}")
    train_dataset = multi_data.iloc[:round(sets * .7) * n_steps_in]
    train_target = multi_data.iloc[:, 0].iloc[:round(sets * .7) * n_steps_in]
    train_dataset.to_csv(f'{col_name}_train.csv')
    print(f"{Fore.MAGENTA}train_dataset.shape: {train_dataset.shape}")
    print(f"{Fore.MAGENTA}train_target.shape: {train_target.shape}")
    test_dataset = multi_data.iloc[round(sets * .7) * n_steps_in:]
    test_dataset.to_csv(f'{col_name}_test.csv')
    test_target = multi_data.iloc[:, 0].iloc[round(sets * .7) * n_steps_in:]
    print(f"{Fore.MAGENTA}test_dataset.shape: {test_dataset.shape}")
    print(f"{Fore.MAGENTA}test_target.shape: {test_target.shape}")
    print_line()

    output_cols = [col_name]

    x_train_multi, y_train_multi = split_sequences(train_dataset.values,
                                                   train_dataset[output_cols].values,
                                                   n_steps_in=n_steps_in,
                                                   n_steps_out=n_steps_out,
                                                   n_sliding_steps=1,
                                                   window_type='sliding')
    x_test_multi, y_test_multi = split_sequences(test_dataset.values,
                                                 test_dataset[output_cols].values,
                                                 n_steps_in=n_steps_in,
                                                 n_steps_out=n_steps_out,
                                                 n_sliding_steps=1,
                                                 window_type='sliding')

    print(f"{Fore.YELLOW}x_train_multi.shape: {x_train_multi.shape}")
    print(f"{Fore.YELLOW}y_train_multi.shape: {y_train_multi.shape}")
    print(f"{Fore.YELLOW}x_test_multi.shape: {x_test_multi.shape}")
    print(f"{Fore.YELLOW}y_test_multi.shape: {y_test_multi.shape}")
    print_line()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    x_train_multi = x_train_multi.reshape(x_train_multi.shape[0], x_train_multi.shape[1] * x_train_multi.shape[2])
    x_train_multi = scaler_X.fit_transform(x_train_multi)
    x_train_multi = x_train_multi.reshape(x_train_multi.shape[0], n_steps_in, features)

    y_train_multi = y_train_multi.reshape(y_train_multi.shape[0], y_train_multi.shape[1] * y_train_multi.shape[2])
    y_train_multi = scaler_y.fit_transform(y_train_multi)
    y_train_multi = y_train_multi.reshape(y_train_multi.shape[0], n_steps_out, 1)

    x_test_multi = x_test_multi.reshape(x_test_multi.shape[0], x_test_multi.shape[1] * x_test_multi.shape[2])
    x_test_multi = scaler_X.transform(x_test_multi)
    x_test_multi = x_test_multi.reshape(x_test_multi.shape[0], n_steps_in, features)

    y_test_multi = y_test_multi.reshape(y_test_multi.shape[0], y_test_multi.shape[1] * y_test_multi.shape[2])
    y_test_multi = scaler_y.transform(y_test_multi)
    y_test_multi = y_test_multi.reshape(y_test_multi.shape[0], n_steps_out, 1)
    print(f"{Fore.YELLOW}x_train_multi.shape: {x_train_multi.shape}")
    print(f"{Fore.YELLOW}y_train_multi.shape: {y_train_multi.shape}")
    print(f"{Fore.YELLOW}x_test_multi.shape: {x_test_multi.shape}")
    print(f"{Fore.YELLOW}y_test_multi.shape: {y_test_multi.shape}")

    ### MODEL
    # reshape to [samples, timesteps, features]
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0
    )
    opt = Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(
        x_train_multi.shape[1], x_train_multi.shape[2])))
    model.add(Dropout(.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    history = model.fit(x_train_multi, y_train_multi, epochs=500, verbose=1,
                        validation_data=(x_test_multi, y_test_multi),
                        callbacks=[callback])
    if plot_history:
        plt.plot(history.history["val_loss"], label="val_loss - validate", color="orange")
        plt.plot(history.history["loss"], label="loss - test", color="blue")
        plt.legend()
        plt.show()

    model.save(f"Models/Multivariate/{col_name}.h5")

    # print_line()
    # prediction_data_raw = raw_data.iloc[-48:-24]
    # prediction_data_raw.index = pd.to_datetime(prediction_data_raw.index)
    #
    # print(prediction_data_raw)
    # prediction_data = create_prediction_data(prediction_data_raw, scaler_X, n_steps_in, features)
    # predictions = []
    # for i in range(0, prediction_data_raw.shape[0]):
    #     if i == 0:
    #         prediction_data = create_prediction_data(prediction_data_raw, scaler_X, n_steps_in, features)
    #         yhat = model.predict(prediction_data)
    #         predictions.append(yhat[0,0])
    #     else:
    #
    #         last_date = prediction_data_raw.index[-1] + pd.Timedelta(hours=1)
    #         a = prediction_data_raw.copy()
    #         a.loc[last_date] = predictions[-1]
    #         prediction_data_raw = a
    #         prediction_data_raw = prediction_data_raw.iloc[1:, :]
    #         prediction_data = create_prediction_data(prediction_data_raw, scaler_X, n_steps_in, features)
    #         print(prediction_data_raw)
    #         # input()
    #         yhat = model.predict(prediction_data)
    #         predictions.append(yhat[0, 0])
    # print(predictions)
    # gt = raw_data.iloc[-24:]
    # gt['predictions'] = predictions
    # gt.plot()
    # plt.show()
    # quit()
    prediction_data_raw = raw_data.iloc[-(n_steps_in + n_steps_out):-n_steps_out]
    prediction = fill_missing(prediction_data_raw)

    prediction = create_multivariate_data(prediction).values
    print(f"prediction.shape: {prediction.shape}")
    # print(prediction.values.reshape(1, -1).shape)

    prediction = scaler_X.transform(prediction.reshape(1, prediction.shape[0] * prediction.shape[1]))
    print(f"prediction.shape: {prediction.shape}")

    # model = tf.keras.models.load_model(f"Models/Multivariate/{col_name}.h5")
    yhat = model.predict(prediction.reshape(1, n_steps_in, features))
    print(f"yhat.shape: {yhat.shape}")
    yhat = scaler_y.inverse_transform(yhat)
    # print(yhat)
    gt = fill_missing(raw_data.iloc[-n_steps_out:])
    # print(gt)

    # y_test_multi = y_test_multi.reshape(y_test_multi.shape[0], y_test_multi.shape[1] * y_test_multi.shape[2])
    plt.plot(gt, label="Ground Truth", color="green")
    plt.plot(gt.index.to_numpy(), yhat.reshape(n_steps_out, 1), label="prediction", color='blue')
    plt.title(f'{col_name} features_in={n_steps_in}')
    plt.legend()
    plt.show()
