import os
import math
from math import floor
import numpy
import sklearn
import tensorflow
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras.layers import MultiHeadAttention, Dense, Input, Dropout, BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclass_global import G
from model import *
from transformer_helper_dc import *
from rolling_and_plot_dc import data_plot, rolling_split, normalize, validate

tf.config.run_functions_eagerly(True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


l = ['AAPL','NVDA','GOOG'] # a list of stock tickers to be processed.

for i in l:
    
    START = "2000-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    df_raw = acquire_ticker(i, START, TODAY)
    df_raw = df_raw.sort_values('Date')
    df_raw = df_raw[['Date','Open','High','Low','Close','Volume','Adj Close']]

    division_rate1 = 0.8
    division_rate2 = 0.9

    seq_len = G.src_len  # 20 how long of a preceeding sequence to collect for training
    tgt = G.tgt_len
    mulpre = G.mulpr_len  # how far into the future are we trying to predict?
    window = G.window_size



    df,list,list1 = get_stock_data(df_raw)
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(df, seq_len, mulpre, division_rate1, division_rate2,tgt)
    tf.keras.backend.clear_session()
    model = Transformer()
    model.build(input_shape=[None, G.window_size, G.num_features])
    #model.summary(expand_nested=True)


    G.T_i = 1
    G.T_mult = 2
    G.T_cur = 0.0

    loss_object = tf.keras.losses.LogCosh()

    optimizer = tf.keras.optimizers.Adam(learning_rate = G.learning_rate,
                                         beta_1 = 0.9,
                                         beta_2 = 0.999
                                        )


    model.compile(loss=up_down_accuracy, optimizer=optimizer, metrics=["mse"])
    

    #early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min', restore_best_weights=True)

    cwd = os.getcwd()  # Get current working directory
    checkpoint_file = 'ckpt/model_checkpoint_' + i + '.keras'
    checkpoint_path = os.path.join(cwd, checkpoint_file)
    #model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=False, mode='auto')

    history = model.fit(X_train,y_train,
                        epochs = G.epochs,
                        batch_size=G.batch_size,
                        verbose = 1,
                        validation_data=(X_valid, y_valid),
                        callbacks=[model_checkpoint]
                        )
    

    # Plot the training history:

    epoch_start_idx = 3 # skip the first 3 epochs for plotting
    train_hist_x = [k for k in range (epoch_start_idx, len(history.history['loss']))]

    plt.plot(train_hist_x,history.history['loss'][epoch_start_idx:])
    plt.plot(train_hist_x,history.history['val_loss'][epoch_start_idx:])

    plt.title('Losses vs Epochs')


    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


    predicted_stock_price_multi_head = model.predict(X_test)


    testdata_len = y_test.shape[0]
    predicted_stock_price_multi_head_dff1, predicted_stock_price_multi_head = denormalize(i, df_raw, predicted_stock_price_multi_head,division_rate1,division_rate2, seq_len, mulpre, testdata_len, cwd, True) # denormalize model predictions of the test set.
    y_test_dff1, y_test = denormalize(i, df_raw, y_test,division_rate1,division_rate2, seq_len, mulpre, testdata_len, cwd, False) # denormalize ground truth of the test set.


    accu = np.multiply(predicted_stock_price_multi_head_dff1,y_test_dff1)

    accu = np.maximum(accu, 0)

    accu = np.sign(accu)

    accu = np.mean(accu) * 100


    #print('predicted_stock_price_multi_head.shape',predicted_stock_price_multi_head.shape)
    print('predicted_stock_price_multi_head',predicted_stock_price_multi_head)

    predicted_stock_price_multi_head = numpy.ravel(predicted_stock_price_multi_head)

    y_test = numpy.ravel(y_test)

    calculate_accuracy(predicted_stock_price_multi_head, y_test)
    print(f'Accuracy of the test dataset is: {accu}%')

    plt.ion()
    plt.figure(figsize = (18,9))
    plt.plot(y_test, color = 'black', label = 'ground truth')
    plt.plot(predicted_stock_price_multi_head, color = 'green', label = 'predicted')
    plt.title('Adj closing Price Prediction', fontsize=30)
    plt.xlabel('Date')
    plt.ylabel('Adj closing Price')
    plt.legend(fontsize=18)
    plt.show()
    plt.close()

    # Next 3 days prediction:

    cwd = os.getcwd()  # Get current working directory
    scaler_file = 'output/standard_train_scaler_' + i + '.pkl'
    scaler_path = os.path.join(cwd, scaler_file)
    standard_scaler = joblib.load(scaler_path)

    y_pred_next_days = predict_next_days(i, model, standard_scaler)

    print(f'Next 3 days predicted prices for {i}:', y_pred_next_days, end = '\n\n\n')


