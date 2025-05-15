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


l = ['000001.SS', 'AAPL', 'BTC-USD' , 'DJI', 'Gold_daily','GSPC','IXIC']

for i in l:
    
    START = "2000-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    df_raw = acquire_ticker('AAPL', START, TODAY)
    df_raw = df_raw.sort_values('Date')
    df_raw = df_raw[['Date','Open','High','Low','Close', 'Adj Close','Volume']]

    division_rate1 = 0.8
    division_rate2 = 0.9

    seq_len = G.src_len  # 20 how long of a preceeding sequence to collect for RNN
    tgt = G.tgt_len
    mulpre = G.mulpr_len  # how far into the future are we trying to predict?
    window = G.window_size



    df,list,list1 = get_stock_data(df_raw)
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(df, seq_len, mulpre, division_rate1, division_rate2,tgt)
    tf.keras.backend.clear_session()
    model = Transformer()
    model.build(input_shape=[None, G.window_size, G.num_features])#输入的格式
    model.summary(expand_nested=True)


    G.T_i = 1
    G.T_mult = 2
    G.T_cur = 0.0

    loss_object = tf.keras.losses.LogCosh()

    optimizer = tf.keras.optimizers.Adam(learning_rate = G.learning_rate,
                                         beta_1 = 0.9,
                                         beta_2 = 0.999
                                        )

    '''#cos_anneal is for the model.fit() call
    cos_anneal = tf.keras.callbacks.LambdaCallback(on_batch_end = schedule)
    
    #progress plot callback
    pp_update = ProgressCallback()'''

    #metrics改过，原来"mean_absolute_percentage_error"报错
    model.compile(loss=up_down_accuracy, optimizer=optimizer, metrics=["mse"])

    #x有问题，要window_size!!!!!!!!!!!!!!!!!!!
    history = model.fit(X_train,y_train,
                        epochs = G.epochs,
                        batch_size=G.batch_size,
                        verbose = 1,
                        validation_data=(X_valid, y_valid)
                        )


    '''model.evaluate(X_test, y_test,
                   verbose = 1
                   )'''



    s = time.time()
    predicted_stock_price_multi_head = model.predict(X_test)
    e = time.time()
    print('时间', e - s)

    testdata_len = y_test.shape[0]
    predicted_stock_price_multi_head_dff1, predicted_stock_price_multi_head = denormalize(df_raw, predicted_stock_price_multi_head,division_rate2, seq_len, mulpre, testdata_len)
    y_test_dff1, y_test = denormalize(df_raw, y_test, division_rate2, seq_len, mulpre, testdata_len)

    stock = i
    model2 = 'Galformer'
    csv_path = 'C:/lyx/learning/期刊论文/程序结果/对比图表/' + stock +'/' + model2 + '.xls'
    df = pd.DataFrame(predicted_stock_price_multi_head)
    df.columns.name = None
    df.to_excel(csv_path,index=False,header=None)


    accu = np.multiply(predicted_stock_price_multi_head_dff1,y_test_dff1)
    print('accu的形状', accu.shape)
    print('accu', accu)
    accu = np.maximum(accu, 0)
    print('accu的形状', accu.shape)
    print('accu', accu)
    accu = np.sign(accu)
    print('accu的形状', accu.shape)
    print('accu', accu)
    accu = np.mean(accu) * 100
    print('accu的形状', accu.shape)
    print('accu', accu)
    print(f'测试集趋势预测准确率为{accu}%')

    print('predicted_stock_price_multi_head.shape',predicted_stock_price_multi_head.shape)
    print('predicted_stock_price_multi_head',predicted_stock_price_multi_head)

    predicted_stock_price_multi_head = numpy.ravel(predicted_stock_price_multi_head)
    y_test = numpy.ravel(y_test)

    calculate_accuracy(predicted_stock_price_multi_head, y_test)
    print(f'测试集趋势预测准确率为{accu}%')

    plt.ion()
    plt.figure(figsize = (18,9))
    plt.plot(y_test, color = 'black', label = 'real')
    plt.plot(predicted_stock_price_multi_head, color = 'green', label = 'pre')
    plt.title('Adj closing Price Prediction', fontsize=30)
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Adj closing Price')
    plt.legend(fontsize=18)
    plt.show()
    plt.close()


    '''
    predicted_stock_price_multi_head = model.predict(X_train)
    
    predicted_stock_price_multi_head = denormalize(predicted_stock_price_multi_head)
    y_test = denormalize(y_train)
    
    print('predicted_stock_price_multi_head.shape',predicted_stock_price_multi_head.shape)
    print('predicted_stock_price_multi_head',predicted_stock_price_multi_head)
    
    predicted_stock_price_multi_head = numpy.ravel(predicted_stock_price_multi_head)
    y_test = numpy.ravel(y_train)
    
    calculate_accuracy(predicted_stock_price_multi_head, y_test)
    
    plt.figure(figsize = (18,9))
    plt.plot(y_test, color = 'black', label = 'real')
    plt.plot(predicted_stock_price_multi_head, color = 'green', label = 'pre')
    plt.title('Adj closing Price Prediction', fontsize=30)
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Adj closing Price')
    plt.legend(fontsize=18)
    plt.show()
    '''