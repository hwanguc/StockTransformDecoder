import os
import joblib
import math
from math import floor
import yfinance as yf
import datetime as dt
from datetime import date
import time
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclass_global import G
from transformer_helper_dc import *
from rolling_and_plot_dc import data_plot, rolling_split, normalize, validate


def acquire_ticker(ticker, START, TODAY):
    data = yf.download(ticker, START, TODAY, auto_adjust=False)
    data.reset_index(inplace=True)
    return data


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def get_stock_data(df): # drop NAs and calculate the 1st order derivatives of Adj Close.
    df.drop(['Date', 'Close'], axis=1, inplace=True)
    list = df['Adj Close']
    list1 = list.diff(1).dropna()
    list = list.iloc[:, 0].tolist()
    list1 = list1.iloc[:, 0].tolist()

    list1 = np.array(list1)
    df = df.drop(0, axis=0)

    df['Adj Close'] = list1
    df = df.reset_index(drop=True)
    print(df.head())
    return df,list,list1



def load_data(df, seq_len , mul, division_rate1, division_rate2, tgt, normalize=True): # Prepare training, validation, and test datasets.
    amount_of_features = G.num_features 
    data = df.values
    row1 = round(division_rate1 * data.shape[0])  #0.8: split 80% for training
    row2 = round(division_rate2 * data.shape[0])  #0.9: split the next 10% for validation

    train = data[:int(row1), :]
    valid = data[int(row1):int(row2), :]
    test = data[int(row2): , :]

    #print('train', train)
    #print('valid', valid)
    #print('test', test)

    if normalize:
        standard_scaler = preprocessing.StandardScaler()
        train = standard_scaler.fit_transform(train)
        valid = standard_scaler.transform(valid)
        test = standard_scaler.transform(test)

    #print('train',train)
    #print('valid', valid)
    #print('test', test)
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    X_test = []
    y_test = []
    train_samples=train.shape[0]-seq_len-mul+1
    valid_samples = valid.shape[0] - seq_len - mul + 1
    test_samples = test.shape[0] - seq_len - mul + 1
    for i in range(0,train_samples,mul):  # maximum date = lastest  date - sequence length  
        X_train.append(train[i:i + seq_len,-amount_of_features:]) # five features per sliding window, i.e., 4 features + 1 target
        y_train.append(train[i + seq_len:i+seq_len+tgt,-1]) # idx -1 is the target, i.e., Adj Close, the last feature in the data

    for i in range(0,valid_samples,mul):  # maximum date = lastest  date - sequence length
        X_valid.append(valid[i:i + seq_len,-amount_of_features:])
        y_valid.append(valid[i+seq_len:i+seq_len+tgt,-1])

    for i in range(0, test_samples,mul):  # maximum date = lastest date - sequence length
        X_test.append(test[i:i + seq_len, -amount_of_features:])
        y_test.append(test[i+seq_len:i+seq_len+tgt, -1])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print('train', train.shape)
    print(train)
    print('valid', valid.shape)
    print(valid)
    print('test', test.shape)
    print(test)

    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_valid', X_valid.shape)
    print('y_valid', y_valid.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)
    print('df', df)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))


    print('X_train', X_train.shape)
    print('X_valid', X_valid.shape)
    print('X_test', X_test.shape)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


#################################

def FullyConnected():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(G.dense_dim, activation='relu',
                                kernel_initializer = tf.keras.initializers.HeNormal(),
                                bias_initializer = tf.keras.initializers.RandomUniform(minval=0.005, maxval = 0.08)
                                ),
        # (G.batch_size, G.window_size, G.dense_dim)


        tf.keras.layers.BatchNormalization(momentum = 0.98, epsilon=5e-4),

        tf.keras.layers.Dense(G.d_model,
                                kernel_initializer = tf.keras.initializers.HeNormal(),
                                bias_initializer = tf.keras.initializers.RandomUniform(minval=0.001, maxval = 0.01)
                                ),

        tf.keras.layers.BatchNormalization(momentum = 0.95, epsilon=5e-4)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """
    The encoder layer is composed by a multi-head self-attention mechanism,
    followed by a simple, positionwise fully connected feed-forward network.
    This archirecture includes a residual connection around each of the two
    sub-layers, followed by batch normalization.
    """

    def __init__(self,
                    num_heads,
                    d_k,
                    dropout_rate,
                    batchnorm_eps,**kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_k,
            dropout=dropout_rate,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.01)
        )

        # feed-forward-network
        self.ffn = FullyConnected()

        self.batchnorm1 = BatchNormalization(momentum=0.95, epsilon=batchnorm_eps)
        self.batchnorm2 = BatchNormalization(momentum=0.95, epsilon=batchnorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, training):
        """
        Forward pass for the Encoder Layer

        Arguments:
            x -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
        Returns:
            encoder_layer_out -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
        """
        # Dropout is added by Keras automatically if the dropout parameter is non-zero during training

        attn_output = self.mha(query=x,
                                value=x)  # Self attention

        out1 = self.batchnorm1(tf.add(x, attn_output))  # (G.batch_size, G.src_len, G.dense_dim)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout_ffn(ffn_output)  # (G.batch_size, G.src_len, G.dense_dim)

        encoder_layer_out = self.batchnorm2(tf.add(ffn_output, out1))
        # (G.batch_size, G.src_len, G.dense_dim)
        return encoder_layer_out


class Encoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the input to an embedding layer
    and using positional encoding to then pass the output through a stack of
    encoder Layers

    """

    def __init__(self,
                    num_layers=G.num_layers,
                    num_heads=G.num_heads,
                    num_features=G.num_features,
                    d_model=G.d_model,
                    d_k=G.d_k,
                    dense_dim=G.dense_dim,
                    maximum_position_encoding=G.src_len,
                    dropout_rate=G.dropout_rate,
                    batchnorm_eps=1e-4,**kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.num_layers = num_layers

        # linear input layer
        self.lin_input = tf.keras.layers.Dense(d_model, activation="relu")

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                d_model)

        self.enc_layers = [EncoderLayer(num_heads=num_heads,
                                        d_k=d_k,
                                        dropout_rate=dropout_rate,
                                        batchnorm_eps=batchnorm_eps)
                            for _ in range(self.num_layers)]

    def call(self, x, training):
        """
        Forward pass for the Encoder

        Arguments:
            x -- Tensor of shape (G.batch_size, G.src_len, G.num_features)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input
        Returns:
            Tensor of shape (G.batch_size, G.src_len, G.dense_dim)
        """
        x = self.lin_input(x)
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]

        #应该concatenate！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)

        return x  # (G.batch_size, G.src_len, G.dense_dim)


class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks,
    one that takes the new input and uses self-attention, and the other
    one that combines it with the output of the encoder, followed by a
    fully connected block.
    """

    def __init__(self,
                    num_heads,
                    d_k,
                    dropout_rate,
                    batchnorm_eps,**kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.mha1 = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_k,
            dropout=dropout_rate,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.01)
        )

        self.mha2 = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_k,
            dropout=dropout_rate,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_initializer=tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.01)
        )

        self.ffn = FullyConnected()

        self.batchnorm1 = BatchNormalization(momentum=0.95, epsilon=batchnorm_eps)
        self.batchnorm2 = BatchNormalization(momentum=0.95, epsilon=batchnorm_eps)
        self.batchnorm3 = BatchNormalization(momentum=0.95, epsilon=batchnorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, y, enc_output, dec_ahead_mask, enc_memory_mask, training):
        """
        Forward pass for the Decoder Layer

        Arguments:
            y -- Tensor of shape (G.batch_size, G.tgt_len, 1) #the soc values for the batches
            enc_output --  Tensor of shape(G.batch_size, G.num_features)
            training -- Boolean, set to true to activate
                        the training mode for dropout and batchnorm layers
        Returns:
            out3 -- Tensor of shape (G.batch_size, G.tgt_len, 1)
        """

        # BLOCK 1
        # Dropout will be applied during training only
        mult_attn_out1 = self.mha1(query=y,
                                    value=y,
                                    attention_mask=dec_ahead_mask,
                                    return_attention_scores=False)
        # (G.batch_size, G.tgt_len, G.dense_dim)

        Q1 = self.batchnorm1(tf.add(y, mult_attn_out1))

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output.
        # Dropout will be applied during training
        mult_attn_out2 = self.mha2(query=Q1,
                                    value=enc_output,
                                    key=enc_output,
                                    attention_mask=enc_memory_mask,
                                    return_attention_scores=False)

        mult_attn_out2 = self.batchnorm2(tf.add(mult_attn_out1, mult_attn_out2))

        # BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(mult_attn_out2)

        # apply a dropout layer to the ffn output
        ffn_output = self.dropout_ffn(ffn_output)

        out3 = self.batchnorm3(tf.add(ffn_output, mult_attn_out2))
        return out3


class Decoder(tf.keras.layers.Layer):
    """

    """

    def __init__(self,
                    num_layers=G.num_layers,
                    num_heads=G.num_heads,
                    num_features=G.num_features,
                    d_model=G.d_model,
                    d_k=G.d_k,
                    dense_dim=G.dense_dim,
                    target_size=G.num_features,
                    maximum_position_encoding=G.dec_len,
                    dropout_rate=G.dropout_rate,
                    batchnorm_eps=1e-5,**kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                d_model)

        # linear input layer
        self.lin_input = tf.keras.layers.Dense(d_model, activation="relu")

        self.dec_layers = [DecoderLayer(num_heads=num_heads,
                                        d_k=d_k,
                                        dropout_rate=dropout_rate,
                                        batchnorm_eps=batchnorm_eps
                                        )
                            for _ in range(self.num_layers)]
        # look_ahead_masks for decoder:
        self.dec_ahead_mask = create_look_ahead_mask(G.dec_len, G.dec_len)
        self.enc_memory_mask = create_look_ahead_mask(G.dec_len, G.src_len)

    def call(self, y, enc_output, training):
        """
        Forward  pass for the Decoder

        Arguments:
            y -- Tensor of shape (G.batch_size, G.tgt_len, G.dense_dim) #the final SOC values in the batches
            enc_output --  Tensor of shape(G.batch_size, G.src_len, G.dense_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
        Returns:
            y -- Tensor of shape (G.batch_size, G.tgt_len, 1)
        """
        y = self.lin_input(y)  # maps to dense_dim, the dimension of all the sublayer outputs.

        dec_len = tf.shape(y)[1]
        print('dec_len',dec_len)
        y += self.pos_encoding[:, :dec_len, :]

        # use a for loop to pass y through a stack of decoder layers and update attention_weights
        for i in range(self.num_layers):
            # pass y and the encoder output through a stack of decoder layers and save attention weights
            y = self.dec_layers[i](y,
                                    enc_output,
                                    self.dec_ahead_mask,
                                    self.enc_memory_mask,
                                    training=training)

        #print('y.shape', y.shape)
        return y


class Transformer(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder
    """

    def __init__(self,
                    num_layers=G.num_layers,
                    num_heads=G.num_heads,
                    dense_dim=G.dense_dim,
                    src_len=G.src_len,
                    dec_len = G.dec_len,
                    tgt_len=G.tgt_len,
                    max_positional_encoding_input=G.src_len,
                    max_positional_encoding_target=G.tgt_len,
                    **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.tgt_len = tgt_len
        self.dec_len = dec_len
        self.src_len = src_len

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.linear_map = tf.keras.Sequential([
            tf.keras.layers.Dense(
                dense_dim, activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer=tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.02)
            ),
            tf.keras.layers.BatchNormalization(momentum=0.97, epsilon=5e-4),

            tf.keras.layers.Dense(1)
        ])

        self.final_dense = tf.keras.layers.Dense(tgt_len)

    def call(self, x, training):
        """
        Forward pass for the entire Transformer
        Arguments:
            x -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
                    An array of the windowed voltage, current and soc data
            training -- Boolean, set to true to activate
                        the training mode for dropout and batchnorm layers
        Returns:
            final_output -- SOC prediction at time t

        """
        enc_input = x[:, :self.src_len, :]   # (G.batch_size, G.src_len, G.num_features)
        dec_input = x[:, -self.dec_len:, ]
        #print(type(dec_input))
        #print('dec_input.shape',dec_input.shape)




        enc_output = self.encoder(enc_input, training=training)  # (G.batch_size, G.src_len, G.num_features)
        #print('enc_output.shape', enc_output.shape)

        dec_output = self.decoder(dec_input, enc_output, training=training)
        #print('dec_output.shape', dec_output.shape)
        # (G.batch_size, G.tgt_len, 32)

        final_output = self.linear_map(dec_output)  # (G.batch_size, G.tgt_len, 1)

        #print('final_output.shape', final_output.shape)
        final_output = tf.transpose(final_output,perm=[0,2,1])
        final_output = self.final_dense(final_output)  # (G.batch_size, G.tgt_len, 1
        final_output = tf.transpose(final_output,perm=[0,2,1])
        #print('final_output.shape', final_output.shape)
        return final_output


def calculate_accuracy(pre, real):
    print('pre.shape', pre.shape)
    print(pre)

    print('real.shape', real.shape)
    print(real)
    # MAPE = np.mean(np.abs((pre - real) / real))
    MAPE = sklearn.metrics.mean_absolute_percentage_error(real,pre)
    #MAPE = calculate_MAPE(pre,real)
    RMSE = np.sqrt(np.mean(np.square(pre - real)))
    MAE = np.mean(np.abs(pre - real))
    R2 = r2_score(pre, real)
    dict = {'MAPE': [MAPE], 'RMSE': [RMSE], 'MAE': [MAE], 'R2': [R2]}
    df = pd.DataFrame(dict)
    print('Performance metrics:\n',df)
    return df

def up_down_accuracy(y_true, y_pred):

    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    trend = tf.multiply(y_true, y_pred)
    trend = tf.nn.relu(trend)
    trend = tf.sign(trend)
    trend_acc = tf.reduce_mean(trend)

    scale = tf.pow(10.0, tf.floor(tf.math.log(tf.abs(mse) + 1e-8) / tf.math.log(10.0)))
    loss = (1.0 - trend_acc) * scale + mse
    return loss


def denormalize(ticker, df, normalized_value, division_rate1, division_rate2, seq_len, mulpre, testdata_len, cwd, scalersave = False):
    list = df['Adj Close']
    list1 = list.diff(1).dropna()  # list 1 is the first order difference of Adj Close, i.e., the difference between Adj Close and the previous day Adj Close

    list1 = list1.iloc[:, 0].tolist()
    list1 = np.array(list1)  
    df1 = df.drop(0, axis=0)
    df1['Adj Close'] = list1
    df1 = df1.reset_index(drop=True)
    print(df.head())
    print(df1.head())
    data = df.values
    data1 = df1.values


    # row indices for splitting the data into train, validation, and test sets
    row1 = round(division_rate1 * list1.shape[0])
    row2 = round(division_rate2 * list1.shape[0])

    train = data1[:int(row1), :]
    test = data1[int(row2): , :]

    test = test[:,-1].reshape(-1, 1)

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(train[:, -1].reshape(-1, 1)) 
    m = standard_scaler.transform(test)

    if scalersave == True:
        scaler_file = 'output/standard_train_scaler_' + ticker + '.pkl'
        scaler_path = os.path.join(cwd, scaler_file)
        joblib.dump(standard_scaler, scaler_path)


    # invderse transform the normalized value
    normalized_value = normalized_value.reshape(-1, 1)
    new = standard_scaler.inverse_transform(normalized_value) # use m to inverse transform the normalized test values.
    print('new',new.shape)

    
    residual = data[int(row2) + seq_len : int(row2) + seq_len +  mulpre * testdata_len, -1 ] #Only -2 (Adj Close) is used for residual for now.
    residual = residual.reshape(-1, 1)  # reshape to (618, 1)
    print('residual', residual.shape)

    sum = new + residual

    return new,sum


def predict_next_days(ticker, model, standard_scaler):

    START = "2000-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")


    df_raw = acquire_ticker(ticker, START, TODAY)
    df_raw = df_raw.sort_values('Date')

    # Step 1: get last 20 days
    df_latest = df_raw.iloc[-21:]
    df_latest = df_latest[['Date','Open','High','Low','Close','Volume','Adj Close']]
    df_latest = df_latest.drop(columns=['Date','Close'])
    features = df_latest.iloc[:, -G.num_features:]
    features_diff = features.diff(1).dropna()
    
    # Step 2: normalize


    X_input = standard_scaler.transform(features_diff.values)
    X_input = np.expand_dims(X_input, axis=0)  # shape (1, 20, 5)

    # Step 3: predict
    y_pred_norm = model(tf.convert_to_tensor(X_input), training=False).numpy()
    y_pred_norm = y_pred_norm.reshape(-1, 1)  # shape (3, 1)

    # Step 4: inverse transform
    y_pred_real = standard_scaler.inverse_transform(y_pred_norm)
    
    # Step 5: if differenced, reconstruct price
    last_price = df_latest["Adj Close"].values[-1]
    y_pred_final = np.cumsum(y_pred_real.flatten()) + last_price

    return y_pred_final  # next 3 days' predicted prices
