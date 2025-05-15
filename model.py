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


def get_stock_data(df):
    df.drop(['Date', 'Close'], axis=1, inplace=True)#由于date不连续,这时候保留5维
    list = df['Adj Close']
    list1 = list.diff(1).dropna()  # list1为list的1阶差分序列,序列的序号从1开始,所以要tolist,这样序号才从0开始. 但是列表不能调用diff
    # 或者list1 = np.diff(list)[1:]
    list = list.iloc[:, 0].tolist()
    list1 = list1.iloc[:, 0].tolist()

    list1 = np.array(list1)#array才能reshape
    df = df.drop(0, axis=0)
    # print(df1.head())
    df['Adj Close'] = list1
    df = df.reset_index(drop=True)
    print(df.head())
    return df,list,list1


#先划分训练集测试集,再标准化归一化,避免数据泄露
def load_data(df, seq_len , mul, division_rate1, division_rate2, tgt, normalize=True):
    amount_of_features = 1  # columns是列索引,index是行索引
    data = df.values
    row1 = round(division_rate1 * data.shape[0])  #0.8  split可改动!!!!!!!#round是四舍五入,0.9可能乘出来小数  #shape[0]是result列表中子列表的个数
    row2 = round(division_rate2 * data.shape[0])  #0.9
    #训练集和测试集划分
    train = data[:int(row1), :]
    valid = data[int(row1):int(row2), :]
    test = data[int(row2): , :]

    print('train', train)
    print('valid', valid)
    print('test', test)

    # 训练集和测试集归一化
    if normalize:
        standard_scaler = preprocessing.StandardScaler()
        train = standard_scaler.fit_transform(train)
        valid = standard_scaler.transform(valid)
        test = standard_scaler.transform(test)

    print('train',train)
    print('valid', valid)
    print('test', test)
    X_train = []  # train列表中4个特征记录
    y_train = []
    X_valid = []
    y_valid = []
    X_test = []
    y_test = []
    train_samples=train.shape[0]-seq_len-mul+1
    valid_samples = valid.shape[0] - seq_len - mul + 1
    test_samples = test.shape[0] - seq_len - mul + 1
    for i in range(0,train_samples,mul):  # maximum date = lastest  date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
        X_train.append(train[i:i + seq_len,-2])#每个滑窗每天四个特征
        y_train.append(train[i + seq_len:i+seq_len+tgt,-2])#-1为成交量,倒数第二个才是adj close

    for i in range(0,valid_samples,mul):  # maximum date = lastest  date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
        X_valid.append(valid[i:i + seq_len,-2])#每个滑窗每天四个特征
        y_valid.append(valid[i+seq_len:i+seq_len+tgt,-2])#-1为成交量,倒数第二个才是adj close

    for i in range(0, test_samples,mul):  # maximum date = lastest date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
        X_test.append(test[i:i + seq_len, -2])  # 每个滑窗每天四个特征
        y_test.append(test[i+seq_len:i+seq_len+tgt, -2])  # -1即取最后一个特征
    # X都对应全部4特征,y都对应adj close   #train都是前百分之90,test都是后百分之10
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
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))  # (90%maximum, seq-1 ,4) #array才能reshape
    X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  # (10%maximum, seq-1 ,4) #array才能reshape、


    print('X_train', X_train.shape)
    print('X_valid', X_valid.shape)
    print('X_test', X_test.shape)
    return X_train, y_train, X_valid, y_valid, X_test, y_test  # x是训练的数据，y是数据对应的标签,也就是说y是要预测的那一个特征!!!!!!


#################################

def FullyConnected():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(G.dense_dim, activation='relu',
                                kernel_initializer = tf.keras.initializers.HeNormal(),
                                bias_initializer = tf.keras.initializers.RandomUniform(minval=0.005, maxval = 0.08)
                                ),
        # (G.batch_size, G.window_size, G.dense_dim)

        #原来是relu
        tf.keras.layers.BatchNormalization(momentum = 0.98, epsilon=5e-4),
        #原来是G.dense_dim
        tf.keras.layers.Dense(G.d_model,
                                kernel_initializer = tf.keras.initializers.HeNormal(),
                                bias_initializer = tf.keras.initializers.RandomUniform(minval=0.001, maxval = 0.01)
                                ),
        # (G.batch_size, G.window_size, G.dense_dim)
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
                    batchnorm_eps):
        super(EncoderLayer, self).__init__()

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
                    batchnorm_eps=1e-4):
        super(Encoder, self).__init__()

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
                    batchnorm_eps):
        super(DecoderLayer, self).__init__()

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
                    batchnorm_eps=1e-5):
        super(Decoder, self).__init__()

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

        print('y.shape', y.shape)
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
                    max_positional_encoding_target=G.tgt_len):
        super(Transformer, self).__init__()

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

#!!!!!!!!!!!!activation原来是sigmoid，bias_initializer=tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.005)
            #!!!!!!!!!!!!!!!!!!dense里面是1，而不是mul（是把d_model数字回归1维最后结果）
            tf.keras.layers.Dense(1)
        ])

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
        dec_input = x[:, -self.dec_len:, ]  # only want the SOC thats why -1 is there!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        print(type(dec_input))
        print('dec_input.shape',dec_input.shape)
        #dec_input = dec_input.resize_as_(dec_input.shape[0],dec_input.shape[1],1)#(128,3)->(128,3,1)




        enc_output = self.encoder(enc_input, training=training)  # (G.batch_size, G.src_len, G.num_features)
        print('enc_output.shape', enc_output.shape)

        dec_output = self.decoder(dec_input, enc_output, training=training)
        print('dec_output.shape', dec_output.shape)
        # (G.batch_size, G.tgt_len, 32)

        final_output = self.linear_map(dec_output)  # (G.batch_size, G.tgt_len, 1)

        print('final_output.shape', final_output.shape)
        final_output = tf.transpose(final_output,perm=[0,2,1])
        final_output = Dense(G.tgt_len)(final_output)
        final_output = tf.transpose(final_output,perm=[0,2,1])
        print('final_output.shape', final_output.shape)
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
    print('最终的准确率和指标如下\n',df)
    return df

def up_down_accuracy(real, pre):
    '''products = []

    print('tf.shape',tf.shape(real))
    print('pre.shape',pre.shape)
    for i in tf.range(tf.shape(real)[0]):
        products.append(real[i] *  pre[i])
    accuracy = (sum([int(x > 0) for x in products])) / len(products)
    return accuracy'''
    print('real.shape', real.shape)
    print('pre.shape', pre.shape)
    mse = tf.reduce_mean(tf.square(pre - real))
    print('MSE is:',K.get_value(mse))
    print('real666.shape', real.shape)
    print('pre666.shape', pre.shape)#real666.shape (None, 3)  pre666.shape (None, 3)
    print('real666', real)
    print('pre666', pre)
    accu = tf.multiply(real,pre)#矩阵点积，不是乘法，得出正负，正的就是趋势预测正确
    accu = tf.nn.relu(accu)#relu(x) = max(0,x)
    accu = tf.sign(accu)#正数变1，0不变
    accu = tf.reduce_mean(accu)#取平均
    print('\n\n\n')
    print('Accuracy is:', K.get_value(accu))#准确率，0.x
    '''result = tf.compat.v1.Session().run(result)

    print('resultnumpy', result)
    accuracy = (sum([int(x > 0) for x in result]))
    print('loss666', tf.math.reduce_mean(tf.square(real - pre)))'''
    accu = 1 - accu#loss越小越好，所以1-准确率S
    #loss = mse + accu * 10 #mse个位数，accu 0.x
    loss = accu * pow(10, floor(math.log(abs(mse), 10))) + mse
    print('Loss is:', K.get_value(loss))
    return loss#个位数


def denormalize(df, normalized_value, division_rate1, division_rate2, seq_len, mulpre, testdata_len):
    list = df['Adj Close']
    list1 = list.diff(1).dropna()  # list1为list的1阶差分序列,序列的序号从1开始,所以要tolist,这样序号才从0开始. 但是列表不能调用diff
    # 或者list1 = np.diff(list)[1:]
    list1 = list1.iloc[:, 0].tolist()
    list1 = np.array(list1)  # array才能reshape
    df1 = df.drop(0, axis=0)
    df1['Adj Close'] = list1
    df1 = df1.reset_index(drop=True)#index从0开始
    print(df.head())
    print(df1.head())
    data = df.values
    data1 = df1.values


    # 训练集和测试集划分
    row1 = round(division_rate1 * list1.shape[0])
    row2 = round(division_rate2 * list1.shape[0])

    train = data1[:int(row1), :]
    test = data1[int(row2): , :]

    test = test[:,-2].reshape(-1, 1)#取原来没有归一化的adj数据作为样本

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(train[:, -2].reshape(-1, 1)) 
    m = standard_scaler.transform(test)  # 利用m对data进行归一化，并储存df的归一化参数. 用测试集的归一化参数来反归一化y_test和预测值


    '反归一化'
    normalized_value = normalized_value.reshape(-1, 1)
    new = standard_scaler.inverse_transform(normalized_value)#利用m对normalized_value进行反归一化
    print('new',new.shape)

    
    residual = data[int(row2) + seq_len : int(row2) + seq_len +  mulpre * testdata_len, -2 ]#差分残差从test的seq-1序号天开始到test的倒数第二天,预测加上前一天的残差对应test[seq:]反归一的真实值,注意y_test和预测值是一致的; Only -2 (Adj Close) is used for residual for now.
    residual = residual.reshape(-1, 1)  # reshape to (618, 1)
    print('residual', residual.shape)

    sum = new + residual
    '归一化'
    '''m = min_max_scaler.fit_transform(train)  # 利用m对train进行归一化，并储存df的归一化参数!!
    new = min_max_scaler.transform(test)  # 利用m对test也进行归一化,注意这里是transform不能是fit_transform!!!1'''
    return new,sum#new是差分预测值，sum是没差分的预测值}}