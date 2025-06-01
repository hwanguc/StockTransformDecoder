# This script defines a dataclass for global parameters used in the Galformer model.

from dataclasses import dataclass

@dataclass

class G:
    # preprocess

    batch_size = 64
    src_len = 20  # encoder input sequence length, the number of previous time steps to consider for prediction.
    dec_len = 3
    tgt_len = 3  # decoder input sequence length, same length as transformer output. Number of next days to predict.
    window_size = src_len
    mulpr_len = tgt_len
    # network
    d_model = 512
    dense_dim = 2048
    num_features = 5  # num of features in the input data (5= Open, High, Low, Volume, Adj Close; 1 = Adj Close)
    num_heads = 8
    d_k = int(d_model/num_heads)
    num_layers = 6
    dropout_rate = 0.1
    # learning_rate_scheduler
    T_i = 1
    T_mult = 2
    T_cur = 0.0
    # training
    epochs = 200 #21 should be T_i + a power of T_mult, ex) T_mult = 2 -> epochs = 2**5 + 1 = 32+1 = 33   257, original epoch number was 5.
    learning_rate = 0.003 #0.0045
    min_learning_rate = 7e-11
    # weight_decay = 0.0 #No weight decay param in the the keras optimizers