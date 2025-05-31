from dataclass_global import G
from model import *
from transformer_helper_dc import *
from rolling_and_plot_dc import data_plot, rolling_split, normalize, validate


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


ticker = 'NVDA'

cwd = os.getcwd()  # Get current working directory
scaler_file = 'output/standard_train_scaler.pkl'
scaler_path = os.path.join(cwd, scaler_file)
standard_scaler = joblib.load(scaler_path)

model =  Transformer()
model.build(input_shape=[None, G.window_size, G.num_features])
model.load_weights(cwd + '/ckpt/model_checkpoint_' + ticker + '.keras')

y_pred_next_days = predict_next_days(ticker, model, standard_scaler)