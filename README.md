# StockTransformDecoder

This repo contains Python scripts for predicting the adjusted close stock price with a Transformer model based on the Galformer architecture ([Ji et al., 2024](https://www.nature.com/articles/s41598-024-72045-3)). The current model generates next 3-day prediction of the Adj Close price based on the a sliding window of Open, High, Low, Adjusted Close prices and Volume from the previous 20 days. You can add more features as needed.

## Scripts

**main.py**: The main pipeline for the analysis.

**model.py**: The Transformer model.

**rolling_and_plot_dc.py**: Data and plotting helper functions from repo [transform_decode_soc](https://github.com/att-ar/transform_decode_soc).

**rolling_and_plot_dc.py**: Transformer helper functions from repo [transform_decode_soc](https://github.com/att-ar/transform_decode_soc).

**./ckpt/**: Checkpoint for the best model (measured by validation loss). Not included due to large file sizes.

**./output/**: Scaler for raw data normalisation.

## Examples

AAPL (Adj Closed for test dataset):<br />
![alt text](https://github.com/hwanguc/StockTransformDecoder/blob/main/output/pred_AAPL_5features_next3days.png "Prediced AAPL Adj Closed for test dataset")
<br /><br />

NVDA (Adj Closed for test dataset):<br />
![alt text](https://github.com/hwanguc/StockTransformDecoder/blob/main/output/pred_NVDA_5features_next3days.png "Prediced NVDA Adj Closed for test dataset")
<br /><br />

GOOG (Adj Closed for test dataset):<br />
![alt text](https://github.com/hwanguc/StockTransformDecoder/blob/main/output/pred_GOOG_5features_next3days.png "Prediced GOOG Adj Closed for test dataset")
<br /><br />