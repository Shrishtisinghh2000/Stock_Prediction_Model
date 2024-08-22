# Stock Price Prediction Using LSTM

This project uses Long Short-Term Memory (LSTM) networks, a type of recurrent neural network, to predict stock prices. The model is trained on historical stock data, with a focus on predicting the closing price of NVIDIA (NVDA) stock.

## Table of Contents

Project Overview

Dataset

Dependencies

Usage

Model Description

Results

Acknowledgements


## Project Overview

This project demonstrates the use of LSTM networks for time series forecasting, specifically in the domain of stock price prediction. The project focuses on predicting the closing price of NVIDIA (NVDA) stock based on historical price data. The model is trained using past stock prices and is evaluated by comparing the predicted prices with the actual prices.


## Dataset

The dataset used in this project consists of historical stock prices for NVIDIA (NVDA) obtained from Yahoo Finance.

Time Period: January 1, 2013 - December 21, 2023

Features: Date, Open, High, Low, Close, Volume

Target: Close price

    
## Dependencies

To run this project, you need the following dependencies:

Python 3.x
numpy
pandas
matplotlib
yfinance
scikit-learn
keras
tensorflow
Jupyter Notebook (optional, for running the notebook)


## Usage

Download historical stock data from Yahoo Finance.

Preprocess the data by calculating moving averages and normalizing the values.

Build and train an LSTM model to predict stock prices.

Evaluate the model by comparing predicted prices with actual prices.

Save the trained model for future use.


## Model Description

The project uses an LSTM model with multiple layers and dropout for regularization to predict the stock price. The model is trained on 100-day windows of stock prices to predict the next day's closing price.

Key Steps:

Data Collection:

Download historical stock data using yfinance.

Calculate 100-day and 200-day moving averages for visualization.

Data Preprocessing:

Normalize the closing prices using MinMaxScaler.

Create sequences of 100 days of stock prices to use as input features.

Model Training:

Build an LSTM model with four LSTM layers and dropout layers to prevent overfitting.

Train the model on the training data for 50 epochs with a batch size of 32.

Model Evaluation:

Predict stock prices on the test set and scale them back to original values.

Visualize the predicted prices against the actual prices.


## Results

The LSTM model is able to predict stock prices with reasonable accuracy, as visualized by comparing the predicted and actual prices. The model shows the potential of LSTM networks in time series forecasting, especially for stock prices, which are inherently noisy and complex.


## Acknowledgements

The stock data used in this project is provided by [Yahoo Finance](https://finance.yahoo.com/quote/NVDA/).

The project is inspired by various tutorials on time series forecasting using LSTM.
