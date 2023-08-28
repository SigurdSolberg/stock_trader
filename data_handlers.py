from transformers.utils import *
from transformers.components import *
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from time import time as t
import numpy as np

def fetch_data(tickers, start_date, end_date, validation_start):
    # Fetch data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # If data is empty, return None
    if data.empty:
        return None
    
    # Sort the data in ascending order of date (if not already)
    data.sort_index(inplace=True)

    data = data.swaplevel(axis=1)
    for ticker in tickers:
        close_data = data.loc[:, (ticker, 'Close')]  # Using .loc to access the 'Close' column

        # Calculate the 5-day moving average for each stock
        moving_average = close_data.rolling(window=5).mean()

        # Assign the moving average values back to the DataFrame
        data.loc[:, (ticker, '5-day Moving Average')] = moving_average
    desired_order = ['Adj Close', 'Close', 'High', 'Low', 'Open', '5-day Moving Average', 'Volume',]
    data = data.swaplevel(axis=1)
    data = data[desired_order]
    data = data.swaplevel(axis=1)
    
    # Split data into training and validation datasets
    training_data = data[data.index < validation_start]#.swaplevel(axis = 1)
    validation_data = data[data.index >= validation_start]#.swaplevel(axis = 1)

    return (tickers, training_data, validation_data)

def get_sample(input_data, seq_length = 64):
    input_sequences = []
    target_sequences = []

    for i in range(0, len(input_data) - seq_length):
        input_sequences.append(input_data[i:i + seq_length])
        tgt = (input_data[i + seq_length][1] / input_sequences[-1][-1][1]) if input_sequences[-1][-1][1] != 0 else 1
        target_sequences.append([[tgt-1],])
    return input_sequences, target_sequences

def create_dataset(sequence_data, ticker_list, device):
    X = []
    y = []
    for ticker in ticker_list:
        data = sequence_data[ticker].values[:, 1:7]
        data = np.nan_to_num(data, 0)
        inputs, targets = get_sample(data, seq_length=8)
        data /= data[0]
        data -= data[0]
        X += (inputs)
        y += (targets)
    X = np.array(X)
    y = np.array(y)
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    return torch.tensor(X, dtype=torch.float32, requires_grad=True, device=device), torch.tensor(y, dtype=torch.float32, requires_grad=True, device=device)

