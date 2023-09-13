import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy.stats
from data_handlers import create_dataset


def custom_loss(predictions,targets, fee = 0.01):
    #print(predictions.shape, targets.shape)
    binary = predictions * targets
    wrong = binary < 0
    #print(predictions[:5], targets[:5])
    return torch.abs(predictions - targets)#torch.sum(torch.abs(targets[wrong])) - torch.sum(torch.abs(targets[~wrong])) #+ fee

def trading_earnings(predictions, targets, alpha=0.5, fee=0.0015):

    #print(min(predictions), max(predictions))

    # Identify where the absolute predictions are smaller than the fee
    no_trade = torch.abs(predictions) < fee
    
    # Calculate earnings for trading scenarios
    trading_earnings = torch.where(predictions * targets > 0, torch.abs(targets) - fee, -torch.abs(targets) - fee)
    trades_made = trading_earnings[~no_trade].cpu().detach().numpy()

    return list(trades_made) if len(trades_made) > 0 else [0]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

class TransformerLMDataset(Dataset):
    def __init__(self, sequence_data, ticker_list, device = 'cpu'):
        self.inputs, self.targets = create_dataset(sequence_data, ticker_list, device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]