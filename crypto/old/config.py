import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, graphical_lasso
from scipy.stats import zscore
from datetime import datetime, timedelta
import warnings
import pytz
import math
import time
from sklearn.cluster import AgglomerativeClustering
from itertools import product

warnings.filterwarnings('ignore')

# Configuration des paramètres
class Config:
    # Paramètres de la stratégie
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
               'ADA-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD',
               'UNI-USD', 'AAVE-USD', 'CRV-USD', 'ATOM-USD', 'LTC-USD']
    start_date = '2022-01-01'
    end_date = '2025-01-01'

    # Paramètres d'horizon multiple
    horizons = {
        'court': {'window': 30, 'weight': 0.25},
        'moyen': {'window': 90, 'weight': 0.50},
        'long': {'window': 180, 'weight': 0.25}
    }
