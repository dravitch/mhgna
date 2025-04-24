# -*- coding: utf-8 -*-
"""
MHGNA Forex - Multi-Horizon Graphical Network Allocation for Forex Trading
===========================================================================

Adaptation of the MHGNA model specifically for Forex trading, integrating
macroeconomic data, interest rate differentials, and other indicators
specific to the currency market.

Author: [Your Name]
Date: April 2025
Version: 1.0.0
"""

import datetime
import json
import os
import re
import warnings
import argparse
import sys
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf
from dateutil.relativedelta import relativedelta
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.stats import zscore
from sklearn.covariance import GraphicalLassoCV

# Dash imports for UI
from dash import Dash, html, dcc
import plotly.graph_objects as go

# Ignore warnings
warnings.filterwarnings('ignore')

# --- Configuration ---

class ForexConfig:
    """Configuration settings for MHGNA Forex."""

    TICKERS: List[str] = [
        'EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD',
        'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY',
        'AUD/JPY', 'EUR/AUD', 'EUR/CHF', 'USD/MXN', 'USD/PLN'
    ]

    CURRENCIES: List[str] = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD', 'MXN', 'PLN']

    CURRENCY_TO_COUNTRY: Dict[str, str] = {
        'USD': 'US', 'EUR': 'EU', 'GBP': 'GB', 'JPY': 'JP',
        'CHF': 'CH', 'AUD': 'AU', 'CAD': 'CA', 'NZD': 'NZ',
        'MXN': 'MX', 'PLN': 'PL'
    }

    SAFE_PAIRS: List[str] = ['USD/CHF', 'USD/JPY', 'EUR/CHF']

    CARRY_PAIRS: List[str] = ['AUD/JPY', 'NZD/JPY', 'AUD/CHF', 'NZD/CHF']

    HORIZONS: Dict[str, int] = {
        'very_short': 1,  # 1 jour
        'short_term': 3,  # 3 jours
        'week': 7,        # 1 semaine
        'short': 10,      # ~2 semaines
        'medium': 60,     # ~3 mois
        'long': 120       # ~6 mois
    }

    DRAWDOWN_ALERT_THRESHOLD: float = -0.03
    VOLATILITY_ALERT_THRESHOLD: float = 0.70
    RANGE_BREAKOUT_THRESHOLD: float = 1.5

    SESSIONS: Dict[str, Dict[str, str]] = {
        'asia': {'start': '22:00', 'end': '08:00'},
        'europe': {'start': '07:00', 'end': '16:00'},
        'america': {'start': '13:00', 'end': '22:00'}
    }

    RECOMMENDED_PAIRS: int = 5

    CHART_STYLE: str = 'darkgrid'
    NETWORK_COLORS: str = 'viridis'
    FIGSIZE: Tuple[int, int] = (14, 10)

    LOOKBACK_PERIOD_YEARS: int = 1
    DATA_INTERVAL: str = '1d'

    API_KEYS: Dict[str, Optional[str]] = {
        'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_ALPHA_VANTAGE_API_KEY'),
        'fred': os.getenv('FRED_API_KEY', '0363ec0e0208840bc7552afaa843117a'),
        'news_api': os.getenv('NEWS_API_KEY', 'YOUR_NEWS_API_KEY'),
        'investing_com': os.getenv('INVESTING_COM_API_KEY', 'YOUR_INVESTING_COM_API_KEY'),
        'oanda': os.getenv('OANDA_API_KEY', 'YOUR_OANDA_API_KEY')
    }

    CACHE_DIR: str = 'data_cache'
    INTEREST_RATE_CACHE_FILE: str = os.path.join(CACHE_DIR, 'interest_rates.json')
    INTEREST_RATE_CACHE_EXPIRY_DAYS: int = 1
    INFLATION_CACHE_FILE: str = os.path.join(CACHE_DIR, 'inflation.json')
    INFLATION_CACHE_EXPIRY_DAYS: int = 7
    ECON_CALENDAR_CACHE_FILE: str = os.path.join(CACHE_DIR, 'economic_calendar.json')
    ECON_CALENDAR_CACHE_EXPIRY_HOURS: int = 12
    COMMODITY_CACHE_FILE: str = os.path.join(CACHE_DIR, 'commodities.json')
    COMMODITY_CACHE_EXPIRY_HOURS: int = 3

# --- Data Utilities ---

class ForexDataUtils:
    """Utilities for manipulating Forex data."""

    @staticmethod
    def standardize_yahoo_data(data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            present_cols = [col for col in required_cols if col in data.columns.get_level_values(0)]
            if not present_cols:
                return pd.DataFrame()
            result = pd.DataFrame(index=data.index)
            for col_name in present_cols:
                result[col_name.lower()] = data[col_name].iloc[:, 0]
            return result.rename(columns={'adj close': 'adj_close'})
        else:
            data.columns = data.columns.str.lower().str.replace(' ', '_', regex=False)
            if 'adj_close' not in data.columns and 'adj close' in data.columns:
                data = data.rename(columns={'adj close': 'adj_close'})
            return data

    @staticmethod
    def convert_to_yahoo_forex_symbol(forex_pair: str) -> str:
        if '/' in forex_pair:
            base, quote = forex_pair.split('/')
            return f"{base}{quote}=X"
        elif forex_pair.endswith('=X'):
            return forex_pair
        elif len(forex_pair) == 6:
            return f"{forex_pair}=X"
        else:
            print(f"Warning: Could not convert '{forex_pair}' to Yahoo format confidently.")
            return forex_pair

    @staticmethod
    def convert_from_yahoo_forex_symbol(yahoo_symbol: str) -> str:
        if yahoo_symbol.endswith('=X'):
            base_quote = yahoo_symbol[:-2]
            if len(base_quote) == 6:
                return f"{base_quote[:3]}/{base_quote[3:]}"
        return yahoo_symbol

    @staticmethod
    def parse_forex_timeframe(timeframe_str: Optional[str]) -> int:
        default_minutes = 1440
        if not timeframe_str:
            return default_minutes
        timeframe_str = timeframe_str.lower().strip()
        match = re.match(r'(\d+)([mhdw])', timeframe_str)
        if not match:
            print(f"Warning: Invalid timeframe format '{timeframe_str}'. Defaulting to 1 day.")
            return default_minutes
        value_str, unit = match.groups()
        try:
            value = int(value_str)
        except ValueError:
            print(f"Warning: Invalid timeframe value '{value_str}'. Defaulting to 1 day.")
            return default_minutes
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        elif unit == 'w':
            return value * 7 * 1440
        else:
            return default_minutes

# --- Macro Data Collection ---

class MacroDataCollector:
    """Collects and integrates relevant macroeconomic data for Forex analysis."""

    def __init__(self, config: ForexConfig = ForexConfig()):
        self.config = config
        self.api_keys = config.API_KEYS
        self.data: Dict[str, Any] = {}
        self.last_updated: Dict[str, str] = {}
        if not os.path.exists(self.config.CACHE_DIR):
            try:
                os.makedirs(self.config.CACHE_DIR)
            except OSError as e:
                print(f"Error creating cache directory '{self.config.CACHE_DIR}': {e}")

    def _read_cache(self, cache_file: str, expiry_seconds: int) -> Optional[Any]:
        if os.path.exists(cache_file):
            try:
                cache_mod_time = os.path.getmtime(cache_file)
                cache_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(cache_mod_time)
                if cache_age.total_seconds() < expiry_seconds:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except (IOError, json.JSONDecodeError, OSError) as e:
                print(f"Error reading cache file '{cache_file}': {e}")
        return None

    def _write_cache(self, cache_file: str, data_to_cache: Any):
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_cache, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"Error writing cache file '{cache_file}': {e}")

    @lru_cache(maxsize=4)
    def fetch_interest_rates(self, currencies: Optional[List[str]] = None) -> Dict[str, float]:
        target_currencies = currencies or self.config.CURRENCIES
        print(f"Fetching interest rates for {len(target_currencies)} currencies...")
        data_key = 'interest_rates'
        cache_file = self.config.INTEREST_RATE_CACHE_FILE
        cache_expiry = self.config.INTEREST_RATE_CACHE_EXPIRY_DAYS * 86400

        cached_data = self._read_cache(cache_file, cache_expiry)
        if cached_data:
            cached_rates = cached_data.get('rates', {})
            if cached_rates:
                self.data[data_key] = cached_rates
                self.last_updated[data_key] = cached_data.get('timestamp', '')
                print(f"Interest rates loaded from cache ({len(self.data[data_key])} currencies).")
                return self.data[data_key]

        interest_rates: Dict[str, float] = {}
        fred_api_key = self.api_keys.get('fred')
        fred_ids: Dict[str, str] = {
            'USD': 'FEDFUNDS', 'EUR': 'ECBDFR', 'GBP': 'BOEBR', 'JPY': 'BOJDPR',
            'CHF': 'SNBPRA', 'AUD': 'RBATCTR', 'CAD': 'BOCWLR', 'NZD': 'RBOKCR',
        }
        if fred_api_key and fred_api_key != 'YOUR_FRED_API_KEY':
            for currency in target_currencies:
                if currency in fred_ids:
                    series_id = fred_ids[currency]
                    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1"
                    try:
                        response = requests.get(url, timeout=10)
                        data = response.json()
                        if 'observations' in data and data['observations']:
                            rate = float(data['observations'][0]['value'])
                            interest_rates[currency] = rate
                    except Exception as e:
                        print(f"Error fetching rate for {currency}: {e}")
        else:
            interest_rates = {
                'USD': 5.33, 'EUR': 4.00, 'GBP': 5.25, 'JPY': 0.10,
                'CHF': 1.50, 'AUD': 4.35, 'CAD': 4.75, 'NZD': 5.50,
                'MXN': 11.00, 'PLN': 5.75
            }
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write_cache(cache_file, {'rates': interest_rates, 'timestamp': current_time_str})
        self.data[data_key] = interest_rates
        self.last_updated[data_key] = current_time_str
        return interest_rates

    @lru_cache(maxsize=4)
    def fetch_gdp_data(self, countries: Optional[List[str]] = None) -> Dict[str, float]:
        target_countries = countries or [self.config.CURRENCY_TO_COUNTRY.get(c) for c in self.config.CURRENCIES]
        target_countries = sorted(list(set(c for c in target_countries if c)))
        cache_file = os.path.join(self.config.CACHE_DIR, 'gdp.json')
        cache_expiry = 30 * 86400

        cached_data = self._read_cache(cache_file, cache_expiry)
        if cached_data:
            self.data['gdp'] = cached_data.get('gdp', {})
            print(f"GDP data loaded from cache for {len(self.data['gdp'])} countries.")
            return self.data['gdp']

        gdp_data = {}
        fred_api_key = self.api_keys.get('fred')
        if fred_api_key and fred_api_key != 'YOUR_FRED_API_KEY':
            fred_ids = {'US': 'GDP', 'EU': 'EUNNGDP', 'GB': 'UKNGDP', 'JP': 'JPNRGDPEXP'}
            for country in target_countries:
                if country in fred_ids:
                    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={fred_ids[country]}&api_key={fred_api_key}&file_type=json&limit=1&sort_order=desc"
                    try:
                        response = requests.get(url, timeout=10)
                        data = response.json()
                        if 'observations' in data and data['observations']:
                            gdp_data[country] = float(data['observations'][0]['value'])
                    except Exception as e:
                        print(f"Error fetching GDP for {country}: {e}")
        else:
            gdp_data = {'US': 27.36, 'EU': 15.81, 'GB': 3.34, 'JP': 4.21}
        self._write_cache(cache_file, {'gdp': gdp_data, 'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.data['gdp'] = gdp_data
        return gdp_data

    def fetch_cot_reports(self) -> Dict[str, float]:
        cache_file = os.path.join(self.config.CACHE_DIR, 'cot.json')
        cached_data = self._read_cache(cache_file, 7 * 86400)
        if cached_data:
            self.data['cot'] = cached_data.get('cot', {})
            return self.data['cot']
        cot_data = {'USD': 0.2, 'EUR': -0.1}
        self._write_cache(cache_file, {'cot': cot_data, 'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.data['cot'] = cot_data
        return cot_data

# --- Core MHGNA Forex Logic ---

class MHGNAForex:
    """MHGNA model tailored for Forex trading."""

    def __init__(self, config: ForexConfig = ForexConfig()):
        self.config = config
        self.utils = ForexDataUtils()
        self.macro = MacroDataCollector(self.config)
        self.forex_pairs: List[str] = self.config.TICKERS
        self.data: Optional[pd.DataFrame] = None
        self.ohlc_data: Optional[Dict[str, pd.DataFrame]] = None
        self.returns: Optional[pd.DataFrame] = None
        self.volatility: Optional[pd.DataFrame] = None
        self.momentum: Optional[pd.DataFrame] = None
        self.network_graph: Optional[nx.Graph] = None
        self.last_recommendations: Optional[pd.DataFrame] = None
        self.technicals: Dict[str, Dict[str, pd.Series]] = {}
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        try:
            sns.set_style(self.config.CHART_STYLE)
            plt.rcParams['figure.figsize'] = self.config.FIGSIZE
        except ValueError:
            sns.set_style('darkgrid')
            plt.rcParams['figure.figsize'] = self.config.FIGSIZE

    def fetch_forex_data(self, end_date: Optional[str] = None, interval: str = ForexConfig.DATA_INTERVAL) -> bool:
        try:
            end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now().normalize()
            if end_dt.tz is not None:
                end_dt = end_dt.tz_localize(None)
        except ValueError as e:
            print(f"Error parsing end_date '{end_date}': {e}. Defaulting to today.")
            end_dt = pd.Timestamp.now().normalize()

        start_dt = end_dt - relativedelta(years=self.config.LOOKBACK_PERIOD_YEARS)
        print(f"Fetching Forex data from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} with interval {interval}...")

        yahoo_symbols_map = {ticker: self.utils.convert_to_yahoo_forex_symbol(ticker) for ticker in self.forex_pairs}
        yahoo_symbols_list = list(yahoo_symbols_map.values())

        try:
            data_raw = yf.download(
                tickers=yahoo_symbols_list,
                start=start_dt,
                end=end_dt + datetime.timedelta(days=1),
                interval=interval,
                progress=True,
                group_by='ticker'
            )
        except Exception as e:
            print(f"Error during yfinance download: {e}")
            return False

        if data_raw.empty:
            print("No data returned from yfinance.")
            return False

        close_prices = pd.DataFrame()
        ohlc_data_dict = {}
        successful_pairs = []

        for ticker, yahoo_symbol in yahoo_symbols_map.items():
            try:
                if isinstance(data_raw.columns, pd.MultiIndex) and yahoo_symbol in data_raw.columns.levels[0]:
                    pair_data_full = data_raw[yahoo_symbol].copy()
                elif not isinstance(data_raw.columns, pd.MultiIndex) and len(yahoo_symbols_list) == 1:
                    pair_data_full = data_raw.copy()
                else:
                    print(f"  No data for {ticker} ({yahoo_symbol}).")
                    continue

                pair_data_full.columns = pair_data_full.columns.str.lower()
                if 'adj close' in pair_data_full.columns:
                    pair_data_full = pair_data_full.rename(columns={'adj close': 'adj_close'})

                price_col = 'adj_close' if 'adj_close' in pair_data_full.columns else 'close'
                pair_prices = pair_data_full[price_col].dropna()
                pair_data_full = pair_data_full.loc[pair_prices.index]

                if not pair_prices.empty:
                    close_prices[ticker] = pair_prices
                    if all(c in pair_data_full.columns for c in ['open', 'high', 'low', 'close']):
                        ohlc_data_dict[ticker] = pair_data_full[['open', 'high', 'low', 'close']].copy()
                    successful_pairs.append(ticker)
            except Exception as e:
                print(f"  Error processing {ticker} ({yahoo_symbol}): {e}")

        if close_prices.empty:
            print("No valid price data processed.")
            return False

        common_index = close_prices.dropna(axis=0, how='all').index
        self.data = close_prices.loc[common_index].copy()
        self.ohlc_data = {ticker: df.loc[common_index].copy() for ticker, df in ohlc_data_dict.items() if df is not None}
        self.forex_pairs = successful_pairs

        self.data = self.data.sort_index()
        self.returns = self.data.pct_change().dropna(axis=0, how='all')
        annualization_factor = 252 if interval == '1d' else 252 * 24 if interval == '1h' else 252
        self.volatility = self.returns.rolling(window=20, min_periods=10).std().dropna() * np.sqrt(annualization_factor)

        momentum_scores = {}
        for horizon_name, horizon_days in self.config.HORIZONS.items():
            min_periods = int(horizon_days * 0.8)
            momentum_scores[horizon_name] = self.returns.rolling(window=horizon_days, min_periods=min_periods).sum()
        composite_momentum = pd.concat(momentum_scores.values(), axis=1).mean(axis=1)
        self.momentum = pd.DataFrame(composite_momentum, columns=['composite_momentum'])

        return True

    def calculate_technical_indicators(self):
        if self.data is None:
            return
        self.technicals = {}
        for pair in self.data.columns:
            prices = self.data[pair].dropna()
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.technicals[pair] = {'RSI': 100 - (100 / (1 + rs))}

    def build_forex_network(self):
        if self.returns is None or self.returns.empty:
            return
        model = GraphicalLassoCV()
        model.fit(self.returns.dropna())
        precision_matrix = model.precision_
        G = nx.Graph()
        for i, ticker1 in enumerate(self.forex_pairs):
            G.add_node(ticker1)
            for j, ticker2 in enumerate(self.forex_pairs[i+1:], start=i+1):
                weight = precision_matrix[i, j]
                if abs(weight) > 0.01:
                    G.add_edge(ticker1, ticker2, weight=weight)
        centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        nx.set_node_attributes(G, centrality, 'centrality')
        self.network_graph = G

    def recommend_forex_pairs(self):
        if self.data is None or self.returns is None:
            return None
        self.macro.fetch_interest_rates()
        self.macro.fetch_gdp_data()
        self.macro.fetch_cot_reports()
        self.calculate_technical_indicators()
        self.build_forex_network()

        metrics = pd.DataFrame(index=self.forex_pairs)
        metrics['momentum'] = self.momentum['composite_momentum'].iloc[-1] if self.momentum is not None else 0
        metrics['volatility'] = self.volatility.iloc[-1] if self.volatility is not None else 1
        metrics['inv_volatility'] = 1 / metrics['volatility']
        metrics['carry_score'] = [self.macro.data['interest_rates'].get(p.split('/')[0], 0) -
                                  self.macro.data['interest_rates'].get(p.split('/')[1], 0)
                                  for p in self.forex_pairs]
        metrics['gdp_diff'] = [self.macro.data['gdp'].get(self.config.CURRENCY_TO_COUNTRY.get(p.split('/')[0]), 0) -
                               self.macro.data['gdp'].get(self.config.CURRENCY_TO_COUNTRY.get(p.split('/')[1]), 0)
                               for p in self.forex_pairs]
        metrics['cot_score'] = [self.macro.data['cot'].get(p.split('/')[0], 0) -
                                self.macro.data['cot'].get(p.split('/')[1], 0)
                                for p in self.forex_pairs]
        metrics['rsi'] = [self.technicals[p]['RSI'].iloc[-1] if p in self.technicals else 50
                          for p in self.forex_pairs]
        metrics['eigenvector'] = [self.network_graph.nodes[p]['centrality'] if self.network_graph and p in self.network_graph else 0
                                  for p in self.forex_pairs]

        metrics['score'] = (metrics['eigenvector'] * 0.2 + metrics['momentum'] * 0.2 +
                            metrics['carry_score'] * 0.2 + metrics['inv_volatility'] * 0.2 +
                            metrics['gdp_diff'] * 0.1 + metrics['cot_score'] * 0.1 +
                            (metrics['rsi'] - 50) * 0.1)
        recommendations = metrics.sort_values(by='score', ascending=False).head(self.config.RECOMMENDED_PAIRS)
        self.last_recommendations = recommendations
        return recommendations

# --- User Interface with Dash ---

def create_dashboard(mhgna: MHGNAForex):
    app = Dash(__name__)
    G = mhgna.network_graph
    if G is not None:
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        node_x, node_y = zip(*[pos[node] for node in G.nodes()])
        node_centrality = [G.nodes[n]['centrality'] * 100 for n in G.nodes()]
        network_fig = go.Figure(data=[
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='gray')),
            go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                       marker=dict(size=15, color=node_centrality, colorscale='Viridis', showscale=True))
        ])
        network_fig.update_layout(title="Forex Dependency Network", showlegend=False)
    else:
        network_fig = go.Figure()

    recommendations = mhgna.last_recommendations
    if recommendations is not None and not recommendations.empty:
        reco_fig = go.Figure(data=[
            go.Bar(x=recommendations.index, y=recommendations['score'], name='Score'),
            go.Bar(x=recommendations.index, y=recommendations['rsi'], name='RSI')
        ])
        reco_fig.update_layout(title="Top Recommended Pairs", barmode='group')
    else:
        reco_fig = go.Figure()

    app.layout = html.Div([
        html.H1("MHGNA Forex Dashboard"),
        dcc.Graph(id='network-graph', figure=network_fig),
        html.H2("Recommendations"),
        dcc.Graph(id='recommendations', figure=reco_fig),
        html.Pre(recommendations.to_string() if recommendations is not None else "No recommendations available.")
    ])
    app.run_server(debug=True)

# --- Command-Line Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MHGNA Forex Analysis Tool')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d, 1h)')
    args = parser.parse_args()

    config = ForexConfig()
    config.DATA_INTERVAL = args.interval

    forex_analyzer = MHGNAForex(config=config)
    if forex_analyzer.fetch_forex_data(interval=args.interval):
        forex_analyzer.recommend_forex_pairs()
        create_dashboard(forex_analyzer)
    else:
        print("Failed to fetch data. Exiting.")