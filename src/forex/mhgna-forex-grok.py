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
import argparse # Added for __main__
import sys # Added for __main__
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

# Ignore warnings
warnings.filterwarnings('ignore')

# --- Configuration ---

class ForexConfig:
    """Configuration settings for MHGNA Forex."""

    # Forex pairs to track (majors and crosses)
    TICKERS: List[str] = [
        'EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD',
        'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY',
        'AUD/JPY', 'EUR/AUD', 'EUR/CHF', 'USD/MXN', 'USD/PLN'
    ]

    # Individual currencies for tracking macro data
    CURRENCIES: List[str] = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD', 'MXN', 'PLN']

    # Countries associated with currencies (for macro data)
    CURRENCY_TO_COUNTRY: Dict[str, str] = {
        'USD': 'US', 'EUR': 'EU', 'GBP': 'GB', 'JPY': 'JP',
        'CHF': 'CH', 'AUD': 'AU', 'CAD': 'CA', 'NZD': 'NZ',
        'MXN': 'MX', 'PLN': 'PL'
    }

    # Safe-haven pairs for periods of volatility
    SAFE_PAIRS: List[str] = ['USD/CHF', 'USD/JPY', 'EUR/CHF']

    # Carry trade pairs (high interest rate differential) - subjective, example
    CARRY_PAIRS: List[str] = ['AUD/JPY', 'NZD/JPY', 'AUD/CHF', 'NZD/CHF']

    # Time horizons adapted for Forex (in trading days)
    HORIZONS: Dict[str, int] = {
        'short': 10,  # Approx. 2 weeks
        'medium': 60,  # Approx. 3 months
        'long': 120   # Approx. 6 months
    }

    # Forex alert parameters
    DRAWDOWN_ALERT_THRESHOLD: float = -0.03  # More sensitive for Forex (3%)
    VOLATILITY_ALERT_THRESHOLD: float = 0.70  # Alert if vol > 70% of historical max
    RANGE_BREAKOUT_THRESHOLD: float = 1.5  # Range breakout threshold (in std devs)

    # Trading session times (UTC)
    SESSIONS: Dict[str, Dict[str, str]] = {
        'asia': {'start': '22:00', 'end': '08:00'},
        'europe': {'start': '07:00', 'end': '16:00'},
        'america': {'start': '13:00', 'end': '22:00'}
    }

    # Number of pairs to recommend
    RECOMMENDED_PAIRS: int = 5

    # Visualization parameters
    CHART_STYLE: str = 'darkgrid' # 'darkgrid' or other seaborn styles
    NETWORK_COLORS: str = 'viridis' # Colormap for network graph
    FIGSIZE: Tuple[int, int] = (14, 10) # Default figure size

    # Data retrieval lookback period (in years)
    LOOKBACK_PERIOD_YEARS: int = 1

    # --- API Keys ---
    # IMPORTANT: Replace placeholders and manage securely (e.g., environment variables)
    API_KEYS: Dict[str, Optional[str]] = {
        'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_ALPHA_VANTAGE_API_KEY'),
        'fred': os.getenv('FRED_API_KEY', '0363ec0e0208840bc7552afaa843117a'),
        'news_api': os.getenv('NEWS_API_KEY', 'YOUR_NEWS_API_KEY'),
        'investing_com': os.getenv('INVESTING_COM_API_KEY', 'YOUR_INVESTING_COM_API_KEY'), # Hypothetical
        'oanda': os.getenv('OANDA_API_KEY', 'YOUR_OANDA_API_KEY')
    }

    # --- Caching ---
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
        """
        Standardizes data from Yahoo Finance, handling potential MultiIndex columns.

        Args:
            data: DataFrame downloaded from yfinance.

        Returns:
            DataFrame with standardized columns ('open', 'high', 'low', 'close', 'volume', 'adj_close').
        """
        if isinstance(data.columns, pd.MultiIndex):
            # Ensure standard column names exist at the top level
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            present_cols = [col for col in required_cols if col in data.columns.get_level_values(0)]

            if not present_cols:
                return pd.DataFrame() # Return empty if no standard columns found

            # Select the first level of the MultiIndex for present columns
            result = pd.DataFrame(index=data.index)
            for col_name in present_cols:
                 # Take the first sub-column for each main column type
                result[col_name.lower()] = data[col_name].iloc[:, 0]

            return result.rename(columns={'adj close': 'adj_close'}) # Standardize adj close name
        else:
            # Standardize column names if not MultiIndex
            data.columns = data.columns.str.lower().str.replace(' ', '_', regex=False)
            # Ensure 'adj close' is named correctly if present
            if 'adj_close' not in data.columns and 'adj close' in data.columns:
                 data = data.rename(columns={'adj close': 'adj_close'})
            return data

    @staticmethod
    def convert_to_yahoo_forex_symbol(forex_pair: str) -> str:
        """
        Converts a standard Forex pair string (e.g., 'EUR/USD') to Yahoo Finance format ('EURUSD=X').

        Args:
            forex_pair: Standard Forex pair string.

        Returns:
            Yahoo Finance formatted symbol string.
        """
        if '/' in forex_pair:
            base, quote = forex_pair.split('/')
            return f"{base}{quote}=X"
        elif forex_pair.endswith('=X'):
            # Already in Yahoo format
            return forex_pair
        elif len(forex_pair) == 6:
             # Assume it's like 'EURUSD' and append '=X'
             return f"{forex_pair}=X"
        else:
            # Return original if format is uncertain
            print(f"Warning: Could not convert '{forex_pair}' to Yahoo format confidently.")
            return forex_pair


    @staticmethod
    def convert_from_yahoo_forex_symbol(yahoo_symbol: str) -> str:
        """
        Converts a Yahoo Finance symbol string ('EURUSD=X') back to standard Forex format ('EUR/USD').

        Args:
            yahoo_symbol: Yahoo Finance formatted symbol string.

        Returns:
            Standard Forex pair string.
        """
        if yahoo_symbol.endswith('=X'):
            base_quote = yahoo_symbol[:-2]
            if len(base_quote) == 6:
                return f"{base_quote[:3]}/{base_quote[3:]}"
        # Return original if conversion fails
        return yahoo_symbol

    @staticmethod
    def parse_forex_timeframe(timeframe_str: Optional[str]) -> int:
        """
        Converts a timeframe string (e.g., '1h', '15m', '4h', '1d') into minutes.

        Args:
            timeframe_str: String representation of the timeframe. Defaults to '1d' if None or invalid.

        Returns:
            Timeframe duration in minutes.
        """
        default_minutes = 1440 # 1 day

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
            # Should not happen due to regex, but as a safeguard
            return default_minutes


# --- Macro Data Collection ---

class MacroDataCollector:
    """
    Collects and integrates relevant macroeconomic data for Forex analysis.
    Uses file caching to minimize redundant API calls.
    """

    def __init__(self, config: ForexConfig = ForexConfig()):
        """
        Initializes the MacroDataCollector.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.api_keys = config.API_KEYS
        self.data: Dict[str, Any] = {}
        self.last_updated: Dict[str, str] = {}

        # Create cache directory if it doesn't exist
        if not os.path.exists(self.config.CACHE_DIR):
            try:
                os.makedirs(self.config.CACHE_DIR)
            except OSError as e:
                print(f"Error creating cache directory '{self.config.CACHE_DIR}': {e}")


    def _read_cache(self, cache_file: str, expiry_seconds: int) -> Optional[Any]:
        """Helper function to read from cache if valid."""
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
        """Helper function to write data to cache."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_cache, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"Error writing cache file '{cache_file}': {e}")

    @lru_cache(maxsize=4) # Cache recent calls within the same run
    # DANS LA CLASSE MacroDataCollector:

    @lru_cache (maxsize=4)  # Cache recent calls within the same run
    def fetch_interest_rates (self,currencies: Optional[List[str]] = None) -> Dict[str,float]:
        """
        Fetches central bank interest rates for specified currencies using FRED API.
        Uses file cache for persistence between runs.

        Args:
            currencies: List of currency codes (e.g., ['USD', 'EUR']). Defaults to config.

        Returns:
            Dictionary mapping currency code to its latest interest rate.
        """
        target_currencies=currencies or self.config.CURRENCIES
        print (f"Fetching interest rates for {len (target_currencies)} currencies...")
        data_key='interest_rates'
        cache_file=self.config.INTEREST_RATE_CACHE_FILE
        cache_expiry=self.config.INTEREST_RATE_CACHE_EXPIRY_DAYS * 86400  # seconds

        # Try reading from cache first
        cached_data=self._read_cache (cache_file,cache_expiry)
        if cached_data:
            # Check if cache actually contains rate data
            cached_rates=cached_data.get ('rates',{})
            if cached_rates:  # Only return if not empty
                self.data[data_key]=cached_rates
                self.last_updated[data_key]=cached_data.get ('timestamp','')
                print (f"Interest rates loaded from cache ({len (self.data[data_key])} currencies).")
                return self.data[data_key]
            else:
                print ("  Cache contained no interest rate data, proceeding to fetch.")

        # --- Fetch from API ---
        interest_rates: Dict[str,float]={}
        fred_api_key=self.api_keys.get ('fred')
        # Vérification robuste (Robust check)
        # is_key_valid=fred_api_key and isinstance (fred_api_key,str) and len (fred_api_key) > 5 and fred_api_key != '0363ec0e0208840bc7552afaa843117a'
        # Par celle-ci (plus simple) :
        is_key_valid=fred_api_key and isinstance (fred_api_key,str) and len (fred_api_key) > 5  # Vérifie juste si une clé d'une certaine longueur existe
        # (Note: Adjust 'YOUR_FRED_API_KEY' if you use a different default placeholder string in ForexConfig)

        fred_ids: Dict[str,str]={  # FRED series IDs for key policy rates
            'USD':'FEDFUNDS','EUR':'ECBDFR','GBP':'BOEBR','JPY':'BOJDPR',
            'CHF':'SNBPRA','AUD':'RBATCTR','CAD':'BOCWLR','NZD':'RBOKCR',  # Using BOC key rate
        }

        # --- Attempt API Fetch if Key is Valid ---
        if is_key_valid:
            print (f"    DEBUG: Using FRED Key: {str (fred_api_key)[:4]}...{str (fred_api_key)[-4:]}")
            print (f"  Attempting to fetch data using FRED API key...")
            for currency in target_currencies:
                if currency in fred_ids:
                    series_id=fred_ids[currency]
                    url=(f"https://api.stlouisfed.org/fred/series/observations?"
                         f"series_id={series_id}&api_key={fred_api_key}"
                         f"&file_type=json&sort_order=desc&limit=1")

                    # *** BLOC TRY/EXCEPT CORRECTEMENT INDENTÉ ***
                    try:
                        # --- Code indenté sous try: ---
                        response=requests.get (url,timeout=10)
                        response.raise_for_status ()  # Raise HTTPError for bad responses (4xx or 5xx)
                        data=response.json ()
                        if 'observations' in data and data['observations']:
                            # Find the first valid observation
                            for obs in data['observations']:
                                if obs.get ('value') and obs['value'] != '.':  # Safer access with .get()
                                    try:
                                        rate=float (obs['value'])
                                        interest_rates[currency]=rate
                                        print (f"    • Fetched {currency} ({series_id}): {rate:.2f}%")
                                        break  # Exit inner loop once valid rate is found
                                    except ValueError:
                                        print (
                                            f"    • Error converting value '{obs['value']}' to float for {currency}.")
                                        continue  # Try next observation if conversion fails
                            # This else corresponds to the inner 'for obs...' loop
                            # else: # Optional: If loop completes without break (no valid obs found)
                            #    print(f"    • {currency} ({series_id}): No valid observations found in response.")

                        else:  # Corresponds to 'if 'observations' in data...'
                            print (
                                f"    • {currency} ({series_id}): 'observations' key missing or empty in API response.")

                    # --- except blocks alignés avec try: ---
                    except requests.exceptions.RequestException as e:
                        print (f"    • Error fetching rate for {currency} ({series_id}): {e}")
                    except (ValueError,KeyError,json.JSONDecodeError) as e:  # Added JSONDecodeError
                        print (f"    • Error parsing data or value for {currency} ({series_id}): {e}")
                    # --- FIN DU BLOC TRY/EXCEPT ---

                # else: # Optional: Handle currencies not in fred_ids dict
                # print(f"  Skipping {currency}: No FRED ID defined.")

            # --- FIN DE LA BOUCLE 'for currency...' ---

        # --- Fallback Data ---
        # S'exécute si la clé n'était pas valide OU si l'appel API (même s'il a été tenté) n'a rempli aucune donnée
        if not interest_rates:  # Check if the dictionary is still empty
            if not is_key_valid:  # Explain why fallback is used
                print ("Warning: FRED API key not configured or is placeholder. Using fallback data.")
            else:
                print ("Warning: FRED API call failed or returned no data for configured IDs. Using fallback data.")

            fallback_interest_rates={  # Example rates - Update periodically
                'USD':5.33,'EUR':4.00,'GBP':5.25,'JPY':0.10,
                'CHF':1.50,'AUD':4.35,'CAD':4.75,'NZD':5.50,
                'MXN':11.00,'PLN':5.75
            }
            # Only include requested currencies in fallback
            interest_rates={k:v for k,v in fallback_interest_rates.items () if k in target_currencies}

        # --- Update Cache and State ---
        current_time_str=datetime.datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')
        self._write_cache (cache_file,{'rates':interest_rates,'timestamp':current_time_str})
        self.data[data_key]=interest_rates
        self.last_updated[data_key]=current_time_str
        print (f"Finished fetching/loading interest rates for {len (interest_rates)} currencies.")

        return interest_rates



    @lru_cache(maxsize=4)
    # DANS LA CLASSE MacroDataCollector:

    def fetch_inflation_data (self,countries: Optional[List[str]] = None) -> Dict[str,float]:
        """
        Fetches latest inflation data (CPI YoY % change) for specified countries using FRED API.
        Uses file cache.

        Args:
            countries: List of country codes (e.g., ['US', 'EU']). Defaults based on config currencies.

        Returns:
            Dictionary mapping country code to its latest inflation rate.
        """
        # --- Début de la fonction (indentation correcte) ---
        if not countries:
            target_countries=[self.config.CURRENCY_TO_COUNTRY.get (c) for c in self.config.CURRENCIES]
            target_countries=sorted (list (set (c for c in target_countries if c)))  # Unique, sorted
        else:
            target_countries=sorted (list (set (countries)))

        print (f"Fetching inflation data for {len (target_countries)} countries...")
        data_key='inflation'
        cache_file=self.config.INFLATION_CACHE_FILE
        cache_expiry=self.config.INFLATION_CACHE_EXPIRY_DAYS * 86400  # seconds

        cached_data=self._read_cache (cache_file,cache_expiry)
        if cached_data:
            self.data[data_key]=cached_data.get ('inflation',{})
            self.last_updated[data_key]=cached_data.get ('timestamp','')
            print (f"Inflation data loaded from cache ({len (self.data[data_key])} countries).")
            # Ensure the loaded data is returned immediately if cache is hit and valid
            if self.data[data_key]:  # Only return if cache actually contains data
                return self.data[data_key]
            else:
                print ("  Cache contained no inflation data, proceeding to fetch.")

        # --- Fetch from API ---
        inflation_data: Dict[str, float] = {}
        # Vérification robuste : la clé existe, est une chaîne, a une longueur > 5 (arbitraire)
        # et n'est pas le placeholder par défaut 'YOUR_FRED_API_KEY'
        # is_key_valid=fred_api_key and isinstance (fred_api_key,str) and len (fred_api_key) > 5 and fred_api_key != '0363ec0e0208840bc7552afaa843117a'
        # Par celle-ci (plus simple) :
        is_key_valid=fred_api_key and isinstance (fred_api_key,str) and len (fred_api_key) > 5  # Vérifie juste si une clé d'une certaine longueur existe

        # --- DÉFINITION DE fred_ids - CORRECTEMENT INDENTÉE (niveau principal de la fonction) ---
        fred_ids: Dict[str,str]={  # FRED CPI series IDs (YoY % Change preferred)
            'US':'CPIAUCSL',  # CPI All Urban Consumers, SA -> Needs YoY calc if using this series directly
            # Use 'CPALTT01USM657N' for pre-calculated YoY if available
            'EU':'CPHPTT01EZM657N',  # HICP - Overall index, pct change YoY (preferred)
            'GB':'CPALTT01GBM657N',  # CPI Total, All Items, Pct Change YoY
            'JP':'JPNCPIALLMINMEI',  # CPI Total, All Items, Index -> Needs YoY calc
            'CH':'CHECPIALLMINMEI',  # CPI Total, All Items, Index -> Needs YoY calc
            'AU':'AUSCPIALLQINMEI',  # CPI Total, All Items, Index (Quarterly) -> Needs YoY calc
            'CA':'CANCPIALLMINMEI',  # CPI Total, All Items, Index -> Needs YoY calc
            'NZ':'NZLCPIALLQINMEI',  # CPI Total, All Items, Index (Quarterly) -> Needs YoY calc
            # MX, PL might need different sources/IDs
        }
        # Note: Some FRED series are indices, requiring manual YoY calculation.
        #       For simplicity here, we might use pre-calculated YoY series if available,
        #       or fallback data if FRED API is complex for YoY. Using fallback for now.
        # --- FIN DÉFINITION DE fred_ids ---

        # --- CONDITION IF - CORRECTEMENT INDENTÉE ---
        if is_key_valid:
            # Code à l'intérieur du if - CORRECTEMENT INDENTÉ
            print (f"  Attempting to fetch inflation data using FRED API key...")  # Message mis à jour
            # Logique d'appel API (actuellement en commentaire/pass)
            print (
                "    (Note: Actual FRED API call for inflation needs implementation/refinement for YoY calculation). Using fallback for now.")
            # Placeholder: Add actual FRED API calls with YoY calculation here if needed
            # Example structure (needs refinement):
            # for country in target_countries:
            #     if country in fred_ids:
            #         series_id = fred_ids[country]
            #         # ... Call FRED API for series_id ...
            #         # ... Perform YoY calculation if needed ...
            #         # ... Store result in inflation_data[country] ...
            #         pass
            # ===> FIN BLOC INDENTÉ SOUS 'if is_key_valid:' <===
        else:
            # Code à l'intérieur du else - CORRECTEMENT INDENTÉ
            print ("Warning: FRED API key not configured or is placeholder.")
            # Proceed to fallback data below

        # --- Fallback Data (indentation correcte, hors du if/else) ---
        # S'exécute si la clé n'est pas valide OU si l'appel API (s'il était implémenté) a échoué
        if not inflation_data:  # Si inflation_data est toujours vide
            print ("Warning: Using placeholder fallback inflation data.")
            fallback_inflation_data={  # Example rates - Update periodically
                'US':3.1,'EU':2.4,'GB':3.2,'JP':2.8,
                'CH':1.4,'AU':3.6,'CA':2.9,'NZ':4.0,
                'MX':4.6,'PL':2.0
            }
            # Assigner les données fallback UNIQUEMENT pour les pays cibles demandés
            inflation_data={k:v for k,v in fallback_inflation_data.items () if k in target_countries}

        # --- Update Cache and State (indentation correcte) ---
        current_time_str=datetime.datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')
        # Écrire dans le cache même si les données fallback ont été utilisées
        self._write_cache (cache_file,{'inflation':inflation_data,'timestamp':current_time_str})
        self.data[data_key]=inflation_data
        self.last_updated[data_key]=current_time_str
        print (f"Finished fetching/loading inflation data for {len (inflation_data)} countries.")

        return inflation_data

    def fetch_economic_calendar(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Fetches upcoming economic events.
        Uses file cache.
        NOTE: Requires a real API or scraping implementation for investing.com or similar.

        Args:
            days_ahead: Number of days ahead to fetch events for.

        Returns:
            List of economic event dictionaries.
        """
        print(f"Fetching economic calendar for the next {days_ahead} days...")
        data_key = 'economic_calendar'
        cache_file = self.config.ECON_CALENDAR_CACHE_FILE
        cache_expiry = self.config.ECON_CALENDAR_CACHE_EXPIRY_HOURS * 3600 # seconds

        # Try reading from cache
        cached_data = self._read_cache(cache_file, cache_expiry)
        if cached_data:
            self.data[data_key] = cached_data.get('events', [])
            self.last_updated[data_key] = cached_data.get('timestamp', '')
            print(f"Economic calendar loaded from cache ({len(self.data[data_key])} events).")
            return self.data[data_key]

        # --- Fetch from API (Hypothetical) ---
        economic_calendar: List[Dict[str, Any]] = []
        investing_api_key = self.api_keys.get('investing_com')

        # IMPORTANT: The following is a placeholder. investing.com does not have an official public API.
        # You would need to use a third-party provider, scrape responsibly, or find an alternative source.
        if investing_api_key and investing_api_key != 'YOUR_INVESTING_COM_API_KEY':
            print("Warning: Investing.com API key found, but the API endpoint is hypothetical. Using fallback.")
            # Placeholder code for hypothetical API call:
            # today = datetime.date.today()
            # end_date = today + datetime.timedelta(days=days_ahead)
            # try:
            #     # url = f"HYPOTHETICAL_API_ENDPOINT?from={today.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&api_key={investing_api_key}"
            #     # response = requests.get(url, timeout=15)
            #     # response.raise_for_status()
            #     # data = response.json()
            #     # economic_calendar = data.get('events', [])
            #     print(f"  -> Hypothetical API call skipped.")
            # except requests.exceptions.RequestException as e:
            #     print(f"Error fetching hypothetical economic calendar: {e}")
            # except (ValueError, KeyError) as e:
            #     print(f"Error parsing hypothetical economic calendar data: {e}")
            pass # Skip actual call
        else:
            print("Warning: Economic calendar API key not configured or placeholder.")
            # Proceed to fallback data

        # --- Fallback Data ---
        if not economic_calendar:
            print("Warning: Using placeholder fallback economic calendar events.")
            today_dt = datetime.datetime.now()
            fallback_economic_calendar = [
                {'date': (today_dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d'), 'time': '12:30 UTC', 'currency': 'USD', 'name': 'Non-Farm Payrolls', 'impact': 'high', 'forecast': '180K', 'previous': '175K'},
                {'date': (today_dt + datetime.timedelta(days=2)).strftime('%Y-%m-%d'), 'time': '08:00 UTC', 'currency': 'EUR', 'name': 'German Factory Orders m/m', 'impact': 'medium', 'forecast': '0.5%', 'previous': '-0.2%'},
                {'date': (today_dt + datetime.timedelta(days=3)).strftime('%Y-%m-%d'), 'time': '18:00 UTC', 'currency': 'USD', 'name': 'FOMC Meeting Minutes', 'impact': 'high', 'forecast': '', 'previous': ''},
                {'date': (today_dt + datetime.timedelta(days=5)).strftime('%Y-%m-%d'), 'time': '06:00 UTC', 'currency': 'GBP', 'name': 'GDP m/m', 'impact': 'medium', 'forecast': '0.2%', 'previous': '0.1%'},
                {'date': (today_dt + datetime.timedelta(days=7)).strftime('%Y-%m-%d'), 'time': '23:50 UTC', 'currency': 'JPY', 'name': 'BoJ Summary of Opinions', 'impact': 'low', 'forecast': '', 'previous': ''},
            ]
            # Filter events to be within days_ahead limit
            end_limit_date = today_dt + datetime.timedelta(days=days_ahead)
            economic_calendar = [e for e in fallback_economic_calendar if e.get('date') <= end_limit_date.strftime('%Y-%m-%d')]


        # --- Update Cache and State ---
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write_cache(cache_file, {'events': economic_calendar, 'timestamp': current_time_str})
        self.data[data_key] = economic_calendar
        self.last_updated[data_key] = current_time_str
        print(f"Finished fetching/loading economic calendar ({len(economic_calendar)} events).")

        return economic_calendar


    def calculate_interest_rate_differentials(self) -> Dict[str, float]:
        """
        Calculates interest rate differentials for configured Forex pairs.

        Returns:
            Dictionary mapping Forex pair string to its interest rate differential (Base Rate - Quote Rate).
        """
        if 'interest_rates' not in self.data or not self.data['interest_rates']:
            print("Interest rates not loaded. Fetching now...")
            self.fetch_interest_rates()

        rates = self.data.get ('interest_rates',{})
        if not rates:
            print("Error: Cannot calculate interest rate differentials, no rates data available.")
            return {}

        differentials: Dict[str,float]={}
        # Boucle sur self.config.TICKERS (qui peut contenir 'EURUSD=X')
        for pair_maybe_yahoo in self.config.TICKERS:
            try:
                # Convertir en format standard ('EUR/USD') AVANT de splitter
                pair=ForexDataUtils.convert_from_yahoo_forex_symbol (pair_maybe_yahoo)
                if '/' not in pair: continue  # Skip si la conversion échoue

                base,quote=pair.split ('/')
                if base in rates and quote in rates:
                    # Stocker avec la clé au format standard
                    differentials[pair]=rates[base] - rates[quote]
                else:
                     # Optionally print warning for missing rates
                     # print(f"Warning: Missing rate for {base} or {quote} to calculate differential for {pair}")
                     pass
            except ValueError:
                pass  # Ignorer les erreurs de split si la conversion a échoué

        self.data['interest_differentials']=differentials
        print (
            f"Calculated interest rate differentials for {len (differentials)} pairs.")  # Devrait maintenant être > 0
        return differentials

    def calculate_inflation_differentials(self) -> Dict[str, float]:
        """
        Calculates inflation differentials for configured Forex pairs.

        Returns:
            Dictionary mapping Forex pair string to its inflation differential (Base Country Rate - Quote Country Rate).
        """
        if 'inflation' not in self.data or not self.data['inflation']:
            print("Inflation data not loaded. Fetching now...")
            self.fetch_inflation_data()

        inflation=self.data.get ('inflation',{})
        if not inflation:
            # ... (gestion d'erreur identique) ...
            return {}

        differentials: Dict[str,float]={}
        # Boucle sur self.config.TICKERS (qui peut contenir 'EURUSD=X')
        for pair_maybe_yahoo in self.config.TICKERS:
            try:
                # Convertir en format standard ('EUR/USD') AVANT de splitter
                pair=ForexDataUtils.convert_from_yahoo_forex_symbol (pair_maybe_yahoo)
                if '/' not in pair: continue  # Skip si la conversion échoue

                base_curr,quote_curr=pair.split ('/')
                base_country=self.config.CURRENCY_TO_COUNTRY.get (base_curr)
                quote_country=self.config.CURRENCY_TO_COUNTRY.get (quote_curr)

                if base_country and quote_country and base_country in inflation and quote_country in inflation:
                    # Stocker avec la clé au format standard
                    differentials[pair]=inflation[base_country] - inflation[quote_country]
                else:
                    # Optionally print warning for missing inflation data
                    # missing = []
                    # if not base_country or base_country not in inflation: missing.append(f"{base_curr}({base_country})")
                    # if not quote_country or quote_country not in inflation: missing.append(f"{quote_curr}({quote_country})")
                    # print(f"Warning: Missing inflation data for {', '.join(missing)} to calculate differential for {pair}")
                    pass

            except ValueError:
                 # Optionally print warning for invalid pair format
                 # print(f"Warning: Invalid pair format '{pair}' for differential calculation.")
                 pass

        self.data['inflation_differentials'] = differentials
        print(f"Calculated inflation differentials for {len(differentials)} pairs.")
        return differentials

    def fetch_commodity_data(self, commodities: List[str] = ['GOLD', 'OIL', 'NATGAS', 'COPPER']) -> Dict[str, float]:
        """
        Fetches latest prices for major commodities using yfinance.
        Uses file cache.

        Args:
            commodities: List of commodity names (e.g., ['GOLD', 'OIL']).

        Returns:
            Dictionary mapping commodity name to its latest price.
        """
        print(f"Fetching data for {len(commodities)} commodities...")
        data_key = 'commodities'
        cache_file = self.config.COMMODITY_CACHE_FILE
        cache_expiry = self.config.COMMODITY_CACHE_EXPIRY_HOURS * 3600 # seconds

        # Try reading from cache
        cached_data = self._read_cache(cache_file, cache_expiry)
        if cached_data and isinstance(cached_data, dict):
            self.data[data_key] = cached_data
            print(f"Commodity data loaded from cache ({len(self.data[data_key])} items).")
            return self.data[data_key]
        elif cached_data:
             print("Warning: Old commodity cache format detected or invalid cache. Fetching fresh data.")

        # --- Fetch from yfinance ---
        commodity_symbols: Dict[str, str] = {
            'GOLD': 'GC=F', 'SILVER': 'SI=F', 'PLATINUM': 'PL=F',
            'OIL': 'CL=F', 'BRENT': 'BZ=F', 'NATGAS': 'NG=F',
            'COPPER': 'HG=F', 'ALUMINUM': 'ALI=F',
            'WHEAT': 'ZW=F', 'CORN': 'ZC=F', 'SOYBEAN': 'ZS=F'
        }
        symbols_to_fetch = [commodity_symbols[com] for com in commodities if com in commodity_symbols]
        commodity_data: Dict[str, float] = {}

        if not symbols_to_fetch:
            print("No valid commodity symbols selected.")
            return {}

        try:
            # Fetch data for the last 5 days to ensure getting the latest close
            data = yf.download(symbols_to_fetch, period='5d', progress=False)

            if data.empty:
                 print("Warning: yfinance returned no data for selected commodities.")
            else:
                # Handle both MultiIndex and single index DataFrames
                price_col_type = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
                close_prices = data[price_col_type] if isinstance(data.columns, pd.MultiIndex) else data['Close'] # Use 'Close' for single

                if isinstance(close_prices, pd.DataFrame):
                    # Get the last available closing price for each symbol
                    latest_prices = close_prices.iloc[-1]
                    for com, symbol in commodity_symbols.items():
                        if symbol in latest_prices.index and not pd.isna(latest_prices[symbol]):
                            commodity_data[com] = float(latest_prices[symbol])
                            print(f"  • {com} ({symbol}): {commodity_data[com]:.2f}")
                elif isinstance(close_prices, pd.Series): # If only one symbol was fetched
                     if not pd.isna(close_prices.iloc[-1]):
                         # Find which commodity this symbol belongs to
                         fetched_symbol = close_prices.name # yfinance usually names the series
                         if fetched_symbol in symbols_to_fetch:
                             for com, sym in commodity_symbols.items():
                                 if sym == fetched_symbol:
                                     commodity_data[com] = float(close_prices.iloc[-1])
                                     print(f"  • {com} ({sym}): {commodity_data[com]:.2f}")
                                     break

        except Exception as e:
            print(f"Error fetching commodity data from yfinance: {e}")


        # --- Fallback Data ---
        if not commodity_data:
             print("Warning: Using placeholder fallback commodity data.")
             fallback_data = {'GOLD': 2350.0, 'OIL': 80.0, 'NATGAS': 2.5, 'COPPER': 4.5}
             commodity_data = {k: v for k, v in fallback_data.items() if k in commodities}

        # --- Update Cache and State ---
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write_cache(cache_file, commodity_data) # Store only the final dictionary
        self.data[data_key] = commodity_data
        self.last_updated[data_key] = current_time_str
        print(f"Finished fetching/loading commodity data for {len(commodity_data)} items.")

        return commodity_data


    def analyze_monetary_policy_bias(self) -> Dict[str, str]:
        """
        Analyzes the monetary policy bias (hawkish/dovish/neutral) of central banks.
        NOTE: This is a simplified version. A real implementation would use NLP on
              central bank statements, speeches, and recent rate decisions.

        Returns:
            Dictionary mapping currency code to perceived policy bias.
        """
        print("Analyzing monetary policy bias (simplified version)...")
        data_key = 'monetary_bias'

        # Placeholder biases - Update based on current macroeconomic climate
        monetary_bias: Dict[str, str] = {
            'USD': 'neutral', # Fed likely on hold after hikes
            'EUR': 'dovish',  # ECB leaning towards cuts
            'GBP': 'neutral', # BoE waiting for more data
            'JPY': 'hawkish', # BoJ slowly normalizing policy
            'CHF': 'neutral', # SNB potentially pausing/cutting
            'AUD': 'neutral', # RBA holding, slight dovish tilt
            'CAD': 'dovish',  # BoC started cutting cycle
            'NZD': 'neutral', # RBNZ holding, data dependent
            'MXN': 'dovish',  # Banxico started cutting cycle
            'PLN': 'dovish'   # NBP likely on hold or easing
        }
        # Filter based on configured currencies
        filtered_bias = {c: bias for c, bias in monetary_bias.items() if c in self.config.CURRENCIES}

        self.data[data_key] = filtered_bias
        self.last_updated[data_key] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print(f"Monetary policy biases determined for {len(filtered_bias)} currencies.") # Less verbose
        return filtered_bias


# --- Core MHGNA Forex Logic ---

class MHGNAForex:
    """
    MHGNA model version tailored for Forex, integrating macro data and FX indicators.
    """

    def __init__(self, config: ForexConfig = ForexConfig()):
        """
        Initializes the MHGNA Forex system.

        Args:
            config: Configuration object (ForexConfig instance).
        """
        self.config = config
        self.utils = ForexDataUtils()
        self.macro = MacroDataCollector(self.config)

        self.forex_pairs: List[str] = self.config.TICKERS
        self.data: Optional[pd.DataFrame] = None # Close price data
        # IMPORTANT: Add attribute to store full OHLC data if ATR needed
        self.ohlc_data: Optional[Dict[str, pd.DataFrame]] = None
        self.returns: Optional[pd.DataFrame] = None
        self.volatility: Optional[pd.DataFrame] = None
        self.momentum: Optional[pd.DataFrame] = None
        self.network_graph: Optional[nx.Graph] = None
        self.last_recommendations: Optional[pd.DataFrame] = None # Store formatted recommendations
        self.alerts: List[Dict[str, Any]] = []
        self.drawdown_history: Dict[str, pd.Series] = {}
        self.session_states: Dict[str, Dict[str, Any]] = {s: {} for s in self.config.SESSIONS}

        self.current_date: Optional[datetime.datetime] = None

        self._setup_plotting_style()

    def _setup_plotting_style(self):
        """Sets the plotting style using seaborn."""
        try:
            sns.set_style(self.config.CHART_STYLE)
            plt.rcParams['figure.figsize'] = self.config.FIGSIZE
        except ValueError:
            print(f"Warning: Invalid chart style '{self.config.CHART_STYLE}'. Using default.")
            sns.set_style('darkgrid') # Fallback
            plt.rcParams['figure.figsize'] = self.config.FIGSIZE


    def fetch_forex_data(self, end_date: Optional[str] = None) -> bool:
        """
        Fetches historical Forex OHLC price data using yfinance and calculates derived metrics.

        Args:
            end_date: The end date for the data retrieval (YYYY-MM-DD format).
                      Defaults to the current date.

        Returns:
            True if data fetching and initial processing was successful, False otherwise.
        """
        # --- Date Handling ---
        # CORRECT INDENTATION STARTS HERE
        try:
            if end_date:
                # Convert string to pandas Timestamp
                end_dt_ts=pd.to_datetime (end_date)
            else:
                # Default to today's date (midnight, local time) if no date specified
                end_dt_ts=pd.Timestamp.now ().normalize ()  # Gets today's date at 00:00:00

            # Ensure the timestamp is timezone-naive for consistent calculations later
            if end_dt_ts.tz is not None:
                # If it has a timezone, remove it
                end_dt=end_dt_ts.tz_localize (None)
                # print(f"Note: Provided end_date had timezone info, converted to naive datetime: {end_dt}") # Optional print
            else:
                # Already naive
                end_dt=end_dt_ts

            # Check if the resulting date is valid
            if pd.isna (end_dt):
                raise ValueError ("Parsed date resulted in NaT (Not a Time).")

        except ValueError as e:  # This except MUST be aligned with the try above
            print (f"⚠️ Error parsing end_date '{end_date}': {e}. Defaulting to today's date (naive).")
            # Fallback to today's date (midnight, local time), ensure it's Timestamp
            end_dt=pd.Timestamp.now ().normalize ()
        # CORRECT INDENTATION FOR THE ABOVE BLOCK ENDS HERE

        # The rest of the function body starts correctly indented from here
        self.current_date=end_dt  # Store the final naive Timestamp
        start_dt=end_dt - relativedelta (years=self.config.LOOKBACK_PERIOD_YEARS)

        print (
            f"Fetching Forex data from {start_dt.strftime ('%Y-%m-%d')} to {end_dt.strftime ('%Y-%m-%d')}...")

        # Convertir les paires Forex au format Yahoo Finance
        yahoo_symbols_map: Dict[str,str]={
            ticker:self.utils.convert_to_yahoo_forex_symbol (ticker)
            for ticker in self.forex_pairs
        }
        yahoo_symbols_list=list (yahoo_symbols_map.values ())

        # --- Download data ---
        # This inner try...except block structure seems correct in your snippet
        try:
            # Fetch OHLCV data
            # Pass naive start/end dates to yfinance
            data_raw=yf.download (
                tickers=yahoo_symbols_list,
                start=start_dt,
                # yfinance usually includes start but excludes end, add 1 day to include end_dt
                end=end_dt + datetime.timedelta (days=1),
                progress=True,
                group_by='ticker'
            )
            # ... (le reste du traitement des données téléchargées) ...
            # ... (le calcul des retours, volatilité, momentum, etc.) ...
            # ... (le bloc except Exception as e: final de la fonction) ...

        except Exception as e:
            print(f"Error during yfinance download: {e}")
            return False


        if data_raw.empty:
            print("❌ No data returned from yfinance.")
            return False

        # --- Process downloaded data ---
        close_prices = pd.DataFrame()
        ohlc_data_dict = {} # Store individual OHLC dataframes
        successful_pairs = []

        for ticker, yahoo_symbol in yahoo_symbols_map.items():
            try:
                # yfinance with group_by='ticker' gives a dict-like object or MultiIndex cols
                if isinstance(data_raw.columns, pd.MultiIndex) and yahoo_symbol in data_raw.columns.levels[0]:
                     # Extract DataFrame for this specific ticker
                     pair_data_full = data_raw[yahoo_symbol].copy()
                elif not isinstance(data_raw.columns, pd.MultiIndex) and len(yahoo_symbols_list) == 1:
                     # Handle case where only one ticker was downloaded (no MultiIndex)
                     pair_data_full = data_raw.copy()
                elif isinstance(data_raw, dict) and yahoo_symbol in data_raw:
                     # Handle case where yfinance returns a dict (less common now?)
                     pair_data_full = data_raw[yahoo_symbol].copy()
                else:
                    print(f"  ✗ No data found for {ticker} ({yahoo_symbol}) in downloaded structure.")
                    continue

                # Standardize column names (Open, High, Low, Close, Adj Close, Volume) -> lowercase
                pair_data_full.columns = pair_data_full.columns.str.lower()
                if 'adj close' in pair_data_full.columns:
                     pair_data_full = pair_data_full.rename(columns={'adj close': 'adj_close'})

                # Select the price column ('adj_close' preferred, fallback to 'close')
                price_col = 'adj_close' if 'adj_close' in pair_data_full.columns else 'close'

                if price_col not in pair_data_full.columns:
                     print(f"  ✗ No 'adj_close' or 'close' column for {ticker}.")
                     continue

                # Drop rows with NaN price for this pair
                pair_prices = pair_data_full[price_col].dropna()
                pair_data_full = pair_data_full.loc[pair_prices.index] # Keep OHLC rows corresponding to valid prices

                if not pair_prices.empty:
                    close_prices[ticker] = pair_prices
                    # Store OHLC data if columns exist (needed for ATR)
                    if all(c in pair_data_full.columns for c in ['open', 'high', 'low', 'close']):
                         ohlc_data_dict[ticker] = pair_data_full[['open', 'high', 'low', 'close']].copy()
                    else:
                         print(f"  Warning: Missing OHLC columns for {ticker}, ATR calculation might fail.")
                         ohlc_data_dict[ticker] = None # Indicate missing OHLC

                    print(f"  ✓ Data processed for {ticker} ({len(pair_prices)} points)")
                    successful_pairs.append(ticker)
                else:
                    print(f"  ! No non-NaN price data for {ticker} after processing.")


            except KeyError:
                print(f"  ✗ Data column not found for {ticker} ({yahoo_symbol}). Skipping.")
            except Exception as e:
                print(f"  ✗ Error processing data for {ticker} ({yahoo_symbol}): {e}")
                import traceback
                traceback.print_exc()


        if close_prices.empty:
            print("❌ No valid price data could be processed for any pair.")
            self.data = None
            self.ohlc_data = None
            return False

        # Align all dataframes to common index (inner join)
        common_index = close_prices.dropna(axis=0, how='all').index
        self.data = close_prices.loc[common_index].copy()
        self.ohlc_data = {ticker: df.loc[common_index].copy() for ticker, df in ohlc_data_dict.items() if df is not None and ticker in self.data.columns}

        self.forex_pairs = successful_pairs # Update list to only include successfully processed pairs
        print(f"\nSuccessfully processed data for {len(successful_pairs)} pairs.")
        print(f"Final close price data shape: {self.data.shape}")

        # --- Calculate Returns, Volatility, Momentum ---
        # This block needs to be inside a TRY except to catch processing errors
        try:
            if self.data is not None and not self.data.empty:
                # Ensure data is sorted by date
                self.data = self.data.sort_index()

                self.returns = self.data.pct_change().dropna(axis=0, how='all')

                if self.returns.empty:
                     print("Error: Returns calculation resulted in empty DataFrame.")
                     return False # Or handle differently

                # Calculate rolling volatility (e.g., 30-day annualized)
                self.volatility = self.returns.rolling(window=30, min_periods=15).std().dropna(axis=0, how='all') * np.sqrt(252)

                # Calculate momentum scores over different periods
                min_periods_factor = 0.8 # Require 80% of window for calculation
                min_periods_short = int(self.config.HORIZONS['short'] * min_periods_factor)
                min_periods_medium = int(self.config.HORIZONS['medium'] * min_periods_factor)
                min_periods_long = int(self.config.HORIZONS['long'] * min_periods_factor)

                momentum_short = self.returns.rolling(window=self.config.HORIZONS['short'], min_periods=min_periods_short).sum()
                momentum_medium = self.returns.rolling(window=self.config.HORIZONS['medium'], min_periods=min_periods_medium).sum()
                momentum_long = self.returns.rolling(window=self.config.HORIZONS['long'], min_periods=min_periods_long).sum()

                # Calculate composite momentum score (example weighting)
                # Apply z-score normalization *within* each timeframe before combining
                # Use fillna(0) after zscore in case a whole column is constant (std dev = 0)
                norm_momentum_short = momentum_short.apply(lambda x: zscore(x.dropna(), nan_policy='omit') if x.dropna().std() > 1e-9 else 0).fillna(0)
                norm_momentum_medium = momentum_medium.apply(lambda x: zscore(x.dropna(), nan_policy='omit') if x.dropna().std() > 1e-9 else 0).fillna(0)
                norm_momentum_long = momentum_long.apply(lambda x: zscore(x.dropna(), nan_policy='omit') if x.dropna().std() > 1e-9 else 0).fillna(0)

                self.momentum = (
                    norm_momentum_short * 0.50 +  # Higher weight for short-term
                    norm_momentum_medium * 0.30 +
                    norm_momentum_long * 0.20
                ).reindex(self.returns.index).fillna(method='ffill').dropna(axis=0, how='all') # Reindex & ffill before dropna

                # --- Initialize drawdown history ---
                print("Calculating initial drawdown history...")
                self.drawdown_history = {} # Ensure it's reset
                for ticker in self.forex_pairs:
                    if ticker in self.data.columns:
                        prices = self.data[ticker].dropna()
                        if not prices.empty:
                            peaks = prices.cummax()
                            # Avoid division by zero if peaks contains zeros
                            peaks[peaks == 0] = np.nan
                            self.drawdown_history[ticker] = (prices / peaks - 1.0).fillna(0) # Fill NaN resulting from zero peak
                        else:
                            self.drawdown_history[ticker] = pd.Series(dtype=np.float64)

                # --- Calculate initial session states (simulated) ---
                self._calculate_session_states()

                print("Calculated returns, volatility, momentum, drawdowns, and session states.")
                return True # Initial processing successful

            else: # self.data is None or empty after processing
                print("Error: Price data is missing or empty after processing.")
                return False

        # This except block catches errors during the returns/vol/momentum/etc calculation
        except Exception as e:
            print(f"⚠️ Error during data processing (returns, vol, momentum calc): {e}")
            import traceback
            traceback.print_exc()
            # Decide if partial data is acceptable or should return False
            # Setting returns etc to None might be safer
            self.returns = None
            self.volatility = None
            self.momentum = None
            return False # Indicate processing failed

    def _calculate_session_states(self):
        """
        Calculates SIMULATED indicators specific to each trading session.
        NOTE: This is a simplification using daily data. A real implementation
              would require intraday data and timestamp filtering.
        """
        if self.data is None or self.returns is None:
            # print("Warning: Cannot calculate session states, data not available.") # Less verbose
            return

        # print("Calculating simulated session states...") # Less verbose

        # Use recent returns for calculation (e.g., last 60 days)
        lookback_days = 60
        if len(self.returns) < lookback_days:
            # print(f"Warning: Less than {lookback_days} days of returns data for session state calculation.")
            recent_returns = self.returns
        else:
            recent_returns = self.returns.iloc[-lookback_days:]

        # Reset session states
        self.session_states = {s: {} for s in self.config.SESSIONS}

        for pair in self.forex_pairs:
            if pair not in recent_returns.columns:
                continue

            pair_returns = recent_returns[pair].dropna()

            # Need sufficient data points for std calculation
            if len(pair_returns) < min(15, len(recent_returns) // 2): # Reduced min period
                continue

            # --- Determine currencies and assign session weights ---
            base_curr, quote_curr = None, None
            if '/' in pair:
                base_curr, quote_curr = pair.split('/')
            elif pair.endswith('=X') and len(pair) == 8: # Yahoo format like EURUSD=X
                base_curr, quote_curr = pair[:3], pair[3:6]

            sessions_weights: Dict[str, float] = {}
            if base_curr and quote_curr:
                # Simplified weighting based on currency geography
                is_asia_base = base_curr in ['JPY', 'AUD', 'NZD', 'CNY', 'HKD', 'SGD']
                is_asia_quote = quote_curr in ['JPY', 'AUD', 'NZD', 'CNY', 'HKD', 'SGD']
                is_eur_base = base_curr in ['EUR', 'GBP', 'CHF', 'SEK', 'NOK', 'PLN']
                is_eur_quote = quote_curr in ['EUR', 'GBP', 'CHF', 'SEK', 'NOK', 'PLN']
                is_usd_base = base_curr == 'USD'
                is_usd_quote = quote_curr == 'USD'
                is_cad_base = base_curr == 'CAD'
                is_cad_quote = quote_curr == 'CAD'
                # Add other relevant currencies (MXN etc.) if needed

                # Assign weights (subjective, adjust as needed)
                # Priority: Asia > Europe > America based on presence
                if is_asia_base or is_asia_quote:
                    sessions_weights = {'asia': 0.6, 'europe': 0.2, 'america': 0.2}
                elif is_eur_base or is_eur_quote:
                    sessions_weights = {'asia': 0.1, 'europe': 0.7, 'america': 0.2}
                elif is_usd_base or is_usd_quote or is_cad_base or is_cad_quote: # Primarily America session
                     sessions_weights = {'asia': 0.1, 'europe': 0.3, 'america': 0.6}
                else: # Default/Fallback if currencies are not in main groups
                     sessions_weights = {'asia': 0.33, 'europe': 0.34, 'america': 0.33}
            else:
                 # print(f"Warning: Could not determine base/quote for '{pair}'. Using default session weights.") # Less verbose
                 sessions_weights = {'asia': 0.33, 'europe': 0.34, 'america': 0.33} # Default fallback

            # --- Simulate data for each session ---
            pair_std_dev = pair_returns.std()
            if pd.isna(pair_std_dev) or pair_std_dev < 1e-9 : continue # Skip if std dev is invalid

            for session, weight in sessions_weights.items():
                # Simulate session volatility based on overall vol and weight
                # Use a non-linear factor maybe? Example: 0.7 base + 0.5 * weight
                sim_session_volatility = pair_std_dev * (0.7 + 0.5 * weight)

                # Simulate session volume (purely fictional scale)
                sim_session_volume = 1_000_000 * weight

                # Store session info
                if pair not in self.session_states[session]:
                    self.session_states[session][pair] = {}

                self.session_states[session][pair]['volatility'] = sim_session_volatility
                self.session_states[session][pair]['volume'] = sim_session_volume
                self.session_states[session][pair]['weight'] = weight # Store the weight itself

    def diagnose_forex_data(self) -> Dict[str, Any]:
        """
        Performs an in-depth diagnosis of the available Forex data.
        Helps identify potential issues before network construction.

        Returns:
            Dictionary containing diagnostic information.
        """
        diagnostics: Dict[str, Any] = {}
        print("\n" + "=" * 50)
        print(" FOREX DATA DIAGNOSTICS")
        print("=" * 50)

        if self.data is None or self.data.empty:
            print("⚠️ No price data available. Run fetch_forex_data() first.")
            return diagnostics

        # 1. General Information
        print("\n1. General Information:")
        start_date = self.data.index.min()
        end_date = self.data.index.max()
        num_days = len(self.data)
        num_pairs = self.data.shape[1]
        print(f"   • Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   • Number of Days: {num_days}")
        print(f"   • Number of Pairs: {num_pairs}")
        diagnostics['period'] = (start_date.isoformat(), end_date.isoformat())
        diagnostics['days'] = num_days
        diagnostics['pairs_count'] = num_pairs
        diagnostics['pairs_list'] = self.data.columns.tolist()

        # 2. Missing Value Analysis (Prices)
        print("\n2. Missing Value Analysis (Prices):")
        missing_prices = self.data.isna().sum()
        missing_pct = (missing_prices / num_days) * 100
        diagnostics['missing_prices'] = {}
        for pair, count in missing_prices.items():
            pct = missing_pct[pair]
            status = "✅ OK" if pct < 5 else "⚠️ Warning" if pct < 20 else "❌ Problem"
            print(f"   • {pair}: {int(count)} missing ({pct:.1f}%) - {status}")
            diagnostics['missing_prices'][pair] = {'count': int(count), 'percentage': float(pct)}

        # 3. Returns Analysis
        print("\n3. Returns Analysis:")
        if self.returns is not None and not self.returns.empty:
            missing_returns = self.returns.isna().sum()
            # Use len(self.returns) which might be 1 less than len(self.data)
            missing_returns_pct = (missing_returns / len(self.returns)) * 100 if len(self.returns) > 0 else pd.Series(0.0, index=self.returns.columns)
            diagnostics['missing_returns'] = {}
            print("   Missing Returns:")
            for pair, count in missing_returns.items():
                 pct = missing_returns_pct.get(pair, 0) # Use .get for safety
                 status = "✅ OK" if pct < 5 else "⚠️ Warning" if pct < 20 else "❌ Problem"
                 print(f"     • {pair}: {int(count)} missing ({pct:.1f}%) - {status}")
                 diagnostics['missing_returns'][pair] = {'count': int(count), 'percentage': float(pct)}

            print("\n   Returns Statistics (Daily %):")
            diagnostics['returns_stats'] = {}
            for pair in self.returns.columns:
                returns_series = self.returns[pair].dropna()
                if len(returns_series) > 1: # Need at least 2 points for std dev
                    mean_pct = returns_series.mean() * 100
                    std_pct = returns_series.std() * 100
                    min_pct = returns_series.min() * 100
                    max_pct = returns_series.max() * 100
                    # Subjective volatility assessment
                    vol_status = "✅ Normal" if std_pct < 1.5 else "⚠️ High" if std_pct < 3 else "❌ Very High"
                    print(f"     • {pair}: Mean={mean_pct:.3f}%, StdDev={std_pct:.3f}% ({vol_status}), Min={min_pct:.2f}%, Max={max_pct:.2f}%")
                    diagnostics['returns_stats'][pair] = {
                        'mean_pct': float(mean_pct), 'std_dev_pct': float(std_pct),
                        'min_pct': float(min_pct), 'max_pct': float(max_pct)
                    }
                else:
                    print(f"     • {pair}: Not enough data points for stats ({len(returns_series)} points).")
                    diagnostics['returns_stats'][pair] = {}
        else:
            print("   ⚠️ No returns data calculated or available.")

        # 4. Data Continuity Check
        print("\n4. Data Continuity Check:")
        # Ensure index is datetime
        if pd.api.types.is_datetime64_any_dtype(self.data.index):
             date_diffs = self.data.index.to_series().diff().dropna()
             # Consider business days? For now, check calendar days > 1
             # Allow for weekend gaps (2 or 3 days)
             gaps = date_diffs[date_diffs > pd.Timedelta(days=3)]
             diagnostics['data_gaps'] = {'count': len(gaps), 'details': []}
             if not gaps.empty:
                 print(f"   ⚠️ {len(gaps)} potential gap(s) > 3 days detected:")
                 for i, (date, gap) in enumerate(gaps.items()):
                     prev_date = date - gap
                     gap_info = f"Gap of {gap.days} days between {prev_date.strftime('%Y-%m-%d')} and {date.strftime('%Y-%m-%d')}"
                     if i < 5: # Show details for the first few
                         print(f"     • {gap_info}")
                     diagnostics['data_gaps']['details'].append({
                         'start_date': prev_date.isoformat(),
                         'end_date': date.isoformat(),
                         'gap_days': gap.days
                     })
                 if len(gaps) > 5:
                     print(f"     • ... and {len(gaps) - 5} other gaps.")
             else:
                 print("   ✅ No significant gaps (> 3 days) detected in data index.")
        else:
            print("   ⚠️ Cannot perform continuity check: Index is not datetime.")
            diagnostics['data_gaps'] = {'count': -1, 'details': []} # Indicate check failed

        # 5. Recommendations
        print("\n5. Recommendations:")
        recommendations = []
        if num_pairs < 3:
            rec = "❌ CRITICAL: Less than 3 pairs available. Network analysis requires at least 3."
            print(f"   {rec}")
            recommendations.append(rec)
        high_missing_price = [p for p, d in diagnostics.get('missing_prices', {}).items() if d['percentage'] > 20]
        if high_missing_price:
            rec = f"⚠️ Consider excluding pairs with >20% missing price data: {', '.join(high_missing_price)}"
            print(f"   {rec}")
            recommendations.append(rec)
        high_missing_ret = [p for p, d in diagnostics.get('missing_returns', {}).items() if d['percentage'] > 20]
        if high_missing_ret:
             rec = f"⚠️ Check pairs with >20% missing returns data: {', '.join(high_missing_ret)}"
             print(f"   {rec}")
             recommendations.append(rec)

        if num_days < 60: # Approx 3 months
            rec = f"⚠️ Short history ({num_days} days). A minimum of 120-180 days is often recommended for robust analysis."
            print(f"   {rec}")
            recommendations.append(rec)
        if diagnostics.get('data_gaps', {}).get('count', 0) > 10:
            rec = f"⚠️ Numerous data gaps ({diagnostics['data_gaps']['count']}) detected. Verify data source or consider imputation/interpolation if appropriate."
            print(f"   {rec}")
            recommendations.append(rec)

        if not recommendations:
             print("   ✅ Data seems generally suitable for analysis based on basic checks.")

        diagnostics['recommendations'] = recommendations
        print("=" * 50)
        return diagnostics

    # --- build_forex_network ---
    def build_forex_network(self) -> nx.Graph:
        """
        Builds the multi-horizon dependency network between Forex pairs
        using GraphicalLassoCV with robust NaN handling via imputation.

        Returns:
            A networkx Graph representing the combined multi-horizon network.
            Returns an empty graph with nodes if analysis cannot be performed.
        """
        print("\nBuilding multi-horizon Forex network...")

        # Ensure returns data is available and valid
        if self.returns is None or self.returns.empty:
            print("❌ Cannot build network: Returns data is missing or empty.")
            G_empty = nx.Graph()
            if self.data is not None: G_empty.add_nodes_from(self.data.columns) # Add nodes even if empty
            self.network_graph = G_empty
            return G_empty

        if self.returns.shape[1] < 3:
            print(f"⚠️ Warning: Only {self.returns.shape[1]} pairs available. Need at least 3 for network analysis.")
            # Create graph with available nodes but no edges
            G = nx.Graph()
            G.add_nodes_from(self.returns.columns)
            self.network_graph = G
            return G

        if self.current_date is None:
            self.current_date = self.returns.index.max() # Use last date from returns
            print(f"Warning: current_date not set, using last date from returns: {self.current_date.strftime('%Y-%m-%d')}")


        horizon_graphs: Dict[str, nx.Graph] = {}
        horizon_matrices: Dict[str, Dict[str, Any]] = {}
        alpha_params: Dict[str, float] = {'short': 0.015, 'medium': 0.01, 'long': 0.005}

        for horizon_name, days in self.config.HORIZONS.items():
            print(f"\n--- Processing Horizon: {horizon_name} ({days} days) ---")
            try:
                # --- 1. Select Data for Horizon ---
                # Ensure indices are compatible (e.g., both timezone-naive)
                current_date_naive = self.current_date.tz_localize(None) if self.current_date.tz is not None else self.current_date
                lookback_date = current_date_naive - relativedelta(days=days)
                returns_index_naive = self.returns.index.tz_localize(None) if self.returns.index.tz is not None else self.returns.index

                horizon_returns = self.returns[returns_index_naive >= lookback_date].copy()

                if len(horizon_returns) < days // 2:
                    print(f"  Skipping: Not enough data points ({len(horizon_returns)} found, {days // 2} required).")
                    continue

                # --- 2. Handle NaNs Robustly ---
                nan_percentage = horizon_returns.isna().mean()
                valid_columns = nan_percentage[nan_percentage < 0.3].index # Keep columns with < 30% NaN
                num_initial_cols = horizon_returns.shape[1]
                num_valid_cols = len(valid_columns)

                if num_valid_cols < 3:
                    print(f"  Skipping: Only {num_valid_cols} pairs have <30% NaN (out of {num_initial_cols}). Minimum 3 required.")
                    continue

                # print(f"  Using {num_valid_cols} pairs with <30% NaN.") # Less verbose
                clean_returns = horizon_returns[valid_columns].copy()

                # Impute remaining NaNs (column-wise mean)
                imputed_cols_count = 0
                for col in clean_returns.columns:
                    if clean_returns[col].isna().any():
                        # nan_count = clean_returns[col].isna().sum() # Less verbose
                        col_mean = clean_returns[col].mean()
                        if pd.isna(col_mean): # Handle case where entire column might be NaN initially
                            col_mean = 0 # Impute with 0 if mean is NaN
                        clean_returns[col] = clean_returns[col].fillna(col_mean)
                        imputed_cols_count += 1
                if imputed_cols_count > 0:
                     print(f"  Imputed NaNs in {imputed_cols_count} column(s).")


                # Drop rows if any NaNs remain after imputation (should be rare)
                if clean_returns.isna().any().any():
                    rows_before = len(clean_returns)
                    clean_returns = clean_returns.dropna(axis=0)
                    rows_after = len(clean_returns)
                    if rows_before > rows_after:
                         print(f"  Warning: Dropped {rows_before - rows_after} rows with persistent NaNs.")

                if clean_returns.shape[0] < 20 or clean_returns.shape[1] < 3:
                    print(f"  Skipping: Insufficient data after cleaning ({clean_returns.shape[0]} rows, {clean_returns.shape[1]} cols).")
                    continue

                # --- 3. Clip Outliers ---
                # Clip column-wise based on each column's quantiles
                qt_01 = clean_returns.quantile(0.01, axis=0)
                qt_99 = clean_returns.quantile(0.99, axis=0)
                clean_returns = clean_returns.clip(lower=qt_01, upper=qt_99, axis=1)


                # --- 4. Fit Graphical Lasso ---
                print(f"  Fitting GraphicalLassoCV ({clean_returns.shape[0]} days, {clean_returns.shape[1]} pairs)...")
                alpha = alpha_params[horizon_name]
                # Check for near-constant columns which cause issues
                if (clean_returns.std() < 1e-9).any():
                     constant_cols = clean_returns.columns[clean_returns.std() < 1e-9].tolist()
                     print(f"  Warning: Constant/Near-constant columns found and removed for horizon '{horizon_name}': {constant_cols}")
                     clean_returns = clean_returns.drop(columns=constant_cols)
                     if clean_returns.shape[1] < 3:
                          print(f"  Skipping: Less than 3 non-constant columns remain for horizon '{horizon_name}'.")
                          continue


                model = GraphicalLassoCV(alphas=[alpha * 0.5, alpha, alpha * 2.0], cv=5, max_iter=1000, n_jobs=-1, assume_centered=True) # Use multiple cores
                model.fit(clean_returns.to_numpy()) # Pass numpy array
                precision_matrix = model.precision_
                selected_alpha = model.alpha_

                # --- 5. Build Horizon Graph ---
                G = nx.Graph()
                nodes = clean_returns.columns.tolist() # Get column names back
                G.add_nodes_from(nodes)
                edge_count = 0
                # Use a threshold slightly above zero to avoid numerical noise
                edge_threshold = 1e-4
                for i, ticker1 in enumerate(nodes):
                    for j, ticker2 in enumerate(nodes):
                        if i < j:
                            # Use absolute value of off-diagonal precision matrix elements
                            weight = abs(precision_matrix[i, j])
                            # Add edge if weight is above threshold
                            if weight > edge_threshold:
                                G.add_edge(ticker1, ticker2, weight=weight, horizon=horizon_name)
                                edge_count += 1

                horizon_graphs[horizon_name] = G
                horizon_matrices[horizon_name] = {'matrix': precision_matrix, 'columns': nodes, 'alpha': selected_alpha}
                print(f"  Network for '{horizon_name}': {G.number_of_nodes()} nodes, {edge_count} edges (alpha={selected_alpha:.4f}).")

            except np.linalg.LinAlgError as lae:
                 print(f"❌ Linear Algebra Error processing horizon '{horizon_name}': {lae}. Skipping.")
                 print("   This might be due to highly collinear data even after cleaning.")
            except Exception as e:
                print(f"❌ Error processing horizon '{horizon_name}': {e}")
                import traceback
                traceback.print_exc()

        # --- Combine Horizon Graphs ---
        if not horizon_graphs:
            print("❌ No horizon networks could be built. Returning empty graph.")
            G_empty = nx.Graph()
            if self.returns is not None: # Add nodes even if no edges
                 G_empty.add_nodes_from(self.returns.columns)
            self.network_graph = G_empty
            return G_empty

        print("\nCombining horizon graphs...")
        combined_graph = nx.Graph()

        # Add all nodes that appeared in any horizon + original nodes
        all_nodes_combined = set(self.forex_pairs) # Start with the target pairs
        for G_h in horizon_graphs.values():
            all_nodes_combined.update(G_h.nodes())
        combined_graph.add_nodes_from(list(all_nodes_combined))

        horizon_weights = {'short': 0.25, 'medium': 0.5, 'long': 0.25}
        combined_edges: Dict[Tuple[str, str], float] = {}

        for horizon_name, graph in horizon_graphs.items():
            horizon_w = horizon_weights.get(horizon_name, 1.0 / len(horizon_graphs)) # Default weight if needed
            for u, v, data in graph.edges(data=True):
                edge = tuple(sorted((u, v))) # Ensure consistent edge representation
                current_weight = combined_edges.get(edge, 0)
                combined_edges[edge] = current_weight + data['weight'] * horizon_w

        # Add weighted edges to the combined graph
        final_edge_count = 0
        final_edge_threshold = 1e-3 # Threshold for combined weight
        for (u, v), weight in combined_edges.items():
            if weight > final_edge_threshold:
                combined_graph.add_edge(u, v, weight=weight)
                final_edge_count += 1

        # --- Enrich with Macro Data ---
        print("Enriching network with macro data...")
        try:
            self._enrich_network_with_macro(combined_graph)
        except Exception as e:
            print(f"⚠️ Warning: Failed to enrich network with macro data: {e}")

        num_nodes = combined_graph.number_of_nodes()
        num_edges = combined_graph.number_of_edges()
        print(f"Combined multi-horizon network built: {num_nodes} nodes, {num_edges} edges.")

        # Check for isolated nodes
        isolated = list(nx.isolates(combined_graph))
        if isolated:
            print(f"  Warning: {len(isolated)} isolated node(s) in the combined graph: {', '.join(isolated[:5])}{'...' if len(isolated)>5 else ''}")


        self.network_graph = combined_graph
        return combined_graph

    def _calculate_atr(self, period: int = 14) -> Optional[pd.Series]:
        """
        Calculates the Average True Range (ATR) for each Forex pair.

        NOTE: Requires self.ohlc_data attribute to be populated during data fetching,
              containing DataFrames with 'high', 'low', 'close' columns (lowercase) for each ticker.

        Args:
            period: The lookback period for ATR calculation.

        Returns:
            A Pandas Series containing the latest ATR value for each pair, or None if data is missing.
            Returns NaNs for pairs where HLC data is unavailable.
        """
        # print(f"Calculating ATR({period})...") # Less verbose
        # Check if the required OHLC data exists
        if not hasattr(self, 'ohlc_data') or not self.ohlc_data:
            print("⚠️ Warning: Cannot calculate ATR. `self.ohlc_data` attribute not found or empty.")
            # Return a series of NaNs matching the main data columns
            if self.data is not None:
                 return pd.Series(np.nan, index=self.data.columns)
            else:
                 return None


        atr_results = {}
        for ticker in self.forex_pairs: # Use the list of successfully processed pairs
             if ticker not in self.ohlc_data or self.ohlc_data[ticker] is None:
                 # print(f"  Skipping ATR for {ticker}: No OHLC data.")
                 atr_results[ticker] = np.nan
                 continue

             ohlc_df = self.ohlc_data[ticker]

             # Check if required columns exist (lowercase)
             required_cols = ['high', 'low', 'close']
             if not all(col in ohlc_df.columns for col in required_cols):
                 # print(f"  Warning: Missing HLC columns for {ticker}. Cannot calculate ATR.")
                 atr_results[ticker] = np.nan
                 continue

             # Ensure data is sorted and sufficient
             ohlc_df = ohlc_df.sort_index()
             if len(ohlc_df) < period + 1:
                 # print(f"  Warning: Not enough data points for {ticker} ({len(ohlc_df)} found, {period+1} required).")
                 atr_results[ticker] = np.nan
                 continue

             # Calculate True Range (TR) - Ensure using lowercase columns
             try:
                 high_low = ohlc_df['high'] - ohlc_df['low']
                 high_close_prev = abs(ohlc_df['high'] - ohlc_df['close'].shift(1))
                 low_close_prev = abs(ohlc_df['low'] - ohlc_df['close'].shift(1))

                 tr_df = pd.DataFrame({'hl': high_low, 'hcp': high_close_prev, 'lcp': low_close_prev})
                 tr = tr_df.max(axis=1)
                 tr = tr.fillna(0) # Fill first NaN in TR

                 # Calculate ATR using Exponential Moving Average (EMA) common method
                 # Use pandas ewm for exponential smoothing:
                 atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

                 # Store the latest ATR value
                 atr_results[ticker] = atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else np.nan

             except Exception as e:
                  print(f"  Error calculating ATR for {ticker}: {e}")
                  atr_results[ticker] = np.nan


        # Convert results to a Pandas Series aligned with self.data columns
        atr_series = pd.Series(atr_results).reindex(self.data.columns if self.data is not None else None)
        # print(f"ATR calculation complete for {atr_series.notna().sum()} pairs.") # Less verbose
        return atr_series


    def calculate_forex_metrics(self) -> Optional[pd.DataFrame]:
        """
        Calculates Forex-specific metrics integrating network centrality,
        technical indicators, and macro data.

        Returns:
            A DataFrame where rows are Forex pairs and columns are calculated metrics,
            including a final composite 'forex_score', or None if analysis cannot proceed.
        """
        if self.network_graph is None:
            print("❌ Cannot calculate metrics: Network graph not available. Run build_forex_network() first.")
            return None
        if self.data is None or self.returns is None:
             print("❌ Cannot calculate metrics: Price/Returns data not available.")
             return None

        # print("\nCalculating combined Forex metrics...") # Less verbose

        # --- 1. Network Centrality ---
        centrality_metrics = self._calculate_network_centrality()
        if centrality_metrics is None:
            # Create DataFrame with zeros if centrality failed but graph exists
            metrics_df = pd.DataFrame(index=list(self.network_graph.nodes()))
            for metric in ['eigenvector', 'betweenness', 'closeness']: metrics_df[metric] = 0.0
            print("⚠️ Warning: Failed to calculate network centrality, using default 0 values.")
        else:
             metrics_df = pd.DataFrame(centrality_metrics)


        # Ensure index is set correctly if starting from centrality_metrics dict
        if not isinstance(metrics_df.index, pd.Index):
             metrics_df = pd.DataFrame(centrality_metrics).T # Transpose if needed

        # Ensure all nodes from graph are in the index, fill missing centrality with 0
        metrics_df = metrics_df.reindex(list(self.network_graph.nodes())).fillna(0.0)


        # --- 2. Technical Indicators ---
        # Add recent momentum (use .get() for safety if momentum is None)
        if self.momentum is not None and not self.momentum.empty:
            latest_momentum = self.momentum.iloc[-1]
            metrics_df['momentum'] = latest_momentum.reindex(metrics_df.index).fillna(0)
        else:
             metrics_df['momentum'] = 0.0

        # Add recent volatility (inverse: higher score = lower vol)
        if self.volatility is not None and not self.volatility.empty:
            latest_vol = self.volatility.iloc[-1]
            # Impute missing vol with column average, then global if still missing
            mean_vol_per_pair = self.volatility.mean(axis=0)
            global_mean_vol = mean_vol_per_pair.mean()
            latest_vol = latest_vol.reindex(metrics_df.index).fillna(mean_vol_per_pair).fillna(global_mean_vol)

            # Replace 0 or negative vol with a small positive number before division
            latest_vol[latest_vol <= 1e-6] = 1e-6 # Floor vol
            metrics_df['volatility'] = latest_vol
            metrics_df['inv_volatility'] = 1.0 / latest_vol
        else:
             metrics_df['volatility'] = 0.01 # Assign default vol
             metrics_df['inv_volatility'] = 1.0 / 0.01

        # Add recent returns
        lookback_periods = {'7d': 7, '30d': 30, '90d': 90}
        for name, period in lookback_periods.items():
            if len(self.data) > period:
                # Ensure pct_change aligns index properly
                returns_period = self.data.pct_change(periods=period).iloc[-1]
                metrics_df[f'return_{name}'] = returns_period.reindex(metrics_df.index).fillna(0)
            else:
                metrics_df[f'return_{name}'] = 0.0

        # Add ATR (Average True Range)
        atr_series = self._calculate_atr(14) # Use standard 14 period
        if atr_series is not None:
             # Impute missing ATR with average of available ATRs
             mean_atr = atr_series.mean()
             metrics_df['atr_14d'] = atr_series.reindex(metrics_df.index).fillna(mean_atr if not pd.isna(mean_atr) else 0.0)
        else:
             metrics_df['atr_14d'] = 0.0


        # --- 3. Macro Indicators (from enriched network nodes) ---
        node_attrs_to_add = ['interest_diff', 'carry_score', 'inflation_diff', 'real_rate', 'policy_bias_diff']
        for attr in node_attrs_to_add:
             # Get attribute from graph nodes, default to NaN if missing
             attr_values = nx.get_node_attributes(self.network_graph, attr)
             metrics_df[attr] = pd.Series(attr_values).reindex(metrics_df.index).fillna(0.0) # Fill missing with 0


        # --- 4. Session Strength ---
        dominant_sessions = {}
        session_strengths = {}
        for pair in metrics_df.index:
             pair_session_weights = {}
             for session_name, session_data in self.session_states.items():
                 if pair in session_data:
                      pair_session_weights[session_name] = session_data[pair].get('weight', 0)

             if pair_session_weights:
                 dominant_session = max(pair_session_weights, key=pair_session_weights.get)
                 dominant_sessions[pair] = dominant_session
                 session_strengths[pair] = pair_session_weights[dominant_session]
             else: # Handle case where pair might not be in session_states
                 dominant_sessions[pair] = 'unknown'
                 session_strengths[pair] = 0.0

        metrics_df['dominant_session'] = pd.Series(dominant_sessions).reindex(metrics_df.index).fillna('unknown')
        metrics_df['session_strength'] = pd.Series(session_strengths).reindex(metrics_df.index).fillna(0.0)


        # --- 5. Normalization (0 to 1 scaling) ---
        metrics_to_normalize = {
             'eigenvector': True, 'betweenness': True, 'closeness': True, # Higher centrality is better
             'inv_volatility': True, # Higher inverse vol (lower vol) is better
             'momentum': True, # Higher momentum is better
             'carry_score': True, # Higher carry is generally better (context needed)
             'real_rate': True, # Higher real rate is better
             'policy_bias_diff': True, # Higher difference (base more hawkish) might be better depending on strategy
             'return_7d': True, 'return_30d': True, 'return_90d': True # Higher recent returns are better
        }

        # print("Normalizing metrics...") # Less verbose
        for metric, higher_is_better in metrics_to_normalize.items():
            norm_col_name = f'{metric}_norm'
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna().astype(float) # Ensure numeric
                if len(values) > 1 and values.max() - values.min() > 1e-9: # Avoid division by zero
                    min_val = values.min()
                    max_val = values.max()
                    normalized = (metrics_df[metric] - min_val) / (max_val - min_val)
                    metrics_df[norm_col_name] = normalized.fillna(0.5) # Fill NaNs resulting from normalization with mid-value
                else:
                     # Handle constant columns or single value case
                    metrics_df[norm_col_name] = 0.5 # Assign neutral score
            else:
                # print(f"  Warning: Metric '{metric}' not found for normalization.") # Less verbose
                metrics_df[norm_col_name] = 0.5 # Assign neutral score if column doesn't exist

        # --- 6. Composite Score ---
        # print("Calculating composite Forex score...") # Less verbose
        metrics_df['forex_score'] = self._calculate_forex_composite_score(metrics_df)


        # print("Forex metrics calculation complete.") # Less verbose
        # Sort by score descending, fill NaN scores with lowest value for sorting
        return metrics_df.sort_values('forex_score', ascending=False, na_position='last')


    def _enrich_network_with_macro(self, graph: nx.Graph):
        """
        Enriches the network graph nodes with macroeconomic data attributes.
        Fetches data via MacroDataCollector if not already loaded.

        Args:
            graph: The networkx graph to enrich.
        """
        macro_fetched = False
        # Check if macro data seems loaded, otherwise fetch essentials
        if not hasattr(self.macro, 'data') or not self.macro.data or \
           'interest_differentials' not in self.macro.data or \
           'inflation_differentials' not in self.macro.data or \
           'monetary_bias' not in self.macro.data:
            print("  Fetching required macro data for network enrichment...")
            try:
                 self.macro.fetch_interest_rates()
                 self.macro.fetch_inflation_data()
                 self.macro.calculate_interest_rate_differentials()
                 self.macro.calculate_inflation_differentials() # Needed for real rate
                 self.macro.analyze_monetary_policy_bias() # Needed for bias diff
                 print("  Macro data fetched.")
                 macro_fetched = True
            except Exception as e:
                 print(f"  Warning: Failed to fetch all macro data for enrichment: {e}")
                 # Continue with whatever data is available


        interest_diffs = self.macro.data.get('interest_differentials', {})
        inflation_diffs = self.macro.data.get('inflation_differentials', {})
        policy_biases = self.macro.data.get('monetary_bias', {})
        # Get latest volatility safely
        latest_vol = pd.Series(dtype=float)
        if self.volatility is not None and not self.volatility.empty:
             latest_vol = self.volatility.iloc[-1]


        for pair in graph.nodes():
             if not isinstance(pair, str) or '/' not in pair: continue # Basic check for valid pair format
             node_data = graph.nodes[pair] # Get mutable node data view

             # --- Interest Rate Differential & Carry Score ---
             interest_diff = interest_diffs.get(pair)
             if interest_diff is not None:
                 node_data['interest_diff'] = interest_diff

                 # Calculate Carry Score (Rate Diff / Volatility)
                 pair_vol = latest_vol.get(pair)
                 # Ensure vol is positive float and not None/NaN
                 if pair_vol is not None and isinstance(pair_vol, (int, float)) and not pd.isna(pair_vol) and pair_vol > 1e-6:
                     carry_score = interest_diff / pair_vol # Volatility is already annualized
                     node_data['carry_score'] = carry_score
                 else:
                     node_data['carry_score'] = 0.0 # Assign 0 if vol is invalid

             # --- Inflation Differential & Real Rate ---
             inflation_diff = inflation_diffs.get(pair)
             if inflation_diff is not None:
                 node_data['inflation_diff'] = inflation_diff

                 # Calculate Real Rate = Interest Diff - Inflation Diff
                 if 'interest_diff' in node_data:
                     real_rate = node_data['interest_diff'] - inflation_diff
                     node_data['real_rate'] = real_rate

             # --- Monetary Policy Bias Differential ---
             try:
                 base, quote = pair.split('/')
                 base_bias_str = policy_biases.get(base)
                 quote_bias_str = policy_biases.get(quote)
                 if base_bias_str and quote_bias_str:
                     bias_map = {'hawkish': 1.0, 'neutral': 0.0, 'dovish': -1.0}
                     base_bias_score = bias_map.get(base_bias_str, 0.0)
                     quote_bias_score = bias_map.get(quote_bias_str, 0.0)
                     # Positive diff = Base currency's central bank is more hawkish
                     node_data['policy_bias_diff'] = base_bias_score - quote_bias_score
             except ValueError:
                  pass # Silently ignore if pair format is wrong


        # print("  Network enrichment with macro attributes complete.") # Less verbose


    def _calculate_network_centrality(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calculates network centrality measures (Eigenvector, Betweenness, Closeness).
        Handles disconnected graphs by analyzing components separately.

        Returns:
            A dictionary where keys are node names (tickers) and values are
            dictionaries containing 'eigenvector', 'betweenness', 'closeness' scores,
            or None if the graph is invalid.
        """
        if self.network_graph is None or self.network_graph.number_of_nodes() == 0:
            print("Warning: Cannot calculate centrality, network graph is missing or empty.")
            return None

        # print(f"Calculating network centrality for {self.network_graph.number_of_nodes()} nodes...") # Less verbose

        # Initialize results dictionary with all nodes from the main graph
        centrality_results = {node: {'eigenvector': 0.0, 'betweenness': 0.0, 'closeness': 0.0}
                              for node in self.network_graph.nodes()}

        # Check if graph is connected - use try-except for robustness as nx.is_connected can be slow/fail
        try:
            is_connected = nx.is_connected(self.network_graph)
        except Exception:
             is_connected = False # Assume disconnected if check fails

        if is_connected:
             # print("  Graph is connected. Calculating centrality directly.") # Less verbose
             components_subgraphs = [(self.network_graph.nodes(), self.network_graph)] # Treat as one component
        else:
            num_components = nx.number_connected_components(self.network_graph)
            print(f"  Graph is disconnected ({num_components} components). Calculating centrality per component.")
            # Create subgraphs generator
            components_subgraphs = [(c, self.network_graph.subgraph(c).copy()) for c in nx.connected_components(self.network_graph)]


        total_nodes_processed = 0
        # Calculate centrality for each component
        for i, (component_nodes, subgraph) in enumerate(components_subgraphs):
             num_nodes_in_comp = subgraph.number_of_nodes()
             total_nodes_processed += num_nodes_in_comp

             if num_nodes_in_comp <= 1: continue # Skip single-node components

             try:
                # Eigenvector Centrality
                try:
                    # Use numpy version, requires conversion but often more stable
                    eigenvector = nx.eigenvector_centrality_numpy(subgraph, weight='weight')
                    for node, score in eigenvector.items():
                        centrality_results[node]['eigenvector'] = score
                except (nx.NetworkXError, nx.PowerIterationFailedConvergence, np.linalg.LinAlgError) as e:
                    print(f"    Warning: Eigenvector centrality failed for component {i+1} ({num_nodes_in_comp} nodes): {e}. Setting to 0.")
                    for node in subgraph.nodes(): centrality_results[node]['eigenvector'] = 0.0

                # Betweenness Centrality (use approximation for larger graphs?)
                # Normalize by default. Consider using edge weights? distance = 1/weight? Default unweighted.
                if num_nodes_in_comp < 1000: # Exact calculation for smaller graphs
                     betweenness = nx.betweenness_centrality(subgraph, normalized=True, weight=None)
                else: # Approximation for larger graphs
                     print(f"    Using approximated betweenness centrality for large component ({num_nodes_in_comp} nodes).")
                     betweenness = nx.betweenness_centrality(subgraph, k=min(num_nodes_in_comp // 10, 100), normalized=True, weight=None)
                for node, score in betweenness.items():
                    centrality_results[node]['betweenness'] = score

                # Closeness Centrality
                # Needs connected component - subgraph is connected
                closeness = nx.closeness_centrality(subgraph, distance=None) # Unweighted shortest path
                for node, score in closeness.items():
                    centrality_results[node]['closeness'] = score

             except Exception as e:
                  print(f"❌ Unexpected error calculating centrality for component {i+1}: {e}")
                  # Assign default 0 values for this component
                  for node in subgraph.nodes():
                      centrality_results[node] = {'eigenvector': 0.0, 'betweenness': 0.0, 'closeness': 0.0}

        # if total_nodes_processed != self.network_graph.number_of_nodes():
        #      print(f"Warning: Processed {total_nodes_processed} nodes, but graph has {self.network_graph.number_of_nodes()}. Check isolated nodes.")

        # print("Network centrality calculation complete.") # Less verbose
        return centrality_results

    # --- Composite Score Calculation ---
    def _calculate_forex_composite_score(self, metrics_df: pd.DataFrame) -> pd.Series:
        """
        Calculates the final composite score based on normalized metrics.
        Weights should be adjusted based on the desired strategy focus.

        Args:
            metrics_df: DataFrame containing metrics, including normalized columns ending in '_norm'.

        Returns:
            A Pandas Series with the composite score for each asset.
        """
        # print("Calculating composite score...") # Less verbose
        composite_score = pd.Series(0.0, index=metrics_df.index)
        total_weight = 0.0

        # Define weights for each normalized metric component
        # ** THESE WEIGHTS ARE EXAMPLES - TUNE BASED ON STRATEGY **
        weights = {
            'eigenvector_norm': 0.15,      # Network importance
            'betweenness_norm': 0.05,      # Bridge role in network
            'closeness_norm': 0.05,        # Network efficiency
            'momentum_norm': 0.25,         # Trend strength
            'inv_volatility_norm': 0.15,   # Stability (low vol)
            'carry_score_norm': 0.10,      # Carry potential (adj. for risk)
            'real_rate_norm': 0.05,        # Real yield differential
            'policy_bias_diff_norm': 0.05, # Favorable policy divergence
            'return_30d_norm': 0.10,       # Medium-term performance
            'return_90d_norm': 0.05,       # Longer-term performance
            # Add other normalized metrics if needed
        }

        # Ensure we only use weights for metrics that actually exist after normalization
        available_norm_cols = [col for col in metrics_df.columns if col.endswith('_norm')]

        for metric_norm, weight in weights.items():
            if metric_norm in available_norm_cols:
                # Fill any potential NaNs in the normalized column with 0.5 (neutral) before weighting
                composite_score += metrics_df[metric_norm].fillna(0.5) * weight
                total_weight += weight

        # Normalize the final score by total weight used, if total_weight is > 0
        if total_weight > 1e-6:
            final_score = composite_score / total_weight
            # print(f"Composite score calculated using total weight: {total_weight:.2f}") # Less verbose
            # Clip score to be strictly between 0 and 1
            return final_score.clip(0.0, 1.0)
        else:
            # print("Warning: No weighted metrics available for composite score calculation.") # Less verbose
            return pd.Series(0.0, index=metrics_df.index) # Return zero score if no weights applied


    # --- Alerting ---
    def check_forex_alerts(self) -> List[Dict[str, Any]]:
        """
        Checks for various alert conditions based on price data, volatility,
        macro data, technical indicators, and upcoming events.

        Returns:
            A list of active alert dictionaries.
        """
        # print("\nChecking for Forex alerts...") # Less verbose
        alerts: List[Dict[str, Any]] = []
        current_pairs = self.forex_pairs # Use the list of pairs successfully processed

        # --- 1. Drawdown Check ---
        if self.drawdown_history:
            threshold = self.config.DRAWDOWN_ALERT_THRESHOLD
            # print(f"  Checking drawdowns (threshold: {threshold:.1%})...") # Less verbose
            for pair, drawdowns in self.drawdown_history.items():
                if pair in current_pairs and not drawdowns.empty: # Check if pair is still valid
                    last_drawdown = drawdowns.iloc[-1]
                    if not pd.isna(last_drawdown) and last_drawdown < threshold:
                        alerts.append({
                            'type': 'DRAWDOWN', 'asset': pair, 'value': last_drawdown, 'threshold': threshold,
                            'message': f"Drawdown: {pair} at {last_drawdown:.2%} (Threshold: {threshold:.1%})"
                        })

        # --- 2. Volatility Check ---
        if self.volatility is not None and not self.volatility.empty:
            threshold = self.config.VOLATILITY_ALERT_THRESHOLD
            # print(f"  Checking volatility (threshold: {threshold:.1%} of max)...") # Less verbose
            try:
                recent_vol = self.volatility.iloc[-30:].mean() if len(self.volatility) >= 30 else self.volatility.mean()
                max_vol = self.volatility.max()

                for pair in current_pairs:
                     if pair in recent_vol.index and pair in max_vol.index and max_vol[pair] > 1e-9:
                        vol_ratio = recent_vol[pair] / max_vol[pair]
                        if not pd.isna(vol_ratio) and vol_ratio > threshold:
                             alerts.append({
                                'type': 'HIGH_VOLATILITY', 'asset': pair, 'value': vol_ratio, 'threshold': threshold,
                                'message': f"High Vol: {pair} ratio {vol_ratio:.1%} (Threshold: {threshold:.1%})"
                            })
            except Exception as e:
                 print(f"    Warning: Error during volatility check: {e}")


        # --- 3. Policy Divergence Check ---
        if hasattr(self.macro, 'data') and 'monetary_bias' in self.macro.data:
            # print("  Checking monetary policy divergence...") # Less verbose
            biases = self.macro.data['monetary_bias']
            for pair in current_pairs:
                 try:
                     base, quote = pair.split('/')
                     if base in biases and quote in biases:
                         base_bias = biases[base]
                         quote_bias = biases[quote]
                         if base_bias == 'hawkish' and quote_bias == 'dovish':
                             alerts.append({
                                'type': 'POLICY_DIVERGENCE', 'asset': pair, 'value': 'H > D', 'threshold': None,
                                'message': f"Policy Divergence: {base}(Hawkish) vs {quote}(Dovish)"
                             })
                         elif base_bias == 'dovish' and quote_bias == 'hawkish':
                              alerts.append({
                                'type': 'POLICY_DIVERGENCE', 'asset': pair, 'value': 'D > H', 'threshold': None,
                                'message': f"Policy Divergence: {base}(Dovish) vs {quote}(Hawkish)"
                             })
                 except ValueError:
                     continue # Skip pairs not in BASE/QUOTE format

        # --- 4. Technical Breakout (Bollinger Bands Example) ---
        if self.data is not None and len(self.data) >= 20:
            # print("  Checking technical breakouts (Bollinger Bands)...") # Less verbose
            for pair in current_pairs:
                if pair not in self.data.columns: continue
                prices = self.data[pair].dropna()
                if len(prices) < 20: continue

                try:
                     rolling_mean = prices.rolling(window=20).mean()
                     rolling_std = prices.rolling(window=20).std()
                     upper_band = rolling_mean + (rolling_std * 2)
                     lower_band = rolling_mean - (rolling_std * 2)

                     # Check latest values safely
                     if not prices.empty and not upper_band.empty and not lower_band.empty:
                         last_price = prices.iloc[-1]
                         last_upper = upper_band.iloc[-1]
                         last_lower = lower_band.iloc[-1]

                         if not pd.isna(last_price):
                             if not pd.isna(last_upper) and last_price > last_upper:
                                 alerts.append({
                                     'type': 'BB_BREAKOUT_UP', 'asset': pair, 'value': last_price, 'threshold': last_upper,
                                     'message': f"BB Breakout Up: {pair} at {last_price:.4f} (Band: {last_upper:.4f})"
                                 })
                             elif not pd.isna(last_lower) and last_price < last_lower:
                                  alerts.append({
                                     'type': 'BB_BREAKOUT_DOWN', 'asset': pair, 'value': last_price, 'threshold': last_lower,
                                     'message': f"BB Breakout Down: {pair} at {last_price:.4f} (Band: {last_lower:.4f})"
                                 })
                except Exception as e:
                     print(f"    Warning: Error calculating Bollinger Bands for {pair}: {e}")

        # --- 5. Upcoming High-Impact Economic Events ---
        if hasattr(self.macro, 'data') and 'economic_calendar' in self.macro.data:
            # print("  Checking upcoming economic events...") # Less verbose
            try:
                today_dt = self.current_date or datetime.datetime.now()
                limit_dt = today_dt + datetime.timedelta(days=7) # Check next 7 days
                today_str = today_dt.strftime('%Y-%m-%d')
                limit_str = limit_dt.strftime('%Y-%m-%d')

                upcoming_events = self.macro.data.get('economic_calendar', [])
                processed_events = set() # Avoid duplicate alerts for same event/pair

                for event in upcoming_events:
                    event_date_str = event.get('date')
                    if not event_date_str: continue

                    # Check if event date is within range and impact is high
                    if today_str <= event_date_str <= limit_str and event.get('impact') == 'high':
                        currency = event.get('currency')
                        event_name = event.get('name', 'N/A')
                        event_key = f"{event_date_str}_{currency}_{event_name}" # Key to track duplicates

                        if not currency or event_key in processed_events: continue

                        affected_pairs = [p for p in current_pairs if p.startswith(f"{currency}/") or p.endswith(f"/{currency}")]
                        if affected_pairs:
                             alerts.append({
                                'type': 'ECONOMIC_EVENT', 'asset': currency, # Alert on the currency itself
                                'value': event_name, 'threshold': None,
                                'message': f"Upcoming Event ({currency}): {event_name} on {event_date_str} {event.get('time', '')}"
                             })
                             processed_events.add(event_key) # Mark as processed

            except Exception as e:
                print(f"    Warning: Error processing economic calendar: {e}")

        # --- 6. Significant Carry Trade Opportunity ---
        if self.network_graph is not None:
             # print("  Checking carry trade opportunities (from network attributes)...") # Less verbose
             carry_scores = nx.get_node_attributes(self.network_graph, 'carry_score')
             carry_threshold = 0.5 # Example threshold for significance

             for pair, score in carry_scores.items():
                 if pair in current_pairs and not pd.isna(score) and abs(score) > carry_threshold: # Check absolute value
                     direction = "Positive" if score > 0 else "Negative"
                     alerts.append({
                        'type': 'CARRY_OPPORTUNITY', 'asset': pair, 'value': score, 'threshold': carry_threshold,
                        'message': f"{direction} Carry Opp: {pair} score={score:.2f} (Threshold: +/-{carry_threshold})"
                    })

        self.alerts = alerts
        # print(f"Alert check complete. Found {len(alerts)} active alerts.") # Less verbose
        return alerts

    # --- Recommendations ---
    def recommend_forex_pairs(self) -> Optional[pd.DataFrame]:
        """
        Generates Forex pair recommendations based on calculated metrics.

        Returns:
            DataFrame of recommended Forex pairs sorted by score, or None if metrics are unavailable.
        """
        # print("\nGenerating Forex pair recommendations...") # Less verbose

        # 1. Calculate comprehensive metrics
        metrics_df = self.calculate_forex_metrics()
        if metrics_df is None or metrics_df.empty:
            print("❌ Cannot generate recommendations: Failed to calculate metrics.")
            return None
        if 'forex_score' not in metrics_df.columns:
             print("❌ Cannot generate recommendations: 'forex_score' column is missing.")
             return None


        # 2. Rank pairs by score
        ranked_pairs = metrics_df.sort_values('forex_score', ascending=False, na_position='last')
        # Remove pairs with NaN score before selecting top N
        ranked_pairs = ranked_pairs.dropna(subset=['forex_score'])


        # 3. Select top N pairs
        num_to_recommend = min(self.config.RECOMMENDED_PAIRS, len(ranked_pairs))
        if num_to_recommend == 0:
             print("No pairs available for recommendation after filtering.")
             return pd.DataFrame()
        top_pairs = ranked_pairs.head(num_to_recommend).copy()


        # 4. Format the output DataFrame
        recommendations = top_pairs
        recommendations['Rank'] = range(1, len(recommendations) + 1)

        # Determine Direction based on momentum score
        if 'momentum' in recommendations.columns:
             # Use a small threshold around zero if needed? For now, simple sign check.
             recommendations['Direction'] = recommendations['momentum'].apply(lambda x: 'LONG' if x > 0 else 'SHORT' if x < 0 else 'NEUTRAL')
        else:
             recommendations['Direction'] = 'N/A'


        # Map dominant session codes to readable names
        if 'dominant_session' in recommendations.columns:
             session_map = {'asia': 'Asia', 'europe': 'Europe', 'america': 'America', 'unknown': 'N/A'}
             recommendations['Optimal Session'] = recommendations['dominant_session'].map(session_map).fillna('N/A')
        else:
             recommendations['Optimal Session'] = 'N/A'


        # Select and rename columns for final presentation
        # Use original metric names here for clarity
        final_cols_map = {
            'Rank': 'Rank',
            'forex_score': 'Score',
            'Direction': 'Direction',
            'eigenvector': 'Centrality',
            'momentum': 'Momentum',
            'inv_volatility': 'Stability', # Higher is better (lower vol)
            'carry_score': 'Carry Score',
            'real_rate': 'Real Rate Diff',
            'return_30d': 'Return 30d',
            'Optimal Session': 'Optimal Session'
        }
        # Only include columns that actually exist in the recommendations DataFrame
        available_cols_raw = [col for col in final_cols_map.keys() if col in recommendations.columns]
        recommendations_final = recommendations[available_cols_raw].copy()
        # Rename using the map, but only for columns that were present
        recommendations_final = recommendations_final.rename(columns={k:v for k,v in final_cols_map.items() if k in available_cols_raw})

        # Format numeric columns (rounding, percentages)
        for col_disp_name, col_orig_name in zip(recommendations_final.columns, available_cols_raw):
             # Apply formatting based on the display name (which is now the column name)
             if col_disp_name in ['Score', 'Centrality', 'Momentum', 'Stability', 'Carry Score', 'Real Rate Diff']:
                 recommendations_final[col_disp_name] = recommendations_final[col_disp_name].map('{:.3f}'.format)
             elif col_disp_name == 'Return 30d':
                  recommendations_final[col_disp_name] = recommendations_final[col_disp_name].map('{:.2%}'.format)


        # Set index name
        recommendations_final.index.name = 'Forex Pair'

        self.last_recommendations = recommendations_final # Store formatted recommendations
        # print(f"Generated {len(recommendations_final)} recommendations.") # Less verbose
        return recommendations_final


    # --- Reporting ---
    def generate_forex_report(self, output_folder: str = 'forex_reports') -> Optional[str]:
        """
        Generates a comprehensive text report of the Forex analysis,
        including recommendations, alerts, and references to saved plots.

        Args:
            output_folder: The directory where report files (including plots) will be saved.

        Returns:
            A string containing the summary text of the report, or None on critical error.
        """
        print(f"\nGenerating Forex report in folder: '{output_folder}'...")

        # --- Setup ---
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                print(f"Created report folder: '{output_folder}'")
            except OSError as e:
                print(f"❌ Error creating report folder '{output_folder}': {e}. Cannot save files.")
                # Continue without saving files? Or return None? Let's continue but warn.
                output_folder = None # Indicate saving is not possible

        report_dt = self.current_date or datetime.datetime.now()
        report_date_str = report_dt.strftime('%Y-%m-%d_%H%M')
        report_id = f"mhgna_forex_{report_dt.strftime('%Y%m%d_%H%M%S')}"
        report_file_path = os.path.join(output_folder, f"{report_id}_report.txt") if output_folder else None


        # --- Generate Visualizations (Call placeholders/implementations) ---
        network_file_path = None
        trends_file_path = None
        if output_folder:
            network_file_path = os.path.join(output_folder, f"{report_id}_network.png")
            trends_file_path = os.path.join(output_folder, f"{report_id}_trends.png")
            print("  Generating visualizations...")
            try:
                 self.visualize_forex_network(filename=network_file_path)
            except Exception as e:
                 print(f"⚠️ Warning: Failed to generate network visualization: {e}")
                 network_file_path = "Generation failed" # Mark as failed

            try:
                 self.visualize_forex_trends(filename=trends_file_path)
            except Exception as e:
                 print(f"⚠️ Warning: Failed to generate trends visualization: {e}")
                 trends_file_path = "Generation failed" # Mark as failed
        else:
             print("  Skipping visualization generation (no output folder).")


        # --- Build Report Text ---
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f" MHGNA FOREX ANALYSIS REPORT - {report_dt.strftime('%Y-%m-%d %H:%M')}")
        report_lines.append("=" * 60)

        # --- 1. Recommendations ---
        report_lines.append("\n--- 1. TOP FOREX PAIR RECOMMENDATIONS ---")
        if self.last_recommendations is None:
            # print("  Recommendations not pre-generated, attempting now...") # Less verbose
            self.recommend_forex_pairs() # Ignore return value, check self.last_recommendations

        if self.last_recommendations is not None and not self.last_recommendations.empty:
            try:
                # Use tabulate for better formatting if available? For now, use to_string
                report_lines.append(self.last_recommendations.to_string())
            except Exception as e:
                 report_lines.append(f"  Error formatting recommendations: {e}")

            report_lines.append("\n  Interpretation Notes:")
            # ... (keep interpretation notes) ...
            report_lines.append("  - Score: Overall attractiveness based on combined metrics.")
            report_lines.append("  - Direction: Suggested trade direction (LONG/SHORT) based on momentum.")
            report_lines.append("  - Centrality: Importance within the network.")
            report_lines.append("  - Momentum: Recent trend strength.")
            report_lines.append("  - Stability: Inverse volatility (higher is more stable).")
            report_lines.append("  - Carry Score: Risk-adjusted interest rate differential.")
            report_lines.append("  - Real Rate Diff: Interest diff minus inflation diff.")
            report_lines.append("  - Optimal Session: Session with highest simulated activity weight.")

        else:
            report_lines.append("\n  No recommendations available (Analysis might have failed or produced no valid pairs).")

        # --- 2. Active Alerts ---
        report_lines.append("\n--- 2. ACTIVE ALERTS ---")
        # Ensure alerts are checked
        if not self.alerts:
             # print("  Alerts not pre-checked, attempting now...") # Less verbose
             self.check_forex_alerts()

        if self.alerts:
            alerts_by_type: Dict[str, list] = {}
            for alert in self.alerts:
                alerts_by_type.setdefault(alert['type'], []).append(alert)

            for alert_type, type_alerts in sorted(alerts_by_type.items()): # Sort by type
                report_lines.append(f"\n  * {alert_type.replace('_', ' ').title()} ({len(type_alerts)}):")
                for alert in type_alerts[:5]: # Limit display per type
                    report_lines.append(f"    - {alert['message']}")
                if len(type_alerts) > 5:
                     report_lines.append("    - ... (and more)")
        else:
            report_lines.append("\n  No active alerts detected.")


        # --- 3. Macroeconomic Analysis ---
        report_lines.append("\n--- 3. MACROECONOMIC ANALYSIS ---")
        # ... (Keep macro section from previous step, it was correct) ...
        if hasattr(self.macro, 'data') and self.macro.data:
            macro_data_available = False
            # Interest Rates
            if 'interest_rates' in self.macro.data and self.macro.data['interest_rates']:
                report_lines.append("\n  Interest Rates:")
                sorted_rates = sorted(self.macro.data['interest_rates'].items(), key=lambda item: item[1], reverse=True)
                for currency, rate in sorted_rates:
                    report_lines.append(f"    - {currency}: {rate:.2f}%")
                macro_data_available = True

            # Rate Differentials
            if 'interest_differentials' in self.macro.data and self.macro.data['interest_differentials']:
                diffs = self.macro.data['interest_differentials']
                # Sort by absolute value, show top 5
                sorted_diffs = sorted(diffs.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
                if sorted_diffs:
                     report_lines.append("\n  Most Significant Interest Rate Differentials:")
                     for pair, diff in sorted_diffs:
                         report_lines.append(f"    - {pair}: {diff:+.2f}%")
                     macro_data_available = True

            # Upcoming Events
            if 'economic_calendar' in self.macro.data and self.macro.data['economic_calendar']:
                 try:
                    today_dt = self.current_date or datetime.datetime.now()
                    today_str = today_dt.strftime('%Y-%m-%d')
                    limit_dt = today_dt + datetime.timedelta(days=7) # Check next 7 days
                    limit_str = limit_dt.strftime('%Y-%m-%d')

                    coming_events = [
                         e for e in self.macro.data.get('economic_calendar', [])
                         if e.get('date') and today_str <= e.get('date') <= limit_str and
                            e.get('impact') == 'high'
                     ]
                    if coming_events:
                        report_lines.append("\n  Upcoming High-Impact Events (Next 7 Days):")
                        for event in sorted(coming_events, key=lambda x: x.get('date'))[:5]: # Show earliest 5
                            report_lines.append(f"    - {event.get('date')} {event.get('time','')} - {event.get('currency','N/A')}: {event.get('name','N/A')}")
                        macro_data_available = True
                 except Exception as e:
                      print(f"    Warning: Error processing economic calendar for report: {e}")


            if not macro_data_available:
                report_lines.append("\n  No relevant macroeconomic data points to display.")

        else:
            report_lines.append("\n  Macroeconomic data module not available.")


        # --- 4. Market Analysis by Session (Simulated) ---
        report_lines.append("\n--- 4. MARKET ANALYSIS BY SESSION (SIMULATED ACTIVITY WEIGHT) ---")
        # ... (Keep session section from previous step, it was correct) ...
        if hasattr(self, 'session_states') and self.session_states:
             session_data_found = False
             session_name_map = {
                 'asia': 'Asia (22:00-08:00 UTC)',
                 'europe': 'Europe (07:00-16:00 UTC)',
                 'america': 'America (13:00-22:00 UTC)'
             }
             for session_code, state_data in self.session_states.items():
                 session_name = session_name_map.get(session_code, session_code)
                 report_lines.append(f"\n  {session_name}:")
                 if state_data:
                     # Sort pairs by simulated activity weight within the session
                     pairs_by_weight = sorted(
                         [(pair, data.get('weight', 0)) for pair, data in state_data.items()],
                         key=lambda item: item[1],
                         reverse=True
                     )
                     if pairs_by_weight:
                         for pair, weight in pairs_by_weight[:3]: # Show top 3 for brevity
                             report_lines.append(f"    - {pair}: Activity Weight = {weight:.2f}")
                         session_data_found = True
                     else:
                          report_lines.append("    - No pair data available for this session.")

                 else:
                     report_lines.append("    - No pair data available for this session.")

             if not session_data_found:
                  report_lines.append("\n  No session-specific data points available.")
        else:
            report_lines.append("\n  Session state data not available.")


        # --- 5. Recommended Actions ---
        report_lines.append("\n--- 5. RECOMMENDED ACTIONS ---")
        # ... (Keep actions section from previous step, it was correct) ...
        actions_generated = False
        # Based on Recommendations
        if self.last_recommendations is not None and not self.last_recommendations.empty:
             top_recs = []
             max_recs_display = min(3, len(self.last_recommendations))
             for i in range(max_recs_display):
                 pair = self.last_recommendations.index[i]
                 # Access column by the renamed value if available
                 direction_col_name = 'Direction' if 'Direction' in self.last_recommendations.columns else None
                 direction = self.last_recommendations.iloc[i].get(direction_col_name, 'N/A') if direction_col_name else 'N/A'
                 top_recs.append(f"{pair} ({direction})")

             if top_recs:
                 report_lines.append(f"\n  * Consider potential positions based on top recommendations: {', '.join(top_recs)}.")
                 actions_generated = True
        else:
             report_lines.append("\n  * Wait for valid recommendations before considering positions.")
             actions_generated = True # Still an action (waiting)


        # Based on Alerts
        alert_actions = []
        if self.alerts:
             if any(a['type'] == 'DRAWDOWN' for a in self.alerts):
                 alert_actions.append("Review/reduce exposure on pairs hitting significant drawdown thresholds.")
             if any(a['type'] == 'HIGH_VOLATILITY' for a in self.alerts):
                  alert_actions.append("Be cautious with pairs showing unusually high volatility; consider smaller position sizes.")
             if any(a['type'] == 'CARRY_OPPORTUNITY' for a in self.alerts):
                  carry_opps = [a['asset'] for a in self.alerts if a['type'] == 'CARRY_OPPORTUNITY' and a.get('value',0) > 0] # Positive carry
                  neg_carry = [a['asset'] for a in self.alerts if a['type'] == 'CARRY_OPPORTUNITY' and a.get('value',0) < 0] # Negative carry
                  if carry_opps: alert_actions.append(f"Explore potential positive carry trades on: {', '.join(sorted(list(set(carry_opps)))[:3])}{'...' if len(set(carry_opps))>3 else ''}.")
                  if neg_carry: alert_actions.append(f"Be aware of negative carry costs on: {', '.join(sorted(list(set(neg_carry)))[:3])}{'...' if len(set(neg_carry))>3 else ''}.")
             if any(a['type'] == 'POLICY_DIVERGENCE' for a in self.alerts):
                 alert_actions.append("Monitor pairs with strong monetary policy divergence for potential trend shifts.")
             if any(a['type'] == 'ECONOMIC_EVENT' for a in self.alerts):
                  alert_actions.append("Prepare for potential volatility around upcoming high-impact economic events.")
             if any(a['type'].startswith('BB_BREAKOUT') for a in self.alerts):
                 alert_actions.append("Note Bollinger Band breakouts as potential entry/exit signals or volatility indicators.")


        if alert_actions:
             actions_generated = True
             for action in alert_actions:
                 report_lines.append(f"  * {action}")

        if not actions_generated:
             report_lines.append("\n  * Monitor market conditions; no specific actions triggered by current analysis.")


        # --- 6. Visualizations Reference ---
        report_lines.append("\n--- 6. VISUALIZATIONS ---")
        if output_folder:
             if network_file_path and "failed" not in network_file_path:
                 report_lines.append(f"\n  - Network Graph: Saved to {os.path.basename(network_file_path)}")
             else:
                 report_lines.append("\n  - Network Graph: Not generated or failed.")
             if trends_file_path and "failed" not in trends_file_path:
                 report_lines.append(f"  - Trend Visuals: Saved to {os.path.basename(trends_file_path)}")
             else:
                 report_lines.append("  - Trend Visuals: Not generated or failed.")
        else:
             report_lines.append("\n  - Visualizations not saved (no output folder specified or created).")


        # --- Footer ---
        report_lines.append("\n" + "=" * 60)
        report_lines.append("End of Report")
        report_lines.append("=" * 60)

        final_report_text = "\n".join(report_lines)

        # --- Save Report to File ---
        if report_file_path:
            try:
                with open(report_file_path, 'w', encoding='utf-8') as f:
                    f.write(final_report_text)
                print(f"Report text saved successfully to: {report_file_path}")
            except (IOError, OSError, UnicodeEncodeError) as e:
                print(f"⚠️ Error saving report file with UTF-8: {e}. Trying ASCII fallback...")
                try:
                    # Basic ASCII replacement
                    ascii_report = final_report_text.encode('ascii', 'replace').decode('ascii')
                    with open(report_file_path, 'w', encoding='ascii') as f:
                        f.write(ascii_report)
                    print(f"Report text saved successfully to: {report_file_path} (ASCII fallback)")
                except Exception as e_fallback:
                    print(f"❌ Failed to save report even with ASCII fallback: {e_fallback}")
        # else: # No need to print this if saving was never attempted
        #     print("Report text not saved (no output folder).")


        return final_report_text

        # DANS LA CLASSE MHGNAForex:

        # --- Visualization Methods ---

    # Imports nécessaires pour ces fonctions (normalement déjà en haut du fichier)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import networkx as nx
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    import os  # Nécessaire pour generate_forex_report
    import datetime  # Nécessaire pour generate_forex_report

    # --- Collez ce bloc à l'intérieur de la classe MHGNAForex ---
    # --- Remplacez les anciennes versions des fonctions visualize_* et generate_forex_report ---

    # --- Visualization Methods ---

    # DANS LA CLASSE MHGNAForex (Remplace les 3 fonctions: trends, heatmap, metrics)

    def visualize_forex_trends (self,filename: Optional[str] = None,**kwargs):
        """
        Creates visualizations of price trends, alerts, and key metrics for selected Forex pairs.
        WARNING: This function combines multiple plots onto 3 axes, overwriting content.
                 Consider using the separated visualize_* functions for clarity.

        Args:
            filename: Path to save the visualization file.
            **kwargs: Additional arguments (currently unused).
        """
        # --- Vérifications initiales ---
        if self.data is None: print ("❌ Cannot visualize: Price data not available."); return
        try:
            import matplotlib.pyplot as plt;
            import matplotlib.dates as mdates
            from matplotlib.colors import LinearSegmentedColormap;
            import numpy as np
        except ImportError:
            print ("❌ Matplotlib or Numpy not installed."); return

        print ("Generating Combined Trends/Signals/Metrics visualization...")

        # --- Définir current_date_str ---
        current_date_str=(self.current_date or datetime.datetime.now ()).strftime ('%Y-%m-%d')

        # --- Select Pairs to Display ---
        display_pairs=[]
        if self.last_recommendations is None: self.recommend_forex_pairs ()
        if self.last_recommendations is not None and not self.last_recommendations.empty:
            display_pairs=self.last_recommendations.index.tolist ()[:5]
        else:
            display_pairs=self.forex_pairs[:min (5,len (self.forex_pairs))]
        eurusd_std='EUR/USD'
        if eurusd_std not in display_pairs and eurusd_std in self.data.columns: display_pairs.append (eurusd_std)
        display_pairs=[p for p in display_pairs if p in self.data.columns]
        if not display_pairs: print ("  No pairs available to display."); return
        print (f"  Visualizing combined info for: {', '.join (display_pairs)}")

        # --- Setup Figure and Subplots ---
        plt.style.use ('dark_background')
        fig,(ax1,ax2,ax3)=plt.subplots (3,1,figsize=(14,18),sharex=False,
                                        # sharex=False car les axes 2 et 3 ne sont pas des séries temporelles
                                        gridspec_kw={'height_ratios':[3,1,2]})
        fig.patch.set_facecolor ('#101010')

        # --- Plotting Data Preparation ---
        plot_data=self.data.iloc[-90:].copy ()
        metrics_df=self.calculate_forex_metrics ()  # Needed for bar plot

        # === AXIS 1: Normalized Price Trends ===
        ax1.clear ();
        ax1.set_facecolor ('#181818');
        ax1.set_title ("Normalized Price Trends (Last 90 Days)",fontsize=14,color='white')
        ax1.set_ylabel ("Price (Normalized to 100)",color='white');
        ax1.grid (True,linestyle=':',alpha=0.4,color='#555555');
        ax1.tick_params (axis='y',colors='white');
        ax1.tick_params (axis='x',labelbottom=False)
        normalized_prices=pd.DataFrame (index=plot_data.index)
        for pair in display_pairs:
            if pair in plot_data.columns:
                series=plot_data[pair].dropna ();
                if len (series) > 0: start_value=series.iloc[0]; normalized_prices[pair]=(
                                                                                                     series / start_value) * 100 if start_value and start_value != 0 else np.nan
        # Plot prices
        for pair in display_pairs:
            if pair in normalized_prices.columns and not normalized_prices[pair].isnull ().all ():
                rank=None;
                if self.last_recommendations is not None and pair in self.last_recommendations.index:
                    try:
                        rank=self.last_recommendations.index.get_loc (pair) + 1
                    except KeyError:
                        pass
                linewidth=2.5 if rank == 1 else 2.0 if rank in [2,3] else 1.5;
                alpha=1.0 if rank is not None and rank <= 3 else 0.75;
                label=f"{pair} (Rank {rank})" if rank is not None else pair
                ax1.plot (normalized_prices.index,normalized_prices[pair],linewidth=linewidth,alpha=alpha,label=label)
        if any (ax1.get_legend_handles_labels ()): ax1.legend (loc='best',fontsize=9,facecolor='#222222',
                                                               labelcolor='white',framealpha=0.7)
        # Add event lines
        if hasattr (self.macro,'data') and 'economic_calendar' in self.macro.data:
            plot_start_date=plot_data.index.min ().tz_localize (
                None) if plot_data.index.tz is not None else plot_data.index.min ()
            plot_end_date=plot_data.index.max ().tz_localize (
                None) if plot_data.index.tz is not None else plot_data.index.max ()
            for event in self.macro.data.get ('economic_calendar',[]):
                try:
                    event_date_naive=pd.to_datetime (event.get ('date')).tz_localize (None);
                    if plot_start_date <= event_date_naive <= plot_end_date and event.get ('impact') == 'high':
                        currency=event.get ('currency');
                        if currency and any (p.startswith (f"{currency}/") or p.endswith (f"/{currency}") for p in
                                             display_pairs): ax1.axvline (x=event_date_naive,color='yellow',alpha=0.3,
                                                                          linestyle='--',linewidth=1)
                except Exception:
                    continue
        # Format X axis for ax1 (dates)
        ax1.xaxis.set_major_formatter (mdates.DateFormatter ('%b %d'))
        ax1.xaxis.set_major_locator (mdates.MonthLocator (interval=1))
        ax1.xaxis.set_minor_locator (mdates.WeekdayLocator (interval=1))

        # === AXIS 2: Alert Heatmap ===
        ax2.clear ();
        ax2.set_facecolor ('#181818');
        ax2.set_title ("Current Alert Signals",fontsize=14,color='white')
        ax2.tick_params (colors='white',top=False,bottom=True,labeltop=False,labelbottom=True,length=0);
        plt.setp (ax2.get_xticklabels (),rotation=0,ha='center')
        if not self.alerts: self.check_forex_alerts ()
        alert_type_map={'DRAWDOWN':{'index':0,'value':1,'label':'Drawdown'},
                        'HIGH_VOLATILITY':{'index':1,'value':1,'label':'High Vol'},
                        'BB_BREAKOUT':{'index':2,'value':-1,'label':'BB Break'},
                        'POLICY_DIVERGENCE':{'index':3,'value':2,'label':'Policy Div.'},
                        'CARRY_OPPORTUNITY':{'index':4,'value':2,'label':'Carry Opp.'},
                        'ECONOMIC_EVENT':{'index':5,'value':1,'label':'Event'}}
        alert_cols=sorted (list (set (d['index'] for d in alert_type_map.values ())));
        alert_labels=[""] * len (alert_cols);
        for d in alert_type_map.values (): alert_labels[d['index']]=d['label']
        heatmap_data=np.zeros ((len (display_pairs),len (alert_cols)));
        y_labels=display_pairs
        for r,pair in enumerate (display_pairs):
            for alert in self.alerts:
                alert_asset=alert.get ('asset');
                alert_type_base=alert.get ('type');
                alert_applies=False
                if alert_asset == pair:
                    alert_applies=True
                elif alert_type_base == 'ECONOMIC_EVENT' and alert_asset and (
                        pair.startswith (f"{alert_asset}/") or pair.endswith (f"/{alert_asset}")):
                    alert_applies=True
                if alert_applies:
                    map_entry=None;
                    alert_type_lookup=alert_type_base
                    if alert_type_base == 'BB_BREAKOUT_UP' or alert_type_base == 'BB_BREAKOUT_DOWN': alert_type_lookup='BB_BREAKOUT'
                    if alert_type_lookup in alert_type_map: map_entry=alert_type_map[alert_type_lookup]
                    if map_entry:
                        col_index=map_entry['index'];
                        value_to_set=map_entry['value']
                        if alert_type_base == 'BB_BREAKOUT_UP':
                            value_to_set=2
                        elif alert_type_base == 'BB_BREAKOUT_DOWN':
                            value_to_set=1
                        elif alert_type_base == 'POLICY_DIVERGENCE':
                            value_to_set=2 if alert.get ('value') == 'H > D' else 1
                        elif alert_type_base == 'CARRY_OPPORTUNITY':
                            value_to_set=2
                        elif alert_type_base == 'ECONOMIC_EVENT':
                            value_to_set=1
                        heatmap_data[r,col_index]=max (heatmap_data[r,col_index],value_to_set)
        if heatmap_data.size > 0:
            cmap=LinearSegmentedColormap.from_list ('alerts',['#444444','#CC3333','#33CC33'],N=3);
            ax2.imshow (heatmap_data,cmap=cmap,aspect='auto',vmin=0,vmax=2,interpolation='nearest')
            ax2.set_yticks (np.arange (len (y_labels)));
            ax2.set_yticklabels (y_labels,fontsize=9);
            ax2.set_xticks (np.arange (len (alert_cols)));
            ax2.set_xticklabels (alert_labels,fontsize=9)
            for r in range (len (y_labels)):
                for c in range (len (alert_cols)):
                    value=heatmap_data[r,c]
                    if value == 1:
                        ax2.text (c,r,"⚠️",ha="center",va="center",color="white",fontsize=10)
                    elif value == 2:
                        ax2.text (c,r,"✓",ha="center",va="center",color="white",fontsize=10)
        else:
            ax2.text (0.5,0.5,"No Alert Data Available",transform=ax2.transAxes,ha='center',va='center',color='gray')

        # === AXIS 3: Key Metrics Bar Plot ===
        ax3.clear ();
        ax3.set_facecolor ('#181818');
        ax3.set_title ("Key Metrics for Top Recommendations",fontsize=14,color='white')
        ax3.grid (True,axis='y',linestyle=':',alpha=0.4,color='#555555');
        ax3.tick_params (colors='white')
        top_pairs_for_bars=[];
        if self.last_recommendations is not None and not self.last_recommendations.empty: top_pairs_for_bars=self.last_recommendations.index.tolist ()[
                                                                                                             :min (3,
                                                                                                                   len (
                                                                                                                       self.last_recommendations))]
        if metrics_df is not None and not metrics_df.empty and top_pairs_for_bars:
            metrics_to_plot={'Centrality':'eigenvector','Momentum':'momentum','Carry Score':'carry_score',
                             'Stability':'inv_volatility'}
            plot_metric_data={};
            actual_metrics_plotted=[]
            for display_name,metric_col in metrics_to_plot.items ():
                if metric_col in metrics_df.columns:
                    valid_pairs=[p for p in top_pairs_for_bars if p in metrics_df.index]
                    if not valid_pairs: continue
                    values=metrics_df.loc[valid_pairs,metric_col].apply (pd.to_numeric,errors='coerce').fillna (
                        0.0).tolist ()
                    if len (values) == len (valid_pairs): plot_metric_data[
                        display_name]=values; actual_metrics_plotted.append (display_name)
            if plot_metric_data and actual_metrics_plotted:
                num_metrics=len (actual_metrics_plotted);
                num_bars=len (top_pairs_for_bars);
                bar_width=0.8 / num_bars;
                x_indices=np.arange (num_metrics);
                plot_pairs=top_pairs_for_bars[:num_bars]
                for i,pair in enumerate (plot_pairs):
                    values_for_pair=[plot_metric_data[metric][i] for metric in actual_metrics_plotted if
                                     i < len (plot_metric_data[metric])]
                    if len (values_for_pair) == num_metrics:
                        offset=bar_width * (i - num_bars / 2.0 + 0.5);
                        bars=ax3.bar (x_indices + offset,values_for_pair,bar_width,label=pair,alpha=0.85)
                ax3.set_ylabel ("Metric Value",color='white');
                tick_pos=x_indices if num_bars == 1 else x_indices + bar_width / 2 * (num_bars % 2 - 1);
                ax3.set_xticks (tick_pos)
                ax3.set_xticklabels (actual_metrics_plotted,fontsize=10);
                ax3.legend (loc='best',fontsize=9,facecolor='#222222',labelcolor='white',framealpha=0.7);
                ax3.margins (x=0.05)
            else:
                ax3.text (0.5,0.5,"Metric Data Error",transform=ax3.transAxes,ha='center',va='center',color='gray')
        else:
            ax3.text (0.5,0.5,"No Recommendations",transform=ax3.transAxes,ha='center',va='center',color='gray')

        # --- Final Touches for Combined Plot ---
        fig.align_ylabels ()
        plt.tight_layout (rect=[0,0.03,1,0.95])
        fig.suptitle (f"MHGNA Forex Trends & Signals - {current_date_str}",fontsize=18,color='white',y=0.99)

        # --- Save Figure ---
        if filename:
            try:
                plt.savefig (filename,dpi=300,bbox_inches='tight',facecolor='#101010')
                print (f"  Combined visualization saved to {filename}")
            except Exception as e:
                print (f"❌ Error saving combined visualization: {e}")
        plt.close (fig)

    '''
    def visualize_forex_network (self,filename: Optional[str] = None,**kwargs):
        """
        Creates and optionally saves a visualization of the Forex dependency network.
        """
        if self.network_graph is None or self.network_graph.number_of_nodes () == 0:
            print ("❌ Cannot visualize network: Graph not available or empty.")
            return
        # Ensure plotting libraries are available
        try:
            # Imports are done locally within methods that use plotting
            # to potentially allow core logic to run without matplotlib/networkx
            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.patches import Patch  # Needed for legend
        except ImportError:
            print ("❌ Matplotlib or NetworkX not installed. Cannot visualize network.")
            return

        print ("Generating Forex network visualization...")

        # --- Prepare Data for Plotting ---
        # Recalculate metrics or ensure they are fresh? Assuming they might be needed.
        metrics_df=self.calculate_forex_metrics ()
        G=self.network_graph.copy ()  # Work on a copy

        # --- Setup Figure ---
        plt.style.use ('dark_background')
        fig,ax=plt.subplots (figsize=self.config.FIGSIZE)
        fig.patch.set_facecolor ('#101010')  # Dark background for figure
        ax.set_facecolor ('#101010')  # Dark background for axes area

        # --- Node Positioning ---
        try:
            # Use weight for spring layout distance (stronger links = closer)
            # Convert weight to distance (e.g., 1/weight, handle potential zero weights)
            valid_edges=[(u,v) for u,v,data in G.edges (data=True) if data.get ('weight',0) > 1e-6]
            if valid_edges:
                distances={(u,v):1.0 / G.edges[u,v].get ('weight',1e-6) for u,v in valid_edges}
                nx.set_edge_attributes (G,values=distances,name='distance')
                pos=nx.spring_layout (G,k=0.4,iterations=75,seed=42,weight='distance')
            else:  # Handle graph with no valid weights or no edges
                pos=nx.spring_layout (G,k=0.4,iterations=75,seed=42)  # Fallback layout
        except Exception as e:
            print (f"  Warning: Spring layout failed ({e}), using random layout.")
            pos=nx.random_layout (G,seed=42)

        # --- Node Attributes (Size, Color) ---
        node_sizes=[]
        node_colors=[]
        default_eigenvector=0.1  # Default score for missing nodes
        default_momentum=0.0

        metrics_present=metrics_df is not None and not metrics_df.empty
        # Use .get() with default for safety when accessing Series/DataFrame index
        eigen_cent=metrics_df['eigenvector'] if metrics_present and 'eigenvector' in metrics_df.columns else pd.Series (
            default_eigenvector,index=G.nodes ())
        node_momentum=metrics_df['momentum'] if metrics_present and 'momentum' in metrics_df.columns else pd.Series (
            default_momentum,index=G.nodes ())

        # Normalize eigenvector centrality (0-1) for scaling
        min_eig,max_eig=eigen_cent.min (),eigen_cent.max ()
        if (max_eig - min_eig) > 1e-9:
            scaled_eigen=(eigen_cent - min_eig) / (max_eig - min_eig)
        else:
            scaled_eigen=pd.Series (0.5,index=eigen_cent.index)  # Mid-value if all same

        # Normalize momentum (-1 to 1 approx) for color mapping
        max_abs_mom=max (abs (node_momentum.min ()),abs (node_momentum.max ()),1e-6)  # Avoid zero division
        norm_momentum=node_momentum / max_abs_mom

        min_node_size,max_node_size=150,1500
        for node in G.nodes ():
            # Use .get() with default value for safety
            size=min_node_size + scaled_eigen.get (node,0.5) * (max_node_size - min_node_size)
            node_sizes.append (size)

            mom=norm_momentum.get (node,0.0)
            # Color map: Red (strong negative) -> Gray (neutral) -> Green (strong positive)
            if mom < -0.1:
                color_val=(0.6 + min (1.0,abs (mom)) * 0.3,0.2,0.2)  # More intense red
            elif mom > 0.1:
                color_val=(0.2,0.6 + min (1.0,mom) * 0.3,0.2)  # More intense green
            else:
                color_val=(0.5,0.5,0.5)  # Gray
            node_colors.append (color_val)

        # --- Edge Attributes (Width, Alpha) ---
        edge_weights=[G.edges[u,v].get ('weight',0.0) for u,v in G.edges ()]  # Default 0 if missing
        max_weight=max (edge_weights) if edge_weights else 1.0
        if max_weight < 1e-6: max_weight=1.0
        edge_widths=[0.5 + (w / max_weight * 5.5) for w in edge_weights]
        edge_alphas=[0.2 + (w / max_weight * 0.5) for w in edge_weights]

        # --- Drawing ---
        if G.number_of_edges () > 0:
            nx.draw_networkx_edges (G,pos,width=edge_widths,alpha=edge_alphas,edge_color='#777777',ax=ax)
        nx.draw_networkx_nodes (G,pos,node_size=node_sizes,node_color=node_colors,alpha=0.9,ax=ax,
                                linewidths=0.5,edgecolors='#DDDDDD')

        # --- Labels for Important Nodes ---
        labels_to_draw={}
        nodes_to_label=set ()
        # Use .get() with default empty list/Series for safety
        if self.last_recommendations is not None: nodes_to_label.update (self.last_recommendations.index[:5].tolist ())
        if metrics_present and 'eigenvector' in metrics_df.columns: nodes_to_label.update (
            metrics_df.get ('eigenvector',pd.Series ()).nlargest (5).index.tolist ())
        if metrics_present and 'momentum' in metrics_df.columns: nodes_to_label.update (
            metrics_df.get ('momentum',pd.Series ()).abs ().nlargest (5).index.tolist ())

        for node in G.nodes ():
            if node in nodes_to_label: labels_to_draw[node]=node

        if labels_to_draw:
            nx.draw_networkx_labels (G,pos,labels=labels_to_draw,font_size=9,font_weight='normal',font_color='white',
                                     ax=ax,
                                     bbox=dict (facecolor='#333333',alpha=0.7,boxstyle='round,pad=0.2',
                                                edgecolor='none'))

        # --- Titles and Annotations ---
        date_str=(self.current_date or datetime.datetime.now ()).strftime ('%Y-%m-%d')
        ax.set_title (f"Forex Pair Dependency Network - {date_str}",fontsize=16,pad=15,color='white')

        # Legend
        red_patch=Patch (color=(0.9,0.2,0.2),label='Negative Momentum')
        green_patch=Patch (color=(0.2,0.9,0.2),label='Positive Momentum')
        gray_patch=Patch (color=(0.5,0.5,0.5),label='Neutral Momentum')
        ax.legend (handles=[red_patch,gray_patch,green_patch],loc='upper right',fontsize=9,facecolor='#222222',
                   labelcolor='white',framealpha=0.7)

        # Info text boxes
        info_bbox=dict (facecolor='#282828',alpha=0.8,boxstyle='round,pad=0.5',edgecolor='#555555')
        ax.text (0.01,0.01,f"Nodes: {G.number_of_nodes ()}\nEdges: {G.number_of_edges ()}",transform=ax.transAxes,
                 color='white',fontsize=9,verticalalignment='bottom',bbox=info_bbox)

        ax.text (0.5,0.01,"Size=Centrality | Color=Momentum | Width=Connection Strength",transform=ax.transAxes,
                 ha='center',fontsize=10,color='white',verticalalignment='bottom',bbox=info_bbox)

        # Top Rate Diffs
        if hasattr (self.macro,'data') and 'interest_differentials' in self.macro.data:
            diffs=self.macro.data['interest_differentials']
            if diffs:
                # Use pairs from graph nodes for sorting diffs
                sorted_diffs=sorted (
                    [(p,diffs.get (p,0.0)) for p in G.nodes () if p in diffs],
                    key=lambda item:abs (item[1]),
                    reverse=True
                )[:3]
                if sorted_diffs:
                    diff_text="Top Rate Diffs:\n" + "\n".join ([f"{p}: {d:+.2f}%" for p,d in sorted_diffs])
                    ax.text (0.99,0.01,diff_text,transform=ax.transAxes,
                             color='white',fontsize=9,verticalalignment='bottom',horizontalalignment='right',
                             bbox=info_bbox)

        # --- Final Touches ---
        plt.axis ('off')
        fig.tight_layout (pad=0.5)  # Use figure's tight_layout

        # --- Save Figure ---
        if filename:
            try:
                plt.savefig (filename,dpi=300,bbox_inches='tight',facecolor='#101010')
                print (f"  Network visualization saved to {filename}")
            except Exception as e:
                print (f"❌ Error saving network visualization: {e}")

        plt.close (fig)  # Close figure to free memory

    def visualize_forex_trends (self,display_pairs: List[str],filename: Optional[str] = None,**kwargs):
        """
        Creates visualizations of price trends and momentum for selected Forex pairs.
        (Version refactorisée)
        """
        if self.data is None:
            print ("❌ Cannot visualize trends: Price data not available.")
            return
        display_pairs=[p for p in display_pairs if p in self.data.columns]  # Filter available
        if not display_pairs:
            print ("❌ Cannot visualize trends: No valid pairs specified or available.")
            return
        # Ensure plotting libraries are available
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print ("❌ Matplotlib not installed. Cannot visualize trends.")
            return

        print (f"Generating Price/Momentum trends visualization for: {', '.join (display_pairs)}...")

        # --- Setup Figure and Subplots ---
        plt.style.use ('dark_background')
        fig,(ax1,ax2)=plt.subplots (
            2,1,figsize=(14,10),sharex=True,  # Share x-axis
            gridspec_kw={'height_ratios':[3,2]}  # More space for price
        )
        fig.patch.set_facecolor ('#101010')  # Dark background for figure too
        current_date_str=(self.current_date or datetime.datetime.now ()).strftime ('%Y-%m-%d')

        # --- Plotting Data Preparation ---
        plot_data=self.data.iloc[-90:].copy ()  # Last 90 days
        plot_momentum=self.momentum.iloc[-90:].copy () if self.momentum is not None else None

        # --- AXIS 1: Normalized Price Trends ---
        ax1.clear ()
        ax1.set_facecolor ('#181818')
        ax1.set_title ("Normalized Price Trends (Last 90 Days)",fontsize=14,color='white')
        ax1.set_ylabel ("Price (Normalized to 100)",color='white')
        ax1.grid (True,linestyle=':',alpha=0.4,color='#555555')
        ax1.tick_params (axis='y',colors='white')
        ax1.tick_params (axis='x',labelbottom=False)  # Hide x-labels

        normalized_prices=pd.DataFrame (index=plot_data.index)
        for pair in display_pairs:
            if pair in plot_data.columns:
                series=plot_data[pair].dropna ()
                if len (series) > 0:
                    start_value=series.iloc[0]
                    if start_value and start_value != 0:
                        normalized_prices[pair]=(series / start_value) * 100
                    else:
                        normalized_prices[pair]=np.nan  # Avoid division by zero

        # Plot prices
        for pair in display_pairs:
            if pair in normalized_prices.columns and not normalized_prices[pair].isnull ().all ():
                rank=None
                if self.last_recommendations is not None and pair in self.last_recommendations.index:
                    try:  # Safer way to get rank
                        rank=self.last_recommendations.index.get_loc (pair) + 1
                    except KeyError:
                        pass

                linewidth=2.5 if rank == 1 else 2.0 if rank in [2,3] else 1.5
                alpha=1.0 if rank is not None and rank <= 3 else 0.75
                linestyle='-'
                label=f"{pair} (Rank {rank})" if rank is not None else pair
                ax1.plot (normalized_prices.index,normalized_prices[pair],
                          linewidth=linewidth,alpha=alpha,label=label,linestyle=linestyle)

        if any (ax1.get_legend_handles_labels ()):  # Show legend only if there are labels
            ax1.legend (loc='best',fontsize=9,facecolor='#222222',labelcolor='white',framealpha=0.7)

        # Add event lines
        if hasattr (self.macro,'data') and 'economic_calendar' in self.macro.data:
            plot_start_date=plot_data.index.min ().tz_localize (
                None) if plot_data.index.tz is not None else plot_data.index.min ()
            plot_end_date=plot_data.index.max ().tz_localize (
                None) if plot_data.index.tz is not None else plot_data.index.max ()
            for event in self.macro.data.get ('economic_calendar',[]):
                try:
                    event_date_naive=pd.to_datetime (event.get ('date')).tz_localize (None)
                    if plot_start_date <= event_date_naive <= plot_end_date and event.get ('impact') == 'high':
                        currency=event.get ('currency')
                        if currency and any (
                                p.startswith (f"{currency}/") or p.endswith (f"/{currency}") for p in display_pairs):
                            ax1.axvline (x=event_date_naive,color='yellow',alpha=0.3,linestyle='--',linewidth=1)
                except Exception:
                    continue  # Ignore errors parsing events

        # --- AXIS 2: Momentum ---
        ax2.clear ()
        ax2.set_facecolor ('#181818')
        ax2.set_title ("Composite Momentum Score",fontsize=14,color='white')
        ax2.set_ylabel ("Momentum Score",color='white')
        ax2.grid (True,linestyle=':',alpha=0.4,color='#555555')
        ax2.tick_params (axis='both',colors='white')
        ax2.axhline (0,color='white',linestyle='--',linewidth=0.7,alpha=0.5)

        if plot_momentum is not None:
            plot_momentum_filtered=plot_momentum[
                [p for p in display_pairs if p in plot_momentum.columns]].copy ()  # Select only needed columns
            if not plot_momentum_filtered.empty:
                for pair in plot_momentum_filtered.columns:
                    rank=None
                    if self.last_recommendations is not None and pair in self.last_recommendations.index:
                        try:
                            rank=self.last_recommendations.index.get_loc (pair) + 1
                        except KeyError:
                            pass
                    linewidth=2.0 if rank == 1 else 1.5 if rank in [2,3] else 1.0
                    alpha=1.0 if rank is not None and rank <= 3 else 0.6
                    ax2.plot (plot_momentum_filtered.index,plot_momentum_filtered[pair],
                              linewidth=linewidth,alpha=alpha,label=f"{pair} Mom.")
                # Optional legend: ax2.legend(loc='best', fontsize=8)
            else:
                ax2.text (0.5,0.5,"Momentum Data Not Available for Selected Pairs",transform=ax2.transAxes,ha='center',
                          va='center',color='gray')
        else:
            ax2.text (0.5,0.5,"Momentum Data Not Available",transform=ax2.transAxes,ha='center',va='center',
                      color='gray')

        # --- Final Touches ---
        ax2.xaxis.set_major_formatter (mdates.DateFormatter ('%b %d, %Y'))
        ax2.xaxis.set_major_locator (mdates.MonthLocator (interval=1))
        plt.setp (ax2.xaxis.get_majorticklabels (),rotation=30,ha='right')

        fig.align_ylabels ()
        plt.tight_layout (rect=[0,0.03,1,0.95])
        fig.suptitle (f"MHGNA Forex Price & Momentum Trends - {current_date_str}",fontsize=18,color='white',y=0.99)

        # --- Save Figure ---
        if filename:
            try:
                plt.savefig (filename,dpi=300,bbox_inches='tight',facecolor='#101010')
                print (f"  Price/Momentum trends visualization saved to {filename}")
            except Exception as e:
                print (f"❌ Error saving price/momentum trends visualization: {e}")

        plt.close (fig)  # Close figure

    def visualize_alert_heatmap (self,display_pairs: List[str],filename: Optional[str] = None,**kwargs):
        """
        Creates a heatmap visualization of current alerts for selected Forex pairs.
        (Nouvelle fonction séparée)
        """
        display_pairs=[p for p in display_pairs if p in self.data.columns] if self.data is not None else []
        if not display_pairs: print ("❌ Cannot visualize heatmap: No valid pairs available or specified."); return
        # Ensure plotting libraries are available
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
        except ImportError:
            print ("❌ Matplotlib or Numpy not installed. Cannot visualize heatmap.")
            return

        print (f"Generating alert heatmap visualization for: {', '.join (display_pairs)}...")

        if not self.alerts: self.check_forex_alerts ()  # Ensure alerts are fresh

        # --- Setup Figure ---
        plt.style.use ('dark_background')
        fig_height=max (4,len (display_pairs) * 0.5)
        fig,ax=plt.subplots (figsize=(10,fig_height))
        fig.patch.set_facecolor ('#101010')
        ax.set_facecolor ('#181818')
        ax.set_title ("Current Alert Signals",fontsize=14,color='white')
        ax.tick_params (colors='white',top=False,bottom=True,labeltop=False,labelbottom=True,length=0)
        plt.setp (ax.get_xticklabels (),rotation=0,ha='center')

        # --- Prepare Heatmap Data ---
        alert_type_map={
            'DRAWDOWN':{'index':0,'value':1,'label':'Drawdown'},
            'HIGH_VOLATILITY':{'index':1,'value':1,'label':'High Vol'},
            'BB_BREAKOUT':{'index':2,'value':-1,'label':'BB Break'},  # Placeholder value
            'POLICY_DIVERGENCE':{'index':3,'value':2,'label':'Policy Div.'},  # Treat as opportunity/info
            'CARRY_OPPORTUNITY':{'index':4,'value':2,'label':'Carry Opp.'},
            'ECONOMIC_EVENT':{'index':5,'value':1,'label':'Event'}  # Treat as warning/info
        }
        # Create labels in the correct order based on index
        alert_cols=sorted (list (set (d['index'] for d in alert_type_map.values ())))
        alert_labels=[""] * len (alert_cols)
        for d in alert_type_map.values ():
            if d['index'] < len (alert_labels):  # Basic check
                alert_labels[d['index']]=d['label']

        heatmap_data=np.zeros ((len (display_pairs),len (alert_cols)))
        y_labels=display_pairs

        for r,pair in enumerate (display_pairs):
            for alert in self.alerts:
                alert_asset=alert.get ('asset')
                alert_type_base=alert.get ('type')
                alert_applies=False

                # Check if alert applies to this pair
                if alert_asset == pair:
                    alert_applies=True
                elif alert_type_base == 'ECONOMIC_EVENT' and alert_asset:  # Event alerts are by currency
                    if pair.startswith (f"{alert_asset}/") or pair.endswith (f"/{alert_asset}"):
                        alert_applies=True

                if alert_applies:
                    map_entry=None
                    # Handle subtypes explicitly for mapping
                    if alert_type_base == 'BB_BREAKOUT_UP' or alert_type_base == 'BB_BREAKOUT_DOWN':
                        alert_type_lookup='BB_BREAKOUT'
                    else:
                        alert_type_lookup=alert_type_base

                    if alert_type_lookup in alert_type_map:
                        map_entry=alert_type_map[alert_type_lookup]

                    if map_entry:
                        col_index=map_entry['index']
                        value_to_set=map_entry['value']  # Default value from map

                        # Adjust value based on specific alert subtypes or values
                        if alert_type_base == 'BB_BREAKOUT_UP':
                            value_to_set=2  # Positive
                        elif alert_type_base == 'BB_BREAKOUT_DOWN':
                            value_to_set=1  # Negative
                        elif alert_type_base == 'POLICY_DIVERGENCE':
                            # Assign 2 for H>D (potentially positive), 1 for D>H (potentially negative)
                            value_to_set=2 if alert.get ('value') == 'H > D' else 1
                        elif alert_type_base == 'CARRY_OPPORTUNITY':
                            value_to_set=2  # Treat any significant carry opp as "positive" signal
                        elif alert_type_base == 'ECONOMIC_EVENT':
                            value_to_set=1  # Treat as warning/info

                        # Use max to prioritize meaningful signals (1 or 2) over default (0)
                        # If multiple alerts map to the same cell, this takes the 'strongest' signal type
                        heatmap_data[r,col_index]=max (heatmap_data[r,col_index],value_to_set)

        # --- Plot Heatmap ---
        if heatmap_data.size > 0:
            cmap=LinearSegmentedColormap.from_list ('alerts',['#444444','#CC3333','#33CC33'],N=3)  # Gray, Red, Green
            ax.imshow (heatmap_data,cmap=cmap,aspect='auto',vmin=0,vmax=2,interpolation='nearest')
            ax.set_yticks (np.arange (len (y_labels)))
            ax.set_yticklabels (y_labels,fontsize=9)
            ax.set_xticks (np.arange (len (alert_cols)))
            ax.set_xticklabels (alert_labels,fontsize=9)
            # Add text annotations (symbols)
            for r in range (len (y_labels)):
                for c in range (len (alert_cols)):
                    value=heatmap_data[r,c]
                    text_symbol=""
                    text_color="white"
                    if value == 1:
                        text_symbol="⚠️"  # Warning/Negative
                    elif value == 2:
                        text_symbol="✓"  # Opportunity/Positive

                    if text_symbol:
                        ax.text (c,r,text_symbol,ha="center",va="center",color=text_color,fontsize=10)
        else:
            ax.text (0.5,0.5,"No Alert Data Available",transform=ax.transAxes,ha='center',va='center',color='gray')

        plt.tight_layout (pad=0.5)
        if filename:
            try:
                plt.savefig (filename,dpi=150,bbox_inches='tight',facecolor='#101010')
                print (f"  Alert heatmap visualization saved to {filename}")
            except Exception as e:
                print (f"❌ Error saving alert heatmap visualization: {e}")
        plt.close (fig)  # Close figure

    def visualize_recommendation_metrics (self,filename: Optional[str] = None,**kwargs):
        """
        Creates a bar plot comparing key metrics for top recommended Forex pairs.
        (Nouvelle fonction séparée)
        """
        if self.last_recommendations is None or self.last_recommendations.empty:
            print ("❌ Cannot visualize metrics: No recommendations available.")
            return
        # Ensure plotting libraries are available
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print ("❌ Matplotlib or Numpy not installed. Cannot visualize metrics.")
            return

        print ("Generating recommendation metrics visualization...")

        # --- Prepare Data ---
        # Recommendations are already formatted, need raw metrics
        metrics_df=self.calculate_forex_metrics ()
        if metrics_df is None or metrics_df.empty:
            print ("  Warning: Failed to get raw metrics for visualization.")
            return

        # Get top pairs from the formatted recommendations index
        top_pairs_for_bars=self.last_recommendations.index.tolist ()
        num_bars=len (top_pairs_for_bars)
        if num_bars == 0: return

        metrics_to_plot={  # Display Name: Raw Metric Column Name
            'Centrality':'eigenvector',
            'Momentum':'momentum',
            'Carry Score':'carry_score',
            'Stability':'inv_volatility',
            'Real Rate':'real_rate'  # Add real rate?
        }
        plot_metric_data={}  # Dict: {Display Name: [value1, value2, ...]}
        actual_metrics_plotted=[]  # List of display names actually plotted

        # Extract numerical data for top pairs and selected metrics from metrics_df
        for display_name,metric_col in metrics_to_plot.items ():
            if metric_col in metrics_df.columns:
                # Get values for the recommended pairs, handle missing pairs
                values=metrics_df.reindex (top_pairs_for_bars)[metric_col].apply (pd.to_numeric,
                                                                                  errors='coerce').fillna (
                    0.0).tolist ()
                # Check if we got values for all requested pairs
                if len (values) == num_bars:
                    plot_metric_data[display_name]=values
                    actual_metrics_plotted.append (display_name)
                else:
                    print (
                        f"  Warning: Mismatch in metric data for {display_name}. Expected {num_bars}, got {len (values)}")

        if not plot_metric_data or not actual_metrics_plotted:
            print ("  No valid metric data found for recommended pairs visualization.")
            return

        # --- Setup Figure ---
        plt.style.use ('dark_background')
        # Adjust height based on number of pairs
        fig_height=max (5,num_bars * 1.5 + 1)
        fig,ax=plt.subplots (figsize=(10,fig_height))
        fig.patch.set_facecolor ('#101010')
        ax.set_facecolor ('#181818')
        ax.set_title ("Key Metrics Comparison for Top Recommendations",fontsize=14,color='white')
        ax.grid (True,axis='y',linestyle=':',alpha=0.4,color='#555555')  # Horizontal grid lines
        ax.tick_params (colors='white')

        # --- Plot Bars ---
        num_metrics=len (actual_metrics_plotted)
        bar_total_width=0.8  # Total width allocated for bars per metric
        bar_width=bar_total_width / num_bars
        x_indices=np.arange (num_metrics)  # X locations for the groups (metrics)

        for i,pair in enumerate (top_pairs_for_bars):
            # Get values for this pair across all plotted metrics
            values_for_pair=[plot_metric_data[metric][i] for metric in actual_metrics_plotted]
            # Calculate offset for this pair's bars within the group
            offset=bar_width * (i - num_bars / 2.0 + 0.5)
            bars=ax.bar (x_indices + offset,values_for_pair,bar_width,label=pair,alpha=0.9)
            # Optional: Add value labels on bars
            # ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8, color='lightgrey')

        ax.set_ylabel ("Metric Value (Scaled/Raw)",color='white')
        # Center ticks between groups if multiple bars, otherwise center on bar
        tick_pos=x_indices if num_bars == 1 else x_indices + bar_total_width / 2 - bar_width / 2
        ax.set_xticks (tick_pos)
        ax.set_xticklabels (actual_metrics_plotted,fontsize=10)
        ax.legend (loc='best',fontsize=9,facecolor='#222222',labelcolor='white',framealpha=0.7)
        ax.margins (x=0.05)  # Add some padding on x-axis

        plt.tight_layout (pad=0.5)
        if filename:
            try:
                plt.savefig (filename,dpi=150,bbox_inches='tight',facecolor='#101010')
                print (f"  Recommendation metrics visualization saved to {filename}")
            except Exception as e:
                print (f"❌ Error saving recommendation metrics visualization: {e}")
        plt.close (fig)  # Close figure

    # --- Mise à jour de generate_forex_report ---
    def generate_forex_report (self,output_folder: str = 'forex_reports') -> Optional[str]:
        """
        Generates a comprehensive text report of the Forex analysis.
        Calls separate visualization functions. (Version refactorisée)
        """
        print (f"\nGenerating Forex report in folder: '{output_folder}'...")
        # --- Setup ---
        if not os.path.exists (output_folder):
            try:
                os.makedirs (output_folder); print (f"Created report folder: '{output_folder}'")
            except OSError as e:
                print (f"❌ Error creating report folder '{output_folder}': {e}."); output_folder=None
        report_dt=self.current_date or datetime.datetime.now ();
        report_id=f"mhgna_forex_{report_dt.strftime ('%Y%m%d_%H%M%S')}"
        report_file_path=os.path.join (output_folder,f"{report_id}_report.txt") if output_folder else None

        # --- Generate Visualizations (Appelle les NOUVELLES fonctions) ---
        network_file_path,trends_file_path,heatmap_file_path,metrics_file_path=None,None,None,None

        # Determine pairs for visualization (safer check for self.data)
        display_pairs_for_vis=[]
        if self.data is not None:
            if self.last_recommendations is not None and not self.last_recommendations.empty:
                display_pairs_for_vis=self.last_recommendations.index.tolist ()[:5]
            else:
                display_pairs_for_vis=self.forex_pairs[:min (5,len (self.forex_pairs))]
            # Add baseline pair if available
            eurusd_std='EUR/USD'
            if eurusd_std not in display_pairs_for_vis and eurusd_std in self.data.columns:
                display_pairs_for_vis.append (eurusd_std)
            # Final filter based on actual data columns
            display_pairs_for_vis=[p for p in display_pairs_for_vis if p in self.data.columns]

        if output_folder:
            network_file_path=os.path.join (output_folder,f"{report_id}_1_network.png")
            trends_file_path=os.path.join (output_folder,f"{report_id}_2_trends.png")
            heatmap_file_path=os.path.join (output_folder,f"{report_id}_3_alerts.png")
            metrics_file_path=os.path.join (output_folder,f"{report_id}_4_metrics.png")

            print ("  Generating visualizations...")
            try:
                self.visualize_forex_network (filename=network_file_path)
            except Exception as e:
                print (f"⚠️ Failed: network vis: {e}"); network_file_path="Failed"

            if display_pairs_for_vis:  # Only call if pairs exist
                try:
                    self.visualize_forex_trends (display_pairs=display_pairs_for_vis,filename=trends_file_path)
                except Exception as e:
                    print (f"⚠️ Failed: trends vis: {e}"); trends_file_path="Failed"
                try:
                    self.visualize_alert_heatmap (display_pairs=display_pairs_for_vis,filename=heatmap_file_path)
                except Exception as e:
                    print (f"⚠️ Failed: alert heatmap: {e}"); heatmap_file_path="Failed"
            else:
                trends_file_path,heatmap_file_path="No pairs","No pairs"

            # Only call metrics plot if recommendations exist
            if self.last_recommendations is not None and not self.last_recommendations.empty:
                try:
                    self.visualize_recommendation_metrics (filename=metrics_file_path)
                except Exception as e:
                    print (f"⚠️ Failed: metrics bar plot: {e}"); metrics_file_path="Failed"
            else:
                metrics_file_path="No recs"
        else:
            print ("  Skipping visualization generation (no output folder).")

        # --- Build Report Text ---
        # ... (Le reste du code pour construire report_lines est identique à la version précédente) ...
        # ... (Sections 1, 2, 3, 4, 5) ...
        # ... (Section 6 Visualizations Reference mise à jour pour les 4 fichiers) ...
        # ... (Footer) ...
        # ... (Sauvegarde du fichier texte) ...
        report_lines=[]
        report_lines.append ("=" * 60);
        report_lines.append (f" MHGNA FOREX ANALYSIS REPORT - {report_dt.strftime ('%Y-%m-%d %H:%M')}");
        report_lines.append ("=" * 60)
        report_lines.append ("\n--- 1. TOP FOREX PAIR RECOMMENDATIONS ---")
        if self.last_recommendations is None: self.recommend_forex_pairs ()
        if self.last_recommendations is not None and not self.last_recommendations.empty:
            try:
                report_lines.append (self.last_recommendations.to_string ())
            except Exception as e:
                report_lines.append (f"  Error formatting: {e}")
            report_lines.append ("\n  Interpretation Notes:");
            report_lines.append ("  - Score: Overall attractiveness.");
            report_lines.append ("  - Direction: Suggested direction.");
            report_lines.append ("  - Centrality: Importance in network.");
            report_lines.append ("  - Momentum: Trend strength.");
            report_lines.append ("  - Stability: Inverse volatility.");
            report_lines.append ("  - Carry Score: Risk-adj. rate diff.");
            report_lines.append ("  - Real Rate Diff: Interest diff - inflation diff.");
            report_lines.append ("  - Optimal Session: Highest simulated activity.")
        else:
            report_lines.append ("\n  No recommendations available.")
        report_lines.append ("\n--- 2. ACTIVE ALERTS ---")
        if not self.alerts: self.check_forex_alerts ()
        if self.alerts:
            alerts_by_type: Dict[str,list]={};
            [alerts_by_type.setdefault (a['type'],[]).append (a) for a in self.alerts]
            for alert_type,type_alerts in sorted (alerts_by_type.items ()):
                report_lines.append (f"\n  * {alert_type.replace ('_',' ').title ()} ({len (type_alerts)}):");
                [report_lines.append (f"    - {a['message']}") for a in type_alerts[:5]];
                if len (type_alerts) > 5: report_lines.append ("    - ... (and more)")
        else:
            report_lines.append ("\n  No active alerts detected.")
        report_lines.append ("\n--- 3. MACROECONOMIC ANALYSIS ---");
        if hasattr (self.macro,'data') and self.macro.data:
            macro_data_available=False
            if 'interest_rates' in self.macro.data and self.macro.data['interest_rates']:
                report_lines.append ("\n  Interest Rates:");
                sorted_rates=sorted (self.macro.data['interest_rates'].items (),key=lambda item:item[1],reverse=True);
                [report_lines.append (f"    - {c}: {r:.2f}%") for c,r in sorted_rates];
                macro_data_available=True
            if 'interest_differentials' in self.macro.data and self.macro.data['interest_differentials']:
                diffs=self.macro.data['interest_differentials'];
                sorted_diffs=sorted ([(p,d) for p,d in diffs.items () if p in self.forex_pairs],
                                     key=lambda item:abs (item[1]),reverse=True)[
                             :5]  # Filter diffs by current forex_pairs
                if sorted_diffs: report_lines.append ("\n  Most Significant Rate Differentials:"); [
                    report_lines.append (f"    - {p}: {d:+.2f}%") for p,d in sorted_diffs]; macro_data_available=True
            if 'economic_calendar' in self.macro.data and self.macro.data['economic_calendar']:
                try:
                    today_dt=self.current_date or datetime.datetime.now ();
                    today_str=today_dt.strftime ('%Y-%m-%d');
                    limit_dt=today_dt + datetime.timedelta (days=7);
                    limit_str=limit_dt.strftime ('%Y-%m-%d')
                    coming_events=[e for e in self.macro.data.get ('economic_calendar',[]) if
                                   e.get ('date') and today_str <= e.get ('date') <= limit_str and e.get (
                                       'impact') == 'high']
                    if coming_events: report_lines.append ("\n  Upcoming High-Impact Events (7d):"); [
                        report_lines.append (
                            f"    - {e.get ('date')} {e.get ('time','')} - {e.get ('currency','N/A')}: {e.get ('name','N/A')}")
                        for e in sorted (coming_events,key=lambda x:x.get ('date'))[:5]]; macro_data_available=True
                except Exception as e:
                    print (f"    Warn: Error processing calendar: {e}")
            if not macro_data_available: report_lines.append ("\n  No relevant macro data.")
        else:
            report_lines.append ("\n  Macro data unavailable.")
        report_lines.append ("\n--- 4. MARKET ANALYSIS BY SESSION (SIMULATED) ---");
        if hasattr (self,'session_states') and self.session_states:
            session_data_found=False;
            session_name_map={'asia':'Asia','europe':'Europe','america':'America'}
            for session_code,state_data in self.session_states.items ():
                session_name=session_name_map.get (session_code,session_code);
                report_lines.append (f"\n  {session_name}:")
                if state_data:
                    pairs_by_weight=sorted (
                        [(p,d.get ('weight',0)) for p,d in state_data.items () if p in self.forex_pairs],
                        key=lambda item:item[1],reverse=True)  # Filter by current pairs
                    if pairs_by_weight:
                        [report_lines.append (f"    - {p}: W={w:.2f}") for p,w in
                         pairs_by_weight[:3]]; session_data_found=True
                    else:
                        report_lines.append ("    - No pair data.")
                else:
                    report_lines.append ("    - No pair data.")
            if not session_data_found: report_lines.append ("\n  No session data.")
        else:
            report_lines.append ("\n  Session data unavailable.")
        report_lines.append ("\n--- 5. RECOMMENDED ACTIONS ---");
        actions_generated=False
        if self.last_recommendations is not None and not self.last_recommendations.empty:
            top_recs=[];
            max_recs=min (3,len (self.last_recommendations));
            for i in range (max_recs): pair=self.last_recommendations.index[i]; d=self.last_recommendations.iloc[
                i].get ('Direction','N/A'); top_recs.append (f"{pair}({d})")
            if top_recs: report_lines.append (
                f"\n  * Consider positions: {', '.join (top_recs)}."); actions_generated=True
        else:
            report_lines.append ("\n  * Wait for valid recommendations."); actions_generated=True
        alert_actions=[]
        if self.alerts:
            active_pairs=set (self.forex_pairs)  # Use current pairs
            if any (a['type'] == 'DRAWDOWN' and a['asset'] in active_pairs for a in self.alerts): alert_actions.append (
                "Review pairs hitting drawdown thresholds.")
            if any (a['type'] == 'HIGH_VOLATILITY' and a['asset'] in active_pairs for a in
                    self.alerts): alert_actions.append ("Caution: High volatility pairs.")
            carry_opps=[a['asset'] for a in self.alerts if
                        a['type'] == 'CARRY_OPPORTUNITY' and a['asset'] in active_pairs and a.get ('value',0) > 0]
            neg_carry=[a['asset'] for a in self.alerts if
                       a['type'] == 'CARRY_OPPORTUNITY' and a['asset'] in active_pairs and a.get ('value',0) < 0]
            if carry_opps: alert_actions.append (
                f"Explore Pos Carry: {', '.join (sorted (list (set (carry_opps)))[:3])}...")
            if neg_carry: alert_actions.append (f"Note Neg Carry: {', '.join (sorted (list (set (neg_carry)))[:3])}...")
            if any (a['type'] == 'POLICY_DIVERGENCE' for a in self.alerts): alert_actions.append (
                "Monitor policy divergence pairs.")
            if any (a['type'] == 'ECONOMIC_EVENT' for a in self.alerts): alert_actions.append (
                "Prepare for event volatility.")
            if any (a['type'].startswith ('BB_BREAKOUT') and a['asset'] in active_pairs for a in
                    self.alerts): alert_actions.append ("Note BB breakouts.")
        if alert_actions: actions_generated=True; report_lines.extend ([f"  * {action}" for action in alert_actions])
        if not actions_generated: report_lines.append ("\n  * Monitor market; no specific actions triggered.")

        # --- 6. Visualizations Reference --- (Updated for multiple files)
        report_lines.append ("\n--- 6. VISUALIZATIONS ---")
        if output_folder:
            vis_files={"Network Graph":network_file_path,"Price/Momentum Trends":trends_file_path,
                       "Alert Heatmap":heatmap_file_path,"Recommendation Metrics":metrics_file_path}
            for name,path in vis_files.items (): report_lines.append (
                f"\n  - {name}: {os.path.basename (path) if path and 'Failed' not in str (path) and 'pairs' not in str (path) and 'recs' not in str (path) else 'Not generated/failed'}")
        else:
            report_lines.append ("\n  - Visualizations not saved.")
        # --- Footer ---
        report_lines.append ("\n" + "=" * 60);
        report_lines.append ("End of Report");
        report_lines.append ("=" * 60)
        final_report_text="\n".join (report_lines)
        # --- Save Report to File ---
        if report_file_path:
            try:
                with open (report_file_path,'w',encoding='utf-8') as f:
                    f.write (final_report_text)
                print (f"Report text saved successfully to: {report_file_path}")
            except Exception as e:
                print (f"⚠️ Error saving report: {e}")
        return final_report_text
    '''
    # --- Analysis Orchestration ---
    def run_complete_forex_analysis(self, end_date: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], List[Dict[str, Any]], Optional[str]]:
        """
        Executes the complete Forex analysis pipeline: data fetching, macro collection,
        network building, metric calculation, recommendations, alerts, and reporting.

        Args:
            end_date: Optional end date for the analysis (YYYY-MM-DD). Defaults to today.

        Returns:
            A tuple containing:
                - DataFrame of recommendations (or None).
                - List of active alerts.
                - String containing the generated text report (or None).
        """
        start_time = datetime.datetime.now()
        print("=" * 80)
        print(f" MHGNA FOREX - COMPLETE ANALYSIS RUNNER - START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        recommendations: Optional[pd.DataFrame] = None
        alerts: List[Dict[str, Any]] = []
        report_text: Optional[str] = None
        analysis_successful = True
        output_dir = 'forex_reports' # Default, can be overridden by args in __main__

        # --- Step 0: Determine Output Directory ---
        # This logic is primarily for when run via __main__
        global args # Access args if run via __main__
        if 'args' in globals() and hasattr(args, 'output'):
             output_dir = args.output


        # --- Step 1: Fetch Price Data ---
        print("\n[1/5] Fetching Forex price data...")
        fetch_success = self.fetch_forex_data(end_date=end_date)
        if not fetch_success or self.data is None or self.data.empty:
            print("❌ Critical Error: Failed to fetch sufficient price data. Analysis stopped.")
            return None, [], "Error: Failed to fetch price data."
        print(f"✅ Price data fetched successfully ({self.data.shape[0]} days, {self.data.shape[1]} pairs).")

        if self.data.shape[1] < 3:
            print(f"⚠️ Warning: Only {self.data.shape[1]} pairs fetched. Network analysis requires at least 3.")


        # --- Step 2: Collect Macro Data ---
        print("\n[2/5] Collecting macroeconomic data...")
        try:
            self.macro.fetch_interest_rates()
            self.macro.fetch_inflation_data()
            self.macro.calculate_interest_rate_differentials()
            self.macro.calculate_inflation_differentials()
            self.macro.analyze_monetary_policy_bias()
            self.macro.fetch_economic_calendar()
            print("✅ Macro data collected.")
        except Exception as e:
            print(f"⚠️ Warning: Error during macro data collection: {e}. Analysis continues.")


        # --- Step 3: Build Network ---
        print("\n[3/5] Building multi-horizon dependency network...")
        network_built = False
        if self.data.shape[1] < 3:
            print("  Skipping network build: Not enough pairs.")
            self.network_graph = nx.Graph()
            self.network_graph.add_nodes_from(self.data.columns) # Add nodes
        else:
            try:
                self.build_forex_network()
                if self.network_graph is not None and self.network_graph.number_of_edges() > 0:
                    print(f"✅ Network built ({self.network_graph.number_of_nodes()} nodes, {self.network_graph.number_of_edges()} edges).")
                    network_built = True
                elif self.network_graph is not None:
                     print("⚠️ Network built, but contains 0 edges.")
                     network_built = True # Graph exists
                else:
                     print("⚠️ Network build failed or resulted in None.")

            except Exception as e:
                print(f"❌ Error building network: {e}. Analysis continues without network features.")
                import traceback
                traceback.print_exc()


        # --- Step 4: Recommendations & Alerts ---
        print("\n[4/5] Calculating recommendations and checking alerts...")
        if self.network_graph is not None: # Need graph for metrics/recommendations
             try:
                 recommendations = self.recommend_forex_pairs()
                 if recommendations is not None and not recommendations.empty:
                     print(f"✅ Generated {len(recommendations)} recommendations.")
                 elif recommendations is not None:
                     print("  No recommendations generated.")
                 else:
                     print("⚠️ Recommendation generation failed.")
             except Exception as e:
                 print(f"⚠️ Error generating recommendations: {e}")
                 recommendations = None
        else:
             print("⚠️ Skipping recommendations: Network graph not available.")

        try:
            alerts = self.check_forex_alerts()
            print(f"✅ Alert check complete ({len(alerts)} alerts found).")
        except Exception as e:
            print(f"⚠️ Error checking alerts: {e}")
            alerts = []


        # --- Step 5: Generate Report ---
        print("\n[5/5] Generating analysis report...")
        try:
            report_text = self.generate_forex_report(output_folder=output_dir)
            if report_text:
                print(f"✅ Report generated.") # Saving message is inside generate_report
            else:
                 print("⚠️ Report generation returned empty.")
                 analysis_successful = False # Consider this a failure?
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
            report_text = f"Error: Report generation failed.\n{e}"
            analysis_successful = False


        # --- Final Summary ---
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print("\n" + "=" * 80)
        print(f" MHGNA FOREX - ANALYSIS COMPLETE - END: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration})")
        print("=" * 80)

        if analysis_successful:
            print("\n--- Summary ---")
            if recommendations is not None and not recommendations.empty:
                 # Access columns safely using .get() on the row Series
                 rec1_pair = recommendations.index[0]
                 rec1_row = recommendations.iloc[0]
                 rec1_dir = rec1_row.get('Direction', 'N/A')
                 rec1_score = rec1_row.get('Score', 'N/A')
                 print(f"Top Recommendation: {rec1_pair} ({rec1_dir}) - Score: {rec1_score}")
            else:
                 print("Recommendations: None generated.")
            print(f"Active Alerts: {len(alerts)}")
            print(f"Report & Visualizations saved to folder: '{output_dir}' (if specified & successful)")
        else:
            print("\n⚠️ Analysis completed with errors. Please review logs.")

        return recommendations, alerts, report_text


# --- Command-Line Execution ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='MHGNA Forex Analysis Tool - Performs multi-horizon network analysis for Forex pairs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--pairs', type=str, default=','.join(ForexConfig.TICKERS), # Default from config
        help='Comma-separated list of Forex pairs to analyze (e.g., EUR/USD,USD/JPY).'
    )
    parser.add_argument(
        '--lookback', type=int, default=ForexConfig.LOOKBACK_PERIOD_YEARS,
        help='Historical data lookback period in years.'
    )
    parser.add_argument(
        '--recommended', type=int, default=ForexConfig.RECOMMENDED_PAIRS,
        help='Number of top pairs to include in recommendations.'
    )
    parser.add_argument(
        '--output', type=str, default='forex_reports',
        help='Output directory for reports and visualizations.'
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='End date for analysis in YYYY-MM-DD format (defaults to current date).'
    )

    args = parser.parse_args()

    # --- Configure ---
    config = ForexConfig()

    # Override from command line arguments
    config.TICKERS = [pair.strip().upper() for pair in args.pairs.split(',') if pair.strip()]
    if args.lookback > 0: config.LOOKBACK_PERIOD_YEARS = args.lookback
    if args.recommended > 0: config.RECOMMENDED_PAIRS = args.recommended
    # Output folder used directly in run_complete_analysis via args

    print("--- Configuration ---")
    print(f"Pairs to Analyze: {config.TICKERS}")
    print(f"Lookback Period: {config.LOOKBACK_PERIOD_YEARS} years")
    print(f"Num Recommendations: {config.RECOMMENDED_PAIRS}")
    print(f"Output Folder: {args.output}")
    print(f"Analysis End Date: {args.date or 'Today'}")
    print(f"FRED API Key Set: {'Yes' if config.API_KEYS.get('fred') and config.API_KEYS['fred'] != '0363ec0e0208840bc7552afaa843117a' else 'No'}")
    print("-" * 20)


    # --- Initialize & Run ---
    try:
        forex_analyzer = MHGNAForex(config=config)
        # Run the analysis
        forex_analyzer.run_complete_forex_analysis(end_date=args.date)

    except KeyboardInterrupt:
         print("\nAnalysis interrupted by user.")
         sys.exit(1)
    except Exception as e:
         print("\n❌ An unexpected critical error occurred during analysis:")
         import traceback
         traceback.print_exc()
         sys.exit(1)

    # --- Exit ---
    print("\nScript finished.")
    sys.exit(0)

    '''
    # Pour Google Colab, remplacer la section main par celle-ci
    if __name__ == "__main__":
    import sys
    # Filtrer les arguments indésirables de Colab
    filtered_argv = [arg for arg in sys.argv if not arg.startswith('-f') and '.json' not in arg]
    sys.argv = filtered_argv  # Remplacer sys.argv par la version filtrée

    parser = argparse.ArgumentParser(description='MHGNA Forex Analysis Tool')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d, 1h)')
    args = parser.parse_args()

    config = ForexConfig()
    config.DATA_INTERVAL = args.interval

    forex_analyzer = MHGNAForex(config=config)
    if forex_analyzer.fetch_forex_data(interval=args.interval):
        forex_analyzer.recommend_forex_pairs()
        forex_analyzer.create_dashboard()
    else:
        print("Failed to fetch data. Exiting.")
    '''

