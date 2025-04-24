# -*- coding: utf-8 -*-
"""
MHGNA Forex - Multi-Horizon Graphical Network Allocation for Forex Trading
===========================================================================

Adaptation of the MHGNA model specifically for Forex trading, integrating
macroeconomic data, interest rate differentials, and other indicators
specific to the currency market.

Author: [Your Name]
Date: April 2025
Version: 1.0.2 (Grok Ploty Draft)
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

    def __init__ (self,config: ForexConfig = ForexConfig ()):
        self.config=config
        self.api_keys=config.API_KEYS
        self.data: Dict[str,Any]={}
        self.last_updated: Dict[str,str]={}
        if not os.path.exists (self.config.CACHE_DIR):
            try:
                os.makedirs (self.config.CACHE_DIR)
            except OSError as e:
                print (f"Error creating cache directory '{self.config.CACHE_DIR}': {e}")

    def _read_cache (self,cache_file: str,expiry_seconds: int) -> Optional[Any]:
        if os.path.exists (cache_file):
            try:
                cache_mod_time=os.path.getmtime (cache_file)
                cache_age=datetime.datetime.now () - datetime.datetime.fromtimestamp (cache_mod_time)
                if cache_age.total_seconds () < expiry_seconds:
                    with open (cache_file,'r',encoding='utf-8') as f:
                        return json.load (f)
            except (IOError,json.JSONDecodeError,OSError) as e:
                print (f"Error reading cache file '{cache_file}': {e}")
        return None

    def _write_cache (self,cache_file: str,data_to_cache: Any):
        try:
            with open (cache_file,'w',encoding='utf-8') as f:
                json.dump (data_to_cache,f,ensure_ascii=False,indent=4)
        except IOError as e:
            print (f"Error writing cache file '{cache_file}': {e}")

    @lru_cache (maxsize=4)
    def fetch_interest_rates (self,currencies: Optional[List[str]] = None) -> Dict[str,float]:
        target_currencies=currencies or self.config.CURRENCIES
        print (f"Fetching interest rates for {len (target_currencies)} currencies...")
        data_key='interest_rates'
        cache_file=self.config.INTEREST_RATE_CACHE_FILE
        cache_expiry=self.config.INTEREST_RATE_CACHE_EXPIRY_DAYS * 86400

        cached_data=self._read_cache (cache_file,cache_expiry)
        if cached_data:
            cached_rates=cached_data.get ('rates',{})
            if cached_rates:
                self.data[data_key]=cached_rates
                self.last_updated[data_key]=cached_data.get ('timestamp','')
                print (f"Interest rates loaded from cache ({len (self.data[data_key])} currencies).")
                return self.data[data_key]

        interest_rates: Dict[str,float]={}
        fred_api_key=self.api_keys.get ('fred')
        fred_ids: Dict[str,str]={
            'USD':'FEDFUNDS','EUR':'ECBDFR','GBP':'BOEBR','JPY':'BOJDPR',
            'CHF':'SNBPRA','AUD':'RBATCTR','CAD':'BOCWLR','NZD':'RBOKCR',
        }
        if fred_api_key and fred_api_key != 'YOUR_FRED_API_KEY':
            for currency in target_currencies:
                if currency in fred_ids:
                    series_id=fred_ids[currency]
                    url=f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_api_key}&file_type=json&sort_order=desc&limit=1"
                    try:
                        response=requests.get (url,timeout=10)
                        data=response.json ()
                        if 'observations' in data and data['observations']:
                            rate=float (data['observations'][0]['value'])
                            interest_rates[currency]=rate
                    except Exception as e:
                        print (f"Error fetching rate for {currency}: {e}")
        else:
            interest_rates={
                'USD':5.33,'EUR':4.00,'GBP':5.25,'JPY':0.10,
                'CHF':1.50,'AUD':4.35,'CAD':4.75,'NZD':5.50,
                'MXN':11.00,'PLN':5.75
            }
        current_time_str=datetime.datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')
        self._write_cache (cache_file,{'rates':interest_rates,'timestamp':current_time_str})
        self.data[data_key]=interest_rates
        self.last_updated[data_key]=current_time_str
        return interest_rates

    @lru_cache (maxsize=4)
    def fetch_gdp_data (self,countries: Optional[List[str]] = None) -> Dict[str,float]:
        target_countries=countries or [self.config.CURRENCY_TO_COUNTRY.get (c) for c in self.config.CURRENCIES]
        target_countries=sorted (list (set (c for c in target_countries if c)))
        cache_file=os.path.join (self.config.CACHE_DIR,'gdp.json')
        cache_expiry=30 * 86400

        cached_data=self._read_cache (cache_file,cache_expiry)
        if cached_data:
            self.data['gdp']=cached_data.get ('gdp',{})
            print (f"GDP data loaded from cache for {len (self.data['gdp'])} countries.")
            return self.data['gdp']

        gdp_data={}
        fred_api_key=self.api_keys.get ('fred')
        if fred_api_key and fred_api_key != 'YOUR_FRED_API_KEY':
            fred_ids={'US':'GDP','EU':'EUNNGDP','GB':'UKNGDP','JP':'JPNRGDPEXP'}
            for country in target_countries:
                if country in fred_ids:
                    url=f"https://api.stlouisfed.org/fred/series/observations?series_id={fred_ids[country]}&api_key={fred_api_key}&file_type=json&limit=1&sort_order=desc"
                    try:
                        response=requests.get (url,timeout=10)
                        data=response.json ()
                        if 'observations' in data and data['observations']:
                            gdp_data[country]=float (data['observations'][0]['value'])
                    except Exception as e:
                        print (f"Error fetching GDP for {country}: {e}")
        else:
            gdp_data={'US':27.36,'EU':15.81,'GB':3.34,'JP':4.21}
        self._write_cache (cache_file,
                           {'gdp':gdp_data,'timestamp':datetime.datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')})
        self.data['gdp']=gdp_data
        return gdp_data

    def fetch_cot_reports (self) -> Dict[str,float]:
        cache_file=os.path.join (self.config.CACHE_DIR,'cot.json')
        cached_data=self._read_cache (cache_file,7 * 86400)
        if cached_data:
            self.data['cot']=cached_data.get ('cot',{})
            return self.data['cot']
        cot_data={'USD':0.2,'EUR':-0.1}
        self._write_cache (cache_file,
                           {'cot':cot_data,'timestamp':datetime.datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')})
        self.data['cot']=cot_data
        return cot_data



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
    """MHGNA model tailored for Forex trading."""

    def __init__ (self,config: ForexConfig = ForexConfig ()):
        self.config=config
        self.utils=ForexDataUtils ()
        self.macro=MacroDataCollector (self.config)
        self.forex_pairs: List[str]=self.config.TICKERS
        self.data: Optional[pd.DataFrame]=None
        self.ohlc_data: Optional[Dict[str,pd.DataFrame]]=None
        self.returns: Optional[pd.DataFrame]=None
        self.volatility: Optional[pd.DataFrame]=None
        self.momentum: Optional[pd.DataFrame]=None
        self.network_graph: Optional[nx.Graph]=None
        self.last_recommendations: Optional[pd.DataFrame]=None
        self.technicals: Dict[str,Dict[str,pd.Series]]={}
        self._setup_plotting_style ()

    def _setup_plotting_style (self):
        try:
            sns.set_style (self.config.CHART_STYLE)
            plt.rcParams['figure.figsize']=self.config.FIGSIZE
        except ValueError:
            sns.set_style ('darkgrid')
            plt.rcParams['figure.figsize']=self.config.FIGSIZE

    def fetch_forex_data (self,end_date: Optional[str] = None,interval: str = ForexConfig.DATA_INTERVAL) -> bool:
        try:
            end_dt=pd.to_datetime (end_date) if end_date else pd.Timestamp.now ().normalize ()
            if end_dt.tz is not None:
                end_dt=end_dt.tz_localize (None)
        except ValueError as e:
            print (f"Error parsing end_date '{end_date}': {e}. Defaulting to today.")
            end_dt=pd.Timestamp.now ().normalize ()

        start_dt=end_dt - relativedelta (years=self.config.LOOKBACK_PERIOD_YEARS)
        print (
            f"Fetching Forex data from {start_dt.strftime ('%Y-%m-%d')} to {end_dt.strftime ('%Y-%m-%d')} with interval {interval}...")

        yahoo_symbols_map={ticker:self.utils.convert_to_yahoo_forex_symbol (ticker) for ticker in self.forex_pairs}
        yahoo_symbols_list=list (yahoo_symbols_map.values ())

        try:
            data_raw=yf.download (
                tickers=yahoo_symbols_list,
                start=start_dt,
                end=end_dt + datetime.timedelta (days=1),
                interval=interval,
                progress=True,
                group_by='ticker'
            )
        except Exception as e:
            print (f"Error during yfinance download: {e}")
            return False

        if data_raw.empty:
            print ("No data returned from yfinance.")
            return False

        close_prices=pd.DataFrame ()
        ohlc_data_dict={}
        successful_pairs=[]

        for ticker,yahoo_symbol in yahoo_symbols_map.items ():
            try:
                if isinstance (data_raw.columns,pd.MultiIndex) and yahoo_symbol in data_raw.columns.levels[0]:
                    pair_data_full=data_raw[yahoo_symbol].copy ()
                elif not isinstance (data_raw.columns,pd.MultiIndex) and len (yahoo_symbols_list) == 1:
                    pair_data_full=data_raw.copy ()
                else:
                    print (f"  No data for {ticker} ({yahoo_symbol}).")
                    continue

                pair_data_full.columns=pair_data_full.columns.str.lower ()
                if 'adj close' in pair_data_full.columns:
                    pair_data_full=pair_data_full.rename (columns={'adj close':'adj_close'})

                price_col='adj_close' if 'adj_close' in pair_data_full.columns else 'close'
                pair_prices=pair_data_full[price_col].dropna ()
                pair_data_full=pair_data_full.loc[pair_prices.index]

                if not pair_prices.empty:
                    close_prices[ticker]=pair_prices
                    if all (c in pair_data_full.columns for c in ['open','high','low','close']):
                        ohlc_data_dict[ticker]=pair_data_full[['open','high','low','close']].copy ()
                    successful_pairs.append (ticker)
            except Exception as e:
                print (f"  Error processing {ticker} ({yahoo_symbol}): {e}")

        if close_prices.empty:
            print ("No valid price data processed.")
            return False

        common_index=close_prices.dropna (axis=0,how='all').index
        self.data=close_prices.loc[common_index].copy ()
        self.ohlc_data={ticker:df.loc[common_index].copy () for ticker,df in ohlc_data_dict.items () if
                        df is not None}
        self.forex_pairs=successful_pairs

        self.data=self.data.sort_index ()
        self.returns=self.data.pct_change ().dropna (axis=0,how='all')
        annualization_factor=252 if interval == '1d' else 252 * 24 if interval == '1h' else 252
        self.volatility=self.returns.rolling (window=20,min_periods=10).std ().dropna () * np.sqrt (
            annualization_factor)

        momentum_scores={}
        for horizon_name,horizon_days in self.config.HORIZONS.items ():
            min_periods=int (horizon_days * 0.8)
            momentum_scores[horizon_name]=self.returns.rolling (window=horizon_days,min_periods=min_periods).sum ()
        composite_momentum=pd.concat (momentum_scores.values (),axis=1).mean (axis=1)
        self.momentum=pd.DataFrame (composite_momentum,columns=['composite_momentum'])

        return True

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

    def recommend_forex_pairs (self):
        if self.data is None or self.returns is None:
            return None
        self.macro.fetch_interest_rates ()
        self.macro.fetch_gdp_data ()
        self.macro.fetch_cot_reports ()
        self.calculate_technical_indicators ()
        self.build_forex_network ()

        metrics=pd.DataFrame (index=self.forex_pairs)
        metrics['momentum']=self.momentum['composite_momentum'].iloc[-1] if self.momentum is not None else 0
        metrics['volatility']=self.volatility.iloc[-1] if self.volatility is not None else 1
        metrics['inv_volatility']=1 / metrics['volatility']
        metrics['carry_score']=[self.macro.data['interest_rates'].get (p.split ('/')[0],0) -
                                self.macro.data['interest_rates'].get (p.split ('/')[1],0)
                                for p in self.forex_pairs]
        metrics['gdp_diff']=[self.macro.data['gdp'].get (self.config.CURRENCY_TO_COUNTRY.get (p.split ('/')[0]),0) -
                             self.macro.data['gdp'].get (self.config.CURRENCY_TO_COUNTRY.get (p.split ('/')[1]),0)
                             for p in self.forex_pairs]
        metrics['cot_score']=[self.macro.data['cot'].get (p.split ('/')[0],0) -
                              self.macro.data['cot'].get (p.split ('/')[1],0)
                              for p in self.forex_pairs]
        metrics['rsi']=[self.technicals[p]['RSI'].iloc[-1] if p in self.technicals else 50
                        for p in self.forex_pairs]
        metrics['eigenvector']=[
            self.network_graph.nodes[p]['centrality'] if self.network_graph and p in self.network_graph else 0
            for p in self.forex_pairs]

        metrics['score']=(metrics['eigenvector'] * 0.2 + metrics['momentum'] * 0.2 +
                          metrics['carry_score'] * 0.2 + metrics['inv_volatility'] * 0.2 +
                          metrics['gdp_diff'] * 0.1 + metrics['cot_score'] * 0.1 +
                          (metrics['rsi'] - 50) * 0.1)
        recommendations=metrics.sort_values (by='score',ascending=False).head (self.config.RECOMMENDED_PAIRS)
        self.last_recommendations=recommendations
        return recommendations

    # --- User Interface with Dash ---
    '''

    #def create_dashboard (mhgna: MHGNAForex):
    def create_dashboard (mhgna: 'MHGNAForex'):
        app=Dash (__name__)
        G=mhgna.network_graph
        if G is not None:
            pos=nx.spring_layout (G)
            edge_x,edge_y=[],[]
            for edge in G.edges ():
                x0,y0=pos[edge[0]]
                x1,y1=pos[edge[1]]
                edge_x.extend ([x0,x1,None])
                edge_y.extend ([y0,y1,None])
            node_x,node_y=zip (*[pos[node] for node in G.nodes ()])
            node_centrality=[G.nodes[n]['centrality'] * 100 for n in G.nodes ()]
            network_fig=go.Figure (data=[
                go.Scatter (x=edge_x,y=edge_y,mode='lines',line=dict (width=1,color='gray')),
                go.Scatter (x=node_x,y=node_y,mode='markers+text',text=list (G.nodes ()),
                            marker=dict (size=15,color=node_centrality,colorscale='Viridis',showscale=True))
            ])
            network_fig.update_layout (title="Forex Dependency Network",showlegend=False)
        else:
            network_fig=go.Figure ()

        recommendations=mhgna.last_recommendations
        if recommendations is not None and not recommendations.empty:
            reco_fig=go.Figure (data=[
                go.Bar (x=recommendations.index,y=recommendations['score'],name='Score'),
                go.Bar (x=recommendations.index,y=recommendations['rsi'],name='RSI')
            ])
            reco_fig.update_layout (title="Top Recommended Pairs",barmode='group')
        else:
            reco_fig=go.Figure ()

        app.layout=html.Div ([
            html.H1 ("MHGNA Forex Dashboard"),
            dcc.Graph (id='network-graph',figure=network_fig),
            html.H2 ("Recommendations"),
            dcc.Graph (id='recommendations',figure=reco_fig),
            html.Pre (recommendations.to_string () if recommendations is not None else "No recommendations available.")
        ])
        app.run (debug=True)
'''
    def create_dashboard(self):  # Changé de mhgna: 'MHGNAForex' à self
        app = Dash(__name__)
        G = self.network_graph
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

        recommendations = self.last_recommendations
        if recommendations is not None and not recommendations.empty:
            reco_fig = go.Figure(data=[
                go.Bar(x=recommendations.index, y=recommendations['score'], name='Score'),
                go.Bar(x=recommendations.index, y=recommendations['rsi'], name='RSI')
            ])
            reco_fig.update_layout(title="Top Recommended Pairs", barmode='group')
        else:
            reco_fig = go.Figure()

        # Ajouter un graphique des tendances des prix
        price_trend_fig = go.Figure()
        if self.data is not None and recommendations is not None:
            top_pairs = recommendations.index[:min(5, len(recommendations))]
            for pair in top_pairs:
                if pair in self.data.columns:
                    price_trend_fig.add_trace(go.Scatter(
                        x=self.data.index[-30:],  # Derniers 30 jours
                        y=self.data[pair][-30:],
                        mode='lines',
                        name=pair
                    ))
            price_trend_fig.update_layout(
                title="Price Trends for Top Recommended Pairs (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(orientation="h")
            )

        app.layout = html.Div([
            html.H1("MHGNA Forex Dashboard"),
            dcc.Graph(id='network-graph', figure=network_fig),
            html.H2("Recommendations"),
            dcc.Graph(id='recommendations', figure=reco_fig),
            html.H2("Price Trends"),
            dcc.Graph(id='price-trends', figure=price_trend_fig),
            html.Pre(recommendations.to_string() if recommendations is not None else "No recommendations available.")
        ])
        app.run(debug=True)

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
    parser = argparse.ArgumentParser(description='MHGNA Forex Analysis Tool')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d, 1h)')
    args = parser.parse_args()

    config = ForexConfig()
    config.DATA_INTERVAL = args.interval

    forex_analyzer = MHGNAForex(config=config)
    if forex_analyzer.fetch_forex_data(interval=args.interval):
        forex_analyzer.recommend_forex_pairs()
        forex_analyzer.create_dashboard()  # Appel correct de la méthode
    else:
        print("Failed to fetch data. Exiting.")