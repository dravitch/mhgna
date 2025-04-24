# Documentation for MHGNA Forex Grok

## Objective
The MHGNA Forex Grok tool is designed to analyze Forex market data, generate trading recommendations, and visualize dependencies between currency pairs through an interactive dashboard. It leverages historical price data, macroeconomic indicators (e.g., interest rates, GDP), and technical indicators (e.g., RSI, scores) to provide insights for traders. The tool aims to assist users in identifying high-potential currency pairs and monitoring market conditions via a web-based interface powered by Dash and Plotly.

## Creating a Virtual Environment
To isolate dependencies and ensure a clean setup, use a virtual environment:

1. Open a terminal and navigate to your project directory:
   ```
   mkdir mhgna_forex
   cd mhgna_forex
   ```
2. Create a virtual environment named `env`:
   ```
   python -m venv env
   ```
3. Activate the virtual environment:
   - **Windows**:
     ```
     env\Scripts\activate
     ```
   - **macOS/Linux**:
     ```
     source env/bin/activate
     ```
4. Deactivate when done:
   ```
   deactivate
   ```

## Installing Dependencies
With the virtual environment activated, install the required libraries:
```
pip install dash matplotlib networkx numpy pandas requests seaborn yfinance python-dateutil scipy scikit-learn plotly
```

This ensures all necessary packages for data fetching, analysis, and visualization are available. If you have a `requirements.txt`, use:
```
pip install -r requirements.txt
```

## File Structure
The project is organized as follows:
- **`run_mhgna_forex_grok.py`**: Main script containing the `MHGNAForex` class, data fetching, analysis logic, and dashboard creation.
- **`data_cache/`**: Directory storing cached data (e.g., `interest_rates.json`, `gdp.json`) to reduce API calls.
- **`README.md`** (recommended): Optional file for project overview and setup instructions.
- **Other potential files**:
  - Configuration files (e.g., for API keys).
  - Additional scripts for specific analyses (if extended).

Ensure `data_cache/` exists in the project root to store cached data.

## User Guide
### Running the Tool
1. Activate the virtual environment:
   ```
   env\Scripts\activate  # Windows
   source env/bin/activate  # macOS/Linux
   ```
2. Run the script with an interval parameter:
   ```
   python run_mhgna_forex_grok.py --interval 1d
   ```
   - `--interval`: Specifies data granularity (`1d` for daily, `1h` for hourly).
3. Access the dashboard:
   - Open a browser and navigate to `http://127.0.0.1:8050/`.
   - View the Forex dependency network, recommended pairs, and price trends.

### Features
- **Data Fetching**: Retrieves historical Forex prices (via yfinance) and macroeconomic data (e.g., interest rates, GDP).
- **Analysis**: Generates recommendations based on scores, RSI, and macroeconomic factors.
- **Visualization**: Displays an interactive dashboard with:
  - A network graph of currency pair dependencies.
  - Bar charts of recommended pairs with scores and RSI.
  - Line charts of price trends for top pairs.
  - Textual summary of recommendations.
- **Diagnostics and Alerts**: Prints data quality checks and market alerts (e.g., high volatility) to the console.

### Example Output
Upon running:
```
Fetching Forex data from 2024-04-16 to 2025-04-16 with interval 1d...
[*********************100%***********************]  15 of 15 completed
Fetching interest rates for 10 currencies...
Interest rates updated for 10 currencies.
GDP data updated for 8 countries.
Active Alerts (2):
 - High volatility detected on EUR/USD
 - Significant drawdown on GBP/JPY
Generated Report:
[Recommendations and analysis details...]
Dash is running on http://127.0.0.1:8050/
```

### Configuration
- **API Keys**: Optionally set a FRED API key for real-time macroeconomic data:
  ```
  export FRED_API_KEY='your_key_here'  # macOS/Linux
  set FRED_API_KEY=your_key_here      # Windows
  ```
  Without a key, fallback data is used.
- **Cache**: Data is cached in `data_cache/` to optimize performance.

## Maintenance
### Clearing the Cache
Cached data (e.g., `interest_rates.json`, `gdp.json`) may become outdated or incomplete, leading to limited results. To refresh:
- **Windows**:
  ```
  del data_cache\*.json
  ```
- **macOS/Linux**:
  ```
  rm data_cache/*.json
  ```
Run the script afterward to fetch fresh data:
```
python run_mhgna_forex_grok.py --interval 1d
```

### Updating Dependencies
Periodically update libraries to maintain compatibility:
```
pip install --upgrade dash matplotlib networkx numpy pandas requests seaborn yfinance python-dateutil scipy scikit-learn plotly
```

### Troubleshooting
- **Dashboard not loading**: Ensure no other process uses port 8050. Change the port in `app.run(port=8051)` if needed.
- **Incomplete data**: Clear the cache and verify internet connectivity.
- **Module errors**: Confirm all dependencies are installed in the active virtual environment.