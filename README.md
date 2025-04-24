# MHGNA: Multi-Horizon Graphical Network Allocation

![MHGNA](https://img.shields.io/badge/version-1.2.0-blue.svg)  
**Last Updated:** April 24, 2025  
**Author:** [Your Name]  
**Contributors:** Claude, Gemini

## Overview

The **Multi-Horizon Graphical Network Allocation (MHGNA)** project is an advanced portfolio management framework that leverages graph theory and conditional dependency analysis to optimize asset allocation. Initially developed for the cryptocurrency market, MHGNA has been extended to the Forex market with specialized features such as macroeconomic data integration, carry trade analysis, and session-based trading insights.

MHGNA uses a multi-horizon approach (short, medium, and long-term) to analyze dependencies between assets, incorporating metrics like momentum, stability, and centrality to generate trading recommendations. The project includes two main implementations:
- **Crypto**: Focuses on cryptocurrency portfolio optimization.
- **Forex**: Tailored for Forex trading with additional macroeconomic and session-based analyses.

The system provides detailed reports, visualizations (e.g., dependency networks, price trends), and an interactive dashboard for Forex analysis.

## Project Structure

```
mhgna/
│
├── docs/                             # Documentation essentielle
│   ├── crypto-user-guide.md         # Guide utilisateur pour crypto
│   └── forex-user-guide.md          # Guide utilisateur pour forex
│
├── src/                              # Code source principal
│   ├── crypto/                      # Code source pour crypto
│   │   └── mhgna-compiled.py       # Script principal pour crypto
│   ├── forex/                      # Code source pour forex
│   │   ├── dashboard/              # Scripts pour le tableau de bord forex
│   │   │   └── mhgna-forex-dash.py # Script pour le tableau de bord
│   │   └── mhgna-forex-grok.py     # Script principal pour l'analyse forex
│   │
├── tests/                            # Tests essentiels
│   └── forex/                      # Tests pour forex
│       └── test-forex-data.py      # Test pour les données forex
│
├── data/                             # Données nécessaires
│   ├── forex/                      # Données pour forex
│   │   └── cache/                 # Données en cache
│   │       ├── cot.json
│   │       ├── gdp.json
│   │       └── interest-rates.json
│   │
├── reports/                          # Rapports et visualisations récents
│   ├── crypto/                     # Rapports pour crypto
│   │   └── trends-20250411.png     # Visualisation récente
│   ├── forex/                      # Rapports pour forex
│   │   ├── report-20250421.txt     # Rapport le plus récent
│   │   └── trends-20250421.png     # Visualisation la plus récente
│   │
├── .gitignore                       # Fichier pour ignorer les fichiers inutiles
├── LICENSE                          # Fichier de licence (MIT)
├── README.md                        # Documentation principale
└── requirements.txt                 # Liste des dépendances
```

## Features

### General Features (Crypto & Forex)
- **Multi-Horizon Analysis**: Analyzes asset dependencies across multiple time horizons (e.g., 30, 90, 180 days for crypto; 10, 60, 120 days for Forex).
- **Graphical Lasso**: Uses the Graphical Lasso algorithm to construct dependency networks.
- **Centrality Metrics**: Calculates eigenvector, betweenness, and closeness centrality to identify influential assets.
- **Momentum Integration**: Incorporates momentum into allocation decisions.
- **Drawdown Protection**: Implements mechanisms to limit exposure during significant market downturns.

### Crypto-Specific Features
- Portfolio optimization for cryptocurrencies (e.g., BTC, ETH, SOL).
- Monthly rebalancing with a portfolio size of 7 assets.
- Turnover limitation (max 30% per rebalance).
- Performance metrics: Achieved a total return of 345.18% (vs Bitcoin's 117.56%) in backtests (v1.1.0).

### Forex-Specific Features
- **Macroeconomic Integration**: Incorporates interest rates, inflation, and monetary policy data via the FRED API.
- **Session-Based Analysis**: Simulates activity during Asian, European, and American sessions.
- **Carry Trade Opportunities**: Identifies opportunities based on risk-adjusted interest rate differentials.
- **Technical Alerts**: Detects drawdowns, high volatility, and Bollinger Band breakouts.
- **Interactive Dashboard**: Built with Dash and Plotly, displaying:
  - A network graph of currency pair dependencies.
  - Bar charts of recommended pairs with scores and RSI.
  - Line charts of price trends.
  - Textual summary of recommendations.

## Installation

### Prerequisites
- **Python**: 3.8 or higher.
- **System**: Tested on Windows, macOS, and Linux.
- **Hardware**: Minimum 4 GB RAM; multi-core CPU recommended.

### Step 1: Clone the Repository
```bash
git clone https://github.com/dravitch/mhgna.git
cd mhgna
```

### Step 2: Set Up a Virtual Environment
1. Create a virtual environment:
   ```powershell
   python -m venv env
   ```
2. Activate the virtual environment:
   ```powershell
   .\env\Scripts\activate
   ```

### Step 3: Install Dependencies
Install the required libraries using the provided `requirements.txt`:
```powershell
pip install -r requirements.txt
```

## Usage

### Running the Crypto Version
1. Navigate to the `src/crypto/` directory:
   ```powershell
   cd src\crypto
   ```
2. Run the main script:
   ```powershell
   python mhgna-compiled.py
   ```
3. Review the generated visualizations in `reports/crypto/`.

### Running the Forex Version
1. Navigate to the `src/forex/` directory:
   ```powershell
   cd src\forex
   ```
2. Run the Forex analysis script:
   ```powershell
   python mhgna-forex-grok.py --interval 1d
   ```
   - `--interval`: Data granularity (`1d` for daily, `1h` for hourly).
3. Access the interactive dashboard:
   - Open a browser and go to `http://127.0.0.1:8050/`.
   - Explore the dependency network, recommended pairs, and price trends in `reports/forex/`.

### Example Output (Forex)
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

## Configuration

### Forex-Specific Configuration
- **API Keys**: Optionally set a FRED API key for macroeconomic data:
  ```powershell
  $env:FRED_API_KEY = 'your_key_here'
  ```
  Without a key, fallback data is used.
- **Cache**: Data is cached in `data/forex/cache/` to optimize performance. To clear the cache:
  ```powershell
  Remove-Item data\forex\cache\*.json
  ```

## Maintenance

Refer to `docs/forex-user-guide.md` for troubleshooting tips and maintenance instructions.

## Roadmap

### v1.3.0 - Optimization & Fine-Tuning (May 2025)
- Analyze performance across bullish/bearish periods.
- Optimize horizon weights and drawdown thresholds.
- Conduct Monte Carlo simulations for robustness.

### v2.0.0 - On-Chain Integration (June 2025, Crypto)
- Integrate on-chain data (e.g., transaction flows).
- Enhance graph edge weights with transfer volumes.

### v3.0.0 - Adaptive Intelligence (Q3 2025)
- Implement machine learning for dynamic parameter optimization.
- Add market regime detection and sentiment analysis.

## Contributing
Contributions are welcome! Please fork the repository, create a new branch for your changes, and submit a pull request. For major changes, open an issue first to discuss the proposed changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Claude**: For contributions to the crypto implementation.
- **Gemini**: For identifying and fixing errors in the Forex version.
- Inspired by works like Friedman et al. (2008) on Graphical Lasso and López de Prado (2018) on financial machine learning.