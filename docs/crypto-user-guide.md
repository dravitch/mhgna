# Créer crypto-user-guide.md
Set-Content -Path "docs\crypto-user-guide.md" -Value @"
# Crypto User Guide for MHGNA

## Overview
The crypto version of MHGNA is designed to optimize cryptocurrency portfolios using a multi-horizon graphical network approach. It analyzes dependencies between assets like BTC, ETH, and SOL across short, medium, and long-term horizons, leveraging metrics such as momentum, stability, and centrality to recommend portfolio allocations.

### Key Features
- Portfolio optimization with monthly rebalancing (7 assets).
- Turnover limitation (max 30% per rebalance).
- Achieved a total return of 345.18% in backtests (v1.1.0).

## Usage
To use the crypto version:
1. Navigate to `src/crypto/`.
2. Run the script:
   ```bash
   python mhgna-compiled.py
      ```
3. Check the generated visualizations in reports/crypto/.

For more details on the methodology, refer to the README.