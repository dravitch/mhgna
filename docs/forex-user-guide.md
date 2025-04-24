# Forex User Guide for MHGNA

## Overview
The Forex version of MHGNA provides advanced trading insights by integrating macroeconomic data (via FRED API), session-based analysis, and carry trade opportunities. It includes an interactive dashboard built with Dash and Plotly to visualize dependency networks, recommended pairs, and price trends.

### Key Features
- Multi-horizon analysis (10, 60, 120 days).
- Macroeconomic integration (interest rates, inflation).
- Interactive dashboard for trading recommendations.

## Usage
To use the Forex version:
1. Navigate to `src/forex/`.
2. Run the script:
   ```bash
   python mhgna-forex-grok.py --interval 1d
   ```
3. Access the dashboard at http://127.0.0.1:8050/.
   Check the generated reports in reports/forex/.

## Troubleshooting

    Dashboard Not Loading: Ensure port 8050 is free. Change the port in app.run(port=8051) if needed.
    Incomplete Data: Clear the cache by running:

- For Windows 11
    Remove-Item data\forex\cache\*.json
- Module Errors: Confirm all dependencies are installed (pip install -r requirements.txt).

#### **4. Créer le fichier `requirements.txt`**

Créez le fichier `requirements.txt` avec les dépendances nécessaires :

Set-Content -Path "requirements.txt" -Value @"
numpy
pandas
yfinance
matplotlib
seaborn
networkx
scikit-learn
scipy
dash
plotly
requests
python-dateutil


# Supprimer les dossiers inutiles
Remove-Item -Path "crypto" -Recurse -Force
Remove-Item -Path "forex" -Recurse -Force
Remove-Item -Path ".idea" -Recurse -Force

# Supprimer les fichiers inutiles à la racine
Remove-Item -Path "temp" -Force