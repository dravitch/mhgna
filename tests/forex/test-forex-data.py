"""
Script de test pour vérifier le téléchargement des données Forex
"""

import pandas as pd
import yfinance as yf
from datetime import datetime,timedelta


def test_yahoo_forex_download ():
    """
    Teste le téléchargement de données Forex depuis Yahoo Finance
    """
    print ("Test de téléchargement de données Forex")
    print ("=======================================")

    # Paires à tester
    forex_pairs=[
        "EURUSD=X",
        "USDJPY=X",
        "GBPUSD=X",
        "AUDUSD=X",
        "USDCAD=X",
        "NZDUSD=X"
    ]

    # Période de test
    end_date=datetime.now ()
    start_date=end_date - timedelta (days=30)  # 30 jours de données

    print (f"Période: {start_date.strftime ('%Y-%m-%d')} à {end_date.strftime ('%Y-%m-%d')}")
    print ("\nTest 1: Téléchargement groupé")
    print ("---------------------------")

    # Test 1: Téléchargement groupé
    try:
        data1=yf.download (forex_pairs,start=start_date,end=end_date)
        print (f"Forme des données: {data1.shape}")

        if isinstance (data1.columns,pd.MultiIndex):
            print ("Structure MultiIndex détectée")
            close_data=data1['Close']
            print (f"Colonnes disponibles: {close_data.columns.tolist ()}")

            # Compter les colonnes non-vides
            non_empty_cols=0
            for col in close_data.columns:
                if not close_data[col].isna ().all ():
                    non_empty_cols+=1
                    print (f"  ✓ {col}: {len (close_data[col].dropna ())} points de données")
                else:
                    print (f"  ✗ {col}: Aucune donnée")

            print (f"\nRésumé: {non_empty_cols}/{len (close_data.columns)} paires avec des données")
        else:
            print ("Structure simple détectée")
            print (f"Colonnes: {data1.columns.tolist ()}")
    except Exception as e:
        print (f"Erreur lors du téléchargement groupé: {e}")

    print ("\nTest 2: Téléchargement individuel")
    print ("-------------------------------")

    # Test 2: Téléchargement individuel
    success_count=0

    for pair in forex_pairs:
        try:
            pair_data=yf.download (pair,start=start_date,end=end_date)
            if not pair_data.empty and 'Close' in pair_data.columns:
                success_count+=1
                print (f"  ✓ {pair}: {len (pair_data)} points de données")
            else:
                print (f"  ✗ {pair}: Aucune donnée valide")
        except Exception as e:
            print (f"  ✗ {pair}: Erreur - {e}")

    print (f"\nRésumé: {success_count}/{len (forex_pairs)} paires téléchargées avec succès")

    print ("\nRecommandations:")
    if success_count > 0:
        print ("- Utilisez le téléchargement individuel pour chaque paire")
        print ("- Vérifiez que les symboles sont corrects pour Yahoo Finance (format XXXYYY=X)")
        if success_count < len (forex_pairs):
            print ("- Certaines paires ne sont pas disponibles, limitez-vous à celles qui fonctionnent")
    else:
        print ("- Vérifiez votre connexion Internet")
        print ("- Essayez un autre fournisseur de données que Yahoo Finance")
        print ("- Vérifiez si Yahoo Finance a modifié son API ou ses symboles")


if __name__ == "__main__":
    test_yahoo_forex_download ()