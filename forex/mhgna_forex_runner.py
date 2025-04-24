#!/usr/bin/env python
"""
MHGNA Forex Runner
=================

Script d'exécution convivial pour le système MHGNA Forex.
Permet aux traders manuels d'exécuter facilement des analyses et de générer des rapports.

Utilisation:
    python mhgna_forex_runner_grok.py --pairs EUR/USD,USD/JPY,GBP/USD --lookback 1 --recommended 5

Auteur: [Votre Nom]
Date: Avril 2025
Version: 1.0.0
"""

import os
import sys
import argparse
import datetime
import pandas as pd
from pathlib import Path
from colorama import init,Fore,Style

# Initialiser colorama pour les affichages colorés en console
init (autoreset=True)

# Importer le module MHGNA Forex
try:
    from run_mhgna_forex import MHGNAForex,ForexConfig,MacroDataCollector
except ImportError:
    print (f"{Fore.RED}Erreur: Impossible d'importer les modules MHGNA Forex.")
    print ("Vérifiez que run_mhgna_forex.py est dans le même répertoire que ce script.")
    sys.exit (1)


def print_header ():
    """Affiche l'en-tête du programme"""
    header=f"""
{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════╗
{Fore.CYAN}║ {Fore.YELLOW}MHGNA FOREX RUNNER {Style.RESET_ALL}{Fore.CYAN}                                               ║
{Fore.CYAN}║ {Style.RESET_ALL}Multi-Horizon Graphical Network Allocation pour le marché Forex{Fore.CYAN} ║
{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════╝
    """
    print (header)


def print_section (title):
    """Affiche le titre d'une section"""
    print (f"\n{Fore.CYAN}▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
    print (f"{Fore.YELLOW}{title}")
    print (f"{Fore.CYAN}▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓\n")


def print_success (message):
    """Affiche un message de succès"""
    print (f"{Fore.GREEN}✓ {message}")


def print_warning (message):
    """Affiche un avertissement"""
    print (f"{Fore.YELLOW}⚠ {message}")


def print_error (message):
    """Affiche une erreur"""
    print (f"{Fore.RED}✗ {message}")


def print_info (message):
    """Affiche une information"""
    print (f"{Fore.BLUE}ℹ {message}")


def print_recommendations (recommendations):
    """Affiche les recommandations de manière formatée"""
    if recommendations is None or len (recommendations) == 0:
        print_warning ("Aucune recommandation disponible.")
        return

    print (f"\n{Fore.GREEN}╔══════════════════════════════════════════════════════════════════╗")
    print (f"{Fore.GREEN}║ {Fore.WHITE}RECOMMANDATIONS DE TRADING                                    {Fore.GREEN}║")
    print (f"{Fore.GREEN}╠══════════════════════════════════════════════════════════════════╣")

    # Afficher les colonnes disponibles
    display_cols=[]
    if 'Rang' in recommendations.columns:
        display_cols.append ('Rang')
    if 'Direction' in recommendations.columns:
        display_cols.append ('Direction')
    if 'Score Global' in recommendations.columns:
        display_cols.append ('Score Global')
    if 'Centralité' in recommendations.columns:
        display_cols.append ('Centralité')
    if 'Momentum' in recommendations.columns:
        display_cols.append ('Momentum')
    if 'Score Carry' in recommendations.columns:
        display_cols.append ('Score Carry')
    if 'Rend. 30j' in recommendations.columns:
        display_cols.append ('Rend. 30j')
    if 'Session Optimale' in recommendations.columns:
        display_cols.append ('Session Optimale')

    # Limiter à 5 colonnes pour l'affichage
    if len (display_cols) > 5:
        display_cols=display_cols[:5]

    # Afficher les données
    for idx,row in recommendations.iterrows ():
        direction=row.get ('Direction','')
        direction_color=Fore.GREEN if direction == 'LONG' else Fore.RED if direction == 'SHORT' else Fore.WHITE

        print (f"{Fore.GREEN}║ {Fore.YELLOW}{idx} {direction_color}({direction}){Fore.GREEN} ║",end='')

        values=[]
        for col in display_cols:
            if col != 'Rang' and col != 'Direction':
                if col in row:
                    values.append (f"{col}: {row[col]}")

        value_str=" | ".join (values)
        print (f" {Fore.WHITE}{value_str}{Fore.GREEN} ║")

    print (f"{Fore.GREEN}╚══════════════════════════════════════════════════════════════════╝")


def print_alerts (alerts):
    """Affiche les alertes de manière formatée"""
    if not alerts:
        print_info ("Aucune alerte détectée.")
        return

    # Organiser les alertes par type
    alerts_by_type={}
    for alert in alerts:
        alert_type=alert['type']
        if alert_type not in alerts_by_type:
            alerts_by_type[alert_type]=[]
        alerts_by_type[alert_type].append (alert)

    print (f"\n{Fore.YELLOW}╔══════════════════════════════════════════════════════════════════╗")
    print (f"{Fore.YELLOW}║ {Fore.WHITE}ALERTES ACTIVES                                              {Fore.YELLOW}║")
    print (f"{Fore.YELLOW}╠══════════════════════════════════════════════════════════════════╣")

    for alert_type,alert_list in alerts_by_type.items ():
        # Déterminer la couleur selon le type d'alerte
        if alert_type in ['DRAWDOWN','VOLATILITY']:
            color=Fore.RED
        elif alert_type in ['CARRY_OPPORTUNITY','OVERSOLD']:
            color=Fore.GREEN
        else:
            color=Fore.YELLOW

        print (f"{Fore.YELLOW}║ {color}{alert_type} ({len (alert_list)}){Fore.YELLOW}")

        # Afficher jusqu'à 3 alertes de chaque type
        for alert in alert_list[:3]:
            message=alert['message']
            # Tronquer le message s'il est trop long
            if len (message) > 60:
                message=message[:57] + "..."
            print (f"{Fore.YELLOW}║   {Fore.WHITE}{message}")

    print (f"{Fore.YELLOW}╚══════════════════════════════════════════════════════════════════╝")


def print_report_summary (recommendations,alerts):
    """Affiche un résumé rapide du rapport"""
    print_section ("RÉSUMÉ DE L'ANALYSE")

    # Résumé des recommandations
    if recommendations is not None and len (recommendations) > 0:
        top_pairs=recommendations.index[:3].tolist ()
        print (f"{Fore.GREEN}Top paires recommandées: {', '.join (top_pairs)}")

        # Direction des paires
        if 'Direction' in recommendations.columns:
            for pair in top_pairs:
                direction=recommendations.loc[pair,'Direction']
                direction_color=Fore.GREEN if direction == 'LONG' else Fore.RED
                print (f"  {pair}: {direction_color}{direction}")
    else:
        print_warning ("Aucune recommandation générée.")

    # Résumé des alertes
    if alerts:
        print (f"\n{Fore.YELLOW}Alertes importantes ({len (alerts)} total):")
        alert_types=set (alert['type'] for alert in alerts)
        for alert_type in alert_types:
            count=sum (1 for alert in alerts if alert['type'] == alert_type)
            if alert_type in ['DRAWDOWN','VOLATILITY']:
                print (f"  {Fore.RED}{alert_type}: {count} alerte(s)")
            elif alert_type in ['CARRY_OPPORTUNITY','OVERSOLD']:
                print (f"  {Fore.GREEN}{alert_type}: {count} alerte(s)")
            else:
                print (f"  {Fore.YELLOW}{alert_type}: {count} alerte(s)")
    else:
        print_info ("\nAucune alerte active.")

    # Session de trading actuellement active
    now=datetime.datetime.utcnow ()
    current_hour=now.hour

    if 22 <= current_hour or current_hour < 8:
        print (f"\n{Fore.CYAN}Session active: {Fore.WHITE}Asie (22h-8h UTC)")
    elif 7 <= current_hour < 16:
        print (f"\n{Fore.CYAN}Session active: {Fore.WHITE}Europe (7h-16h UTC)")
    elif 13 <= current_hour < 22:
        print (f"\n{Fore.CYAN}Session active: {Fore.WHITE}Amérique (13h-22h UTC)")

    # Rappel des fichiers générés
    print (f"\n{Fore.BLUE}Fichiers générés:")
    print (f"  Rapport complet: forex_reports/mhgna_forex_*.txt")
    print (f"  Visualisations: forex_reports/mhgna_forex_*_network.png, forex_reports/mhgna_forex_*_trends.png")

    print ("\n" + "=" * 80)


def main ():
    """Fonction principale du programme"""
    print_header ()

    # Configurer les arguments de ligne de commande
    parser=argparse.ArgumentParser (description='MHGNA Forex Analysis Tool')
    parser.add_argument ('--pairs',type=str,default=None,
                         help='Liste des paires Forex à analyser, séparées par des virgules')
    parser.add_argument ('--lookback',type=int,default=1,
                         help='Période d\'historique en années (défaut: 1)')
    parser.add_argument ('--recommended',type=int,default=5,
                         help='Nombre de paires à recommander (défaut: 5)')
    parser.add_argument ('--output',type=str,default='forex_reports',
                         help='Dossier de sortie pour les rapports (défaut: forex_reports)')
    parser.add_argument ('--date',type=str,default=None,
                         help='Date d\'analyse au format YYYY-MM-DD (défaut: aujourd\'hui)')
    parser.add_argument ('--detail',action='store_true',
                         help='Affiche les détails complets de l\'analyse')
    parser.add_argument ('--api-keys',type=str,default=None,
                         help='Chemin vers un fichier de configuration des clés API (JSON)')
    parser.add_argument ('--no-visuals',action='store_true',
                         help='Désactive la génération des visualisations')

    args=parser.parse_args ()

    # Créer le dossier de sortie s'il n'existe pas
    output_dir=Path (args.output)
    output_dir.mkdir (exist_ok=True,parents=True)

    # Créer et configurer l'instance ForexConfig
    config=ForexConfig ()

    # Charger les clés API si spécifiées
    if args.api_keys:
        try:
            import json
            with open (args.api_keys,'r') as f:
                api_keys=json.load (f)
                config.api_keys=api_keys
                print_success (f"Clés API chargées depuis {args.api_keys}")
        except Exception as e:
            print_error (f"Erreur lors du chargement des clés API: {e}")

    # Configurer les paires si spécifiées
    if args.pairs:
        config.tickers=args.pairs.split (',')
        print_info (f"Paires sélectionnées: {', '.join (config.tickers)}")

    # Configurer la période d'historique
    config.lookback_period=args.lookback
    print_info (f"Période d'historique: {args.lookback} an(s)")

    # Configurer le nombre de paires à recommander
    config.recommended_pairs=args.recommended
    print_info (f"Nombre de recommandations: {args.recommended}")

    # Afficher l'heure et la session de trading actuelles
    now=datetime.datetime.utcnow ()
    print_info (f"Date et heure: {now.strftime ('%Y-%m-%d %H:%M')} UTC")

    current_hour=now.hour
    if 22 <= current_hour or current_hour < 8:
        print_info ("Session de trading active: Asie (22h-8h UTC)")
    elif 7 <= current_hour < 16:
        print_info ("Session de trading active: Europe (7h-16h UTC)")
    elif 13 <= current_hour < 22:
        print_info ("Session de trading active: Amérique (13h-22h UTC)")

    # Créer l'instance MHGNA Forex
    print_section ("INITIALISATION DU SYSTÈME")
    try:
        forex=MHGNAForex (config)
        print_success ("Système MHGNA Forex initialisé avec succès")
    except Exception as e:
        print_error (f"Erreur lors de l'initialisation du système: {e}")
        sys.exit (1)

    # Exécuter l'analyse complète
    print_section ("EXÉCUTION DE L'ANALYSE")
    end_date=args.date if args.date else None
    if end_date:
        print_info (f"Date d'analyse: {end_date}")

    try:
        recommendations,alerts,report=forex.run_complete_forex_analysis (end_date=end_date)

        if recommendations is not None:
            print_recommendations (recommendations)

        if alerts:
            print_alerts (alerts)

        # Afficher un résumé du rapport
        print_report_summary (recommendations,alerts)

    except Exception as e:
        print_error (f"Erreur lors de l'exécution de l'analyse: {e}")
        import traceback
        if args.detail:
            traceback.print_exc ()
        sys.exit (1)

    print (f"\n{Fore.GREEN}Analyse terminée avec succès.{Style.RESET_ALL}")
    print (f"Pour consulter le rapport complet: {args.output}/mhgna_forex_*.txt")


if __name__ == "__main__":
    try:
        main ()
    except KeyboardInterrupt:
        print (f"\n\n{Fore.YELLOW}Analyse interrompue par l'utilisateur.{Style.RESET_ALL}")
        sys.exit (0)
    except Exception as e:
        print (f"\n\n{Fore.RED}Erreur inattendue: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc ()
        sys.exit (1)