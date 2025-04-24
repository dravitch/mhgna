"""
Multi-Horizon Graphical Network Allocation (MHGNA)
=================================================

Une stratégie d'allocation de portefeuille crypto basée sur la théorie des graphes
et l'analyse des dépendances conditionnelles à travers différents horizons temporels.

Cette implémentation permet:
- L'analyse multi-horizon des dépendances conditionnelles entre actifs
- La visualisation des graphes de dépendance
- La sélection d'actifs basée sur la structure topologique du marché
- Une allocation optimisée avec contrôle du risque et des drawdowns
- Un backtest complet avec métriques de performance

Auteur: [Votre Nom]
Date: Avril 2025
Version: 1.1.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
from scipy.stats import zscore
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
from itertools import product
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import time
import math

# Ignorer les avertissements
warnings.filterwarnings('ignore')

#==============================================================================
# CLASSES DE CONFIGURATION
#==============================================================================

class Config:
    """
    Configuration globale pour la stratégie MHGNA.
    Contient tous les paramètres ajustables de l'algorithme.
    """
    # Paramètres de la stratégie
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
               'ADA-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD',
               'UNI-USD', 'AAVE-USD', 'CRV-USD', 'ATOM-USD', 'LTC-USD']
    start_date = '2022-01-01'
    end_date = '2024-01-01'
    
    # Paramètres d'horizon multiple
    horizons = {
        'court': {'window': 30, 'weight': 0.25},
        'moyen': {'window': 90, 'weight': 0.50},
        'long': {'window': 180, 'weight': 0.25}
    }
    
    # Fréquence de rééquilibrage
    rebalance_freq = 21  # Mensuel (en jours)
    
    # Sélection d'actifs
    portfolio_size = 7  # Nombre d'actifs dans le portefeuille
    
    # Paramètres de régularisation selon l'horizon
    alpha_short = 0.02    # Plus fort pour horizon court (plus sparse)
    alpha_medium = 0.01   # Moyen pour horizon moyen
    alpha_long = 0.005    # Plus faible pour horizon long (plus dense)
    
    # Paramètres de momentum
    momentum_window = 60  # Fenêtre pour calculer le momentum (jours)
    momentum_weight = 0.3  # Influence du momentum dans l'allocation
    
    # Paramètres de turnover
    max_turnover = 0.3  # Maximum 30% de changement par rééquilibrage
    
    # Capital initial
    initial_capital = 10000  # Capital initial en USD
    benchmark = 'BTC-USD'    # Benchmark pour comparaison
    
    # Risk management
    max_asset_weight = 0.35  # Poids maximum par actif
    min_asset_weight = 0.05  # Poids minimum par actif
    
    # Paramètres de drawdown
    max_drawdown_threshold = -0.15  # -15% de drawdown déclenche un ajustement
    risk_reduction_factor = 0.5     # Réduction de 50% de l'exposition
    recovery_threshold = 0.10       # +10% de récupération pour revenir à l'exposition normale

class PreservationStrategy(Enum):
    """
    Stratégies de préservation des gains en stablecoin.
    """
    THRESHOLD_BASED = "threshold"      # Basé sur des seuils de profit
    VOLATILITY_BASED = "volatility"    # Basé sur la volatilité du marché
    DRAWDOWN_BASED = "drawdown"        # Basé sur les drawdowns
    TIME_BASED = "time"                # Basé sur des intervalles temporels
    HYBRID = "hybrid"                  # Combinaison de plusieurs stratégies

#==============================================================================
# FONCTIONS D'ACQUISITION ET DE PRÉPARATION DES DONNÉES
#==============================================================================

def standardize_yahoo_data(data):
    """
    Standardise les données de Yahoo Finance en gérant le MultiIndex et en normalisant les noms de colonnes.
    
    Args:
        data (pd.DataFrame): DataFrame brut provenant de yf.download()
        
    Returns:
        pd.DataFrame: DataFrame avec colonnes standardisées
    """
    # Handle MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        # Extraire seulement les données de prix (première colonne de chaque catégorie)
        data = pd.DataFrame({
            'open': data['Open'].iloc[:, 0],
            'high': data['High'].iloc[:, 0],
            'low': data['Low'].iloc[:, 0],
            'close': data['Close'].iloc[:, 0],
            'volume': data['Volume'].iloc[:, 0],
            'adj close': data['Adj Close'].iloc[:, 0] if 'Adj Close' in data.columns else data['Close'].iloc[:, 0]
        })
    else:
        # Standardize column names
        data.columns = data.columns.str.lower()
    
    return data

def validate_yahoo_data(df):
    """
    Valide que les données contiennent toutes les colonnes requises.
    
    Args:
        df (pd.DataFrame): DataFrame à valider
        
    Returns:
        bool: True si valide
        
    Raises:
        ValueError: Si des colonnes sont manquantes
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True

def get_data():
    """
    Télécharge et prépare les données de prix pour tous les actifs configurés.
    
    Returns:
        tuple: (data, returns, volatility, momentum_scores)
            - data: DataFrame avec les prix de clôture
            - returns: DataFrame avec les rendements quotidiens
            - volatility: DataFrame avec la volatilité glissante sur 30 jours
            - momentum_scores: DataFrame avec les scores de momentum composites
    """
    print("Récupération des données de prix...")
    
    # Télécharger les données
    raw_data = yf.download(Config.tickers, start=Config.start_date, end=Config.end_date)
    
    # Standardiser les données selon le format Yahoo Finance
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Extraire seulement les données de clôture et créer un DataFrame simple
        data = pd.DataFrame()
        for ticker in Config.tickers:
            try:
                # Extraire les données de clôture pour chaque ticker
                if ('Close', ticker) in raw_data.columns:
                    data[ticker] = raw_data['Close', ticker]
            except Exception as e:
                print(f"Erreur lors de l'extraction des données pour {ticker}: {e}")
    else:
        # Si pas de MultiIndex, supposer que c'est un seul ticker avec toutes les colonnes
        data = raw_data['Close'] if 'Close' in raw_data.columns else raw_data['close']
    
    # Calculer les rendements
    returns = data.pct_change().dropna()
    
    # Calculer la volatilité
    volatility = returns.rolling(30).std()
    
    # Calculer les indicateurs de momentum sur différentes périodes
    momentum_short = returns.rolling(20).sum()
    momentum_medium = returns.rolling(60).sum()
    momentum_long = returns.rolling(120).sum()
    
    # Normaliser et combiner pour créer un score de momentum composite
    momentum_score = (
        zscore(momentum_short, nan_policy='omit') * 0.5 + 
        zscore(momentum_medium, nan_policy='omit') * 0.3 + 
        zscore(momentum_long, nan_policy='omit') * 0.2
    )
    
    print(f"Données récupérées: {len(returns)} jours pour {len(Config.tickers)} actifs.")
    
    # Valider que les données sont bien formées
    if len(returns) == 0:
        raise ValueError("Aucune donnée de rendement n'a été récupérée.")
    
    return data, returns, volatility, momentum_score

#==============================================================================
# FONCTIONS DE CONSTRUCTION DU GRAPHE DE DÉPENDANCE
#==============================================================================

def build_multi_horizon_dependency_graph(returns, current_date):
    """
    Construit un graphe de dépendance multi-horizon en combinant différentes échelles temporelles.
    
    Args:
        returns (pd.DataFrame): DataFrame des rendements journaliers
        current_date (pd.Timestamp): Date jusqu'à laquelle les données doivent être considérées
    
    Returns:
        tuple: (G, precision_matrix)
            - G: Graphe NetworkX représentant les dépendances entre actifs
            - precision_matrix: Matrice de précision combinée
    """
    graph_models = {}
    precision_matrices = {}
    
    # Construire un graphe pour chaque horizon temporel
    for horizon_name, horizon_config in Config.horizons.items():
        window_size = horizon_config['window']
        
        # Déterminer l'alpha approprié pour l'horizon
        if horizon_name == 'court':
            alpha = Config.alpha_short
        elif horizon_name == 'moyen':
            alpha = Config.alpha_medium
        else:
            alpha = Config.alpha_long
            
        # Extraire la fenêtre glissante pour cet horizon
        window_start = current_date - timedelta(days=window_size)
        returns_window = returns.loc[window_start:current_date].dropna()
        
        # S'assurer qu'il y a suffisamment de données
        if len(returns_window) < window_size // 2:
            continue
            
        # Gérer les NaN et les valeurs infinies
        returns_window = returns_window.replace([np.inf, -np.inf], np.nan).dropna()
        
        # S'il reste trop peu d'échantillons, passer à l'horizon suivant
        if returns_window.shape[0] < 20:
            continue
            
        try:
            # Appliquer le Graphical Lasso
            model = GraphicalLassoCV(alphas=[alpha*0.5, alpha, alpha*2], cv=5, max_iter=1000)
            model.fit(returns_window)
            precision_matrix = model.precision_
            
            # Création du graphe pour cet horizon
            G = nx.from_numpy_array(np.abs(precision_matrix))
            mapping = {i: name for i, name in enumerate(returns_window.columns)}
            G = nx.relabel_nodes(G, mapping)
            
            graph_models[horizon_name] = G
            precision_matrices[horizon_name] = precision_matrix
            
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Erreur pour l'horizon {horizon_name}: {e}")
            continue
    
    # Si aucun modèle n'a pu être construit, renvoyer un graphe vide
    if not graph_models:
        print("Aucun modèle de graphe n'a pu être construit.")
        G = nx.Graph()
        for ticker in Config.tickers:
            G.add_node(ticker)
        return G, np.zeros((len(Config.tickers), len(Config.tickers)))
    
    # Combiner les différents horizons en un seul graphe
    combined_G = nx.Graph()
    
    # Ajouter tous les nœuds
    for ticker in Config.tickers:
        combined_G.add_node(ticker)
    
    # Construire la matrice de précision combinée, pondérée par les poids des horizons
    combined_precision = np.zeros((len(Config.tickers), len(Config.tickers)))
    total_weight = 0
    
    for horizon_name, precision_matrix in precision_matrices.items():
        horizon_weight = Config.horizons[horizon_name]['weight']
        # Ajouter cette matrice de précision, pondérée par le poids de l'horizon
        combined_precision += precision_matrix * horizon_weight
        total_weight += horizon_weight
    
    # Normaliser par le poids total
    if total_weight > 0:
        combined_precision /= total_weight
    
    # Ajouter les arêtes au graphe combiné
    ticker_indices = {ticker: i for i, ticker in enumerate(Config.tickers)}
    for i, j in product(range(len(Config.tickers)), range(len(Config.tickers))):
        if i < j:  # Éviter les doublons
            weight = abs(combined_precision[i, j])
            if weight > 0.01:  # Seuil minimal pour ajouter une arête
                source = Config.tickers[i]
                target = Config.tickers[j]
                combined_G.add_edge(source, target, weight=weight)
    
    return combined_G, combined_precision

#==============================================================================
# FONCTIONS DE VISUALISATION
#==============================================================================
def calculate_centrality_for_disconnected_graph (G):
    """
    Calcule les métriques de centralité même pour les graphes déconnectés
    en traitant chaque composante connexe séparément.
    Calcule les métriques de centralité pour graphes déconnectés en évitant l'erreur scipy.
    """
    # Initialiser les dictionnaires pour stocker les centralités
    eigenvector_centrality={}
    betweenness_centrality={}
    closeness_centrality={}

    # Trouver toutes les composantes connexes
    connected_components=list (nx.connected_components (G))

    # Pour chaque composante
    for component in connected_components:
        # Créer un sous-graphe pour cette composante
        subgraph=G.subgraph (component)

        # Si la composante a plus d'un nœud
        if len (subgraph) > 1:
            try:
                # Version adaptée pour matrices denses au lieu de sparse
                # Convertir explicitement en tableau dense
                adj_matrix=nx.to_numpy_array (subgraph,weight='weight')

                # Calculer les centralités pour cette composante
                # Éviter l'erreur en calculant manuellement avec scipy.linalg.eig
                if len (component) > 2:  # Suffisamment grand pour calculer les vecteurs propres
                    # Calcul de centralité eigenvector
                    try:
                        eigenvalues,eigenvectors=np.linalg.eig (adj_matrix)
                        # Trouver l'indice de la plus grande valeur propre
                        idx=np.argmax (eigenvalues.real)
                        # Normaliser le vecteur propre
                        eigvec=np.abs (eigenvectors[:,idx].real)
                        eigvec=eigvec / np.linalg.norm (eigvec)
                        # Créer un dictionnaire pour ce composant
                        nodes=list (subgraph.nodes ())
                        for i,node in enumerate (nodes):
                            eigenvector_centrality[node]=eigvec[i]
                    except Exception as e:
                        print (f"Erreur eigenvector sur composante de taille {len (component)}: {e}")
                        for node in component:
                            eigenvector_centrality[node]=1.0 / len (component)
                else:
                    for node in component:
                        eigenvector_centrality[node]=1.0 / len (component)

                # Utiliser les méthodes standard pour betweenness et closeness
                comp_between=nx.betweenness_centrality (subgraph,weight='weight',normalized=True)
                comp_close=nx.closeness_centrality (subgraph,distance='weight')

                # Ajouter les valeurs aux dictionnaires globaux
                betweenness_centrality.update (comp_between)
                closeness_centrality.update (comp_close)
            except Exception as e:
                print (f"Erreur générale sur composante de taille {len (component)}: {e}")
                # Valeurs par défaut pour cette composante
                for node in component:
                    eigenvector_centrality[node]=1.0 / len (component)
                    betweenness_centrality[node]=1.0 / len (component)
                    closeness_centrality[node]=1.0 / len (component)
        else:
            # Pour les nœuds isolés
            node=list (component)[0]
            eigenvector_centrality[node]=0.5  # Valeur par défaut
            betweenness_centrality[node]=0.0
            closeness_centrality[node]=0.0

    return eigenvector_centrality,betweenness_centrality,closeness_centrality

def plot_dependency_network(G, precision_matrix, date, momentum_scores=None, volatility=None, threshold=0.01):
    """
    Visualise le réseau de dépendance conditionnelle avec information de momentum et volatilité.
    
    Args:
        G (nx.Graph): Graphe de dépendance
        precision_matrix (np.ndarray): Matrice de précision pour les arêtes
        date (pd.Timestamp): Date de l'analyse
        momentum_scores (pd.DataFrame, optional): Scores de momentum par actif
        volatility (pd.DataFrame, optional): Volatilité par actif
        threshold (float, optional): Seuil pour filtrer les arêtes faibles
    
    Returns:
        plt.Figure: Figure matplotlib du graphe
    """

    plt.figure (figsize=(16,14))

    # Vérifier si le graphe est vide ou a trop peu de nœuds
    if len (G.nodes ()) < 2:
        plt.text (0.5,0.5,"Graphe insuffisant pour visualisation",
                  horizontalalignment='center',verticalalignment='center',
                  fontsize=16,fontweight='bold',transform=plt.gca ().transAxes)
        plt.title (f"Réseau de dépendance - {date.strftime ('%Y-%m-%d')}",fontsize=18,fontweight='bold')
        plt.axis ('off')
        return plt

    # Filtrer les liens faibles
    edges=[(u,v) for (u,v,d) in G.edges (data=True) if d['weight'] > threshold]
    filtered_G=G.edge_subgraph (edges).copy () if edges else G.copy ()

    # S'assurer que tous les nœuds sont dans le graphe filtré
    for node in G.nodes ():
        if node not in filtered_G.nodes ():
            filtered_G.add_node (node)

    # Vérifier la connectivité du graphe
    connected_components=list (nx.connected_components (filtered_G))
    num_components=len (connected_components)

    # Préparation des données pour les couleurs des nœuds basées sur le momentum
    if momentum_scores is not None:
        latest_scores=momentum_scores.loc[momentum_scores.index <= date].iloc[
            -1] if not momentum_scores.empty else pd.Series (0,index=G.nodes ())
        # Normalisation entre 0 et 1
        min_score=latest_scores.min ()
        max_score=latest_scores.max ()
        if max_score > min_score:
            normalized_scores=(latest_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores=pd.Series (0.5,index=latest_scores.index)

        # Création d'une colormap personnalisée: rouge (négatif) à vert (positif)
        node_colors=[]
        for node in filtered_G.nodes ():
            if node in normalized_scores:
                score=normalized_scores[node]
                if score < 0.5:  # Momentum négatif
                    r=0.8
                    g=score * 2 * 0.8
                    b=0.1
                else:  # Momentum positif
                    r=(1 - score) * 2 * 0.8
                    g=0.8
                    b=0.1
                node_colors.append ((r,g,b))
            else:
                node_colors.append ((0.5,0.5,0.5))  # Gris pour données manquantes
    else:
        # Utiliser une colormap par défaut si pas de données de momentum
        node_colors=list (range (len (filtered_G.nodes ())))
        node_colors=plt.cm.viridis (np.array (node_colors) / max (len (node_colors) - 1,1))

    # Calcul des métriques de centralité avec gestion des erreurs pour graphes déconnectés
    try:
        # Utiliser notre fonction adaptée pour les graphes déconnectés
        eigenvector_centrality,_,_=calculate_centrality_for_disconnected_graph (filtered_G)
        centrality=eigenvector_centrality
    except Exception as e:
        print (f"Erreur lors du calcul de la centralité pour la visualisation: {e}")
        # Fallback à une valeur par défaut si le calcul échoue
        centrality={node:0.5 for node in filtered_G.nodes ()}

    # Déterminer la taille des nœuds en fonction de la centralité
    node_sizes=[]
    for node in filtered_G.nodes ():
        base_size=300  # Taille de base
        if node in centrality:
            # Échelle de taille: 300 à 3000 en fonction de la centralité
            size=base_size + centrality[node] * 2700
        else:
            size=base_size
        node_sizes.append (size)

    # Tenter de générer un layout optimisé
    try:
        if num_components == 1:
            # Si graphe connecté, utiliser spring_layout
            pos=nx.spring_layout (filtered_G,k=1.5 / np.sqrt (len (filtered_G.nodes ())),seed=42)
        else:
            # Si graphe déconnecté, positionner chaque composante séparément
            pos={}
            y_offset=0
            max_width=0

            for i,component in enumerate (connected_components):
                subgraph=filtered_G.subgraph (component)
                # Calculer la largeur et hauteur approximatives en fonction du nombre de nœuds
                width=max (1.0,np.sqrt (len (subgraph)) * 0.4)
                height=width
                max_width=max (max_width,width)

                # Positionner la composante avec un décalage vertical
                component_pos=nx.spring_layout (subgraph,k=0.3,seed=42 + i,scale=min (width,height))

                # Ajouter un décalage pour chaque composante
                for node,coords in component_pos.items ():
                    pos[node]=np.array ([coords[0],coords[1] + y_offset])

                # Augmenter le décalage pour la prochaine composante
                y_offset+=height * 1.5
    except Exception as e:
        print (f"Erreur lors de la génération du layout: {e}")
        # Fallback en cas d'erreur
        pos={node:(np.random.rand () * 2 - 1,np.random.rand () * 2 - 1) for node in filtered_G.nodes ()}

    # Dessiner les nœuds
    nx.draw_networkx_nodes (filtered_G,pos,node_size=node_sizes,node_color=node_colors,alpha=0.85)

    # Dessiner les liens avec une épaisseur proportionnelle au poids et une couleur basée sur le poids
    edge_colors=[]
    edge_widths=[]
    for u,v in filtered_G.edges ():
        weight=filtered_G[u][v]['weight']
        edge_widths.append (weight * 10)  # Échelle proportionnelle
        # Couleur basée sur le poids: plus l'arête est forte, plus elle est foncée
        edge_colors.append (plt.cm.Blues (min (weight * 10,0.9)))

    nx.draw_networkx_edges (filtered_G,pos,width=edge_widths,edge_color=edge_colors,alpha=0.7)

    # Ajouter les étiquettes des nœuds avec un meilleur formatage
    labels={}
    for node in filtered_G.nodes ():
        # Format d'étiquette plus riche si les données de momentum sont disponibles
        if momentum_scores is not None and node in latest_scores:
            momentum_value=latest_scores[node]
            momentum_sign="+" if momentum_value >= 0 else ""
            labels[node]=f"{node}\n{momentum_sign}{momentum_value:.2f}"
        else:
            labels[node]=node

    # Position légèrement ajustée pour les étiquettes
    label_pos={node:(coords[0],coords[1] + 0.02) for node,coords in pos.items ()}
    nx.draw_networkx_labels (filtered_G,label_pos,labels=labels,font_size=10,
                             font_weight='bold',bbox=dict (facecolor='white',alpha=0.7,edgecolor='none',pad=1))

    # Ajouter des informations sur la structure du graphe
    plt.text (0.02,0.02,
              f"Nœuds: {len (filtered_G.nodes ())}\nArêtes: {len (filtered_G.edges ())}\nComposantes: {num_components}",
              transform=plt.gca ().transAxes,fontsize=10,
              verticalalignment='bottom',horizontalalignment='left',
              bbox=dict (facecolor='white',alpha=0.7,boxstyle='round,pad=0.3'))

    # Titre avec date et informations supplémentaires
    plt.title (f"Réseau de Dépendance Conditionnelle - {date.strftime ('%Y-%m-%d')}",
               fontsize=18,fontweight='bold')

    # Informations sur les composantes déconnectées si applicable
    if num_components > 1:
        plt.suptitle (f"Graphe déconnecté ({num_components} composantes)",
                      fontsize=14,y=0.98)

    plt.axis ('off')
    plt.tight_layout ()
    return plt


class GainPreservationModule:
    """
    Module de préservation des gains en stablecoin pour MHGNA.
    """

    def __init__ (self,strategy=PreservationStrategy.HYBRID,profit_threshold=0.15,
                  max_stablecoin_allocation=0.5,base_preservation_rate=0.3,
                  drawdown_sensitivity=2.0,time_interval=30,stablecoin_assets=None,
                  reinvestment_threshold=-0.1):
        """
        Initialise le module de préservation des gains.
        """
        self.strategy=strategy
        self.profit_threshold=profit_threshold
        self.max_stablecoin_allocation=max_stablecoin_allocation
        self.base_preservation_rate=base_preservation_rate
        self.drawdown_sensitivity=drawdown_sensitivity
        self.time_interval=time_interval
        self.stablecoin_assets=stablecoin_assets or ["USDT","USDC","DAI","BUSD"]
        self.reinvestment_threshold=reinvestment_threshold

        # État interne
        self.initial_portfolio_value=None
        self.last_high_value=None
        self.last_rebalance_date=None
        self.current_stablecoin_allocation=0.0
        self.preserved_gains=0.0
        self.reinvestment_ready=False
        self.last_preservation_date=None
        self.preservation_history=[]

    def initialize (self,initial_value,current_date):
        """
        Initialise le module avec la valeur initiale du portefeuille.
        """
        self.initial_portfolio_value=initial_value
        self.last_high_value=initial_value
        self.last_rebalance_date=current_date
        self.last_preservation_date=current_date

    def calculate_preservation_allocation (self,current_value,current_date,market_drawdown=0.0,
                                           volatility=None,current_weights=None):
        """
        Calcule l'allocation recommandée en stablecoins pour préserver les gains.
        """
        if self.initial_portfolio_value is None:
            self.initialize (current_value,current_date)
            return 0.0,{}

        # Mettre à jour le plus haut de portefeuille
        if current_value > self.last_high_value:
            self.last_high_value=current_value

        # Calculer le profit actuel
        profit_pct=(current_value / self.initial_portfolio_value) - 1.0
        drawdown_pct=(current_value / self.last_high_value) - 1.0

        # Allocation recommandée selon la stratégie
        if self.strategy == PreservationStrategy.THRESHOLD_BASED:
            stablecoin_allocation=self._threshold_based_allocation (profit_pct)
        elif self.strategy == PreservationStrategy.VOLATILITY_BASED:
            stablecoin_allocation=self._volatility_based_allocation (profit_pct,volatility)
        elif self.strategy == PreservationStrategy.DRAWDOWN_BASED:
            stablecoin_allocation=self._drawdown_based_allocation (profit_pct,drawdown_pct)
        elif self.strategy == PreservationStrategy.TIME_BASED:
            stablecoin_allocation=self._time_based_allocation (profit_pct,current_date)
        elif self.strategy == PreservationStrategy.HYBRID:
            stablecoin_allocation=self._hybrid_allocation (
                profit_pct,drawdown_pct,volatility,current_date,market_drawdown
            )
        else:
            stablecoin_allocation=0.0

        # Limiter l'allocation à la valeur maximale configurée
        stablecoin_allocation=min (stablecoin_allocation,self.max_stablecoin_allocation)

        # Vérifier si nous devrions réinvestir plutôt que préserver
        if market_drawdown <= self.reinvestment_threshold and self.current_stablecoin_allocation > 0.05:
            self.reinvestment_ready=True
            stablecoin_allocation=max (0,stablecoin_allocation - 0.1)  # Réduire progressivement
        else:
            self.reinvestment_ready=False

        # Répartition entre les différents stablecoins
        stablecoin_weights=self._distribute_stablecoin_allocation (stablecoin_allocation,current_weights)

        # Mettre à jour l'état interne
        if abs (stablecoin_allocation - self.current_stablecoin_allocation) > 0.02:
            self.last_rebalance_date=current_date
            self.current_stablecoin_allocation=stablecoin_allocation

            # Enregistrer l'historique
            self.preservation_history.append ({
                'date':current_date,
                'portfolio_value':current_value,
                'profit_pct':profit_pct,
                'drawdown_pct':drawdown_pct,
                'stablecoin_allocation':stablecoin_allocation
            })

        return stablecoin_allocation,stablecoin_weights

    def _threshold_based_allocation (self,profit_pct):
        """
        Calcule l'allocation en stablecoin basée sur des seuils de profit.
        """
        if profit_pct < self.profit_threshold:
            return 0.0

        # Plus le profit est élevé, plus nous préservons
        preservation_rate=self.base_preservation_rate * (1 + (profit_pct - self.profit_threshold))
        preservation_rate=min (preservation_rate,0.8)  # Plafonner à 80%

        return profit_pct * preservation_rate

    def _volatility_based_allocation (self,profit_pct,volatility):
        """
        Calcule l'allocation en stablecoin basée sur la volatilité du marché.
        """
        if profit_pct < self.profit_threshold or volatility is None:
            return 0.0

        # Ajuster le taux de préservation en fonction de la volatilité
        # Plus la volatilité est élevée, plus nous préservons
        vol_factor=min (3.0,max (0.5,volatility / 0.02))  # 2% comme référence
        preservation_rate=self.base_preservation_rate * vol_factor

        return profit_pct * preservation_rate

    def _drawdown_based_allocation (self,profit_pct,drawdown_pct):
        """
        Calcule l'allocation en stablecoin basée sur le drawdown actuel.
        """
        if profit_pct < self.profit_threshold:
            return 0.0

        # Plus le drawdown est important, plus nous préservons agressivement
        drawdown_factor=1.0
        if drawdown_pct < 0:
            # Convertir le drawdown en positif pour le calcul
            abs_drawdown=abs (drawdown_pct)
            drawdown_factor=1.0 + (abs_drawdown * self.drawdown_sensitivity)

        preservation_rate=self.base_preservation_rate * drawdown_factor

        return profit_pct * preservation_rate

    def _time_based_allocation (self,profit_pct,current_date):
        """
        Calcule l'allocation en stablecoin basée sur des intervalles temporels.
        """
        if profit_pct < self.profit_threshold:
            return 0.0

        # Vérifier si le temps écoulé depuis la dernière préservation est suffisant
        days_elapsed=(current_date - self.last_preservation_date).days
        if days_elapsed < self.time_interval:
            return self.current_stablecoin_allocation

        # Préserver un peu plus à chaque intervalle de temps
        intervals_passed=days_elapsed // self.time_interval
        additional_rate=0.05 * min (intervals_passed,5)  # Maximum +25% après 5 intervalles

        preservation_rate=self.base_preservation_rate + additional_rate
        self.last_preservation_date=current_date

        return profit_pct * preservation_rate

    def _hybrid_allocation (self,profit_pct,drawdown_pct,volatility,current_date,market_drawdown):
        """
        Calcule l'allocation en stablecoin en combinant plusieurs stratégies.
        """
        if profit_pct < self.profit_threshold:
            return 0.0

        # Calculer l'allocation selon chaque stratégie
        threshold_alloc=self._threshold_based_allocation (profit_pct)
        drawdown_alloc=self._drawdown_based_allocation (profit_pct,drawdown_pct)

        # Calculer l'allocation basée sur la volatilité si disponible
        vol_alloc=0.0
        if volatility is not None:
            vol_alloc=self._volatility_based_allocation (profit_pct,volatility)

        # Calculer l'allocation basée sur le temps
        time_alloc=self._time_based_allocation (profit_pct,current_date)

        # Facteur de marché global - réduire la préservation si le marché est déjà en forte baisse
        # car c'est potentiellement un bon moment pour rester investi
        market_factor=1.0
        if market_drawdown < -0.15:  # Drawdown de plus de 15%
            market_factor=0.7  # Réduire la préservation de 30%
        elif market_drawdown < -0.25:  # Drawdown de plus de 25%
            market_factor=0.5  # Réduire la préservation de 50%

        # Pondération des différentes stratégies selon les conditions actuelles
        weights={
            'threshold':0.3,
            'drawdown':0.3,
            'volatility':0.2 if volatility is not None else 0,
            'time':0.2
        }

        # Normaliser les poids
        weight_sum=sum (weights.values ())
        weights={k:v / weight_sum for k,v in weights.items ()}

        # Combiner les allocations
        combined_allocation=(
                weights['threshold'] * threshold_alloc +
                weights['drawdown'] * drawdown_alloc +
                weights['volatility'] * vol_alloc +
                weights['time'] * time_alloc
        )

        # Appliquer le facteur de marché
        adjusted_allocation=combined_allocation * market_factor

        return adjusted_allocation

    def _distribute_stablecoin_allocation (self,stablecoin_allocation,current_weights=None):
        """
        Répartit l'allocation en stablecoin entre les différents stablecoins disponibles.
        """
        if stablecoin_allocation <= 0:
            return {}

        # Par défaut, utiliser uniquement le premier stablecoin disponible
        stablecoin_weights={self.stablecoin_assets[0]:stablecoin_allocation}

        # Si des poids actuels sont fournis, tenter de minimiser les transactions
        if current_weights:
            # Calculer les allocations actuelles en stablecoins
            current_stablecoin_weights={
                asset:weight for asset,weight in current_weights.items ()
                if asset in self.stablecoin_assets
            }

            # Si des stablecoins sont déjà présents, ajuster progressivement
            if current_stablecoin_weights:
                current_total=sum (current_stablecoin_weights.values ())

                if current_total > 0:
                    # Distribuer la nouvelle allocation proportionnellement aux allocations actuelles
                    scale_factor=stablecoin_allocation / current_total
                    stablecoin_weights={
                        asset:min (weight * scale_factor,stablecoin_allocation)
                        for asset,weight in current_stablecoin_weights.items ()
                    }

                    # S'assurer que la somme est correcte
                    total=sum (stablecoin_weights.values ())
                    if total > 0:
                        stablecoin_weights={
                            asset:(weight / total) * stablecoin_allocation
                            for asset,weight in stablecoin_weights.items ()
                        }
                    else:
                        stablecoin_weights={self.stablecoin_assets[0]:stablecoin_allocation}
            else:
                # Aucun stablecoin présent, choisir celui par défaut
                stablecoin_weights={self.stablecoin_assets[0]:stablecoin_allocation}

        return stablecoin_weights

    def adjust_allocation_weights (self,target_weights,current_value,current_date,
                                   market_drawdown=0.0,volatility=None,current_weights=None):
        """
        Ajuste les poids d'allocation cibles pour intégrer la préservation en stablecoin.
        """
        # Calculer l'allocation recommandée en stablecoin
        stablecoin_allocation,stablecoin_weights=self.calculate_preservation_allocation (
            current_value,current_date,market_drawdown,volatility,current_weights
        )

        # Si aucune allocation en stablecoin, retourner les poids d'origine
        if stablecoin_allocation <= 0 or not stablecoin_weights:
            return target_weights

        # Calculer le facteur de réduction pour les poids non-stablecoin
        non_stablecoin_allocation=1.0 - stablecoin_allocation
        if non_stablecoin_allocation <= 0:
            return stablecoin_weights  # 100% en stablecoin

        # Ajuster les poids des actifs non-stablecoin
        adjusted_weights={}

        # Ajouter d'abord les stablecoins
        for asset,weight in stablecoin_weights.items ():
            adjusted_weights[asset]=weight

        # Filtrer les stablecoins des poids cibles
        crypto_weights={
            asset:weight for asset,weight in target_weights.items ()
            if asset not in self.stablecoin_assets
        }

        # Redistribuer les poids restants
        total_crypto_weight=sum (crypto_weights.values ())
        if total_crypto_weight > 0:
            for asset,weight in crypto_weights.items ():
                adjusted_weights[asset]=(weight / total_crypto_weight) * non_stablecoin_allocation

        return adjusted_weights

    def calculate_preserved_capital (self,initial_capital,current_value):
        """
        Calcule le capital préservé grâce à la stratégie de stablecoin.
        """
        if not self.preservation_history:
            return 0.0

        # Calculer le profit total réalisé
        profit=current_value - initial_capital
        if profit <= 0:
            return 0.0

        # Estimation du capital préservé basée sur l'historique d'allocation
        preserved_capital=0.0
        for i,record in enumerate (self.preservation_history[:-1]):
            next_record=self.preservation_history[i + 1]

            # Calcul de l'augmentation de l'allocation en stablecoin
            allocation_increase=max (0,next_record['stablecoin_allocation'] - record['stablecoin_allocation'])
            portfolio_value=record['portfolio_value']

            # Le capital préservé est la valeur du portefeuille au moment de l'augmentation
            # multiplié par l'augmentation de l'allocation
            preserved_capital+=portfolio_value * allocation_increase

        return preserved_capital

    def generate_report (self):
        """
        Génère un rapport sur la performance de la stratégie de préservation.
        """
        if not self.preservation_history:
            return {"status":"No preservation activity yet"}

        # Calcul des métriques
        initial_value=self.initial_portfolio_value
        last_record=self.preservation_history[-1]
        current_value=last_record['portfolio_value']
        current_allocation=last_record['stablecoin_allocation']

        # Calcul du profit et du montant préservé
        profit_pct=(current_value / initial_value) - 1.0
        preserved_amount=self.calculate_preserved_capital (initial_value,current_value)
        preservation_ratio=preserved_amount / max (1,current_value - initial_value)

        # Statistiques d'allocation
        avg_allocation=np.mean ([r['stablecoin_allocation'] for r in self.preservation_history])
        max_allocation=max ([r['stablecoin_allocation'] for r in self.preservation_history])

        return {
            "strategy":self.strategy.value,
            "initial_value":initial_value,
            "current_value":current_value,
            "profit_percentage":f"{profit_pct * 100:.2f}%",
            "current_stablecoin_allocation":f"{current_allocation * 100:.2f}%",
            "preserved_capital":preserved_amount,
            "preservation_ratio":f"{preservation_ratio * 100:.2f}%",
            "average_allocation":f"{avg_allocation * 100:.2f}%",
            "maximum_allocation":f"{max_allocation * 100:.2f}%",
            "preservation_events":len (self.preservation_history),
            "last_preservation_date":self.last_preservation_date,
            "ready_for_reinvestment":self.reinvestment_ready
        }
#==============================================================================
# FONCTIONS DE SÉLECTION DES ACTIFS
#==============================================================================

def select_portfolio_assets(G, momentum_scores, volatility, current_date):
    """
    Sélectionne les actifs en combinant centralité, communautés et momentum.
    
    Args:
        G (nx.Graph): Graphe de dépendance
        momentum_scores (pd.DataFrame): Scores de momentum par actif
        volatility (pd.DataFrame): Volatilité par actif
        current_date (pd.Timestamp): Date de l'analyse
    
    Returns:
        list: Liste des actifs sélectionnés pour le portefeuille
    """

    # Vérifier que le graphe contient des nœuds
    if len (G.nodes ()) == 0:
        # Sélection par défaut basée uniquement sur le momentum récent
        latest_momentum=momentum_scores.loc[momentum_scores.index <= current_date]
        if len (latest_momentum) > 0:
            latest_momentum=latest_momentum.iloc[-1]
            sorted_assets=latest_momentum.sort_values (ascending=False)
            return sorted_assets.index[:Config.portfolio_size].tolist ()
        else:
            # Si aucune donnée n'est disponible, renvoyer une sélection par défaut
            return Config.tickers[:Config.portfolio_size]

    # Calculer les métriques de centralité en tenant compte des graphes déconnectés
    try:
        eigenvector_centrality,betweenness_centrality,closeness_centrality=calculate_centrality_for_disconnected_graph (
            G)
    except Exception as e:
        print (f"Erreur lors du calcul des centralités avec la nouvelle méthode: {e}")
        # Fallback à des valeurs par défaut
        eigenvector_centrality={node:1.0 / len (G.nodes ()) for node in G.nodes ()}
        betweenness_centrality={node:1.0 / len (G.nodes ()) for node in G.nodes ()}
        closeness_centrality={node:1.0 / len (G.nodes ()) for node in G.nodes ()}

    # Combiner les métriques en un score composite de centralité
    composite_centrality={}
    for node in G.nodes ():
        composite_centrality[node]=(
                eigenvector_centrality.get (node,0) * 0.5 +
                betweenness_centrality.get (node,0) * 0.3 +
                closeness_centrality.get (node,0) * 0.2
        )

    # Le reste de la fonction reste inchangé...
    # Obtenir le momentum récent
    latest_momentum=momentum_scores.loc[momentum_scores.index <= current_date]
    if len (latest_momentum) > 0:
        latest_momentum=latest_momentum.iloc[-1]
    else:
        latest_momentum=pd.Series (0,index=G.nodes ())

    # Obtenir la volatilité récente
    latest_volatility=volatility.loc[volatility.index <= current_date]
    if len (latest_volatility) > 0:
        latest_volatility=latest_volatility.iloc[-1]
    else:
        latest_volatility=pd.Series (0.01,index=G.nodes ())

    # Calculer un score combiné pour chaque actif
    combined_scores={}
    for node in G.nodes ():
        momentum_component=latest_momentum.get (node,0) if node in latest_momentum.index else 0
        volatility_component=1.0 / (latest_volatility.get (node,0.01) if node in latest_volatility.index else 0.01)

        # Normaliser les composants entre 0 et 1
        momentum_min=latest_momentum.min ()
        momentum_max=latest_momentum.max ()
        momentum_component=(momentum_component - momentum_min) / (
                    momentum_max - momentum_min) if momentum_max > momentum_min else 0.5

        # Normaliser la volatilité inversée (lower volatility = higher score)
        vol_inverse_values=[1.0 / v for v in latest_volatility if v > 0]
        if vol_inverse_values:
            vol_min=min (vol_inverse_values)
            vol_max=max (vol_inverse_values)
            volatility_component=(volatility_component - vol_min) / (vol_max - vol_min) if vol_max > vol_min else 0.5
        else:
            volatility_component=0.5

        # Combiner en un score final
        combined_scores[node]=(
                composite_centrality[node] * 0.5 +
                momentum_component * 0.3 +
                volatility_component * 0.2
        )

    # Trier les actifs par score combiné
    sorted_assets=sorted (combined_scores.items (),key=lambda x:x[1],reverse=True)

    # Détecter les communautés pour assurer la diversification
    try:
        # Détection de communautés adaptée pour les graphes déconnectés
        communities=[]
        connected_components=list (nx.connected_components (G))

        # Pour chaque composante connexe de plus de 1 nœud, détecter les communautés
        for component in connected_components:
            if len (component) > 2:
                # Appliquer l'algorithme de détection de communautés
                subgraph=G.subgraph (component)
                component_communities=list (nx.community.greedy_modularity_communities (subgraph))
                communities.extend (component_communities)
            else:
                # Ajouter les petites composantes directement comme "communautés"
                communities.append (component)

        # Sélectionner les meilleurs actifs de chaque communauté
        selected_assets=[]
        community_counts={}

        # D'abord, prendre le meilleur actif de chaque communauté
        for community in communities:
            community_assets=[(asset,combined_scores[asset]) for asset in community if asset in combined_scores]
            if community_assets:
                community_assets.sort (key=lambda x:x[1],reverse=True)
                best_asset=community_assets[0][0]
                selected_assets.append (best_asset)
                community_counts[tuple (sorted (community))]=1

        # Ensuite, compléter avec les meilleurs actifs restants
        remaining_slots=Config.portfolio_size - len (selected_assets)
        if remaining_slots > 0:
            # Prendre les meilleurs actifs non encore sélectionnés
            for asset,score in sorted_assets:
                if asset not in selected_assets and len (selected_assets) < Config.portfolio_size:
                    selected_assets.append (asset)

        return selected_assets

    except Exception as e:
        print (f"Erreur lors de la détection des communautés: {e}")
        # Fallback: prendre simplement les meilleurs actifs par score
        return [asset for asset,_ in sorted_assets[:Config.portfolio_size]]

#==============================================================================
# FONCTIONS D'ALLOCATION DE PORTEFEUILLE
#==============================================================================

def allocate_portfolio(selected_assets, precision_matrix, returns, momentum_scores, volatility, current_date, previous_weights=None):
    """
    Alloue le portefeuille en combinant structure de risque, momentum et stabilité.
    
    Args:
        selected_assets (list): Liste des actifs sélectionnés
        precision_matrix (np.ndarray): Matrice de précision
        returns (pd.DataFrame): Rendements historiques
        momentum_scores (pd.DataFrame): Scores de momentum
        volatility (pd.DataFrame): Volatilité historique
        current_date (pd.Timestamp): Date de l'analyse
        previous_weights (dict, optional): Poids précédents pour limiter le turnover
    
    Returns:
        dict: Poids alloués à chaque actif
    """
    if not selected_assets:
        return {}
    
    # Si la matrice de précision est nulle ou vide, utiliser une allocation simplifiée
    if precision_matrix.size == 0 or np.all(precision_matrix == 0):
        # Allocation équipondérée par défaut
        return {asset: 1.0 / len(selected_assets) for asset in selected_assets}
    
    # Extraire la sous-matrice de précision pour les actifs sélectionnés
    asset_indices = [Config.tickers.index(asset) for asset in selected_assets if asset in Config.tickers]
    
    # S'assurer qu'il y a des indices valides
    if not asset_indices:
        return {asset: 1.0 / len(selected_assets) for asset in selected_assets}
    
    sub_precision = precision_matrix[np.ix_(asset_indices, asset_indices)]
    
    # Vérifier que la matrice est inversible
    try:
        # Inverse de la matrice de précision = matrice de covariance estimée
        sub_covariance = np.linalg.inv(sub_precision)
    except np.linalg.LinAlgError:
        # Si non inversible, ajouter une petite perturbation à la diagonale
        diagonal_adjustment = np.eye(len(asset_indices)) * 1e-5
        try:
            sub_covariance = np.linalg.inv(sub_precision + diagonal_adjustment)
        except:
            # Si toujours non inversible, utiliser une allocation équipondérée
            return {asset: 1.0 / len(selected_assets) for asset in selected_assets}
    
    # Base de l'allocation: poids minimum variance
    ones = np.ones(len(selected_assets))
    try:
        # Calculer les poids de minimum variance
        min_var_weights = np.dot(sub_covariance, ones)
        min_var_weights = min_var_weights / np.sum(min_var_weights)
        # Assurer que tous les poids sont positifs
        min_var_weights = np.maximum(min_var_weights, 0)
        min_var_weights = min_var_weights / np.sum(min_var_weights)
    except:
        # Fallback: allocation équipondérée
        min_var_weights = np.ones(len(selected_assets)) / len(selected_assets)
    
    # Intégrer le momentum si disponible
    if momentum_scores is not None:
        latest_momentum = momentum_scores.loc[momentum_scores.index <= current_date]
        if len(latest_momentum) > 0:
            latest_momentum = latest_momentum.iloc[-1]
            
            # Extraire le momentum pour les actifs sélectionnés
            selected_momentum = [latest_momentum.get(asset, 0) for asset in selected_assets]
            
            # Normaliser les scores de momentum entre 0 et 1
            if max(selected_momentum) > min(selected_momentum):
                normalized_momentum = [(m - min(selected_momentum)) / (max(selected_momentum) - min(selected_momentum)) for m in selected_momentum]
            else:
                normalized_momentum = [0.5 for _ in selected_momentum]
            
            # Combiner avec les poids de min variance
            momentum_weights = np.array(normalized_momentum)
            momentum_weights = momentum_weights / np.sum(momentum_weights) if np.sum(momentum_weights) > 0 else np.ones(len(selected_assets)) / len(selected_assets)
            
            # Pondération entre min variance et momentum
            combined_weights = (1 - Config.momentum_weight) * min_var_weights + Config.momentum_weight * momentum_weights
        else:
            combined_weights = min_var_weights
    else:
        combined_weights = min_var_weights
    
    # Appliquer les contraintes de poids min/max
    constrained_weights = np.clip(combined_weights, Config.min_asset_weight, Config.max_asset_weight)
    
    # Renormaliser
    if np.sum(constrained_weights) > 0:
        constrained_weights = constrained_weights / np.sum(constrained_weights)
    else:
        constrained_weights = np.ones(len(selected_assets)) / len(selected_assets)
    
    # Appliquer les contraintes de turnover si des poids précédents sont disponibles
    if previous_weights:
        new_weights_dict = dict(zip(selected_assets, constrained_weights))
        final_weights = {}
        
        # Calculer le turnover total
        turnover = 0
        for asset in set(previous_weights.keys()) | set(new_weights_dict.keys()):
            prev_weight = previous_weights.get(asset, 0)
            new_weight = new_weights_dict.get(asset, 0)
            turnover += abs(new_weight - prev_weight)
        
        # Si le turnover dépasse le maximum autorisé, ajuster
        if turnover > Config.max_turnover:
            adjustment_factor = Config.max_turnover / turnover
            for asset in set(previous_weights.keys()) | set(new_weights_dict.keys()):
                prev_weight = previous_weights.get(asset, 0)
                new_weight = new_weights_dict.get(asset, 0)
                
                # Limiter l'ampleur du changement
                weight_change = (new_weight - prev_weight) * adjustment_factor
                final_weights[asset] = prev_weight + weight_change
        else:
            final_weights = new_weights_dict
        
        # Ne conserver que les actifs avec un poids positif
        final_weights = {asset: weight for asset, weight in final_weights.items() if weight > 0}
        
        # Renormaliser
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {asset: weight / total_weight for asset, weight in final_weights.items()}
        else:
            final_weights = {asset: 1.0 / len(selected_assets) for asset in selected_assets}
        
        return final_weights
    else:
        # Premier allocation
        return dict(zip(selected_assets, constrained_weights))

def apply_drawdown_control(portfolio_value, current_weights, portfolio_drawdown):
    """
    Applique un contrôle du drawdown en réduisant l'exposition lors des périodes de baisse.
    
    Args:
        portfolio_value (float): Valeur actuelle du portefeuille
        current_weights (dict): Poids actuels
        portfolio_drawdown (float): Drawdown actuel du portefeuille
    
    Returns:
        tuple: (adjusted_weights, cash_weight, is_protected)
            - adjusted_weights: Poids ajustés pour les actifs
            - cash_weight: Poids alloué au cash (0-1)
            - is_protected: True si protection active, False sinon
    """
    # Vérifier si le drawdown dépasse le seuil
    if portfolio_drawdown < Config.max_drawdown_threshold:
        # Réduire l'exposition
        adjusted_weights = {asset: weight * Config.risk_reduction_factor for asset, weight in current_weights.items()}
        # Le reste en "cash" (représenté par la différence à 1)
        cash_weight = 1 - sum(adjusted_weights.values())
        return adjusted_weights, cash_weight, True
    
    # Vérifier si on est en période de récupération
    elif portfolio_drawdown < 0 and portfolio_drawdown > Config.max_drawdown_threshold:
        # Récupération progressive
        recovery_factor = (portfolio_drawdown - Config.max_drawdown_threshold) / (0 - Config.max_drawdown_threshold)
        exposure_factor = Config.risk_reduction_factor + (1 - Config.risk_reduction_factor) * recovery_factor
        
        adjusted_weights = {asset: weight * exposure_factor for asset, weight in current_weights.items()}
        cash_weight = 1 - sum(adjusted_weights.values())
        return adjusted_weights, cash_weight, True
    
    # Pas de modification nécessaire
    else:
        return current_weights, 0, False

#==============================================================================
# FONCTIONS DE PRÉSERVATION DES GAINS
#==============================================================================

class GainPreservationModule:
    """
    Module de préservation des gains en stablecoin pour MHGNA.
    
    Permet de sécuriser progressivement les gains réalisés en les convertissant
    en stablecoin, tout en gardant une exposition au marché pour capturer
    les opportunités de croissance.
    """
    
    def __init__(
        self,
        strategy: PreservationStrategy = PreservationStrategy.HYBRID,
        profit_threshold: float = 0.15,         # 15% de profit déclenche une prise partielle
        max_stablecoin_allocation: float = 0.5, # Maximum 50% en stablecoin
        base_preservation_rate: float = 0.3,    # 30% des gains sont préservés par défaut
        drawdown_sensitivity: float = 2.0,      # Multiplicateur de sensibilité aux drawdowns
        time_interval: int = 30,                # Intervalle en jours pour la stratégie temporelle
        stablecoin_assets: List[str] = None,    # Liste des stablecoins disponibles
        reinvestment_threshold: float = -0.1    # -10% du marché pour réinvestir
    ):
        """
        Initialise le module de préservation des gains.
        
        Args:
            strategy (PreservationStrategy): Stratégie de préservation à utiliser
            profit_threshold (float): Seuil de profit pour commencer la préservation
            max_stablecoin_allocation (float): Allocation maximale en stablecoin (0-1)
            base_preservation_rate (float): Taux de base pour la préservation
            drawdown_sensitivity (float): Sensibilité aux drawdowns
            time_interval (int): Intervalle en jours pour la stratégie temporelle
            stablecoin_assets (List[str], optional): Liste des stablecoins
            reinvestment_threshold (float): Seuil de baisse pour réinvestir
        """
        self.strategy = strategy
        self.profit_threshold = profit_threshold
        self.max_stablecoin_allocation = max_stablecoin_allocation
        self.base_preservation_rate = base_preservation_rate
        self.drawdown_sensitivity = drawdown_sensitivity
        self.time_interval = time_interval
        self.stablecoin_assets = stablecoin_assets or ["USDT", "USDC", "DAI", "BUSD"]
        self.reinvestment_threshold = reinvestment_threshold
        
        # État interne
        self.initial_portfolio_value = None
        self.last_high_value = None
        self.last_rebalance_date = None
        self.current_stablecoin_allocation = 0.0
        self.preserved_gains = 0.0
        self.reinvestment_ready = False
        self.last_preservation_date = None
        self.preservation_history = []
    
    def initialize(self, initial_value: float, current_date: pd.Timestamp):
        """
        Initialise le module avec la valeur initiale du portefeuille.
        
        Args:
            initial_value (float): Valeur initiale du portefeuille
            current_date (pd.Timestamp): Date courante
        """
        self.initial_portfolio_value = initial_value
        self.last_high_value = initial_value
        self.last_rebalance_date = current_date
        self.last_preservation_date = current_date
    
    def calculate_preservation_allocation(
        self,
        current_value: float,
        current_date: pd.Timestamp,
        market_drawdown: float = 0.0,
        volatility: Optional[float] = None,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcule l'allocation recommandée en stablecoins pour préserver les gains.
        
        Args:
            current_value (float): Valeur actuelle du portefeuille
            current_date (pd.Timestamp): Date courante
            market_drawdown (float): Drawdown actuel du marché
            volatility (float, optional): Volatilité récente
            current_weights (Dict[str, float], optional): Poids actuels
        
        Returns:
            Tuple[float, Dict[str, float]]: 
                - Pourcentage d'allocation en stablecoin (0-1)
                - Répartition entre différents stablecoins
        """
        if self.initial_portfolio_value is None:
            self.initialize(current_value, current_date)
            return 0.0, {}
        
        # Mettre à jour le plus haut de portefeuille
        if current_value > self.last_high_value:
            self.last_high_value = current_value
        
        # Calculer le profit actuel
        profit_pct = (current_value / self.initial_portfolio_value) - 1.0
        drawdown_pct = (current_value / self.last_high_value) - 1.0
        
        # Allocation recommandée selon la stratégie
        if self.strategy == PreservationStrategy.THRESHOLD_BASED:
            stablecoin_allocation = self._threshold_based_allocation(profit_pct)
        elif self.strategy == PreservationStrategy.VOLATILITY_BASED:
            stablecoin_allocation = self._volatility_based_allocation(profit_pct, volatility)
        elif self.strategy == PreservationStrategy.DRAWDOWN_BASED:
            stablecoin_allocation = self._drawdown_based_allocation(profit_pct, drawdown_pct)
        elif self.strategy == PreservationStrategy.TIME_BASED:
            stablecoin_allocation = self._time_based_allocation(profit_pct, current_date)
        elif self.strategy == PreservationStrategy.HYBRID:
            stablecoin_allocation = self._hybrid_allocation(
                profit_pct, drawdown_pct, volatility, current_date, market_drawdown
            )
        else:
            stablecoin_allocation = 0.0
        
        # Limiter l'allocation à la valeur maximale configurée
        stablecoin_allocation = min(stablecoin_allocation, self.max_stablecoin_allocation)
        
        # Vérifier si nous devrions réinvestir plutôt que préserver
        if market_drawdown <= self.reinvestment_threshold and self.current_stablecoin_allocation > 0.05:
            self.reinvestment_ready = True
            stablecoin_allocation = max(0, stablecoin_allocation - 0.1)  # Réduire progressivement
        else:
            self.reinvestment_ready = False
        
        # Répartition entre les différents stablecoins
        stablecoin_weights = self._distribute_stablecoin_allocation(stablecoin_allocation, current_weights)
        
        # Mettre à jour l'état interne
        if abs(stablecoin_allocation - self.current_stablecoin_allocation) > 0.02:
            self.last_rebalance_date = current_date
            self.current_stablecoin_allocation = stablecoin_allocation
            
            # Enregistrer l'historique
            self.preservation_history.append({
                'date': current_date,
                'portfolio_value': current_value,
                'profit_pct': profit_pct,
                'drawdown_pct': drawdown_pct,
                'stablecoin_allocation': stablecoin_allocation
            })
        
        return stablecoin_allocation, stablecoin_weights
    
    def _threshold_based_allocation(self, profit_pct: float) -> float:
        """
        Calcule l'allocation en stablecoin basée sur des seuils de profit.
        
        Args:
            profit_pct (float): Pourcentage de profit actuel
            
        Returns:
            float: Allocation recommandée en stablecoin (0-1)
        """
        if profit_pct < self.profit_threshold:
            return 0.0
        
        # Plus le profit est élevé, plus nous préservons
        preservation_rate = self.base_preservation_rate * (1 + (profit_pct - self.profit_threshold))
        preservation_rate = min(preservation_rate, 0.8)  # Plafonner à 80%
        
        return profit_pct * preservation_rate
    
    def _volatility_based_allocation(self, profit_pct: float, volatility: Optional[float]) -> float:
        """
        Calcule l'allocation en stablecoin basée sur la volatilité du marché.
        
        Args:
            profit_pct (float): Pourcentage de profit actuel
            volatility (float, optional): Volatilité récente du marché
            
        Returns:
            float: Allocation recommandée en stablecoin (0-1)
        """
        if profit_pct < self.profit_threshold or volatility is None:
            return 0.0
        
        # Ajuster le taux de préservation en fonction de la volatilité
        # Plus la volatilité est élevée, plus nous préservons
        vol_factor = min(3.0, max(0.5, volatility / 0.02))  # 2% comme référence
        preservation_rate = self.base_preservation_rate * vol_factor
        
        return profit_pct * preservation_rate
    
    def _drawdown_based_allocation(self, profit_pct: float, drawdown_pct: float) -> float:
        """
        Calcule l'allocation en stablecoin basée sur le drawdown actuel.
        
        Args:
            profit_pct (float): Pourcentage de profit actuel
            drawdown_pct (float): Pourcentage de drawdown actuel
            
        Returns:
            float: Allocation recommandée en stablecoin (0-1)
        """
        if profit_pct < self.profit_threshold:
            return 0.0
        
        # Plus le drawdown est important, plus nous préservons agressivement
        drawdown_factor = 1.0
        if drawdown_pct < 0:
            # Convertir le drawdown en positif pour le calcul
            abs_drawdown = abs(drawdown_pct)
            drawdown_factor = 1.0 + (abs_drawdown * self.drawdown_sensitivity)
        
        preservation_rate = self.base_preservation_rate * drawdown_factor
        
        return profit_pct * preservation_rate
    
    def _time_based_allocation(self, profit_pct: float, current_date: pd.Timestamp) -> float:
        """
        Calcule l'allocation en stablecoin basée sur des intervalles temporels.
        
        Args:
            profit_pct (float): Pourcentage de profit actuel
            current_date (pd.Timestamp): Date courante
            
        Returns:
            float: Allocation recommandée en stablecoin (0-1)
        """
        if profit_pct < self.profit_threshold:
            return 0.0
        
        # Vérifier si le temps écoulé depuis la dernière préservation est suffisant
        days_elapsed = (current_date - self.last_preservation_date).days
        if days_elapsed < self.time_interval:
            return self.current_stablecoin_allocation
        
        # Préserver un peu plus à chaque intervalle de temps
        intervals_passed = days_elapsed // self.time_interval
        additional_rate = 0.05 * min(intervals_passed, 5)  # Maximum +25% après 5 intervalles
        
        preservation_rate = self.base_preservation_rate + additional_rate
        self.last_preservation_date = current_date
        
        return profit_pct * preservation_rate
    
    def _hybrid_allocation(
        self,
        profit_pct: float,
        drawdown_pct: float,
        volatility: Optional[float],
        current_date: pd.Timestamp,
        market_drawdown: float
    ) -> float:
        """
        Calcule l'allocation en stablecoin en combinant plusieurs stratégies.
        
        Args:
            profit_pct (float): Pourcentage de profit actuel
            drawdown_pct (float): Pourcentage de drawdown actuel
            volatility (float, optional): Volatilité récente du marché
            current_date (pd.Timestamp): Date courante
            market_drawdown (float): Drawdown actuel du marché global
            
        Returns:
            float: Allocation recommandée en stablecoin (0-1)
        """
        if profit_pct < self.profit_threshold:
            return 0.0
        
        # Calculer l'allocation selon chaque stratégie
        threshold_alloc = self._threshold_based_allocation(profit_pct)
        drawdown_alloc = self._drawdown_based_allocation(profit_pct, drawdown_pct)
        
        # Calculer l'allocation basée sur la volatilité si disponible
        vol_alloc = 0.0
        if volatility is not None:
            vol_alloc = self._volatility_based_allocation(profit_pct, volatility)
        
        # Calculer l'allocation basée sur le temps
        time_alloc = self._time_based_allocation(profit_pct, current_date)
        
        # Facteur de marché global - réduire la préservation si le marché est déjà en forte baisse
        # car c'est potentiellement un bon moment pour rester investi
        market_factor = 1.0
        if market_drawdown < -0.15:  # Drawdown de plus de 15%
            market_factor = 0.7  # Réduire la préservation de 30%
        elif market_drawdown < -0.25:  # Drawdown de plus de 25%
            market_factor = 0.5  # Réduire la préservation de 50%
        
        # Pondération des différentes stratégies selon les conditions actuelles
        weights = {
            'threshold': 0.3,
            'drawdown': 0.3,
            'volatility': 0.2 if volatility is not None else 0,
            'time': 0.2
        }
        
        # Normaliser les poids
        weight_sum = sum(weights.values())
        weights = {k: v / weight_sum for k, v in weights.items()}
        
        # Combiner les allocations
        combined_allocation = (
            weights['threshold'] * threshold_alloc +
            weights['drawdown'] * drawdown_alloc +
            weights['volatility'] * vol_alloc +
            weights['time'] * time_alloc
        )
        
        # Appliquer le facteur de marché
        adjusted_allocation = combined_allocation * market_factor
        
        return adjusted_allocation
    
    def _distribute_stablecoin_allocation(
            self,
            stablecoin_allocation: float,
            current_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
            """
            Répartit l'allocation en stablecoin entre les différents stablecoins disponibles.

            Args:
                stablecoin_allocation (float): Allocation totale en stablecoin (0-1)
                current_weights (Dict[str, float], optional): Poids actuels du portefeuille

            Returns:
                Dict[str, float]: Répartition entre différents stablecoins
            """
            if stablecoin_allocation <= 0:
                return {}

            # Par défaut, utiliser uniquement le premier stablecoin disponible
            stablecoin_weights = {self.stablecoin_assets[0]: stablecoin_allocation}

            # Si des poids actuels sont fournis, tenter de minimiser les transactions
            if current_weights:
                # Calculer les allocations actuelles en stablecoins
                current_stablecoin_weights = {
                    asset: weight for asset, weight in current_weights.items()
                    if asset in self.stablecoin_assets
                }

                # Si des stablecoins sont déjà présents, ajuster progressivement
                if current_stablecoin_weights:
                    current_total = sum(current_stablecoin_weights.values())

                    if current_total > 0:
                        # Distribuer la nouvelle allocation proportionnellement aux allocations actuelles
                        scale_factor = stablecoin_allocation / current_total
                        stablecoin_weights = {
                            asset: min(weight * scale_factor, stablecoin_allocation)
                            for asset, weight in current_stablecoin_weights.items()
                        }

                        # S'assurer que la somme est correcte
                        total = sum(stablecoin_weights.values())
                        if total > 0:
                            stablecoin_weights = {
                                asset: (weight / total) * stablecoin_allocation
                                for asset, weight in stablecoin_weights.items()
                            }
                        else:
                            stablecoin_weights = {self.stablecoin_assets[0]: stablecoin_allocation}
                else:
                    # Aucun stablecoin présent, choisir celui par défaut
                    stablecoin_weights = {self.stablecoin_assets[0]: stablecoin_allocation}

            return stablecoin_weights

    def _distribute_stablecoin_allocation(
            self,
            stablecoin_allocation: float,
            current_weights: Optional[Dict[str, float]] = None
        ) -> Dict[str, float]:
            """
            Répartit l'allocation en stablecoin entre les différents stablecoins disponibles.

            Args:
                stablecoin_allocation (float): Allocation totale en stablecoin (0-1)
                current_weights (Dict[str, float], optional): Poids actuels du portefeuille

            Returns:
                Dict[str, float]: Répartition entre différents stablecoins
            """
            if stablecoin_allocation <= 0:
                return {}

            # Par défaut, utiliser uniquement le premier stablecoin disponible
            stablecoin_weights = {self.stablecoin_assets[0]: stablecoin_allocation}

            # Si des poids actuels sont fournis, tenter de minimiser les transactions
            if current_weights:
                # Calculer les allocations actuelles en stablecoins
                current_stablecoin_weights = {
                    asset: weight for asset, weight in current_weights.items()
                    if asset in self.stablecoin_assets
                }

                # Si des stablecoins sont déjà présents, ajuster progressivement
                if current_stablecoin_weights:
                    current_total = sum(current_stablecoin_weights.values())

                    if current_total > 0:
                        # Distribuer la nouvelle allocation proportionnellement aux allocations actuelles
                        scale_factor = stablecoin_allocation / current_total
                        stablecoin_weights = {
                            asset: min(weight * scale_factor, stablecoin_allocation)
                            for asset, weight in current_stablecoin_weights.items()
                        }

                        # S'assurer que la somme est correcte
                        total = sum(stablecoin_weights.values())
                        if total > 0:
                            stablecoin_weights = {
                                asset: (weight / total) * stablecoin_allocation
                                for asset, weight in stablecoin_weights.items()
                            }
                        else:
                            stablecoin_weights = {self.stablecoin_assets[0]: stablecoin_allocation}
                else:
                    # Aucun stablecoin présent, choisir celui par défaut
                    stablecoin_weights = {self.stablecoin_assets[0]: stablecoin_allocation}

            return stablecoin_weights

    def adjust_allocation_weights(
            self,
            target_weights: Dict[str, float],
            current_value: float,
            current_date: pd.Timestamp,
            market_drawdown: float = 0.0,
            volatility: Optional[float] = None,
            current_weights: Optional[Dict[str, float]] = None
        ) -> Dict[str, float]:
            """
            Ajuste les poids d'allocation cibles pour intégrer la préservation en stablecoin.

            Args:
                target_weights (Dict[str, float]): Poids cibles d'origine
                current_value (float): Valeur actuelle du portefeuille
                current_date (pd.Timestamp): Date courante
                market_drawdown (float): Drawdown actuel du marché
                volatility (float, optional): Volatilité récente
                current_weights (Dict[str, float], optional): Poids actuels

            Returns:
                Dict[str, float]: Poids cibles ajustés incluant les stablecoins
            """
            # Calculer l'allocation recommandée en stablecoin
            stablecoin_allocation, stablecoin_weights = self.calculate_preservation_allocation(
                current_value, current_date, market_drawdown, volatility, current_weights
            )

            # Si aucune allocation en stablecoin, retourner les poids d'origine
            if stablecoin_allocation <= 0 or not stablecoin_weights:
                return target_weights

            # Calculer le facteur de réduction pour les poids non-stablecoin
            non_stablecoin_allocation = 1.0 - stablecoin_allocation
            if non_stablecoin_allocation <= 0:
                return stablecoin_weights  # 100% en stablecoin

            # Ajuster les poids des actifs non-stablecoin
            adjusted_weights = {}

            # Ajouter d'abord les stablecoins
            for asset, weight in stablecoin_weights.items():
                adjusted_weights[asset] = weight

            # Filtrer les stablecoins des poids cibles
            crypto_weights = {
                asset: weight for asset, weight in target_weights.items()
                if asset not in self.stablecoin_assets
            }

            # Redistribuer les poids restants
            total_crypto_weight = sum(crypto_weights.values())
            if total_crypto_weight > 0:
                for asset, weight in crypto_weights.items():
                    adjusted_weights[asset] = (weight / total_crypto_weight) * non_stablecoin_allocation

            return adjusted_weights

    def calculate_preserved_capital(self, initial_capital: float, current_value: float) -> float:
            """
            Calcule le capital préservé grâce à la stratégie de stablecoin.

            Args:
                initial_capital (float): Capital initial
                current_value (float): Valeur actuelle du portefeuille

            Returns:
                float: Montant du capital préservé en valeur absolue
            """
            if not self.preservation_history:
                return 0.0

            # Calculer le profit total réalisé
            profit = current_value - initial_capital
            if profit <= 0:
                return 0.0

            # Estimation du capital préservé basée sur l'historique d'allocation
            preserved_capital = 0.0
            for i, record in enumerate(self.preservation_history[:-1]):
                next_record = self.preservation_history[i+1]

                # Calcul de l'augmentation de l'allocation en stablecoin
                allocation_increase = max(0, next_record['stablecoin_allocation'] - record['stablecoin_allocation'])
                portfolio_value = record['portfolio_value']

                # Le capital préservé est la valeur du portefeuille au moment de l'augmentation
                # multiplié par l'augmentation de l'allocation
                preserved_capital += portfolio_value * allocation_increase

            return preserved_capital


    def generate_report(self) -> Dict[str, Union[float, str, List]]:
            """
            Génère un rapport sur la performance de la stratégie de préservation.

            Returns:
                Dict: Rapport contenant les métriques clés
            """
            if not self.preservation_history:
                return {"status": "No preservation activity yet"}

            # Calcul des métriques
            initial_value = self.initial_portfolio_value
            last_record = self.preservation_history[-1]
            current_value = last_record['portfolio_value']
            current_allocation = last_record['stablecoin_allocation']

            # Calcul du profit et du montant préservé
            profit_pct = (current_value / initial_value) - 1.0
            preserved_amount = self.calculate_preserved_capital(initial_value, current_value)
            preservation_ratio = preserved_amount / max(1, current_value - initial_value)

            # Statistiques d'allocation
            avg_allocation = np.mean([r['stablecoin_allocation'] for r in self.preservation_history])
            max_allocation = max([r['stablecoin_allocation'] for r in self.preservation_history])

            return {
                "strategy": self.strategy.value,
                "initial_value": initial_value,
                "current_value": current_value,
                "profit_percentage": f"{profit_pct * 100:.2f}%",
                "current_stablecoin_allocation": f"{current_allocation * 100:.2f}%",
                "preserved_capital": preserved_amount,
                "preservation_ratio": f"{preservation_ratio * 100:.2f}%",
                "average_allocation": f"{avg_allocation * 100:.2f}%",
                "maximum_allocation": f"{max_allocation * 100:.2f}%",
                "preservation_events": len(self.preservation_history),
                "last_preservation_date": self.last_preservation_date,
                "ready_for_reinvestment": self.reinvestment_ready
            }

#==============================================================================
# FONCTIONS DE BACKTEST
#==============================================================================

# Modifiez la fonction backtest_strategy pour supprimer la vérification problématique
# ou déplacez-la à un endroit approprié
def backtest_strategy (data,returns,volatility,momentum_scores,use_stablecoins=False):
    """
    Backteste la stratégie MHGNA avec contrôle du risque.

    Args:
        data (pd.DataFrame): Données de prix
        returns (pd.DataFrame): Rendements journaliers
        volatility (pd.DataFrame): Volatilité glissante
        momentum_scores (pd.DataFrame): Scores de momentum
        use_stablecoins (bool): Si True, utilise le module de préservation des gains

    Returns:
        tuple: (results, graph_history, weight_history, rebalance_dates, drawdown_history)
    """
    print ("Démarrage du backtest...")

    # Vérification des données
    if data.empty or returns.empty:
        print ("ERREUR: Les données sont vides. Vérification des données de Yahoo Finance nécessaire.")
        return pd.DataFrame (),[],[],[],[]

    # Afficher quelques informations sur les données
    print (f"Vérification des données: data contient {len (data)} jours")
    print (f"Premier jour: {data.index[0]}, dernier jour: {data.index[-1]}")
    print (f"Exemple de 3 premiers tickers:")
    print (data.iloc[:3,:3])

    # Initialisation
    portfolio_value=[Config.initial_capital]
    benchmark_value=[Config.initial_capital]
    current_date=pd.Timestamp (Config.start_date) + timedelta (days=max (Config.horizons['long']['window'],90))
    end_date=pd.Timestamp (Config.end_date)

    # S'assurer que la date de début est dans la plage des données
    current_date=max (current_date,data.index.min ())
    print (f"Date de début du backtest: {current_date}")

    last_rebalance_date=current_date
    current_portfolio={}
    portfolio_history=[]
    graph_history=[]
    weight_history=[]

    # Suivi du cash pour le contrôle du drawdown
    cash_allocation=0.0
    in_drawdown_protection=False

    # Historique du drawdown
    portfolio_cumulative_return=1.0
    portfolio_peak=1.0
    portfolio_drawdown=0.0
    drawdown_history=[]

    # Dates de rebalancement
    rebalance_dates=[]

    # Module de préservation des gains (si activé)
    preservation_module=None
    if use_stablecoins:
        preservation_module=GainPreservationModule (
            strategy=PreservationStrategy.HYBRID,
            profit_threshold=0.15,
            max_stablecoin_allocation=0.3
        )
        preservation_module.initialize (Config.initial_capital,current_date)

    # Liste pour stocker les dates effectives
    actual_dates=[]

    # Compteur pour le suivi de progression
    day_counter=0

    # Avancer jour par jour
    while current_date <= end_date and current_date <= data.index.max ():
        # Montrer la progression tous les 30 jours
        day_counter+=1
        if day_counter % 30 == 0:
            print (f"Traitement du jour {day_counter}: {current_date}")

        # Trouver la date réelle la plus proche dans les données (pour gérer les jours sans données)
        available_dates=data.index[data.index <= current_date]
        if len (available_dates) == 0:
            # Avancer à la première date disponible
            next_dates=data.index[data.index > current_date]
            if len (next_dates) == 0:
                break  # Plus de données disponibles
            current_date=next_dates[0]
            continue

        actual_date=available_dates[-1]
        actual_dates.append (actual_date)  # Stocker la date effective

        # Récupérer les données jusqu'à la date actuelle
        data_until_current=data.loc[:actual_date]

        # Vérifier si on doit rééquilibrer (basé sur la fréquence ou un événement de drawdown)
        days_since_rebalance=(current_date - last_rebalance_date).days
        should_rebalance=days_since_rebalance >= Config.rebalance_freq
        forced_rebalance=False

        # Vérifier s'il y a un drawdown significatif
        if len (portfolio_value) > 1:
            portfolio_cumulative_return=portfolio_value[-1] / Config.initial_capital
            portfolio_peak=max (portfolio_value) / Config.initial_capital
            portfolio_drawdown=(portfolio_cumulative_return - portfolio_peak) / portfolio_peak
            drawdown_history.append (portfolio_drawdown)

            # Si on est en protection contre le drawdown et qu'on récupère
            if in_drawdown_protection and portfolio_drawdown > Config.recovery_threshold:
                should_rebalance=True
                forced_rebalance=True
                in_drawdown_protection=False
                print (f"Sortie de la protection de drawdown à {current_date.strftime ('%Y-%m-%d')}")

            # Si on n'est pas en protection et qu'on a un drawdown significatif
            elif not in_drawdown_protection and portfolio_drawdown < Config.max_drawdown_threshold:
                should_rebalance=True
                forced_rebalance=True
                in_drawdown_protection=True
                print (f"Activation de la protection de drawdown à {current_date.strftime ('%Y-%m-%d')}")

        if should_rebalance and len (data_until_current) > Config.horizons['moyen']['window']:
            try:
                # Construire le graphe de dépendance multi-horizon
                G,precision_matrix=build_multi_horizon_dependency_graph (returns,actual_date)

                # Pour quelques dates, sauvegarder le graphe pour visualisation
                if len (graph_history) < 5 or forced_rebalance:
                    graph_history.append ((actual_date,G,precision_matrix,momentum_scores,volatility))

                # Sélectionner les actifs pour le portefeuille
                current_prices=data_until_current.iloc[-1]
                selected_assets=select_portfolio_assets (G,momentum_scores,volatility,actual_date)

                # Afficher les actifs sélectionnés
                print (f"Actifs sélectionnés pour {current_date}: {selected_assets}")

                # Allouer le portefeuille
                weights=allocate_portfolio (selected_assets,precision_matrix,returns,
                                            momentum_scores,volatility,actual_date,
                                            previous_weights=current_portfolio if current_portfolio else None)

                # Appliquer la préservation des gains si activée
                if use_stablecoins and preservation_module:
                    # Calculer le drawdown du marché (en utilisant le benchmark)
                    if len (benchmark_value) > 1:
                        benchmark_peak=max (benchmark_value) / Config.initial_capital
                        benchmark_current=benchmark_value[-1] / Config.initial_capital
                        market_drawdown=(benchmark_current - benchmark_peak) / benchmark_peak
                    else:
                        market_drawdown=0.0

                    # Récupérer la volatilité moyenne récente
                    recent_volatility=None
                    if not volatility.empty:
                        recent_volatility_values=volatility.loc[volatility.index <= actual_date]
                        if not recent_volatility_values.empty:
                            recent_volatility=recent_volatility_values.iloc[-1].mean ()

                    # Ajuster les poids avec la préservation des gains
                    try:
                        weights=preservation_module.adjust_allocation_weights (
                            weights,portfolio_value[-1],actual_date,
                            market_drawdown,recent_volatility,current_portfolio
                        )
                    except Exception as e:
                        print (f"Erreur lors de l'ajustement des poids pour la préservation: {e}")

                # Appliquer le contrôle du drawdown si nécessaire
                weights,cash,is_protected=apply_drawdown_control (
                    portfolio_value[-1],weights,portfolio_drawdown
                )
                in_drawdown_protection=is_protected
                cash_allocation=cash

                # Mettre à jour le portefeuille
                current_portfolio=weights
                last_rebalance_date=current_date
                rebalance_dates.append (current_date)

                # Enregistrer l'historique
                weight_history.append ((actual_date,weights,cash_allocation))

                portfolio_history.append ({
                    'date':actual_date,
                    'assets':selected_assets,
                    'weights':weights,
                    'cash':cash_allocation,
                    'drawdown':portfolio_drawdown
                })

                assets_str=', '.join ([f'{a}: {w:.2f}' for a,w in weights.items ()])
                print (
                    f"Rééquilibrage à {actual_date.strftime ('%Y-%m-%d')}: {assets_str}, Cash: {cash_allocation:.2f}")

            except Exception as e:
                print (f"Erreur lors du rééquilibrage à {actual_date}: {e}")
                import traceback
                traceback.print_exc ()
                # Conserver le portefeuille actuel

        # Calculer la performance quotidienne
        if current_portfolio and actual_date in returns.index:
            daily_returns_data=returns.loc[actual_date]

            # Calculer le rendement du portefeuille (partie investie)
            portfolio_return=0.0
            for asset,weight in current_portfolio.items ():
                if asset in daily_returns_data.index:
                    portfolio_return+=weight * daily_returns_data[asset]

            # Prendre en compte la partie cash (qui ne génère pas de rendement)
            portfolio_return=portfolio_return * (1 - cash_allocation)

            # Limiter les rendements extrêmes (pour éviter les bugs potentiels)
            if abs (portfolio_return) > 0.5:  # 50% max par jour
                portfolio_return=np.sign (portfolio_return) * 0.5

            # Mettre à jour la valeur du portefeuille
            portfolio_value.append (portfolio_value[-1] * (1 + portfolio_return))

            # Mettre à jour la valeur du benchmark
            benchmark_return=daily_returns_data.get (Config.benchmark,0)
            benchmark_value.append (benchmark_value[-1] * (1 + benchmark_return))
        else:
            # Si pas de portfolio ou pas de données pour cette date, valeur inchangée
            if len (portfolio_value) > 0:
                portfolio_value.append (portfolio_value[-1])
            else:
                portfolio_value.append (Config.initial_capital)

            if len (benchmark_value) > 0:
                benchmark_value.append (benchmark_value[-1])
            else:
                benchmark_value.append (Config.initial_capital)

        # Avancer d'un jour
        current_date+=timedelta (days=1)

    print (f"Fin du backtest. Traitement de {len (actual_dates)} jours.")

    # Créer un DataFrame avec les résultats en utilisant les dates effectives
    if len (actual_dates) == 0:
        print ("Pas de dates valides pour le backtest!")
        return pd.DataFrame (),graph_history,weight_history,rebalance_dates,drawdown_history

    # S'assurer que toutes les listes ont la même longueur
    min_length=min (len (actual_dates),len (portfolio_value) - 1,len (benchmark_value) - 1)

    # Utiliser seulement le nombre d'entrées correspondant à la longueur minimale
    trimmed_dates=actual_dates[:min_length]
    trimmed_portfolio=portfolio_value[1:min_length + 1]  # On saute la première valeur (capital initial)
    trimmed_benchmark=benchmark_value[1:min_length + 1]

    # Créer le DataFrame avec les dates et valeurs ajustées
    results=pd.DataFrame ({
        'Portfolio Value':trimmed_portfolio,
        'Benchmark Value':trimmed_benchmark
    },index=trimmed_dates)

    # Ajouter les drawdowns si disponibles
    if drawdown_history:
        # Assurer que drawdown_history a aussi la bonne longueur
        trimmed_drawdown=drawdown_history[:min_length]
        if len (trimmed_drawdown) == len (results):
            results['Drawdown']=trimmed_drawdown

    # Vérifier les rendements du portefeuille
    portfolio_returns_check=results['Portfolio Value'].pct_change ().dropna ()
    if portfolio_returns_check.isna ().all () or portfolio_returns_check.eq (0).all ():
        print ("AVERTISSEMENT: Tous les rendements sont NaN ou proches de 0. Les données pourraient être mal traitées.")

    # Afficher quelques statistiques de base
    print ("\nStatistiques du portefeuille:")
    print (f"Valeur initiale: {Config.initial_capital:.2f}")
    print (f"Valeur finale: {results['Portfolio Value'].iloc[-1]:.2f}")
    print (f"Rendement total: {(results['Portfolio Value'].iloc[-1] / Config.initial_capital - 1) * 100:.2f}%")

    return results,graph_history,weight_history,rebalance_dates,drawdown_history

def analyze_performance(results):
    """
    Analyse les performances de la stratégie et calcule les métriques clés.
    
    Args:
        results (pd.DataFrame): DataFrame avec les valeurs du portefeuille et du benchmark
    
    Returns:
        tuple: (perf_df, portfolio_drawdown, benchmark_drawdown, portfolio_returns, benchmark_returns)
            - perf_df: DataFrame avec les métriques de performance
            - portfolio_drawdown: Série des drawdowns du portefeuille
            - benchmark_drawdown: Série des drawdowns du benchmark
            - portfolio_returns: Série des rendements quotidiens du portefeuille
            - benchmark_returns: Série des rendements quotidiens du benchmark
    """
    # Vérifier que results contient des données
    if results.empty:
        print("Pas de données pour analyser les performances!")
        empty_df = pd.DataFrame({
            'Métrique': ['Rendement Total', 'Rendement Annualisé', 'Volatilité Annualisée',
                        'Ratio de Sharpe', 'Ratio de Sortino', 'Maximum Drawdown', 'Ratio de Calmar'],
            'Stratégie MHGNA': ['N/A'] * 7,
            'Benchmark': ['N/A'] * 7
        })
        return empty_df, pd.Series(), pd.Series(), pd.Series(), pd.Series()
    
    # Calcul des rendements quotidiens
    portfolio_returns = results['Portfolio Value'].pct_change().dropna()
    benchmark_returns = results['Benchmark Value'].pct_change().dropna()
    
    # Rendements cumulés
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    # Volatilité annualisée (ajustée pour les marchés crypto qui opèrent 365 jours/an)
    portfolio_vol = portfolio_returns.std() * np.sqrt(365)
    benchmark_vol = benchmark_returns.std() * np.sqrt(365)
    
    # Rendement annualisé
    days = (results.index[-1] - results.index[0]).days
    if days <= 0:
        days = 1  # Éviter la division par zéro
    
    portfolio_annual_return = ((results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0]) ** (365 / days)) - 1
    benchmark_annual_return = ((results['Benchmark Value'].iloc[-1] / results['Benchmark Value'].iloc[0]) ** (365 / days)) - 1
    
    # Sharpe Ratio (supposant un taux sans risque de 0%)
    portfolio_sharpe = portfolio_annual_return / portfolio_vol if portfolio_vol != 0 else 0
    benchmark_sharpe = benchmark_annual_return / benchmark_vol if benchmark_vol != 0 else 0
    
    # Sortino Ratio (volatilité des rendements négatifs seulement)
    neg_portfolio_returns = portfolio_returns[portfolio_returns < 0]
    neg_benchmark_returns = benchmark_returns[benchmark_returns < 0]
    
    portfolio_downside_vol = neg_portfolio_returns.std() * np.sqrt(365) if not neg_portfolio_returns.empty else 0.001
    benchmark_downside_vol = neg_benchmark_returns.std() * np.sqrt(365) if not neg_benchmark_returns.empty else 0.001
    
    portfolio_sortino = portfolio_annual_return / portfolio_downside_vol if portfolio_downside_vol != 0 else 0
    benchmark_sortino = benchmark_annual_return / benchmark_downside_vol if benchmark_downside_vol != 0 else 0
    
    # Maximum Drawdown
    portfolio_peak = portfolio_cumulative.cummax()
    benchmark_peak = benchmark_cumulative.cummax()
    
    portfolio_drawdown = (portfolio_cumulative - portfolio_peak) / portfolio_peak
    benchmark_drawdown = (benchmark_cumulative - benchmark_peak) / benchmark_peak
    
    portfolio_max_drawdown = portfolio_drawdown.min()
    benchmark_max_drawdown = benchmark_drawdown.min()
    
    # Calmar Ratio (rendement annualisé / max drawdown absolu)
    portfolio_calmar = portfolio_annual_return / abs(portfolio_max_drawdown) if portfolio_max_drawdown != 0 else 0
    benchmark_calmar = benchmark_annual_return / abs(benchmark_max_drawdown) if benchmark_max_drawdown != 0 else 0
    
    # Créer un DataFrame de résultats
    performance = {
        'Métrique': [
            'Rendement Total', 
            'Rendement Annualisé', 
            'Volatilité Annualisée',
            'Ratio de Sharpe', 
            'Ratio de Sortino', 
            'Maximum Drawdown', 
            'Ratio de Calmar'
        ],
        'Stratégie MHGNA': [
            f"{(results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0] - 1) * 100:.2f}%",
            f"{portfolio_annual_return * 100:.2f}%",
            f"{portfolio_vol * 100:.2f}%",
            f"{portfolio_sharpe:.2f}",
            f"{portfolio_sortino:.2f}",
            f"{portfolio_max_drawdown * 100:.2f}%",
            f"{portfolio_calmar:.2f}"
        ],
        'Benchmark': [
            f"{(results['Benchmark Value'].iloc[-1] / results['Benchmark Value'].iloc[0] - 1) * 100:.2f}%",
            f"{benchmark_annual_return * 100:.2f}%",
            f"{benchmark_vol * 100:.2f}%",
            f"{benchmark_sharpe:.2f}",
            f"{benchmark_sortino:.2f}",
            f"{benchmark_max_drawdown * 100:.2f}%",
            f"{benchmark_calmar:.2f}"
        ]
    }
    
    perf_df = pd.DataFrame(performance)
    
    return perf_df, portfolio_drawdown, benchmark_drawdown, portfolio_returns, benchmark_returns

def plot_results(results, perf_df, portfolio_drawdown, benchmark_drawdown, graph_history, weight_history, rebalance_dates):
    """
    Visualise les résultats du backtest avec des graphiques détaillés.
    
    Args:
        results (pd.DataFrame): DataFrame avec les résultats du backtest
        perf_df (pd.DataFrame): DataFrame avec les métriques de performance
        portfolio_drawdown (pd.Series): Série des drawdowns du portefeuille
        benchmark_drawdown (pd.Series): Série des drawdowns du benchmark
        graph_history (list): Liste de tuples (date, G, precision_matrix, ...)
        weight_history (list): Liste de tuples (date, weights, cash)
        rebalance_dates (list): Liste des dates de rebalancement
    
    Returns:
        plt.Figure: Figure matplotlib avec les graphiques
    """
    plt.figure(figsize=(15, 25))

    # 1. Performance cumulée
    plt.subplot(5, 1, 1)
    plt.plot(results['Portfolio Value'] / results['Portfolio Value'].iloc[0], label='Stratégie MHGNA', linewidth=2, color='#1f77b4')
    plt.plot(results['Benchmark Value'] / results['Benchmark Value'].iloc[0], label=f'Benchmark ({Config.benchmark})', linewidth=2, alpha=0.7, color='#ff7f0e')
    
    # Ajouter des annotations pour les dates de rebalancement
    if rebalance_dates:
        for date in rebalance_dates:
            if date in results.index:
                rel_val = results.loc[date, 'Portfolio Value'] / results['Portfolio Value'].iloc[0]
                plt.scatter([date], [rel_val], color='red', s=30, zorder=5)
    
    plt.title('Performance Cumulée', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Valeur Relative', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # 2. Drawdowns
    plt.subplot(5, 1, 2)
    plt.plot(portfolio_drawdown, label='Stratégie MHGNA', linewidth=2, color='#1f77b4')
    plt.plot(benchmark_drawdown, label=f'Benchmark ({Config.benchmark})', linewidth=2, alpha=0.7, color='#ff7f0e')
    
    # Ajouter une ligne horizontale pour le seuil de drawdown
    plt.axhline(y=Config.max_drawdown_threshold, color='r', linestyle='--', alpha=0.7, label=f'Seuil de protection ({Config.max_drawdown_threshold*100}%)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    plt.title('Drawdowns', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # 3. Évolution des poids du portefeuille
    plt.subplot(5, 1, 3)

    # Créer un DataFrame pour l'évolution des poids
    all_assets = set()
    cash_available = False
    
    # Vérifier si weight_history contient des tuples à 2 ou 3 éléments
    has_cash = False
    if weight_history and len(weight_history[0]) >= 3:
        has_cash = True
    
    for item in weight_history:
        if has_cash:
            date, weights, cash = item  # Pour les tuples à 3 éléments
            all_assets.update(weights.keys())
            cash_available = True
        else:
            date, weights = item  # Pour les tuples à 2 éléments
            all_assets.update(weights.keys())
    
    # Ajouter une colonne pour le cash si disponible
    if cash_available:
        all_assets.add('CASH')
    
    # Créer un DataFrame avec tous les actifs
    weight_df = pd.DataFrame(index=[item[0] for item in weight_history], columns=list(all_assets))
    weight_df = weight_df.fillna(0)
    
# Remplir le DataFrame avec les poids
    for item in weight_history:
        if has_cash:
            date, weights, cash = item
            for asset, weight in weights.items():
                weight_df.loc[date, asset] = weight
            if cash_available:
                weight_df.loc[date, 'CASH'] = cash
        else:
            date, weights = item
            for asset, weight in weights.items():
                weight_df.loc[date, asset] = weight
    
    # Plot stacked area chart
    weight_df.plot(kind='area', stacked=True, ax=plt.gca(), cmap='viridis')
    plt.title('Évolution des Allocations du Portefeuille', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Allocation', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # 4. Rendements mensuels
    plt.subplot(5, 1, 4)
    
    # Calculer les rendements mensuels
    monthly_portfolio = results['Portfolio Value'].resample('M').last().pct_change()
    monthly_benchmark = results['Benchmark Value'].resample('M').last().pct_change()
    
    # Créer un DataFrame pour le bar plot
    monthly_df = pd.DataFrame({
        'MHGNA': monthly_portfolio,
        'Benchmark': monthly_benchmark
    })
    
    # Plot les barres avec des couleurs basées sur la valeur
    ax = plt.gca()
    
    # Afficher les barres pour chaque colonne séparément
    bar_width = 0.35
    index = np.arange(len(monthly_df.index))
    
    # MHGNA bars
    mhgna_bars = ax.bar(index - bar_width/2, monthly_df['MHGNA'], bar_width, 
                       label='MHGNA', alpha=0.8)
    # Benchmark bars
    btc_bars = ax.bar(index + bar_width/2, monthly_df['Benchmark'], bar_width, 
                     label='Benchmark', alpha=0.8)
    
    # Coloriser individuellement chaque barre selon qu'elle est positive ou négative
    for i, bar in enumerate(mhgna_bars):
        if i < len(monthly_df['MHGNA']):
            value = monthly_df['MHGNA'].iloc[i]
            if pd.notna(value):  # Vérifier si la valeur n'est pas NaN
                bar.set_color('green' if value >= 0 else 'red')
    
    for i, bar in enumerate(btc_bars):
        if i < len(monthly_df['Benchmark']):
            value = monthly_df['Benchmark'].iloc[i]
            if pd.notna(value):  # Vérifier si la valeur n'est pas NaN
                bar.set_color('green' if value >= 0 else 'red')
    
    # Configuration du graphique
    plt.title('Rendements Mensuels Comparés', fontsize=16, fontweight='bold')
    plt.xlabel('Mois', fontsize=12)
    plt.ylabel('Rendement (%)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    
    # Ajuster les ticks pour afficher les mois
    plt.xticks(index, [date.strftime('%Y-%m') for date in monthly_df.index], rotation=45)
    
    # Ajouter des étiquettes de valeur pour chaque barre
    for i, v in enumerate(monthly_df['MHGNA']):
        if pd.notna(v):  # Vérifier si la valeur n'est pas NaN
            plt.text(i - bar_width/2, v + 0.01 if v > 0 else v - 0.03, f'{v:.1%}', 
                     color='black', fontweight='bold', fontsize=9, rotation=90)
    
    for i, v in enumerate(monthly_df['Benchmark']):
        if pd.notna(v):  # Vérifier si la valeur n'est pas NaN
            plt.text(i + bar_width/2, v + 0.01 if v > 0 else v - 0.03, f'{v:.1%}', 
                     color='black', fontweight='bold', fontsize=9, rotation=90)

    # 5. Tableau de performance
    plt.subplot(5, 1, 5)
    plt.axis('off')
    
    # Afficher directement le tableau sans coloration conditionnelle
    table = plt.table(cellText=perf_df.values, colLabels=perf_df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Vérifier le nombre de colonnes avant de tenter une coloration conditionnelle
    if perf_df.shape[1] > 2:  # S'il y a au moins 3 colonnes (Métrique, MHGNA, Benchmark)
        # Colorer les cellules selon la performance
        for i in range(1, len(perf_df)):
            # Ignorer les lignes qui ne peuvent pas être directement comparées
            if 'N/A' in perf_df.iloc[i, 1] or 'N/A' in perf_df.iloc[i, 2]:
                continue
                
            # Extraire les valeurs numériques pour la comparaison
            strat_val = float(perf_df.iloc[i, 1].replace('%', '')) if '%' in perf_df.iloc[i, 1] else float(perf_df.iloc[i, 1])
            bench_val = float(perf_df.iloc[i, 2].replace('%', '')) if '%' in perf_df.iloc[i, 2] else float(perf_df.iloc[i, 2])
            
            # Déterminer quelle valeur est meilleure (en tenant compte du sens de la métrique)
            metrics_higher_is_better = [
                'Rendement Total', 'Rendement Annualisé', 'Ratio de Sharpe', 
                'Ratio de Sortino', 'Ratio de Calmar'
            ]
            metrics_lower_is_better = [
                'Volatilité Annualisée', 'Maximum Drawdown'
            ]
            
            metric_name = perf_df.iloc[i, 0]
            
            if metric_name in metrics_higher_is_better:
                better_is_strat = strat_val > bench_val
            elif metric_name in metrics_lower_is_better:
                better_is_strat = strat_val < bench_val
            else:
                better_is_strat = None
                
            if better_is_strat is not None:
                if better_is_strat:
                    table[(i+1, 1)].set_facecolor('#d8f3dc')  # Vert clair pour la stratégie
                    table[(i+1, 2)].set_facecolor('#ffccd5')  # Rouge clair pour le benchmark
                else:
                    table[(i+1, 1)].set_facecolor('#ffccd5')  # Rouge clair pour la stratégie
                    table[(i+1, 2)].set_facecolor('#d8f3dc')  # Vert clair pour le benchmark
    
    plt.title('Métriques de Performance', fontsize=16, fontweight='bold')

    plt.tight_layout(pad=3.0)
    return plt

def print_summary_report(results, perf_df, rebalance_dates):
    """
    Génère un rapport textuel synthétique des performances.
    
    Args:
        results (pd.DataFrame): DataFrame avec les résultats du backtest
        perf_df (pd.DataFrame): DataFrame avec les métriques de performance
        rebalance_dates (list): Liste des dates de rebalancement
    """
    print("\n" + "="*80)
    print("RAPPORT DE SYNTHÈSE")
    print("="*80)
    
    try:
        # Extraire les métriques clés
        strategy_return = float(perf_df.iloc[0, 1].replace('%', '')) / 100
        benchmark_return = float(perf_df.iloc[0, 2].replace('%', '')) / 100
        outperformance = strategy_return - benchmark_return
        
        strategy_sharpe = float(perf_df.iloc[3, 1])
        benchmark_sharpe = float(perf_df.iloc[3, 2])
        sharpe_improvement = strategy_sharpe - benchmark_sharpe
        
        strategy_drawdown = float(perf_df.iloc[5, 1].replace('%', '')) / 100
        benchmark_drawdown = float(perf_df.iloc[5, 2].replace('%', '')) / 100
        drawdown_improvement = benchmark_drawdown - strategy_drawdown
        
        # Informations générales
        print(f"Période d'analyse: {results.index[0].strftime('%Y-%m-%d')} à {results.index[-1].strftime('%Y-%m-%d')}")
        print(f"Nombre de jours: {len(results)}")
        print(f"Nombre de rebalancements: {len(rebalance_dates)}")
        print(f"Fréquence moyenne de rebalancement: {len(results)/max(1, len(rebalance_dates)):.1f} jours")
        
        # Performances
        print(f"\nPerformance MHGNA: {strategy_return*100:.2f}%")
        print(f"Performance {Config.benchmark}: {benchmark_return*100:.2f}%")
        print(f"Surperformance: {outperformance*100:.2f}%")
        
        # Risque
        print(f"\nRatio de Sharpe MHGNA: {strategy_sharpe:.2f}")
        print(f"Ratio de Sharpe {Config.benchmark}: {benchmark_sharpe:.2f}")
        print(f"Amélioration du Sharpe: {sharpe_improvement:.2f}")
        
        print(f"\nDrawdown maximum MHGNA: {strategy_drawdown*100:.2f}%")
        print(f"Drawdown maximum {Config.benchmark}: {benchmark_drawdown*100:.2f}%")
        print(f"Réduction du drawdown: {drawdown_improvement*100:.2f}%")
        
        # Conclusion
        print("\nCONCLUSION:")
        if outperformance > 0:
            print(f"✅ La stratégie MHGNA a SURPERFORMÉ le benchmark de {outperformance*100:.2f}%")
        else:
            print(f"⚠️ La stratégie MHGNA a SOUS-PERFORMÉ le benchmark de {-outperformance*100:.2f}%")
            
        if sharpe_improvement > 0:
            print(f"✅ Ratio de Sharpe AMÉLIORÉ de {sharpe_improvement:.2f}")
        else:
            print(f"⚠️ Ratio de Sharpe DÉTÉRIORÉ de {-sharpe_improvement:.2f}")
            
        if drawdown_improvement > 0:
            print(f"✅ Drawdown maximum RÉDUIT de {drawdown_improvement*100:.2f}%")
        else:
            print(f"⚠️ Drawdown maximum AUGMENTÉ de {-drawdown_improvement*100:.2f}%")
    
    except Exception as e:
        print(f"Erreur lors de la génération du rapport de synthèse: {e}")

#==============================================================================
# FONCTION DEBOGAGE
#==============================================================================
# Exécution avec débogage
def run_mhgna_with_debug (config_updates=None,use_stablecoins=True):
    """
    Exécute la stratégie MHGNA avec des options de débogage activées
    """
    # Configuration par défaut avec un petit ensemble d'actifs pour commencer
    default_config={
        'start_date':'2022-06-01',
        'end_date':'2023-12-31',
        'tickers':['BTC-USD','ETH-USD','SOL-USD','USDT-USD','BNB-USD'],
        'rebalance_freq':30,
        'portfolio_size':3
    }

    # Appliquer les mises à jour spécifiques
    if config_updates:
        for key,value in config_updates.items ():
            default_config[key]=value

    # Mettre à jour la configuration
    for key,value in default_config.items ():
        if hasattr (Config,key):
            setattr (Config,key,value)
            print (f"Configuration mise à jour: {key} = {value}")

    # Récupérer les données
    try:
        data,returns,volatility,momentum_scores=get_data ()
    except Exception as e:
        print (f"Erreur lors de la récupération des données: {e}")
        import traceback
        traceback.print_exc ()
        return None,None,None,None

    # Exécuter le backtest
    start_time=time.time ()
    try:
        results,graph_history,weight_history,rebalance_dates,drawdown_history=backtest_strategy (
            data,returns,volatility,momentum_scores,use_stablecoins=use_stablecoins
        )
    except Exception as e:
        print (f"Erreur lors du backtest: {e}")
        import traceback
        traceback.print_exc ()
        return None,None,None,None

    elapsed_time=time.time () - start_time
    print (f"Backtest terminé en {elapsed_time:.2f} secondes")

    # Analyser les performances
    perf_df,_,_,_,_=analyze_performance (results)
    print ("\n--- Résultats de la Stratégie MHGNA ---")
    print (perf_df)

    # Imprimer le rapport de synthèse
    print_summary_report (results,perf_df,rebalance_dates)

    return results,perf_df,graph_history,weight_history

#==============================================================================
# FONCTION PRINCIPALE
#==============================================================================

def run_mhgna_strategy(config_updates=None, use_stablecoins=False, save_visualizations=True):
    """
    Exécute la stratégie MHGNA complète avec visualisations.
    
    Args:
        config_updates (dict, optional): Mises à jour optionnelles des paramètres
        use_stablecoins (bool, optional): Si True, utilise la préservation des gains
        save_visualizations (bool, optional): Si True, sauvegarde les visualisations
        
    Returns:
        tuple: (results, perf_df, graph_history, weight_history)
            - results: DataFrame avec les résultats
            - perf_df: DataFrame avec les métriques de performance
            - graph_history: Liste des graphes de dépendance
            - weight_history: Liste des allocations
    """
    print("\n" + "="*80)
    print("EXÉCUTION DE LA STRATÉGIE MHGNA")
    print("="*80)
    
    # Mettre à jour la configuration si nécessaire
    if config_updates:
        for key, value in config_updates.items():
            if hasattr(Config, key):
                setattr(Config, key, value)
                print(f"Configuration mise à jour: {key} = {value}")
    
    # Récupérer les données
    print("\nChargement des données...")
    data, returns, volatility, momentum_scores = get_data()
    print(f"Données chargées: {len(returns)} jours pour {len(returns.columns)} actifs")
    
    # Exécuter le backtest
    print("\nExécution du backtest...")
    start_time = time.time()
    results, graph_history, weight_history, rebalance_dates, drawdown_history = backtest_strategy(
        data, returns, volatility, momentum_scores, use_stablecoins
    )
    elapsed_time = time.time() - start_time
    print(f"Backtest terminé en {elapsed_time:.2f} secondes")
    
    if results.empty:
        print("Le backtest n'a pas produit de résultats valides.")
        return None, None, None, None
    
    # Analyser les performances
    print("\nAnalyse des performances...")
    perf_df, portfolio_drawdown, benchmark_drawdown, portfolio_returns, benchmark_returns = analyze_performance(results)
    
    # Afficher les résultats
    print("\n--- Résultats de la Stratégie MHGNA ---")
    print(perf_df)
    
    # Imprimer le rapport de synthèse
    print_summary_report(results, perf_df, rebalance_dates)
    
    # Générer les visualisations
    if save_visualizations:
        print("\nGénération des visualisations...")
        
        # Performance et allocation
        plt_results = plot_results(
            results, perf_df, portfolio_drawdown, benchmark_drawdown, 
            graph_history, weight_history, rebalance_dates
        )
        plt_results.savefig('mhgna_results.png', dpi=300, bbox_inches='tight')
        
        # Visualiser les graphes de dépendance
        if graph_history:
            dependency_graphs = []
            for i, item in enumerate(graph_history[:3]):  # Limiter à 3 graphes pour éviter de surcharger
                date, G, precision_matrix = item[0], item[1], item[2]
                momentum_scores_at_date = item[3] if len(item) > 3 else None
                volatility_at_date = item[4] if len(item) > 4 else None
                
                try:
                    plt_graph = plot_dependency_network(
                        G, precision_matrix, date, 
                        momentum_scores_at_date, volatility_at_date
                    )
                    plt_graph.savefig(f'dependency_graph_{i}_{date.strftime("%Y%m%d")}.png', dpi=300, bbox_inches='tight')
                    dependency_graphs.append(plt_graph)
                except Exception as e:
                    print(f"Erreur lors de la visualisation du graphe {i}: {e}")
        
        print("Visualisations sauvegardées.")
    
    return results, perf_df, graph_history, weight_history

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration personnalisée
    custom_config = {
        'start_date': '2022-01-01',
        'end_date': '2023-12-31',
        'rebalance_freq': 21,  # Rebalancement mensuel
        'portfolio_size': 7
    }
    
    # Exécuter la stratégie
    results, perf_df, graph_history, weight_history = run_mhgna_strategy(
        config_updates=custom_config,
        use_stablecoins=True,  # Activer la préservation des gains
        save_visualizations=True
    )