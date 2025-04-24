"""
MHGNA Simplifié pour Trading Manuel
===================================

Une version simplifiée du Multi-Horizon Graphical Network Allocation
optimisée pour le trading manuel de cryptomonnaies.

Cette version fournit:
1. Une sélection mensuelle des 3-5 actifs les plus prometteurs
2. Des signaux d'alerte clairs pour la réduction de l'exposition
3. Des visuels de la structure du marché pour comprendre les interdépendances

Auteur: [Votre Nom]
Date: Avril 2025
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
from scipy.stats import zscore
import warnings
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Ignorer les avertissements
warnings.filterwarnings ('ignore')


class SimpleConfig:
    """Configuration pour MHGNA Simplifié"""
    # Actifs à suivre (ajoutez ou supprimez selon vos préférences)
    tickers=[
        'BTC-USD','ETH-USD','SOL-USD','BNB-USD','XRP-USD',
        'ADA-USD','AVAX-USD','DOT-USD','MATIC-USD','LINK-USD',
        'UNI-USD','AAVE-USD','ATOM-USD','LTC-USD','DOGE-USD'
    ]

    # Stablecoins pour les allocations défensives
    stablecoins=['USDT-USD','USDC-USD','DAI-USD','BUSD-USD']

    # Horizons temporels
    horizons={
        'court':30,  # 30 jours
        'moyen':90,  # 90 jours
        'long':180  # 180 jours
    }

    # Paramètres d'alerte et de risque
    drawdown_alert_threshold=-0.15  # Alerte à -15% de drawdown
    volatility_alert_threshold=0.8  # Alerte quand vol > 80% de la vol max historique

    # Nombre d'actifs à recommander
    recommended_assets=5

    # Paramètres de visualisation
    chart_style='darkgrid'
    network_colors='viridis'
    figsize=(12,8)

    # Périodes de récupération de données (en années)
    lookback_period=1  # 1 an d'historique


class MHGNASimplified:
    """
    Version simplifiée de MHGNA pour le trading manuel.
    Fournit des recommandations d'actifs et des signaux d'alerte
    basés sur l'analyse de réseau multi-horizon.
    """

    def __init__ (self,config=None):
        """
        Initialise le système MHGNA simplifié.

        Args:
            config: Configuration personnalisée (utilise SimpleConfig par défaut)
        """
        self.config=config if config else SimpleConfig ()
        self.data=None
        self.returns=None
        self.volatility=None
        self.momentum=None
        self.current_date=None
        self.all_tickers=self.config.tickers + self.config.stablecoins
        self.drawdown_history={}
        self.network_graph=None
        self.last_recommendations=None
        self.alerts=[]

        # Configurer le style des graphiques
        sns.set_style (self.config.chart_style)

    def fetch_data (self,end_date=None):
        """
        Récupère les données historiques pour tous les actifs.

        Args:
            end_date: Date de fin (aujourd'hui par défaut)
        """
        if end_date is None:
            end_date=datetime.datetime.now ()
        else:
            end_date=pd.to_datetime (end_date)

        self.current_date=end_date

        # Calculer la date de début
        start_date=end_date - relativedelta (years=self.config.lookback_period)

        print (f"Récupération des données de {start_date.strftime ('%Y-%m-%d')} à {end_date.strftime ('%Y-%m-%d')}...")

        # Télécharger les données
        try:
            data=yf.download (self.all_tickers,start=start_date,end=end_date)

            # Traitement du cas où il y a un MultiIndex
            if isinstance (data.columns,pd.MultiIndex):
                # Extraire seulement les données de clôture
                self.data=pd.DataFrame ()
                for ticker in self.all_tickers:
                    if ('Close',ticker) in data.columns:
                        self.data[ticker]=data['Close',ticker]
            else:
                self.data=data['Close']

            # Calculer les rendements, la volatilité et le momentum
            self.returns=self.data.pct_change ().dropna ()
            self.volatility=self.returns.rolling (window=30).std ().dropna ()

            # Calculer le momentum sur plusieurs périodes
            momentum_short=self.returns.rolling (20).sum ()
            momentum_medium=self.returns.rolling (60).sum ()
            momentum_long=self.returns.rolling (120).sum ()

            # Score de momentum composite normalisé
            self.momentum=(
                    zscore (momentum_short,nan_policy='omit') * 0.5 +
                    zscore (momentum_medium,nan_policy='omit') * 0.3 +
                    zscore (momentum_long,nan_policy='omit') * 0.2
            )

            # Initialiser l'historique des drawdowns
            for ticker in self.config.tickers:
                if ticker in self.data.columns:
                    prices=self.data[ticker].dropna ()
                    peaks=prices.cummax ()
                    self.drawdown_history[ticker]=(prices / peaks - 1.0)

            print (f"Données récupérées pour {len (self.data.columns)} actifs sur {len (self.data)} jours.")
            return True

        except Exception as e:
            print (f"Erreur lors de la récupération des données: {e}")
            return False

    def build_network (self):
        """
        Construit le réseau multi-horizon de dépendances entre actifs.

        Returns:
            nx.Graph: Graphe représentant le réseau de dépendances
        """
        if self.data is None or self.returns is None:
            print ("Aucune donnée disponible. Exécutez fetch_data() d'abord.")
            return None

        print ("Construction du réseau multi-horizon...")

        # Graphes pour chaque horizon
        horizon_graphs={}
        horizon_matrices={}

        # Paramètres de régularisation par horizon
        alpha_params={
            'court':0.02,  # Plus sparse pour le court terme
            'moyen':0.01,  # Intermédiaire pour le moyen terme
            'long':0.005  # Plus dense pour le long terme
        }

        # Construction des graphes pour chaque horizon
        for horizon_name,days in self.config.horizons.items ():
            try:
                # Extraire les données pour cet horizon
                lookback_date=self.current_date - relativedelta (days=days)
                horizon_returns=self.returns[self.returns.index >= lookback_date].copy ()

                if len (horizon_returns) < days // 2:
                    print (f"Pas assez de données pour l'horizon {horizon_name}. Minimum requis: {days // 2} jours.")
                    continue

                # Construire le modèle Graphical Lasso pour cet horizon
                alpha=alpha_params[horizon_name]
                model=GraphicalLassoCV (alphas=[alpha * 0.5,alpha,alpha * 2],cv=5,max_iter=1000)

                # Filtrer pour n'inclure que les cryptos (pas les stablecoins)
                crypto_returns=horizon_returns[self.config.tickers].dropna (axis=1)

                # S'assurer qu'il reste suffisamment d'actifs
                if crypto_returns.shape[1] < 3:
                    print (f"Pas assez d'actifs avec des données pour l'horizon {horizon_name}.")
                    continue

                # Ajustement du modèle
                model.fit (crypto_returns)
                precision_matrix=model.precision_

                # Création du graphe pour cet horizon
                G=nx.Graph ()

                # Ajouter tous les nœuds
                for ticker in crypto_returns.columns:
                    G.add_node (ticker)

                # Ajouter les arêtes avec leur poids
                for i,ticker1 in enumerate (crypto_returns.columns):
                    for j,ticker2 in enumerate (crypto_returns.columns):
                        if i < j:  # Éviter les doublons
                            weight=abs (precision_matrix[i,j])
                            if weight > 0.01:  # Seuil minimal pour les arêtes
                                G.add_edge (ticker1,ticker2,weight=weight)

                horizon_graphs[horizon_name]=G
                horizon_matrices[horizon_name]=precision_matrix

                print (
                    f"Réseau pour horizon {horizon_name} construit: {G.number_of_nodes ()} nœuds, {G.number_of_edges ()} arêtes.")

            except Exception as e:
                print (f"Erreur lors de la construction du réseau pour l'horizon {horizon_name}: {e}")

        if not horizon_graphs:
            print ("Aucun graphe n'a pu être construit.")
            return None

        # Combiner les graphes en un seul réseau multi-horizon
        combined_graph=nx.Graph ()
        for ticker in self.config.tickers:
            if ticker in self.data.columns:
                combined_graph.add_node (ticker)

        # Poids des horizons
        horizon_weights={'court':0.25,'moyen':0.5,'long':0.25}

        # Combiner les arêtes de tous les horizons
        all_edges={}
        for horizon_name,graph in horizon_graphs.items ():
            weight=horizon_weights.get (horizon_name,0.33)

            for u,v,d in graph.edges (data=True):
                edge=tuple (sorted ([u,v]))
                if edge not in all_edges:
                    all_edges[edge]=0
                all_edges[edge]+=d['weight'] * weight

        # Ajouter les arêtes au graphe combiné
        for (u,v),weight in all_edges.items ():
            if weight > 0.01:  # Seuil minimal
                combined_graph.add_edge (u,v,weight=weight)

        print (
            f"Réseau multi-horizon combiné: {combined_graph.number_of_nodes ()} nœuds, {combined_graph.number_of_edges ()} arêtes.")
        self.network_graph=combined_graph
        return combined_graph

    def calculate_centrality_metrics (self):
        """
        Calcule les métriques de centralité pour chaque actif dans le réseau.

        Returns:
            pd.DataFrame: DataFrame avec les métriques de centralité
        """
        if self.network_graph is None:
            print ("Aucun réseau disponible. Exécutez build_network() d'abord.")
            return None

        print ("Calcul des métriques de centralité...")

        # Gérer les graphes déconnectés
        def calculate_centrality_for_components (G):
            # Initialiser les métriques
            eigenvector_cent={}
            betweenness_cent={}
            closeness_cent={}

            # Traiter chaque composante connectée séparément
            for component in nx.connected_components (G):
                subgraph=G.subgraph (component)

                # Calculer les métriques seulement si la composante a plus d'un nœud
                if len (subgraph) > 1:
                    try:
                        # Eigenvector centrality
                        if len (subgraph) > 2:
                            # Utiliser la matrice d'adjacence
                            adj_matrix=nx.to_numpy_array (subgraph,weight='weight')
                            eigenvalues,eigenvectors=np.linalg.eig (adj_matrix)
                            idx=np.argmax (eigenvalues.real)
                            eigvec=np.abs (eigenvectors[:,idx].real)
                            eigvec=eigvec / np.linalg.norm (eigvec)

                            # Assigner aux nœuds
                            for i,node in enumerate (subgraph.nodes ()):
                                eigenvector_cent[node]=eigvec[i]
                        else:
                            # Valeur par défaut pour 2 nœuds
                            for node in subgraph.nodes ():
                                eigenvector_cent[node]=0.5

                        # Autres centralités
                        betw=nx.betweenness_centrality (subgraph,weight='weight',normalized=True)
                        close=nx.closeness_centrality (subgraph,distance='weight')

                        betweenness_cent.update (betw)
                        closeness_cent.update (close)

                    except Exception as e:
                        print (f"Erreur dans le calcul des centralités pour une composante: {e}")
                        # Valeurs par défaut
                        for node in subgraph.nodes ():
                            eigenvector_cent[node]=1.0 / len (subgraph)
                            betweenness_cent[node]=1.0 / len (subgraph)
                            closeness_cent[node]=1.0 / len (subgraph)

                else:
                    # Nœud isolé
                    node=list (component)[0]
                    eigenvector_cent[node]=0.1
                    betweenness_cent[node]=0.0
                    closeness_cent[node]=0.0

            return eigenvector_cent,betweenness_cent,closeness_cent

        # Calculer les centralités
        try:
            eigenvector_centrality,betweenness_centrality,closeness_centrality=calculate_centrality_for_components (
                self.network_graph)

            # Créer un DataFrame avec toutes les métriques
            centrality_df=pd.DataFrame (index=self.network_graph.nodes ())
            centrality_df['eigenvector']=pd.Series (eigenvector_centrality)
            centrality_df['betweenness']=pd.Series (betweenness_centrality)
            centrality_df['closeness']=pd.Series (closeness_centrality)

            # Ajouter les derniers scores de momentum
            latest_momentum=self.momentum.iloc[-1] if not self.momentum.empty else pd.Series (0,
                                                                                              index=self.network_graph.nodes ())
            centrality_df['momentum']=latest_momentum

            # Ajouter la volatilité récente (inversée pour valoriser la faible volatilité)
            latest_volatility=self.volatility.iloc[-1] if not self.volatility.empty else pd.Series (0.01,
                                                                                                    index=self.network_graph.nodes ())
            # Inverser pour que les valeurs faibles soient meilleures
            inversed_volatility=1.0 / latest_volatility
            # Normaliser entre 0 et 1
            min_inv_vol=inversed_volatility.min ()
            max_inv_vol=inversed_volatility.max ()
            if max_inv_vol > min_inv_vol:
                normalized_inv_vol=(inversed_volatility - min_inv_vol) / (max_inv_vol - min_inv_vol)
            else:
                normalized_inv_vol=pd.Series (0.5,index=inversed_volatility.index)

            centrality_df['inverse_volatility']=normalized_inv_vol

            # Calculer la performance récente
            if self.data is not None:
                returns_30d=self.data.pct_change (periods=30).iloc[-1]
                returns_90d=self.data.pct_change (periods=90).iloc[-1]

                centrality_df['return_30d']=returns_30d
                centrality_df['return_90d']=returns_90d

            # Score composite (combinaison pondérée des métriques)
            centrality_df['composite_score']=(
                    centrality_df['eigenvector'].fillna (0) * 0.3 +
                    centrality_df['betweenness'].fillna (0) * 0.2 +
                    centrality_df['closeness'].fillna (0) * 0.1 +
                    centrality_df['momentum'].fillna (0) * 0.25 +
                    centrality_df['inverse_volatility'].fillna (0) * 0.15
            )

            return centrality_df

        except Exception as e:
            print (f"Erreur lors du calcul des métriques de centralité: {e}")
            return None

    def check_alerts (self):
        """
        Vérifie les conditions d'alerte et génère les signaux appropriés.

        Returns:
            list: Liste des alertes actives
        """
        if self.data is None:
            return []

        alerts=[]

        # 1. Vérifier les drawdowns excessifs
        for ticker,drawdown_series in self.drawdown_history.items ():
            if len (drawdown_series) > 0:
                current_drawdown=drawdown_series.iloc[-1]
                if current_drawdown < self.config.drawdown_alert_threshold:
                    alerts.append ({
                        'type':'DRAWDOWN',
                        'asset':ticker,
                        'value':current_drawdown,
                        'threshold':self.config.drawdown_alert_threshold,
                        'message':f"Alerte drawdown: {ticker} est en baisse de {current_drawdown:.2%} (seuil: {self.config.drawdown_alert_threshold:.2%})"
                    })

        # 2. Vérifier la volatilité excessive du marché
        if not self.volatility.empty:
            recent_volatility=self.volatility.iloc[-30:].mean ()
            max_volatility=self.volatility.max ()
            volatility_ratio=recent_volatility / max_volatility

            if volatility_ratio > self.config.volatility_alert_threshold:
                alerts.append ({
                    'type':'VOLATILITY',
                    'asset':'MARKET',
                    'value':volatility_ratio,
                    'threshold':self.config.volatility_alert_threshold,
                    'message':f"Alerte volatilité: Le marché est actuellement à {volatility_ratio:.2%} de sa volatilité maximale (seuil: {self.config.volatility_alert_threshold:.2%})"
                })

        # 3. Vérifier les divergences réseau-prix (actifs surachetés/survendus)
        if self.network_graph is not None and not self.momentum.empty:
            centrality_df=self.calculate_centrality_metrics ()
            if centrality_df is not None:
                # Identifier les cas où le momentum et la centralité sont très divergents
                for ticker in centrality_df.index:
                    if ticker in centrality_df['eigenvector'] and ticker in centrality_df['momentum']:
                        centrality=centrality_df.loc[ticker,'eigenvector']
                        momentum=centrality_df.loc[ticker,'momentum']

                        # Si actif central avec momentum très négatif (potentiellement survendu)
                        if centrality > 0.7 and momentum < -1.5:
                            alerts.append ({
                                'type':'OVERSOLD',
                                'asset':ticker,
                                'value':momentum,
                                'threshold':-1.5,
                                'message':f"Opportunité potentielle: {ticker} est potentiellement survendu (momentum: {momentum:.2f}, centralité: {centrality:.2f})"
                            })

                        # Si actif périphérique avec momentum très positif (potentiellement suracheté)
                        if centrality < 0.3 and momentum > 1.5:
                            alerts.append ({
                                'type':'OVERBOUGHT',
                                'asset':ticker,
                                'value':momentum,
                                'threshold':1.5,
                                'message':f"Attention risque: {ticker} est potentiellement suracheté (momentum: {momentum:.2f}, centralité: {centrality:.2f})"
                            })

        self.alerts=alerts
        return alerts

    def recommend_assets (self):
        """
        Génère des recommandations d'actifs basées sur les métriques de réseau et de marché.

        Returns:
            DataFrame: Actifs recommandés avec leurs scores
        """
        if self.network_graph is None:
            print ("Aucun réseau disponible. Exécutez build_network() d'abord.")
            return None

        centrality_df=self.calculate_centrality_metrics ()
        if centrality_df is None:
            return None

        # Trier par score composite
        ranked_assets=centrality_df.sort_values ('composite_score',ascending=False)

        # Sélectionner les N meilleurs actifs
        num_assets=min (self.config.recommended_assets,len (ranked_assets))
        top_assets=ranked_assets.head (num_assets)

        # Formater le résultat pour la présentation
        recommendations=top_assets.copy ()

        # Formater les rendements en pourcentage
        if 'return_30d' in recommendations.columns:
            recommendations['return_30d']=recommendations['return_30d'].map ('{:.2%}'.format)
        if 'return_90d' in recommendations.columns:
            recommendations['return_90d']=recommendations['return_90d'].map ('{:.2%}'.format)

        # Arrondir les scores
        numeric_columns=['eigenvector','betweenness','closeness','momentum','inverse_volatility','composite_score']
        recommendations[numeric_columns]=recommendations[numeric_columns].round (3)

        # Ajouter le rang
        recommendations['rank']=range (1,len (recommendations) + 1)

        # Réorganiser les colonnes
        column_order=['rank','composite_score','eigenvector','momentum','inverse_volatility','return_30d','return_90d']
        recommendations=recommendations[column_order]

        # Renommer les colonnes pour plus de clarté
        recommendations.columns=[
            'Rang','Score Global','Centralité','Momentum',
            'Stabilité','Rend. 30j','Rend. 90j'
        ]

        self.last_recommendations=recommendations
        return recommendations

    def visualize_network (self,filename=None):
        """
        Crée une visualisation du réseau de dépendances entre actifs.

        Args:
            filename (str, optional): Nom du fichier pour sauvegarder l'image

        Returns:
            plt.Figure: Figure matplotlib du réseau
        """
        if self.network_graph is None:
            print ("Aucun réseau disponible. Exécutez build_network() d'abord.")
            return None

        print ("Création de la visualisation du réseau...")

        # Créer la figure
        plt.figure (figsize=self.config.figsize)

        # Obtenir les métriques pour le graphe
        centrality_df=self.calculate_centrality_metrics ()
        G=self.network_graph

        # Définir le layout
        pos=nx.spring_layout (G,k=0.3,seed=42,iterations=100)

        # Déterminer les tailles des nœuds basées sur la centralité eigenvector
        node_sizes=[]
        for node in G.nodes ():
            if node in centrality_df.index:
                # Échelle: 100 à 1000 selon la centralité
                size=100 + centrality_df.loc[node,'eigenvector'] * 900
            else:
                size=100
            node_sizes.append (size)

        # Déterminer les couleurs des nœuds basées sur le momentum
        node_colors=[]
        for node in G.nodes ():
            if node in centrality_df.index:
                momentum=centrality_df.loc[node,'momentum']
                # Rouge pour momentum négatif, vert pour positif
                if momentum < 0:
                    # Rouge plus fort pour plus négatif
                    intensity=min (1.0,abs (momentum) / 2)
                    node_colors.append ((0.9,0.2 + (1 - intensity) * 0.6,0.2))
                else:
                    # Vert plus fort pour plus positif
                    intensity=min (1.0,momentum / 2)
                    node_colors.append ((0.2,0.7 + intensity * 0.3,0.2))
            else:
                # Gris par défaut
                node_colors.append ((0.7,0.7,0.7))

        # Déterminer les épaisseurs des arêtes basées sur leur poids
        edge_widths=[]
        for (u,v,d) in G.edges (data=True):
            # Échelle: 1 à 5 selon le poids
            width=1 + d['weight'] * 40
            edge_widths.append (width)

        # Dessiner le graphe
        nx.draw_networkx_nodes (G,pos,node_size=node_sizes,node_color=node_colors,alpha=0.8)
        nx.draw_networkx_edges (G,pos,width=edge_widths,alpha=0.5,edge_color='gray')

        # Ajouter les étiquettes uniquement pour les nœuds importants
        labels={}
        for node in G.nodes ():
            if node in centrality_df.index:
                if centrality_df.loc[node,'eigenvector'] > 0.3 or centrality_df.loc[node,'momentum'] > 1.0:
                    # Ajouter l'étiquette avec le score momentum
                    labels[node]=f"{node}\n({centrality_df.loc[node,'momentum']:.2f})"
                else:
                    labels[node]=node
            else:
                labels[node]=node

        # Dessiner les étiquettes
        nx.draw_networkx_labels (G,pos,labels=labels,font_size=10,font_weight='bold')

        # Ajouter des indications pour les couleurs
        red_patch=Patch (color=(0.9,0.2,0.2),label='Momentum Négatif')
        green_patch=Patch (color=(0.2,0.9,0.2),label='Momentum Positif')
        plt.legend (handles=[red_patch,green_patch],loc='upper right')

        # Ajouter un titre
        date_str=self.current_date.strftime ('%Y-%m-%d') if self.current_date else "Date actuelle"
        plt.title (f"Réseau de dépendance des cryptomonnaies - {date_str}",fontsize=16,pad=20)

        # Ajouter des détails sur les nœuds et arêtes
        plt.text (0.01,0.01,
                  f"Nœuds: {G.number_of_nodes ()}, Arêtes: {G.number_of_edges ()}",
                  transform=plt.gca ().transAxes,fontsize=10,
                  verticalalignment='bottom',horizontalalignment='left',
                  bbox=dict (facecolor='white',alpha=0.7,boxstyle='round,pad=0.5'))

        # Ajouter des indications pour l'interprétation
        plt.figtext (0.5,0.01,
                     "Taille = Centralité | Couleur = Momentum | Épaisseur des lignes = Force de la relation",
                     ha='center',fontsize=12,bbox=dict (facecolor='white',alpha=0.8,boxstyle='round,pad=0.5'))

        plt.axis ('off')
        plt.tight_layout ()

        # Sauvegarder si un nom de fichier est fourni
        if filename:
            plt.savefig (filename,dpi=300,bbox_inches='tight')
            print (f"Visualisation sauvegardée dans {filename}")

        return plt.gcf ()

    def visualize_market_trends (self,filename=None):
        """
        Crée une visualisation des tendances de marché avec des indicateurs d'alerte.

        Args:
            filename (str, optional): Nom du fichier pour sauvegarder l'image

        Returns:
            plt.Figure: Figure matplotlib des tendances
        """
        if self.data is None:
            print ("Aucune donnée disponible. Exécutez fetch_data() d'abord.")
            return None

        # Obtenir les recommandations si pas encore calculées
        if self.last_recommendations is None:
            self.recommend_assets ()

        # Sélectionner les actifs à afficher
        display_assets=list (
            self.last_recommendations.index) if self.last_recommendations is not None else self.config.tickers[:5]

        # Ajouter Bitcoin comme référence s'il n'est pas déjà inclus
        if 'BTC-USD' not in display_assets:
            display_assets.append ('BTC-USD')

        # Créer la figure avec 3 sous-graphiques
        fig,(ax1,ax2,ax3)=plt.subplots (3,1,figsize=(12,15),gridspec_kw={'height_ratios':[3,1,2]})

        # 1. Prix normalisés
        # ------------------
        # Sélectionner les 90 derniers jours de données
        last_90d=self.data.iloc[-90:].copy ()

        # Normaliser les prix à 100 pour comparer plus facilement
        normalized_data=last_90d[display_assets].div (last_90d[display_assets].iloc[0]) * 100

        # Tracer les prix normalisés
        for asset in display_assets:
            if asset in normalized_data.columns:
                asset_data=normalized_data[asset].dropna ()
                if len (asset_data) > 0:
                    # Mettre en évidence les actifs recommandés
                    if self.last_recommendations is not None and asset in self.last_recommendations.index:
                        rank=self.last_recommendations.loc[asset,'Rang']
                        linewidth=3 if rank <= 3 else 2
                        ax1.plot (asset_data.index,asset_data,linewidth=linewidth,
                                  label=f"{asset} (Rang {int (rank)})")
                    else:
                        ax1.plot (asset_data.index,asset_data,linewidth=1.5,alpha=0.7,label=asset)

        ax1.set_title ("Évolution des prix sur 90 jours (normalisée à 100)",fontsize=14)
        ax1.set_ylabel ("Prix normalisé")
        ax1.legend (loc='upper left')
        ax1.grid (True,alpha=0.3)

        # Formatter l'axe des dates
        ax1.xaxis.set_major_formatter (mdates.DateFormatter ('%d-%m'))
        ax1.xaxis.set_major_locator (mdates.MonthLocator ())

        # Ajouter les zones d'alerte pour les drawdowns du Bitcoin
        if 'BTC-USD' in self.drawdown_history:
            btc_drawdown=self.drawdown_history['BTC-USD'].iloc[-90:].copy ()
            for i,dd in enumerate (btc_drawdown):
                if dd < self.config.drawdown_alert_threshold:
                    ax1.axvspan (btc_drawdown.index[i],
                                 btc_drawdown.index[i + 1] if i + 1 < len (btc_drawdown) else btc_drawdown.index[-1],
                                 alpha=0.2,color='red')

        # 2. Signaux d'alerte
        # -----------------
        # Vérifier les alertes si pas encore fait
        if not self.alerts:
            self.check_alerts ()

        # Dessiner les alertes sous forme de heatmap
        alert_types=['DRAWDOWN','VOLATILITY','OVERSOLD','OVERBOUGHT']

        # Créer une matrice binaire pour la heatmap
        alert_data=[]
        y_labels=[]

        # Ajouter chaque actif avec ses alertes
        for asset in display_assets:
            asset_alerts=[0] * len (alert_types)
            for i,alert_type in enumerate (alert_types):
                # Vérifier si cet actif a cette alerte
                for alert in self.alerts:
                    if alert['type'] == alert_type and (alert['asset'] == asset or alert['asset'] == 'MARKET'):
                        asset_alerts[i]=1 if alert_type in ['DRAWDOWN','OVERBOUGHT',
                                                            'VOLATILITY'] else 2  # 2 pour les signaux positifs

            alert_data.append (asset_alerts)
            y_labels.append (asset)

        # Créer la heatmap seulement s'il y a des données
        if alert_data:
            # Créer une colormap personnalisée: gris, rouge, vert
            cmap=LinearSegmentedColormap.from_list ('alert_cmap',[(0.9,0.9,0.9),(0.9,0.2,0.2),(0.2,0.8,0.2)])

            # Tracer la heatmap
            im=ax2.imshow (alert_data,cmap=cmap,aspect='auto',vmin=0,vmax=2)

            # Configurer les étiquettes
            ax2.set_yticks (range (len (y_labels)))
            ax2.set_yticklabels (y_labels)
            ax2.set_xticks (range (len (alert_types)))
            ax2.set_xticklabels (['Drawdown','Volatilité','Survendu\n(opportunité)','Suracheté\n(risque)'])

            # Annoter les cellules
            for i in range (len (y_labels)):
                for j in range (len (alert_types)):
                    if alert_data[i][j] == 1:  # Alerte négative
                        ax2.text (j,i,"⚠️",ha="center",va="center",color="black")
                    elif alert_data[i][j] == 2:  # Alerte positive
                        ax2.text (j,i,"✓",ha="center",va="center",color="black")

            ax2.set_title ("Signaux d'alerte actuels",fontsize=14)
        else:
            ax2.text (0.5,0.5,"Aucun signal d'alerte actif",
                      horizontalalignment='center',verticalalignment='center',
                      fontsize=14,transform=ax2.transAxes)
            ax2.set_yticks ([])
            ax2.set_xticks ([])

        # 3. Métriques clés des actifs recommandés
        # --------------------------------------
        if self.last_recommendations is not None:
            # Extraire les métriques clés
            metrics=['Centralité','Momentum','Stabilité']
            top_assets=self.last_recommendations.index.tolist ()

            # Créer un tableau de données pour le graphique en barres
            bar_data=[]
            for asset in top_assets:
                asset_metrics=[self.last_recommendations.loc[asset,metric] for metric in metrics]
                bar_data.append (asset_metrics)

            # Tracer le graphique en barres
            x=np.arange (len (metrics))
            width=0.15

            for i,asset in enumerate (top_assets):
                offset=width * (i - len (top_assets) / 2 + 0.5)
                ax3.bar (x + offset,bar_data[i],width,label=asset)

            ax3.set_title ("Métriques clés des actifs recommandés",fontsize=14)
            ax3.set_xticks (x)
            ax3.set_xticklabels (metrics)
            ax3.set_ylabel ("Score normalisé")
            ax3.legend (loc='upper right')
            ax3.grid (True,alpha=0.3,axis='y')
        else:
            ax3.text (0.5,0.5,"Aucune recommandation disponible",
                      horizontalalignment='center',verticalalignment='center',
                      fontsize=14,transform=ax3.transAxes)
            ax3.set_yticks ([])
            ax3.set_xticks ([])

        # Ajuster la mise en page
        plt.tight_layout ()

        # Sauvegarder si un nom de fichier est fourni
        if filename:
            plt.savefig (filename,dpi=300,bbox_inches='tight')
            print (f"Visualisation des tendances sauvegardée dans {filename}")

        return fig

    def generate_report (self,output_folder='.'):
        """
        Génère un rapport complet avec recommandations, alertes et visualisations.

        Args:
            output_folder (str): Dossier où sauvegarder les fichiers du rapport

        Returns:
            str: Résumé du rapport
        """
        import os
        from datetime import datetime

        # Créer le dossier de sortie s'il n'existe pas
        if not os.path.exists (output_folder):
            os.makedirs (output_folder)

        # Date du rapport
        report_date=self.current_date.strftime ('%Y-%m-%d') if self.current_date else datetime.now ().strftime (
            '%Y-%m-%d')
        report_id=f"mhgna_report_{report_date.replace ('-','')}"

        # Générer les visualisations
        network_file=os.path.join (output_folder,f"{report_id}_network.png")
        trends_file=os.path.join (output_folder,f"{report_id}_trends.png")

        self.visualize_network (filename=network_file)
        self.visualize_market_trends (filename=trends_file)

        # Générer le texte du rapport
        report_text=f"""
RAPPORT MHGNA SIMPLIFIÉ - {report_date}
=====================================

1. ACTIFS RECOMMANDÉS
--------------------
"""

        if self.last_recommendations is not None:
            report_text+="\n" + self.last_recommendations.to_string () + "\n\n"
        else:
            recommendations=self.recommend_assets ()
            report_text+="\n" + recommendations.to_string () + "\n\n"

        report_text+="""
Interprétation:
- "Score Global": Score combiné basé sur toutes les métriques (plus élevé = meilleur)
- "Centralité": Position de l'actif dans le réseau (plus élevé = plus central et influent)
- "Momentum": Force et direction de la tendance récente (plus élevé = momentum plus fort)
- "Stabilité": Inverse de la volatilité (plus élevé = plus stable)

2. SIGNAUX D'ALERTE
------------------
"""

        if not self.alerts:
            self.check_alerts ()

        if self.alerts:
            for alert in self.alerts:
                report_text+=f"• {alert['message']}\n"
        else:
            report_text+="Aucun signal d'alerte actif actuellement.\n"

        report_text+="""

3. RÉSUMÉ DU MARCHÉ
------------------
"""

        # Ajouter un résumé du marché
        if self.data is not None:
            # Calculer des statistiques récentes
            recent_returns=self.data.pct_change (periods=30).iloc[-1]
            btc_return=recent_returns.get ('BTC-USD',0)
            eth_return=recent_returns.get ('ETH-USD',0)

            report_text+=f"""
Bitcoin (BTC): {btc_return:.2%} sur 30 jours
Ethereum (ETH): {eth_return:.2%} sur 30 jours

Structure du marché: Le réseau comprend {self.network_graph.number_of_nodes ()} actifs et {self.network_graph.number_of_edges ()} connections entre eux.
"""

        # Résumé des actions recommandées
        report_text+="""
4. ACTIONS RECOMMANDÉES
---------------------
"""

        if self.last_recommendations is not None:
            top_3=self.last_recommendations.head (3).index.tolist ()
            report_text+=f"""
• Considérer une position sur: {', '.join (top_3)}
"""

        # Ajouter des recommandations basées sur les alertes
        if any (alert['type'] == 'DRAWDOWN' for alert in self.alerts):
            report_text+="• Réduire l'exposition globale au marché en raison des drawdowns significatifs\n"

        if any (alert['type'] == 'VOLATILITY' for alert in self.alerts):
            report_text+="• Prudence conseillée en raison de la volatilité élevée du marché\n"

        oversold_assets=[alert['asset'] for alert in self.alerts if alert['type'] == 'OVERSOLD']
        if oversold_assets:
            report_text+=f"• Opportunités potentielles sur actifs survendus: {', '.join (oversold_assets)}\n"

        overbought_assets=[alert['asset'] for alert in self.alerts if alert['type'] == 'OVERBOUGHT']
        if overbought_assets:
            report_text+=f"• Vigilance sur actifs potentiellement surachetés: {', '.join (overbought_assets)}\n"

        # Si aucune recommandation spécifique n'a été ajoutée
        if "•" not in report_text.split ("4. ACTIONS RECOMMANDÉES")[1]:
            report_text+="• Maintenir les allocations actuelles et surveiller l'évolution du marché\n"

        # Ajouter la référence aux visualisations
        report_text+=f"""
5. VISUALISATIONS
---------------
• Réseau de dépendances: {network_file}
• Tendances et signaux: {trends_file}

Rapport généré le {datetime.now ().strftime ('%Y-%m-%d %H:%M')}
"""

        # Sauvegarder le rapport texte
        report_file=os.path.join (output_folder,f"{report_id}_summary.txt")
        with open (report_file,'w') as f:
            f.write (report_text)

        print (f"Rapport complet sauvegardé dans {report_file}")
        return report_text

    def run_monthly_analysis (self,end_date=None):
        """
        Exécute l'analyse complète pour un trading mensuel.

        Args:
            end_date: Date de fin de l'analyse (aujourd'hui par défaut)

        Returns:
            str: Résumé des recommandations mensuelles
        """
        # Récupérer les données
        self.fetch_data (end_date=end_date)

        # Construire le réseau
        self.build_network ()

        # Générer les recommandations
        recommendations=self.recommend_assets ()

        # Vérifier les alertes
        alerts=self.check_alerts ()

        # Générer et retourner le rapport
        return self.generate_report ()


# Fonction d'aide pour démarrer rapidement
def start_mhgna_analysis (tickers=None,lookback_period=1,recommended_assets=5,output_folder='.'):
    """
    Démarre rapidement une analyse MHGNA simplifiée.

    Args:
        tickers (list): Liste personnalisée de cryptomonnaies à analyser
        lookback_period (int): Période d'historique en années
        recommended_assets (int): Nombre d'actifs à recommander
        output_folder (str): Dossier pour les sorties

    Returns:
        tuple: (mhgna, recommendations, report)
    """
    # Configuration personnalisée
    config=SimpleConfig ()

    if tickers:
        config.tickers=tickers

    config.lookback_period=lookback_period
    config.recommended_assets=recommended_assets

    # Créer et exécuter l'analyseur
    mhgna=MHGNASimplified (config)

    try:
        # Exécuter l'analyse complète
        report=mhgna.run_monthly_analysis ()

        # Obtenir les recommandations
        recommendations=mhgna.last_recommendations

        print ("\n" + "=" * 50)
        print ("RECOMMANDATIONS PRINCIPALES:")
        print ("-" * 50)
        print (recommendations.head (3).to_string ())
        print ("=" * 50)

        if mhgna.alerts:
            print ("\nALERTES ACTIVES:")
            print ("-" * 50)
            for alert in mhgna.alerts:
                print (f"• {alert['message']}")

        return mhgna,recommendations,report

    except Exception as e:
        print (f"Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc ()
        return mhgna,None,None


# Exemple d'utilisation
if __name__ == "__main__":
    print ("MHGNA Simplified - Outil d'analyse pour le trading manuel de cryptomonnaies")
    print ("=" * 80)

    # Exécuter l'analyse
    mhgna,recommendations,report=start_mhgna_analysis (
        # Liste personnalisée (facultatif)
        tickers=['BTC-USD','ETH-USD','SOL-USD','ADA-USD','AVAX-USD',
                 'MATIC-USD','LINK-USD','DOT-USD','UNI-USD','ATOM-USD'],
        # Autres paramètres
        lookback_period=1,
        recommended_assets=5,
        output_folder='./reports'
    )

    # Les visualisations sont automatiquement sauvegardées dans le dossier 'reports'
    print ("\nAnalyse terminée. Consultez les fichiers générés dans le dossier './reports'")