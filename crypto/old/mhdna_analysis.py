import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
from scipy.stats import zscore
from datetime import datetime, timedelta
import warnings
import time  # Correction: importation du module time
from sklearn.cluster import AgglomerativeClustering
warnings.filterwarnings('ignore')


# Configuration des paramètres
class Config:
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
    
    # Fréquence de rééquilibrage augmentée pour stabilité
    rebalance_freq = 21  # Mensuel au lieu de 7 jours
    
    # Sélection d'actifs plus large
    portfolio_size = 7   # Augmenté de 5 à 7
    
    # Paramètres de régularisation
    alpha_short = 0.02  # Plus fort pour horizon court (plus sparse)
    alpha_medium = 0.01  # Moyen pour horizon moyen
    alpha_long = 0.005  # Plus faible pour horizon long (plus dense)
    
    # Paramètres de momentum
    momentum_window = 60  # Fenêtre pour calculer le momentum
    momentum_weight = 0.3  # Intégration du momentum dans l'allocation
    
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

# 1. Téléchargement et préparation des données
def get_data():
    print("Récupération des données de prix...")
    data = yf.download(Config.tickers, start=Config.start_date, end=Config.end_date)['Close']
    returns = data.pct_change().dropna()
    
    # Ajouter des informations sur la volatilité
    volatility = returns.rolling(30).std()
    
    # Ajouter des indicateurs de momentum
    momentum_short = returns.rolling(20).sum()
    momentum_medium = returns.rolling(60).sum()
    momentum_long = returns.rolling(120).sum()
    
    # Normaliser pour créer un score de momentum composite
    momentum_score = (
        zscore(momentum_short, nan_policy='omit') * 0.5 + 
        zscore(momentum_medium, nan_policy='omit') * 0.3 + 
        zscore(momentum_long, nan_policy='omit') * 0.2
    )
    
    print(f"Données récupérées: {len(returns)} jours pour {len(Config.tickers)} actifs.")
    return data, returns, volatility, momentum_score

# 2. Fonction améliorée pour appliquer le Graphical Lasso et construire le graphe
def build_multi_horizon_dependency_graph(returns, current_date):
    """
    Construit un graphe de dépendance multi-horizon en combinant différentes échelles temporelles
    """
    graph_models = {}
    precision_matrices = {}
    
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
            # Utiliser GraphicalLassoCV pour une sélection automatique de l'alpha
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

# 3. Visualisation améliorée du graphe de dépendance
def plot_dependency_network(G, precision_matrix, date, momentum_scores=None, volatility=None, threshold=0.01):
    """
    Visualisation améliorée du réseau de dépendance avec information de momentum et volatilité
    """
    plt.figure(figsize=(14, 12))

    # Filtrer les liens faibles
    edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > threshold]
    filtered_G = G.edge_subgraph(edges).copy()

    # Calculer la centralité et l'importance des noeuds
    centrality = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=1000)
    
    # Déterminer la taille des nœuds en fonction de la centralité
    node_sizes = [centrality[n] * 4000 + 500 for n in filtered_G.nodes()]
    
    # Déterminer la couleur des nœuds en fonction du momentum (si disponible)
    if momentum_scores is not None:
        latest_scores = momentum_scores.loc[momentum_scores.index <= date].iloc[-1] if not momentum_scores.empty else pd.Series(0, index=G.nodes())
        normalized_scores = (latest_scores - latest_scores.min()) / (latest_scores.max() - latest_scores.min()) if latest_scores.max() > latest_scores.min() else pd.Series(0.5, index=G.nodes())
        
        # Créer un dégradé de couleurs: rouge pour négatif, vert pour positif
        node_colors = []
        for node in filtered_G.nodes():
            if node in normalized_scores:
                score = normalized_scores[node]
                # Couleur: du rouge (0) au vert (1)
                if score < 0.5:  # Momentum négatif
                    red = 0.9
                    green = score * 2 * 0.9
                    blue = 0.1
                else:  # Momentum positif
                    red = (1 - score) * 2 * 0.9
                    green = 0.9
                    blue = 0.1
                node_colors.append((red, green, blue))
            else:
                node_colors.append((0.5, 0.5, 0.5))  # Gris pour données manquantes
    else:
        node_colors = 'skyblue'
    
    # Layout du graphe
    pos = nx.spring_layout(filtered_G, seed=42, k=0.5)
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(filtered_G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

    # Dessiner les liens avec une épaisseur proportionnelle au poids
    edge_weights = [filtered_G[u][v]['weight'] * 5 for u, v in filtered_G.edges()]
    nx.draw_networkx_edges(filtered_G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    
    # Ajouter les étiquettes des nœuds
    nx.draw_networkx_labels(filtered_G, pos, font_size=10, font_weight='bold')
    
    # Ajouter des informations sur les communautés détectées
    if len(filtered_G.nodes()) > 1:
        try:
            # Calculer la matrice d'adjacence pour le clustering
            adj_matrix = nx.to_numpy_array(filtered_G)
            
            # Déterminer automatiquement le nombre de clusters (entre 2 et 4)
            n_clusters = min(max(2, int(len(filtered_G.nodes()) / 3)), 4)
            
            # Appliquer un clustering hiérarchique
            clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
            
            # Transformer la matrice d'adjacence en matrice de distance (1 - similarité)
            distance_matrix = 1 - adj_matrix / adj_matrix.max()
            np.fill_diagonal(distance_matrix, 0)  # Diagonale à zéro (distance à soi-même)
            
            # Fit le clustering
            node_list = list(filtered_G.nodes())
            labels = clustering.fit_predict(distance_matrix)
            
            # Ajouter une annotation pour chaque communauté
            communities = {}
            for node, label in zip(node_list, labels):
                if label not in communities:
                    communities[label] = []
                communities[label].append(node)
            
            # Ajouter des cercles autour des communautés
            for label, nodes in communities.items():
                if len(nodes) > 1:  # Seulement pour les communautés avec plus d'un nœud
                    # Calculer le centre et le rayon
                    node_positions = np.array([pos[node] for node in nodes])
                    center = node_positions.mean(axis=0)
                    radius = np.max(np.linalg.norm(node_positions - center, axis=1)) + 0.1
                    
                    # Dessiner le cercle
                    circle = plt.Circle(center, radius, fill=False, linestyle='--', 
                                        edgecolor=f'C{label}', linewidth=2, alpha=0.7)
                    plt.gca().add_patch(circle)
                    
                    # Ajouter l'étiquette de la communauté
                    plt.text(center[0], center[1] + radius + 0.05, f"Cluster {label+1}", 
                             ha='center', fontsize=12, fontweight='bold', color=f'C{label}')
        
        except Exception as e:
            print(f"Erreur lors de la détection des communautés: {e}")

    plt.title(f"Réseau de dépendance conditionnelle - {date.strftime('%Y-%m-%d')}")
    plt.axis('off')
    plt.tight_layout()
    return plt

# 4. Sélection des actifs améliorée pour le portefeuille
def select_portfolio_assets(G, momentum_scores, volatility, current_date):
    """
    Sélectionne les actifs en combinant centralité, communautés et momentum
    """
    # Vérifier que le graphe contient des nœuds
    if len(G.nodes()) == 0:
        # Sélection par défaut basée uniquement sur le momentum récent
        latest_momentum = momentum_scores.loc[momentum_scores.index <= current_date]
        if len(latest_momentum) > 0:
            latest_momentum = latest_momentum.iloc[-1]
            sorted_assets = latest_momentum.sort_values(ascending=False)
            return sorted_assets.index[:Config.portfolio_size].tolist()
        else:
            # Si aucune donnée n'est disponible, renvoyer une sélection par défaut
            return Config.tickers[:Config.portfolio_size]
    
    # Calculer les métriques de centralité
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=1000)
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
        closeness_centrality = nx.closeness_centrality(G, distance='weight')
    except:
        # Si le calcul échoue, utiliser des dictionnaires vides
        eigenvector_centrality = {node: 1.0/len(G.nodes()) for node in G.nodes()}
        betweenness_centrality = {node: 1.0/len(G.nodes()) for node in G.nodes()}
        closeness_centrality = {node: 1.0/len(G.nodes()) for node in G.nodes()}
    
    # Combiner les métriques en un score composite de centralité
    composite_centrality = {}
    for node in G.nodes():
        composite_centrality[node] = (
            eigenvector_centrality.get(node, 0) * 0.5 +
            betweenness_centrality.get(node, 0) * 0.3 +
            closeness_centrality.get(node, 0) * 0.2
        )
    
    # Obtenir le momentum récent
    latest_momentum = momentum_scores.loc[momentum_scores.index <= current_date]
    if len(latest_momentum) > 0:
        latest_momentum = latest_momentum.iloc[-1]
    else:
        latest_momentum = pd.Series(0, index=G.nodes())
    
    # Obtenir la volatilité récente
    latest_volatility = volatility.loc[volatility.index <= current_date]
    if len(latest_volatility) > 0:
        latest_volatility = latest_volatility.iloc[-1]
    else:
        latest_volatility = pd.Series(0.01, index=G.nodes())
    
    # Calculer un score combiné pour chaque actif
    combined_scores = {}
    for node in G.nodes():
        momentum_component = latest_momentum.get(node, 0) if node in latest_momentum.index else 0
        volatility_component = 1.0 / (latest_volatility.get(node, 0.01) if node in latest_volatility.index else 0.01)
        
        # Normaliser les composants entre 0 et 1
        momentum_component = (momentum_component - latest_momentum.min()) / (latest_momentum.max() - latest_momentum.min()) if latest_momentum.max() > latest_momentum.min() else 0.5
        volatility_component = (volatility_component - min(1.0/v for v in latest_volatility if v > 0)) / (max(1.0/v for v in latest_volatility if v > 0) - min(1.0/v for v in latest_volatility if v > 0)) if len(latest_volatility) > 0 else 0.5
        
        # Combiner en un score final
        combined_scores[node] = (
            composite_centrality[node] * 0.5 +
            momentum_component * 0.3 +
            volatility_component * 0.2
        )
    
    # Trier les actifs par score combiné
    sorted_assets = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Détecter les communautés pour assurer la diversification
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # Sélectionner les meilleurs actifs de chaque communauté
        selected_assets = []
        community_counts = {}
        
        # D'abord, prendre le meilleur actif de chaque communauté
        for community in communities:
            community_assets = [(asset, combined_scores[asset]) for asset in community if asset in combined_scores]
            if community_assets:
                community_assets.sort(key=lambda x: x[1], reverse=True)
                best_asset = community_assets[0][0]
                selected_assets.append(best_asset)
                community_counts[tuple(community)] = 1
        
        # Ensuite, compléter avec les meilleurs actifs restants
        remaining_slots = Config.portfolio_size - len(selected_assets)
        if remaining_slots > 0:
            # Prendre les meilleurs actifs non encore sélectionnés
            for asset, score in sorted_assets:
                if asset not in selected_assets and len(selected_assets) < Config.portfolio_size:
                    selected_assets.append(asset)
        
        return selected_assets
    
    except Exception as e:
        print(f"Erreur lors de la détection des communautés: {e}")
        # Fallback: prendre simplement les meilleurs actifs par score
        return [asset for asset, _ in sorted_assets[:Config.portfolio_size]]

# 5. Allocation du portefeuille améliorée
def allocate_portfolio(selected_assets, precision_matrix, returns, momentum_scores, volatility, current_date, previous_weights=None):
    """
    Allocation de portefeuille combinant la structure de risque, le momentum et des contraintes de stabilité
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

# 6. Mécanisme de contrôle des drawdowns
def apply_drawdown_control(portfolio_value, current_weights, portfolio_drawdown):
    """
    Applique un contrôle du drawdown en réduisant l'exposition en cas de drawdown important
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

# 7. Backtesting de la stratégie
def backtest_strategy(data, returns, volatility, momentum_scores):
    """
    Backtest de la stratégie multi-horizon avec contrôle du risque
    """
    print("Démarrage du backtest...")

    # Initialisation
    portfolio_value = [Config.initial_capital]
    benchmark_value = [Config.initial_capital]
    current_date = pd.Timestamp(Config.start_date) + timedelta(days=max(Config.horizons['long']['window'], 90))
    end_date = pd.Timestamp(Config.end_date)
    
    # Ensure current_date is within the data range:
    current_date = max(current_date, data.index.min())
    
    last_rebalance_date = current_date
    current_portfolio = {}
    portfolio_history = []
    graph_history = []
    weight_history = []
    
    # Suivi du cash pour le contrôle du drawdown
    cash_allocation = 0.0
    in_drawdown_protection = False
    
    # Historique du drawdown
    portfolio_cumulative_return = 1.0
    portfolio_peak = 1.0
    portfolio_drawdown = 0.0
    drawdown_history = []

    # Dates de rééquilibrage
    rebalance_dates = []
    
    # Liste pour stocker les dates effectives
    actual_dates = []
    
    # Avancer jour par jour
    while current_date <= end_date and current_date <= data.index.max():
        # Trouver la date réelle la plus proche dans les données (pour gérer les jours sans données)
        available_dates = data.index[data.index <= current_date]
        if len(available_dates) == 0:
            # Avancer à la première date disponible
            next_dates = data.index[data.index > current_date]
            if len(next_dates) == 0:
                break  # Plus de données disponibles
            current_date = next_dates[0]
            continue
        
        actual_date = available_dates[-1]
        actual_dates.append(actual_date)  # Stocker la date effective
        
        # Récupérer les données jusqu'à la date actuelle
        data_until_current = data.loc[:actual_date]
        
        # Vérifier si on doit rééquilibrer (basé sur la fréquence ou un événement de drawdown)
        days_since_rebalance = (current_date - last_rebalance_date).days
        should_rebalance = days_since_rebalance >= Config.rebalance_freq
        forced_rebalance = False
        
        # Vérifier s'il y a un drawdown significatif
        if len(portfolio_value) > 1:
            portfolio_cumulative_return = portfolio_value[-1] / Config.initial_capital
            portfolio_peak = max(portfolio_value) / Config.initial_capital
            portfolio_drawdown = (portfolio_cumulative_return - portfolio_peak) / portfolio_peak
            drawdown_history.append(portfolio_drawdown)
            
            # Si on est en protection contre le drawdown et qu'on récupère
            if in_drawdown_protection and portfolio_drawdown > Config.recovery_threshold:
                should_rebalance = True
                forced_rebalance = True
                in_drawdown_protection = False
                print(f"Sortie de la protection de drawdown à {current_date.strftime('%Y-%m-%d')}")
            
            # Si on n'est pas en protection et qu'on a un drawdown significatif
            elif not in_drawdown_protection and portfolio_drawdown < Config.max_drawdown_threshold:
                should_rebalance = True
                forced_rebalance = True
                in_drawdown_protection = True
                print(f"Activation de la protection de drawdown à {current_date.strftime('%Y-%m-%d')}")

        if should_rebalance and len(data_until_current) > Config.horizons['moyen']['window']:
            try:
                # Construire le graphe de dépendance multi-horizon
                G, precision_matrix = build_multi_horizon_dependency_graph(returns, actual_date)
                
                # Pour quelques dates, sauvegarder le graphe pour visualisation
                if len(graph_history) < 5 or forced_rebalance:
                    graph_history.append((actual_date, G, precision_matrix, momentum_scores, volatility))
                
                # Sélectionner les actifs pour le portefeuille
                current_prices = data_until_current.iloc[-1]
                selected_assets = select_portfolio_assets(G, momentum_scores, volatility, actual_date)
                
                # Allouer le portefeuille
                weights = allocate_portfolio(selected_assets, precision_matrix, returns, 
                                            momentum_scores, volatility, actual_date, 
                                            previous_weights=current_portfolio if current_portfolio else None)
                
                # Appliquer le contrôle du drawdown si nécessaire
                weights, cash_allocation, is_protected = apply_drawdown_control(
                    portfolio_value[-1], weights, portfolio_drawdown
                )
                in_drawdown_protection = is_protected
                
                # Mettre à jour le portefeuille
                current_portfolio = weights
                last_rebalance_date = current_date
                rebalance_dates.append(current_date)
                
                # Enregistrer l'historique
                weight_history.append((actual_date, weights, cash_allocation))
                
                portfolio_history.append({
                    'date': actual_date,
                    'assets': selected_assets,
                    'weights': weights,
                    'cash': cash_allocation,
                    'drawdown': portfolio_drawdown
                })
                
                assets_str = ', '.join([f'{a}: {w:.2f}' for a, w in weights.items()])
                print(f"Rééquilibrage à {actual_date.strftime('%Y-%m-%d')}: {assets_str}, Cash: {cash_allocation:.2f}")
                
            except Exception as e:
                print(f"Erreur lors du rééquilibrage à {actual_date}: {e}")
                # Conserver le portefeuille actuel
        
        # Calculer la performance quotidienne
        if current_portfolio and actual_date in returns.index:
            daily_returns_data = returns.loc[actual_date]
            
            # Calculer le rendement du portefeuille (partie investie)
            portfolio_return = sum(current_portfolio.get(asset, 0) * daily_returns_data.get(asset, 0) 
                                  for asset in current_portfolio)
            
            # Prendre en compte la partie cash (qui ne génère pas de rendement)
            portfolio_return = portfolio_return * (1 - cash_allocation)
            
            # Mettre à jour la valeur du portefeuille
            portfolio_value.append(portfolio_value[-1] * (1 + portfolio_return))
            
            # Mettre à jour la valeur du benchmark
            benchmark_return = daily_returns_data.get(Config.benchmark, 0)
            benchmark_value.append(benchmark_value[-1] * (1 + benchmark_return))
        else:
            # Si pas de portfolio ou pas de données pour cette date, valeur inchangée
            if len(portfolio_value) > 0:
                portfolio_value.append(portfolio_value[-1])
            else:
                portfolio_value.append(Config.initial_capital)
                
            if len(benchmark_value) > 0:
                benchmark_value.append(benchmark_value[-1])
            else:
                benchmark_value.append(Config.initial_capital)
        
        # Avancer d'un jour
        current_date += timedelta(days=1)
    
    # Créer un DataFrame avec les résultats en utilisant les dates effectives
    if len(actual_dates) == 0:
        print("Pas de dates valides pour le backtest!")
        return pd.DataFrame(), graph_history, weight_history, rebalance_dates, drawdown_history
    
    # Utiliser les dates effectives comme index du DataFrame résultat
    results = pd.DataFrame(index=actual_dates,
                          columns=['Portfolio Value', 'Benchmark Value', 'Drawdown'])
    
    # S'assurer que les longueurs correspondent 
    min_length = min(len(results), len(portfolio_value)-1, len(benchmark_value)-1)
    
    # Assigner seulement le nombre de valeurs qui correspond à la longueur de l'index
    results['Portfolio Value'] = portfolio_value[1:min_length+1]
    results['Benchmark Value'] = benchmark_value[1:min_length+1]
    
    # S'assurer que drawdown_history a la bonne longueur
    if drawdown_history:
        results['Drawdown'] = drawdown_history[:min_length]
    
    return results, graph_history, weight_history, rebalance_dates, drawdown_history

# Fonction plot_results mise à jour pour prendre en charge les tuples à 3 éléments
def plot_results_fixed(results, perf_df, portfolio_drawdown, benchmark_drawdown, graph_history, weight_history, rebalance_dates):
    """
    Visualisation des résultats de la stratégie (version corrigée)
    """
    plt.figure(figsize=(15, 24))

    # 1. Performance cumulée
    plt.subplot(4, 1, 1)
    plt.plot(results['Portfolio Value'] / results['Portfolio Value'].iloc[0], label='Stratégie MHGNA', linewidth=2)
    plt.plot(results['Benchmark Value'] / results['Benchmark Value'].iloc[0], label=f'Benchmark ({Config.benchmark})', linewidth=2, alpha=0.7)
    plt.title('Performance Cumulée', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Valeur Relative')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. Drawdowns
    plt.subplot(4, 1, 2)
    plt.plot(portfolio_drawdown, label='Stratégie MHGNA', linewidth=2)
    plt.plot(benchmark_drawdown, label=f'Benchmark ({Config.benchmark})', linewidth=2, alpha=0.7)
    plt.title('Drawdowns', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3. Évolution des poids du portefeuille
    plt.subplot(4, 1, 3)

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
    plt.title('Évolution des Poids du Portefeuille', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Allocation')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 4. Tableau de performance
    plt.subplot(4, 1, 4)
    plt.axis('off')
    table = plt.table(cellText=perf_df.values, colLabels=perf_df.columns,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title('Métriques de Performance', fontsize=14)

    plt.tight_layout(pad=3.0)
    plt.savefig('mhgna_backtest_results.png', dpi=300, bbox_inches='tight')

    # Visualiser un des graphes de dépendance
    if graph_history:
        for i, item in enumerate(graph_history):
            if len(item) >= 3:  # Minimum requis : date, G, precision_matrix
                date, G, precision_matrix = item[0], item[1], item[2]
                momentum_scores = item[3] if len(item) > 3 else None
                volatility = item[4] if len(item) > 4 else None
                
                try:
                    plt_graph = plot_dependency_network(G, precision_matrix, date, momentum_scores, volatility)
                    plt_graph.savefig(f'dependency_graph_{i}.png', dpi=300, bbox_inches='tight')
                    plt_graph.close()
                except Exception as e:
                    print(f"Erreur lors de la visualisation du graphe {i}: {e}")

    return plt

# Modification de la fonction minimal_mhgna_analysis pour utiliser plot_results_fixed
def minimal_mhgna_analysis_fixed():
    """
    Version simplifiée et corrigée de l'analyse MHGNA avec visualisation fixe
    """
    # [... tout le reste du code reste identique ...]
    
    # Remplacer cette ligne dans la fonction original:
    # plot_results(results, perf_df, portfolio_drawdown, benchmark_drawdown, 
    #            graph_history, weight_history, rebalance_dates)
    
    # Par celle-ci:
    plot_results_fixed(results, perf_df, portfolio_drawdown, benchmark_drawdown, 
                      graph_history, weight_history, rebalance_dates)
    
    # [... le reste du code reste identique ...]
    
def main():
    """
    Fonction principale pour exécuter le backtest MHGNA avec gestion des erreurs
    """
    print("="*80)
    print("ANALYSE MULTI-HORIZON GRAPHICAL NETWORK ALLOCATION (MHGNA)")
    print("="*80)
    
    try:
        # Configuration des périodes de test
        test_configs = [
            {
                'start_date': '2022-01-01',
                'end_date': '2024-01-01',
                'description': 'Période complète (2022-2024)'
            },
            {
                'start_date': '2022-01-01',
                'end_date': '2023-01-01',
                'description': 'Marché baissier (2022)'
            },
            {
                'start_date': '2023-01-01',
                'end_date': '2024-01-01',
                'description': 'Marché haussier (2023)'
            }
        ]
        
        # Exécuter l'analyse pour chaque configuration
        results_by_period = {}
        
        for config in test_configs:
            print(f"\n{'-'*80}")
            print(f"ANALYSE POUR: {config['description']}")
            print(f"{'-'*80}")
            
            # Mettre à jour les paramètres de configuration
            Config.start_date = config['start_date']
            Config.end_date = config['end_date']
            
            # Exécuter l'analyse avec gestion d'erreurs
            try:
                results, perf = run_single_analysis(Config)
                results_by_period[config['description']] = (results, perf)
            except Exception as e:
                print(f"Erreur lors de l'analyse pour {config['description']}: {e}")
                print("Passage à la configuration suivante...")
                continue
        
        # Afficher la comparaison des performances
        if results_by_period:
            print_performance_comparison(results_by_period)
        else:
            print("Aucune analyse n'a pu être complétée avec succès.")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution principale: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExécution terminée.")

def run_single_analysis(config):
    """
    Exécute une seule instance d'analyse avec la configuration donnée
    """
    print(f"Démarrage de l'analyse pour la période {config.start_date} - {config.end_date}")
    
    # Récupération des données avec gestion d'erreurs
    try:
        print("Chargement des données...")
        data, returns, volatility, momentum_scores = get_data()
        print(f"Données chargées avec succès: {len(returns)} jours pour {len(returns.columns)} actifs")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        raise
    
    # Exécuter le backtest
    try:
        print("Exécution du backtest...")
        start_time = time.time()
        results, graph_history, weight_history, rebalance_dates, drawdown_history = backtest_strategy(
            data, returns, volatility, momentum_scores
        )
        elapsed_time = time.time() - start_time
        print(f"Backtest terminé en {elapsed_time:.2f} secondes")
        
        if results.empty:
            print("Le backtest n'a pas produit de résultats valides.")
            return None, None
    except Exception as e:
        print(f"Erreur lors du backtest: {e}")
        raise
    
    # Analyser les performances
    try:
        print("Analyse des performances...")
        perf_df, portfolio_drawdown, benchmark_drawdown, portfolio_returns, benchmark_returns = analyze_performance(results)
        
        # Afficher les résultats
        print("\n--- Résultats de la Stratégie MHGNA ---")
        print(perf_df)
    except Exception as e:
        print(f"Erreur lors de l'analyse des performances: {e}")
        raise
    
    # Générer les visualisations
    try:
        print("Génération des visualisations...")
        #plot_results(
        #    results, perf_df, portfolio_drawdown, benchmark_drawdown, 
        #    graph_history, weight_history, rebalance_dates, drawdown_history
        #)
        plot_results_fixed(results, perf_df, portfolio_drawdown, benchmark_drawdown, 
                  graph_history, weight_history, rebalance_dates)
        print("Visualisations enregistrées avec succès")
    except Exception as e:
        print(f"Erreur lors de la génération des visualisations: {e}")
        print("Poursuite de l'analyse sans visualisations...")
    
    # Rapport synthétique
    try:
        print_summary_report(results, perf_df, rebalance_dates)
    except Exception as e:
        print(f"Erreur lors de la génération du rapport de synthèse: {e}")
    
    return results, perf_df

def print_performance_comparison(results_by_period):
    """
    Affiche et enregistre une comparaison des performances pour différentes périodes
    """
    print("\n\n" + "="*80)
    print("COMPARAISON DES PERFORMANCES PAR PÉRIODE")
    print("="*80)
    
    performance_comparison = {}
    
    for period, (_, perf) in results_by_period.items():
        if perf is not None:
            try:
                # Extraire les principales métriques
                total_return = perf.iloc[0, 1].replace('%', '')
                sharpe = perf.iloc[3, 1]
                max_drawdown = perf.iloc[5, 1].replace('%', '')
                
                bench_return = perf.iloc[0, 2].replace('%', '')
                bench_sharpe = perf.iloc[3, 2]
                bench_drawdown = perf.iloc[5, 2].replace('%', '')
                
                # Calculer la surperformance
                return_diff = float(total_return) - float(bench_return)
                sharpe_diff = float(sharpe) - float(bench_sharpe)
                drawdown_diff = float(bench_drawdown) - float(max_drawdown)
                
                performance_comparison[period] = {
                    'Rendement MHGNA': f"{total_return}%",
                    'Rendement Bitcoin': f"{bench_return}%",
                    'Surperformance': f"{return_diff:.2f}%",
                    'Δ Sharpe': f"{sharpe_diff:.2f}",
                    'Δ Drawdown': f"{drawdown_diff:.2f}%"
                }
            except Exception as e:
                print(f"Erreur lors de l'extraction des métriques pour {period}: {e}")
                continue
    
    if performance_comparison:
        comparison_df = pd.DataFrame(performance_comparison).T
        print("\n")
        print(comparison_df)
        
        # Sauvegarder la comparaison
        try:
            comparison_df.to_csv('mhgna_performance_comparison.csv')
            print("\nComparaison des performances sauvegardée dans 'mhgna_performance_comparison.csv'")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la comparaison: {e}")
    else:
        print("Pas de données de performance valides à comparer.")

def print_summary_report(results, perf_df, rebalance_dates):
    """
    Génère un rapport de synthèse des performances
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
        print(f"Performance Bitcoin: {benchmark_return*100:.2f}%")
        print(f"Surperformance: {outperformance*100:.2f}%")
        
        # Risque
        print(f"\nRatio de Sharpe MHGNA: {strategy_sharpe:.2f}")
        print(f"Ratio de Sharpe Bitcoin: {benchmark_sharpe:.2f}")
        print(f"Amélioration du Sharpe: {sharpe_improvement:.2f}")
        
        print(f"\nDrawdown maximum MHGNA: {strategy_drawdown*100:.2f}%")
        print(f"Drawdown maximum Bitcoin: {benchmark_drawdown*100:.2f}%")
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

# Exécution du programme
if __name__ == "__main__":
    main()
    