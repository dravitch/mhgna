# Copier ce code et l'exécuter directement dans votre notebook pour résoudre l'erreur
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

def minimal_mhgna_analysis_debug():
    """
    Version simplifiée et corrigée de l'analyse MHGNA avec débogage des valeurs
    """
    print("\n" + "="*80)
    print("EXÉCUTION DE L'ANALYSE MHGNA (VERSION AVEC DÉBOGAGE)")
    print("="*80)

    # Fonction de débogage pour surveiller les valeurs du portefeuille
    def debug_portfolio_values(date, portfolio_value, portfolio_return=None):
        """
        Fonction de débogage pour surveiller les valeurs du portefeuille
        """
        if portfolio_return is not None and abs(portfolio_return) > 0.5:  # 50% de changement
            print(f"ALERTE: Rendement très élevé à {date}: {portfolio_return*100:.2f}%")
            print(f"Valeur du portefeuille: {portfolio_value:.2f}")

            # Pour tracer l'origine du rendement élevé (quels actifs contribuent le plus)
            if 'current_portfolio' in locals() and 'daily_returns_data' in locals():
                contributions = {}
                for asset in current_portfolio:
                    if asset in daily_returns_data:
                        asset_return = daily_returns_data[asset]
                        asset_weight = current_portfolio[asset]
                        contribution = asset_return * asset_weight
                        contributions[asset] = contribution

                # Afficher les contributions les plus significatives
                significant_contributions = {k: v for k, v in contributions.items() if abs(v) > 0.05}
                if significant_contributions:
                    print("Contributions significatives:")
                    for asset, contrib in sorted(significant_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
                        print(f"  {asset}: {contrib*100:.2f}% (Rendement: {daily_returns_data[asset]*100:.2f}%, Poids: {current_portfolio[asset]*100:.2f}%)")

    # Récupérer les données
    print("Chargement des données...")
    data, returns, volatility, momentum_scores = get_data()
    print(f"Données chargées: {len(returns)} jours pour {len(returns.columns)} actifs")

    # Initialisation pour le backtest
    print("Préparation du backtest...")
    initial_capital = Config.initial_capital
    start_date = pd.Timestamp(Config.start_date) + timedelta(days=max(Config.horizons['long']['window'], 90))
    end_date = pd.Timestamp(Config.end_date)
    start_date = max(start_date, data.index.min())

    # Identifiez les dates à utiliser (toutes les dates disponibles entre start_date et end_date)
    backtest_dates = data.index[(data.index >= start_date) & (data.index <= end_date)]

    # Structures de données pour le backtest
    portfolio_values = []
    benchmark_values = []
    current_portfolio = {}
    cash_allocation = 0.0
    last_rebalance_date = backtest_dates[0]
    rebalance_dates = []
    weight_history = []
    graph_history = []

    # Valeur initiale
    portfolio_values.append(initial_capital)
    benchmark_values.append(initial_capital)

    # Variables pour le suivi du drawdown
    in_drawdown_protection = False
    portfolio_peak = initial_capital

    # Compteurs pour le débogage
    num_high_returns = 0
    total_trading_days = 0

    # Exécution du backtest
    print("Exécution du backtest...")
    for i, current_date in enumerate(backtest_dates):
        if i == 0:  # Ignorer la première date (utilisée pour initialisation)
            continue

        total_trading_days += 1

        # Vérifier si on doit rééquilibrer
        days_since_rebalance = (current_date - last_rebalance_date).days
        should_rebalance = days_since_rebalance >= Config.rebalance_freq

        # Vérifier le drawdown
        if len(portfolio_values) > 1:
            current_value = portfolio_values[-1]
            portfolio_peak = max(portfolio_peak, current_value)
            drawdown = (current_value - portfolio_peak) / portfolio_peak

            # Protection contre le drawdown
            if in_drawdown_protection and drawdown > Config.recovery_threshold:
                should_rebalance = True
                in_drawdown_protection = False
                print(f"Sortie de la protection de drawdown à {current_date.strftime('%Y-%m-%d')}")

            elif not in_drawdown_protection and drawdown < Config.max_drawdown_threshold:
                should_rebalance = True
                in_drawdown_protection = True
                print(f"Activation de la protection de drawdown à {current_date.strftime('%Y-%m-%d')}")

        # Rééquilibrage si nécessaire
        if should_rebalance and i > Config.horizons['moyen']['window']:
            try:
                # Construire le graphe
                data_window = data.loc[:current_date]
                G, precision_matrix = build_multi_horizon_dependency_graph(returns, current_date)

                # Sélectionner les actifs
                selected_assets = select_portfolio_assets(G, momentum_scores, volatility, current_date)

                # Allouer le portefeuille
                weights = allocate_portfolio(selected_assets, precision_matrix, returns,
                                           momentum_scores, volatility, current_date,
                                           previous_weights=current_portfolio)

                # Appliquer le contrôle du drawdown si nécessaire
                current_value = portfolio_values[-1]
                drawdown = (current_value - portfolio_peak) / portfolio_peak
                weights, cash, is_protected = apply_drawdown_control(current_value, weights, drawdown)
                in_drawdown_protection = is_protected
                cash_allocation = cash

                # Mettre à jour le portefeuille
                current_portfolio = weights
                last_rebalance_date = current_date
                rebalance_dates.append(current_date)
                weight_history.append((current_date, weights, cash_allocation))

                # Sauvegarder le graphe pour visualisation (limité à 5)
                if len(graph_history) < 5:
                    graph_history.append((current_date, G, precision_matrix, momentum_scores, volatility))

                # Afficher le rebalancement
                assets_str = ', '.join([f'{a}: {w:.2f}' for a, w in weights.items()])
                print(f"Rééquilibrage à {current_date.strftime('%Y-%m-%d')}: {assets_str}, Cash: {cash_allocation:.2f}")

            except Exception as e:
                print(f"Erreur lors du rééquilibrage à {current_date}: {e}")
                # Continuer avec le portefeuille actuel

        # Calculer la performance quotidienne
        if current_portfolio and current_date in returns.index:
            daily_returns_data = returns.loc[current_date]

            # Rendement du portefeuille (partie investie)
            portfolio_return = sum(current_portfolio.get(asset, 0) * daily_returns_data.get(asset, 0)
                                  for asset in current_portfolio)

            # Ajuster pour la partie cash
            portfolio_return = portfolio_return * (1 - cash_allocation)

            # AJOUT: Limiter les rendements extrêmes (pour debug)
            if abs(portfolio_return) > 2.0:  # Limiter à 200% max par jour
                old_return = portfolio_return
                portfolio_return = np.sign(portfolio_return) * 2.0
                print(f"ATTENTION: Rendement journalier limité de {old_return*100:.2f}% à {portfolio_return*100:.2f}% à {current_date}")

            # Mettre à jour la valeur
            new_value = portfolio_values[-1] * (1 + portfolio_return)

            # Débogage des valeurs du portefeuille
            debug_portfolio_values(current_date, new_value, portfolio_return)

            # Compter les rendements élevés
            if abs(portfolio_return) > 0.5:
                num_high_returns += 1

            portfolio_values.append(new_value)

            # Mettre à jour la valeur du benchmark
            benchmark_return = daily_returns_data.get(Config.benchmark, 0)
            benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
        else:
            # Si pas de portfolio ou pas de données pour cette date, valeur inchangée
            portfolio_values.append(portfolio_values[-1])
            benchmark_values.append(benchmark_values[-1])

    # Afficher les statistiques de débogage
    print("\n=== Statistiques de Débogage ===")
    print(f"Jours de trading: {total_trading_days}")
    print(f"Nombre de jours avec rendement > 50%: {num_high_returns} ({num_high_returns/total_trading_days*100:.2f}%)")
    print(f"Rendement total calculé: {portfolio_values[-1]/initial_capital-1:.2f}x")

    # Créer le DataFrame de résultats
    print("\nCréation du DataFrame de résultats...")

    # S'assurer que les longueurs correspondent
    min_length = min(len(backtest_dates) - 1, len(portfolio_values) - 1)
    results_dates = backtest_dates[1:min_length+1]

    results = pd.DataFrame(index=results_dates)
    results['Portfolio Value'] = portfolio_values[1:min_length+1]
    results['Benchmark Value'] = benchmark_values[1:min_length+1]

    # Calculer et afficher la valeur du portefeuille par mois
    monthly_values = results['Portfolio Value'].resample('M').last()
    monthly_returns = monthly_values.pct_change()

    print("\n=== Valeurs Mensuelles du Portefeuille ===")
    for date, value in monthly_values.items():
        if date > monthly_values.index[0]:
            ret = monthly_returns[date]
            print(f"{date.strftime('%Y-%m')}: {value:.2f} ({ret*100:+.2f}%)")

    # Calculer les drawdowns pour l'affichage
    portfolio_returns = results['Portfolio Value'].pct_change().fillna(0)
    benchmark_returns = results['Benchmark Value'].pct_change().fillna(0)

    # Limiter les rendements extrêmes pour l'analyse
    portfolio_returns = portfolio_returns.clip(-0.5, 0.5)  # Limiter entre -50% et +50%

    portfolio_cum = (1 + portfolio_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    portfolio_peak = portfolio_cum.cummax()
    benchmark_peak = benchmark_cum.cummax()

    portfolio_drawdown = (portfolio_cum - portfolio_peak) / portfolio_peak
    benchmark_drawdown = (benchmark_cum - benchmark_peak) / benchmark_peak

    # Analyser la performance (avec des rendements plus réalistes)
    print("\nAnalyse des performances (ajustée)...")
    perf_df, _, _, _, _ = analyze_performance(results)

    # Afficher les résultats
    print("\n--- Résultats de la Stratégie MHGNA ---")
    print(perf_df)

    return results, perf_df, portfolio_values, weight_history

def run_mhgna_analysis_complete():
    """
    Exécute une analyse complète MHGNA avec visualisations améliorées
    """
    print("\n" + "="*80)
    print("ANALYSE MHGNA COMPLÈTE AVEC VISUALISATION AMÉLIORÉE")
    print("="*80)
    
    # ===== 1. FONCTIONS UTILITAIRES =====
    
    def debug_portfolio_values(date, portfolio_value, portfolio_return=None, current_portfolio=None, daily_returns_data=None):
        """
        Fonction de débogage pour surveiller les valeurs du portefeuille
        """
        if portfolio_return is not None and abs(portfolio_return) > 0.5:  # 50% de changement
            print(f"ALERTE: Rendement très élevé à {date}: {portfolio_return*100:.2f}%")
            print(f"Valeur du portefeuille: {portfolio_value:.2f}")
            
            # Pour tracer l'origine du rendement élevé
            if current_portfolio is not None and daily_returns_data is not None:
                contributions = {}
                for asset in current_portfolio:
                    if asset in daily_returns_data:
                        asset_return = daily_returns_data[asset]
                        asset_weight = current_portfolio[asset]
                        contribution = asset_return * asset_weight
                        contributions[asset] = contribution
                        
                # Afficher les contributions les plus significatives
                significant_contributions = {k: v for k, v in contributions.items() if abs(v) > 0.05}
                if significant_contributions:
                    print("Contributions significatives:")
                    for asset, contrib in sorted(significant_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
                        print(f"  {asset}: {contrib*100:.2f}% (Rendement: {daily_returns_data[asset]*100:.2f}%, Poids: {current_portfolio[asset]*100:.2f}%)")
    
    # ===== 2. RÉCUPÉRATION DES DONNÉES =====
    
    print("Chargement des données...")
    data, returns, volatility, momentum_scores = get_data()
    print(f"Données chargées: {len(returns)} jours pour {len(returns.columns)} actifs")
    
    # ===== 3. INITIALISATION DU BACKTEST =====
    
    print("Préparation du backtest...")
    initial_capital = Config.initial_capital
    start_date = pd.Timestamp(Config.start_date) + timedelta(days=max(Config.horizons['long']['window'], 90))
    end_date = pd.Timestamp(Config.end_date)
    start_date = max(start_date, data.index.min())
    
    # Identifiez les dates à utiliser (toutes les dates disponibles entre start_date et end_date)
    backtest_dates = data.index[(data.index >= start_date) & (data.index <= end_date)]
    
    # Structures de données pour le backtest
    portfolio_values = []
    benchmark_values = []
    current_portfolio = {}
    cash_allocation = 0.0
    last_rebalance_date = backtest_dates[0]
    rebalance_dates = []
    weight_history = []
    graph_history = []
    
    # Valeur initiale
    portfolio_values.append(initial_capital)
    benchmark_values.append(initial_capital)
    
    # Variables pour le suivi du drawdown
    in_drawdown_protection = False
    portfolio_peak = initial_capital
    
    # Compteurs pour le débogage
    num_high_returns = 0
    total_trading_days = 0
    
    # ===== 4. EXÉCUTION DU BACKTEST =====
    
    print("Exécution du backtest...")
    for i, current_date in enumerate(backtest_dates):
        if i == 0:  # Ignorer la première date (utilisée pour initialisation)
            continue
            
        total_trading_days += 1
        
        # Vérifier si on doit rééquilibrer
        days_since_rebalance = (current_date - last_rebalance_date).days
        should_rebalance = days_since_rebalance >= Config.rebalance_freq
        
        # Vérifier le drawdown
        if len(portfolio_values) > 1:
            current_value = portfolio_values[-1]
            portfolio_peak = max(portfolio_peak, current_value)
            drawdown = (current_value - portfolio_peak) / portfolio_peak
            
            # Protection contre le drawdown
            if in_drawdown_protection and drawdown > Config.recovery_threshold:
                should_rebalance = True
                in_drawdown_protection = False
                print(f"Sortie de la protection de drawdown à {current_date.strftime('%Y-%m-%d')}")
                
            elif not in_drawdown_protection and drawdown < Config.max_drawdown_threshold:
                should_rebalance = True
                in_drawdown_protection = True
                print(f"Activation de la protection de drawdown à {current_date.strftime('%Y-%m-%d')}")
        
        # Rééquilibrage si nécessaire
        if should_rebalance and i > Config.horizons['moyen']['window']:
            try:
                # Construire le graphe
                data_window = data.loc[:current_date]
                G, precision_matrix = build_multi_horizon_dependency_graph(returns, current_date)
                
                # Sélectionner les actifs
                selected_assets = select_portfolio_assets(G, momentum_scores, volatility, current_date)
                
                # Allouer le portefeuille
                weights = allocate_portfolio(selected_assets, precision_matrix, returns, 
                                           momentum_scores, volatility, current_date, 
                                           previous_weights=current_portfolio)
                
                # Appliquer le contrôle du drawdown si nécessaire
                current_value = portfolio_values[-1]
                drawdown = (current_value - portfolio_peak) / portfolio_peak
                weights, cash, is_protected = apply_drawdown_control(current_value, weights, drawdown)
                in_drawdown_protection = is_protected
                cash_allocation = cash
                
                # Mettre à jour le portefeuille
                current_portfolio = weights
                last_rebalance_date = current_date
                rebalance_dates.append(current_date)
                weight_history.append((current_date, weights, cash_allocation))
                
                # Sauvegarder le graphe pour visualisation (limité à 5)
                if len(graph_history) < 5:
                    graph_history.append((current_date, G, precision_matrix, momentum_scores, volatility))
                
                # Afficher le rebalancement
                assets_str = ', '.join([f'{a}: {w:.2f}' for a, w in weights.items()])
                print(f"Rééquilibrage à {current_date.strftime('%Y-%m-%d')}: {assets_str}, Cash: {cash_allocation:.2f}")
                
            except Exception as e:
                print(f"Erreur lors du rééquilibrage à {current_date}: {e}")
                # Continuer avec le portefeuille actuel
        
        # Calculer la performance quotidienne
        if current_portfolio and current_date in returns.index:
            daily_returns_data = returns.loc[current_date]
            
            # Rendement du portefeuille (partie investie)
            portfolio_return = sum(current_portfolio.get(asset, 0) * daily_returns_data.get(asset, 0) 
                                  for asset in current_portfolio)
            
            # Ajuster pour la partie cash
            portfolio_return = portfolio_return * (1 - cash_allocation)
            
            # Limiter les rendements extrêmes
            if abs(portfolio_return) > 2.0:  # Limiter à 200% max par jour
                old_return = portfolio_return
                portfolio_return = np.sign(portfolio_return) * 2.0
                print(f"ATTENTION: Rendement journalier limité de {old_return*100:.2f}% à {portfolio_return*100:.2f}% à {current_date}")
            
            # Mettre à jour la valeur
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            
            # Débogage des valeurs du portefeuille
            debug_portfolio_values(current_date, new_value, portfolio_return, current_portfolio, daily_returns_data)
            
            # Compter les rendements élevés
            if abs(portfolio_return) > 0.5:
                num_high_returns += 1
            
            portfolio_values.append(new_value)
            
            # Mettre à jour la valeur du benchmark
            benchmark_return = daily_returns_data.get(Config.benchmark, 0)
            benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
        else:
            # Si pas de portfolio ou pas de données pour cette date, valeur inchangée
            portfolio_values.append(portfolio_values[-1])
            benchmark_values.append(benchmark_values[-1])
    
    # Afficher les statistiques de débogage
    print("\n=== Statistiques de Débogage ===")
    print(f"Jours de trading: {total_trading_days}")
    print(f"Nombre de jours avec rendement > 50%: {num_high_returns} ({num_high_returns/total_trading_days*100:.2f}%)")
    print(f"Rendement total calculé: {portfolio_values[-1]/initial_capital-1:.2f}x")
    
    # ===== 5. CRÉATION DU DATAFRAME DE RÉSULTATS =====
    
    print("\nCréation du DataFrame de résultats...")
    
    # S'assurer que les longueurs correspondent
    min_length = min(len(backtest_dates) - 1, len(portfolio_values) - 1)
    results_dates = backtest_dates[1:min_length+1]
    
    results = pd.DataFrame(index=results_dates)
    results['Portfolio Value'] = portfolio_values[1:min_length+1]
    results['Benchmark Value'] = benchmark_values[1:min_length+1]
    
    # Calculer et afficher la valeur du portefeuille par mois
    monthly_values = results['Portfolio Value'].resample('M').last()
    monthly_returns = monthly_values.pct_change()
    
    print("\n=== Valeurs Mensuelles du Portefeuille ===")
    for date, value in monthly_values.items():
        if date > monthly_values.index[0]:
            ret = monthly_returns[date]
            print(f"{date.strftime('%Y-%m')}: {value:.2f} ({ret*100:+.2f}%)")
    
    # ===== 6. ANALYSE DE PERFORMANCE =====
    
    print("\nAnalyse des performances...")
    
    # Calculer les rendements et drawdowns pour l'analyse
    portfolio_returns_series = results['Portfolio Value'].pct_change().fillna(0)
    benchmark_returns_series = results['Benchmark Value'].pct_change().fillna(0)
    
    # Limiter les rendements extrêmes pour l'analyse
    portfolio_returns_series = portfolio_returns_series.clip(-0.5, 0.5)
    
    portfolio_cum = (1 + portfolio_returns_series).cumprod()
    benchmark_cum = (1 + benchmark_returns_series).cumprod()
    
    portfolio_peak = portfolio_cum.cummax()
    benchmark_peak = benchmark_cum.cummax()
    
    portfolio_drawdown = (portfolio_cum - portfolio_peak) / portfolio_peak
    benchmark_drawdown = (benchmark_cum - benchmark_peak) / benchmark_peak
     

def plot_dependency_network_improved(G, precision_matrix, date, momentum_scores=None, volatility=None, threshold=0.01):
    """
    Visualisation améliorée du réseau de dépendance avec gestion des graphes déconnectés
    et formatage avancé des visualisations.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import networkx as nx
    import numpy as np
    from matplotlib.cm import get_cmap
    from sklearn.cluster import AgglomerativeClustering
    import warnings
    
    # Ignorer les avertissements
    warnings.filterwarnings('ignore')
    
    # Créer une figure de grande taille pour plus de détails
    plt.figure(figsize=(16, 14))
    
    # Vérifier si le graphe est vide ou a trop peu de nœuds
    if len(G.nodes()) < 2:
        plt.text(0.5, 0.5, "Graphe insuffisant pour visualisation", 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
        plt.title(f"Réseau de dépendance - {date.strftime('%Y-%m-%d')}", fontsize=18, fontweight='bold')
        plt.axis('off')
        return plt
    
    # Filtrer les liens faibles
    edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > threshold]
    filtered_G = G.edge_subgraph(edges).copy() if edges else G.copy()
    
    # S'assurer que tous les nœuds sont dans le graphe filtré
    for node in G.nodes():
        if node not in filtered_G.nodes():
            filtered_G.add_node(node)
    
    # Vérifier la connectivité du graphe
    connected_components = list(nx.connected_components(filtered_G))
    num_components = len(connected_components)
    
    # Préparation des données pour les couleurs des nœuds basées sur le momentum
    if momentum_scores is not None:
        latest_scores = momentum_scores.loc[momentum_scores.index <= date].iloc[-1] if not momentum_scores.empty else pd.Series(0, index=G.nodes())
        # Normalisation entre 0 et 1
        min_score = latest_scores.min()
        max_score = latest_scores.max()
        if max_score > min_score:
            normalized_scores = (latest_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = pd.Series(0.5, index=latest_scores.index)
        
        # Création d'une colormap personnalisée: rouge (négatif) à vert (positif)
        node_colors = []
        for node in filtered_G.nodes():
            if node in normalized_scores:
                score = normalized_scores[node]
                if score < 0.5:  # Momentum négatif
                    r = 0.8
                    g = score * 2 * 0.8
                    b = 0.1
                else:  # Momentum positif
                    r = (1 - score) * 2 * 0.8
                    g = 0.8
                    b = 0.1
                node_colors.append((r, g, b))
            else:
                node_colors.append((0.5, 0.5, 0.5))  # Gris pour données manquantes
    else:
        # Utiliser une colormap par défaut si pas de données de momentum
        node_colors = list(range(len(filtered_G.nodes())))
        node_colors = plt.cm.viridis(np.array(node_colors) / max(len(node_colors)-1, 1))
    
    # Calcul des métriques de centralité avec gestion des erreurs pour graphes déconnectés
    try:
        # Tenter de calculer la centralité pour chaque composante connectée
        centrality = {}
        for component in connected_components:
            subgraph = filtered_G.subgraph(component)
            if len(subgraph) > 1:
                # Pour les composantes avec au moins 2 nœuds
                comp_centrality = nx.eigenvector_centrality_numpy(subgraph, weight='weight', max_iter=1000)
                # Normaliser les valeurs au sein de chaque composante
                if len(comp_centrality) > 0:
                    max_cent = max(comp_centrality.values())
                    for node, value in comp_centrality.items():
                        centrality[node] = value / max_cent if max_cent > 0 else 0
            else:
                # Pour les nœuds isolés
                for node in component:
                    centrality[node] = 0.1  # Valeur arbitraire faible
    except Exception as e:
        print(f"Erreur lors du calcul de la centralité: {e}")
        # Fallback à une valeur par défaut si le calcul échoue
        centrality = {node: 0.5 for node in filtered_G.nodes()}
    
    # Déterminer la taille des nœuds en fonction de la centralité
    node_sizes = []
    for node in filtered_G.nodes():
        base_size = 300  # Taille de base
        if node in centrality:
            # Échelle de taille: 300 à 3000 en fonction de la centralité
            size = base_size + centrality[node] * 2700
        else:
            size = base_size
        node_sizes.append(size)
    
    # Tenter de générer un layout optimisé
    try:
        if num_components == 1:
            # Si graphe connecté, utiliser spring_layout
            pos = nx.spring_layout(filtered_G, k=1.5/np.sqrt(len(filtered_G.nodes())), seed=42)
        else:
            # Si graphe déconnecté, positionner chaque composante séparément
            pos = {}
            y_offset = 0
            max_width = 0
            
            for i, component in enumerate(connected_components):
                subgraph = filtered_G.subgraph(component)
                # Calculer la largeur et hauteur approximatives en fonction du nombre de nœuds
                width = max(1.0, np.sqrt(len(subgraph)) * 0.4)
                height = width
                max_width = max(max_width, width)
                
                # Positionner la composante avec un décalage vertical
                component_pos = nx.spring_layout(subgraph, k=0.3, seed=42+i, scale=min(width, height))
                
                # Ajouter un décalage pour chaque composante
                for node, coords in component_pos.items():
                    pos[node] = np.array([coords[0], coords[1] + y_offset])
                
                # Augmenter le décalage pour la prochaine composante
                y_offset += height * 1.5
    except Exception as e:
        print(f"Erreur lors de la génération du layout: {e}")
        # Fallback en cas d'erreur
        pos = {node: (np.random.rand()*2-1, np.random.rand()*2-1) for node in filtered_G.nodes()}
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(filtered_G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    
    # Dessiner les liens avec une épaisseur proportionnelle au poids et une couleur basée sur le poids
    edge_colors = []
    edge_widths = []
    for u, v in filtered_G.edges():
        weight = filtered_G[u][v]['weight']
        edge_widths.append(weight * 10)  # Échelle proportionnelle
        # Couleur basée sur le poids: plus l'arête est forte, plus elle est foncée
        edge_colors.append(plt.cm.Blues(min(weight * 10, 0.9)))
    
    nx.draw_networkx_edges(filtered_G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
    
    # Ajouter les étiquettes des nœuds avec un meilleur formatage
    labels = {}
    for node in filtered_G.nodes():
        # Format d'étiquette plus riche si les données de momentum sont disponibles
        if momentum_scores is not None and node in latest_scores:
            momentum_value = latest_scores[node]
            momentum_sign = "+" if momentum_value >= 0 else ""
            labels[node] = f"{node}\n{momentum_sign}{momentum_value:.2f}"
        else:
            labels[node] = node
    
    # Position légèrement ajustée pour les étiquettes
    label_pos = {node: (coords[0], coords[1] + 0.02) for node, coords in pos.items()}
    nx.draw_networkx_labels(filtered_G, label_pos, labels=labels, font_size=10, 
                          font_weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Détecter et visualiser les communautés
    if num_components == 1 and len(filtered_G.nodes()) > 2:
        try:
            # Préparation pour le clustering
            adj_matrix = nx.to_numpy_array(filtered_G)
            
            # Déterminer le nombre optimal de clusters (entre 2 et 5)
            n_clusters = min(max(2, int(len(filtered_G.nodes()) / 3)), 5)
            
            # Appliquer un clustering hiérarchique
            clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
            
            # Transformer la matrice d'adjacence en matrice de distance
            distance_matrix = 1 - adj_matrix / (adj_matrix.max() + 1e-10)
            np.fill_diagonal(distance_matrix, 0)
            
            # Fit le clustering
            node_list = list(filtered_G.nodes())
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Organiser les nœuds par communauté
            communities = {}
            for node, label in zip(node_list, cluster_labels):
                if label not in communities:
                    communities[label] = []
                communities[label].append(node)
            
            # Couleurs pour les communautés
            community_colors = plt.cm.tab10.colors
            
            # Dessiner des cercles autour de chaque communauté
            for label, nodes in communities.items():
                if len(nodes) > 2:
                    node_positions = np.array([pos[node] for node in nodes])
                    center = np.mean(node_positions, axis=0)
                    radius = np.max(np.linalg.norm(node_positions - center, axis=1)) + 0.1
                    
                    circle = plt.Circle(center, radius, fill=False, linestyle='--', 
                                      edgecolor=community_colors[label % len(community_colors)], 
                                      linewidth=2, alpha=0.7)
                    plt.gca().add_patch(circle)
                    
                    plt.text(center[0], center[1] + radius + 0.05, f"Cluster {label+1}", 
                             ha='center', fontsize=12, fontweight='bold', 
                             color=community_colors[label % len(community_colors)])
            
        except Exception as e:
            print(f"Erreur lors de la détection des communautés: {e}")
    
    # Section d'information
    plt.text(0.02, 0.02, 
             f"Nœuds: {len(filtered_G.nodes())}\nArêtes: {len(filtered_G.edges())}\nComposantes: {num_components}", 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Ajouter une légende pour le momentum
    if momentum_scores is not None:
        # Créer une légende personnalisée pour le momentum
        momentum_cmap = mcolors.LinearSegmentedColormap.from_list(
            'momentum_cmap', [(0.8, 0.1, 0.1), (0.4, 0.4, 0.1), (0.1, 0.8, 0.1)])
        sm = plt.cm.ScalarMappable(cmap=momentum_cmap, 
                                  norm=plt.Normalize(vmin=min_score, vmax=max_score))
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.7, pad=0.05)
        cbar.set_label('Momentum Score', fontsize=12)
    
    # Titre avec date et informations supplémentaires
    plt.title(f"Réseau de Dépendance Conditionnelle - {date.strftime('%Y-%m-%d')}", 
              fontsize=18, fontweight='bold')
    
    # Informations sur les composantes déconnectées si applicable
    if num_components > 1:
        plt.suptitle(f"Graphe déconnecté ({num_components} composantes)", 
                     fontsize=14, y=0.98)
    
    plt.axis('off')
    plt.tight_layout()
    return plt

def plot_results_improved(results, perf_df, portfolio_drawdown, benchmark_drawdown, graph_history, weight_history, rebalance_dates):
    """
    Visualisation améliorée des résultats de la stratégie
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
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

    # 3. Évolution des poids du portefeuille avec cash
    plt.subplot(5, 1, 3)

    # Créer un DataFrame pour l'évolution des poids
    all_assets = set()
    for item in weight_history:
        weights = item[1]
        all_assets.update(weights.keys())

    # Ajouter une colonne pour le cash
    all_assets.add('CASH')

    # Créer un DataFrame avec tous les actifs
    weight_df = pd.DataFrame(index=[item[0] for item in weight_history], columns=list(all_assets))
    weight_df = weight_df.fillna(0)

    # Remplir le DataFrame avec les poids
    for item in weight_history:
        date, weights, cash = item
        for asset, weight in weights.items():
            weight_df.loc[date, asset] = weight
        weight_df.loc[date, 'CASH'] = cash

    # Plot stacked area chart
    weight_df.plot(kind='area', stacked=True, ax=plt.gca(), cmap='viridis')
    plt.title('Évolution des Allocations du Portefeuille', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Allocation', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

# Correction de la partie problématique dans plot_results_improved
# Remplacez le bloc concernant les rendements mensuels par celui-ci:

    # 4. Rendements mensuels
    plt.subplot(5, 1, 4)
    
    # Calculer les rendements mensuels
    monthly_portfolio = results['Portfolio Value'].resample('M').last().pct_change()
    monthly_benchmark = results['Benchmark Value'].resample('M').last().pct_change()
    
    # Créer un DataFrame pour le bar plot
    monthly_df = pd.DataFrame({
        'MHGNA': monthly_portfolio,
        'Bitcoin': monthly_benchmark
    })
    
    # Plot les barres avec des couleurs basées sur la valeur
    ax = plt.gca()
    
    # Afficher les barres pour chaque colonne séparément
    bar_width = 0.35
    index = np.arange(len(monthly_df.index))
    
    # MHGNA bars
    mhgna_bars = ax.bar(index - bar_width/2, monthly_df['MHGNA'], bar_width, 
                       label='MHGNA', alpha=0.8)
    # Bitcoin bars
    btc_bars = ax.bar(index + bar_width/2, monthly_df['Bitcoin'], bar_width, 
                     label='Bitcoin', alpha=0.8)
    
    # Coloriser individuellement chaque barre selon qu'elle est positive ou négative
    for i, bar in enumerate(mhgna_bars):
        if i < len(monthly_df['MHGNA']):
            value = monthly_df['MHGNA'].iloc[i]
            if pd.notna(value):  # Vérifier si la valeur n'est pas NaN
                bar.set_color('green' if value >= 0 else 'red')
    
    for i, bar in enumerate(btc_bars):
        if i < len(monthly_df['Bitcoin']):
            value = monthly_df['Bitcoin'].iloc[i]
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
    
    for i, v in enumerate(monthly_df['Bitcoin']):
        if pd.notna(v):  # Vérifier si la valeur n'est pas NaN
            plt.text(i + bar_width/2, v + 0.01 if v > 0 else v - 0.03, f'{v:.1%}', 
                     color='black', fontweight='bold', fontsize=9, rotation=90)

# Remplacez cette partie de la fonction plot_results_improved:

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

# Ajoutez cette fonction au niveau global de votre script (en dehors de toute autre fonction)
def analyze_performance(results):
    """
    Analyse détaillée des performances de la stratégie
    
    Parameters:
    -----------
    results : pandas.DataFrame
        DataFrame contenant les valeurs du portefeuille et du benchmark
        
    Returns:
    --------
    pandas.DataFrame
        Table des métriques de performance
    pandas.Series
        Drawdown du portefeuille
    pandas.Series
        Drawdown du benchmark
    pandas.Series
        Rendements du portefeuille
    pandas.Series
        Rendements du benchmark
    """
    import numpy as np
    import pandas as pd
    
    # Vérifier que results contient des données
    if results.empty:
        print("Pas de données pour analyser les performances!")
        empty_df = pd.DataFrame({
            'Métrique': ['Rendement Total', 'Rendement Annualisé', 'Volatilité Annualisée',
                         'Ratio de Sharpe', 'Ratio de Sortino', 'Maximum Drawdown', 'Ratio de Calmar'],
            'Stratégie MHGNA': ['N/A'] * 7,
            'Benchmark (Bitcoin)': ['N/A'] * 7
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
    days = max((results.index[-1] - results.index[0]).days, 1)  # Éviter la division par zéro
    portfolio_annual_return = ((results['Portfolio Value'].iloc[-1] / results['Portfolio Value'].iloc[0]) ** (365 / days)) - 1
    benchmark_annual_return = ((results['Benchmark Value'].iloc[-1] / results['Benchmark Value'].iloc[0]) ** (365 / days)) - 1

    # Sharpe Ratio (supposant un taux sans risque de 0%)
    portfolio_sharpe = portfolio_annual_return / portfolio_vol if portfolio_vol != 0 else 0
    benchmark_sharpe = benchmark_annual_return / benchmark_vol if benchmark_vol != 0 else 0

    # Sortino Ratio (volatilité des rendements négatifs seulement)
    neg_portfolio_returns = portfolio_returns[portfolio_returns < 0]
    neg_benchmark_returns = benchmark_returns[benchmark_returns < 0]

    portfolio_downside_vol = neg_portfolio_returns.std() * np.sqrt(365) if len(neg_portfolio_returns) > 0 else 0.0001
    benchmark_downside_vol = neg_benchmark_returns.std() * np.sqrt(365) if len(neg_benchmark_returns) > 0 else 0.0001

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
    
    # Créer un rapport de performance complet
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
    }

    perf_df = pd.DataFrame(performance)

    return perf_df, portfolio_drawdown, benchmark_drawdown, portfolio_returns, benchmark_returns
    
# Fonction principale pour exécuter l'analyse et générer les visualisations
def execute_mhgna_analysis():
    """
    Exécute l'analyse MHGNA et génère les visualisations améliorées
    """
    # Récupérer les données
    print("Chargement des données...")
    data, returns, volatility, momentum_scores = get_data()
    
    # Exécuter le backtest avec la fonction minimale
    print("Exécution du backtest...")
    results, perf_df, portfolio_values, weight_history = minimal_mhgna_analysis_debug()
    
    # Calculer les drawdowns pour l'affichage
    portfolio_returns = results['Portfolio Value'].pct_change().fillna(0)
    benchmark_returns = results['Benchmark Value'].pct_change().fillna(0)
    
    portfolio_cum = (1 + portfolio_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()
    
    portfolio_peak = portfolio_cum.cummax()
    benchmark_peak = benchmark_cum.cummax()
    
    portfolio_drawdown = (portfolio_cum - portfolio_peak) / portfolio_peak
    benchmark_drawdown = (benchmark_cum - benchmark_peak) / benchmark_peak
    
    # Déterminer les dates de rebalancement à partir de l'historique des poids
    rebalance_dates = [date for date, _, _ in weight_history]
    
    # Récupérer les graphes de l'historique
    graph_history = []
    for i, date in enumerate(rebalance_dates[:5]):  # Limiter à 5 graphes
        try:
            G, precision_matrix = build_multi_horizon_dependency_graph(returns, date)
            graph_history.append((date, G, precision_matrix, momentum_scores, volatility))
        except Exception as e:
            print(f"Erreur lors de la création du graphe pour {date}: {e}")
    
    # Générer les visualisations améliorées
    print("Génération des visualisations améliorées...")
    plot_improved = plot_results_improved(results, perf_df, portfolio_drawdown, benchmark_drawdown, 
                                        graph_history, weight_history, rebalance_dates)
    
    print("Analyse MHGNA complétée avec succès.")
    print("Visualisations sauvegardées dans le répertoire courant.")
    
    return results, perf_df, portfolio_values, weight_history, graph_history

# Exécuter l'analyse si le script est exécuté directement
if __name__ == "__main__":
    try:
        results, perf_df, portfolio_values, weight_history, graph_history = execute_mhgna_analysis()
        print("Analyse terminée avec succès!")
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum

class PreservationStrategy(Enum):
    """
    Stratégies de préservation des gains en stablecoin.
    """
    THRESHOLD_BASED = "threshold"      # Basé sur des seuils de profit
    VOLATILITY_BASED = "volatility"    # Basé sur la volatilité du marché
    DRAWDOWN_BASED = "drawdown"        # Basé sur les drawdowns
    TIME_BASED = "time"                # Basé sur des intervalles temporels
    HYBRID = "hybrid"                  # Combinaison de plusieurs stratégies

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

        Parameters:
        -----------
        strategy : PreservationStrategy
            Stratégie de préservation à utiliser
        profit_threshold : float, default 0.15
            Seuil de profit à partir duquel commencer à préserver
        max_stablecoin_allocation : float, default 0.5
            Allocation maximale en stablecoin
        base_preservation_rate : float, default 0.3
            Taux de base pour la préservation des gains
        drawdown_sensitivity : float, default 2.0
            Sensibilité aux drawdowns (plus élevé = préservation plus agressive)
        time_interval : int, default 30
            Intervalle en jours pour la stratégie temporelle
        stablecoin_assets : List[str], optional
            Liste des stablecoins disponibles pour la préservation
        reinvestment_threshold : float, default -0.1
            Seuil de baisse du marché pour réinvestir les stablecoins
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

        Parameters:
        -----------
        initial_value : float
            Valeur initiale du portefeuille
        current_date : pd.Timestamp
            Date courante
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

        Parameters:
        -----------
        current_value : float
            Valeur actuelle du portefeuille
        current_date : pd.Timestamp
            Date courante
        market_drawdown : float, default 0.0
            Drawdown actuel du marché global
        volatility : float, optional
            Volatilité récente du marché
        current_weights : Dict[str, float], optional
            Poids actuels du portefeuille

        Returns:
        --------
        float
            Pourcentage recommandé d'allocation en stablecoin
        Dict[str, float]
            Répartition recommandée entre différents stablecoins
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

        Parameters:
        -----------
        profit_pct : float
            Pourcentage de profit actuel

        Returns:
        --------
        float
            Allocation recommandée en stablecoin
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

        Parameters:
        -----------
        profit_pct : float
            Pourcentage de profit actuel
        volatility : float, optional
            Volatilité récente du marché

        Returns:
        --------
        float
            Allocation recommandée en stablecoin
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

        Parameters:
        -----------
        profit_pct : float
            Pourcentage de profit actuel
        drawdown_pct : float
            Pourcentage de drawdown actuel

        Returns:
        --------
        float
            Allocation recommandée en stablecoin
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

        Parameters:
        -----------
        profit_pct : float
            Pourcentage de profit actuel
        current_date : pd.Timestamp
            Date courante

        Returns:
        --------
        float
            Allocation recommandée en stablecoin
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

        Parameters:
        -----------
        profit_pct : float
            Pourcentage de profit actuel
        drawdown_pct : float
            Pourcentage de drawdown actuel
        volatility : float, optional
            Volatilité récente du marché
        current_date : pd.Timestamp
            Date courante
        market_drawdown : float
            Drawdown actuel du marché global

        Returns:
        --------
        float
            Allocation recommandée en stablecoin
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
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Répartit l'allocation en stablecoin entre les différents stablecoins disponibles.

        Parameters:
        -----------
        stablecoin_allocation : float
            Allocation totale recommandée en stablecoin
        current_weights : Dict[str, float], optional
            Poids actuels du portefeuille

        Returns:
        --------
        Dict[str, float]
            Répartition recommandée entre différents stablecoins
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

        Parameters:
        -----------
        target_weights : Dict[str, float]
            Poids cibles d'origine
        current_value : float
            Valeur actuelle du portefeuille
        current_date : pd.Timestamp
            Date courante
        market_drawdown : float, default 0.0
            Drawdown actuel du marché global
        volatility : float, optional
            Volatilité récente du marché
        current_weights : Dict[str, float], optional
            Poids actuels du portefeuille

        Returns:
        --------
        Dict[str, float]
            Poids cibles ajustés incluant les stablecoins
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

        Parameters:
        -----------
        initial_capital : float
            Capital initial
        current_value : float
            Valeur actuelle du portefeuille

        Returns:
        --------
        float
            Montant du capital préservé en valeur absolue
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
        --------
        Dict
            Rapport contenant les métriques clés
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

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Importer les classes à tester
#from transaction_optimizer import TransactionOptimizer
#from preservation_strategy import GainPreservationModule, PreservationStrategy

def main():
    # Initialisation des données fictives
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)  # Pour la reproductibilité
    prices = pd.DataFrame(np.random.rand(len(date_range), len(tickers)) * 100, index=date_range, columns=tickers)

    # Calcul des rendements fictifs
    returns = prices.pct_change().dropna()

    # Initialisation de TransactionOptimizer
    fee_structure = {ticker: 0.002 for ticker in tickers}  # Exemple de structure de frais
    transaction_optimizer = TransactionOptimizer(fee_structure=fee_structure)

    # Exemple de poids actuels et cibles
    current_weights = {ticker: 1 / len(tickers) for ticker in tickers}
    target_weights = {ticker: np.random.random() for ticker in tickers}
    target_weights = {ticker: weight / sum(target_weights.values()) for ticker, weight in target_weights.items()}

    # Valeur fictive du portefeuille
    portfolio_value = 1000000

    # Calcul des coûts de transaction
    total_cost, transaction_costs = transaction_optimizer.calculate_transaction_costs(
        current_weights, target_weights, portfolio_value
    )
    print(f"Total Transaction Cost: {total_cost}")
    print(f"Transaction Costs: {transaction_costs}")

    # Vérification de la nécessité de rebalancement
    expected_improvement = 0.05  # Augmenter l'amélioration attendue
    current_date = pd.Timestamp(datetime.now())
    should_rebalance, optimized_weights, rebalance_cost = transaction_optimizer.should_rebalance(
        current_weights, target_weights, portfolio_value, expected_improvement, current_date
    )
    print(f"Should Rebalance: {should_rebalance}")
    print(f"Optimized Weights: {optimized_weights}")
    print(f"Rebalance Cost: {rebalance_cost}")

    # Initialisation de GainPreservationModule
    stablecoin_assets = ["USDT", "USDC"]
    preservation_module = GainPreservationModule(
        strategy=PreservationStrategy.HYBRID,
        profit_threshold=0.10,  # Réduire le seuil de profit
        stablecoin_assets=stablecoin_assets
    )

    # Initialisation avec une valeur initiale fictive
    initial_value = 500000
    preservation_module.initialize(initial_value, current_date)

    # Calcul de l'allocation en stablecoins
    market_drawdown = -0.2  # Augmenter le drawdown du marché
    volatility = pd.Series({ticker: np.random.random() * 0.05 for ticker in tickers})  # Augmenter la volatilité
    stablecoin_allocation, stablecoin_weights = preservation_module.calculate_preservation_allocation(
        initial_value, current_date, market_drawdown, volatility, current_weights
    )
    print(f"Stablecoin Allocation: {stablecoin_allocation * 100:.2f}%")
    print(f"Stablecoin Weights: {stablecoin_weights}")

    # Ajustement des poids cibles pour inclure les stablecoins
    adjusted_weights = preservation_module.adjust_allocation_weights(
        target_weights, initial_value, current_date, market_drawdown, volatility, current_weights
    )
    print(f"Adjusted Weights: {adjusted_weights}")

    # Génération d'un rapport
    report = preservation_module.generate_report()
    print("Preservation Report:", report)

if __name__ == "__main__":
    main()

