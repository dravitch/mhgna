# enhanced_mhgna_backtest.py

def enhanced_mhgna_backtest(data, returns, volatility, momentum_scores):
    """
    Version améliorée du backtest MHGNA intégrant l'optimisation des transactions
    et la préservation des gains en stablecoin.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant les données de prix
    returns : pd.DataFrame
        DataFrame contenant les rendements quotidiens
    volatility : pd.DataFrame
        DataFrame contenant la volatilité des actifs
    momentum_scores : pd.DataFrame
        DataFrame contenant les scores de momentum
        
    Returns:
    --------
    Résultats du backtest et statistiques d'optimisation
    """
    print("Démarrage du backtest avancé MHGNA...")
    
    # Initialisation des modules d'optimisation
    from datetime import timedelta
    import numpy as np
    import pandas as pd
    
    # Initialiser le module d'optimisation des transactions
    transaction_optimizer = TransactionOptimizer(
        base_fee_rate=0.001,  # 0.1% de frais de transaction par défaut
        min_improvement_threshold=0.0025,  # 0.25% d'amélioration minimale requise
        observation_period=3,  # 3 jours d'observation après rebalancement
        adaptive_turnover=True,  # Ajustement dynamique du turnover
        market_impact_model=simple_market_impact  # Modèle simple d'impact de marché
    )
    
    # Initialiser le module de préservation des gains
    gain_preserver = GainPreservationModule(
        strategy=PreservationStrategy.HYBRID,
        profit_threshold=0.1,  # 10% de profit pour commencer à préserver
        max_stablecoin_allocation=0.5,  # Maximum 50% en stablecoin
        base_preservation_rate=0.25,  # 25% des gains préservés par défaut
        stablecoin_assets=["USDT-USD", "USDC-USD"]  # Liste des stablecoins disponibles
    )
    
    # Initialisation du backtest
    portfolio_value = [Config.initial_capital]
    benchmark_value = [Config.initial_capital]
    current_date = pd.Timestamp(Config.start_date) + timedelta(days=max(Config.horizons['long']['window'], 90))
    end_date = pd.Timestamp(Config.end_date)
    
    # S'assurer que current_date est dans la plage de données
    current_date = max(current_date, data.index.min())
    
    last_rebalance_date = current_date
    current_portfolio = {}
    portfolio_history = []
    graph_history = []
    weight_history = []
    
    # Suivi du cash pour le contrôle du drawdown et la préservation des gains
    cash_allocation = 0.0
    stablecoin_allocation = 0.0
    in_drawdown_protection = False
    
    # Historique du drawdown
    portfolio_cumulative_return = 1.0
    portfolio_peak = 1.0
    portfolio_drawdown = 0.0
    drawdown_history = []
    
    # Statistiques étendues
    transaction_costs = []
    skipped_rebalances = 0
    executed_rebalances = 0
    preservation_events = 0
    stablecoin_history = []

    # Dates de rééquilibrage
    rebalance_dates = []
    
    # Initialiser le module de préservation des gains
    gain_preserver.initialize(Config.initial_capital, current_date)
    
    # Avancer jour par jour
    print("Traitement des données historiques...")
    while current_date <= end_date and current_date <= data.index.max():
        # Trouver la date réelle la plus proche dans les données
        actual_dates = data.index[data.index <= current_date]
        if len(actual_dates) == 0:
            # Avancer à la première date disponible
            next_dates = data.index[data.index > current_date]
            if len(next_dates) == 0:
                break  # Plus de données disponibles
            current_date = next_dates[0]
            continue
        
        actual_date = actual_dates[-1]
        
        # Récupérer les données jusqu'à la date actuelle
        data_until_current = data.loc[:actual_date]
        
        # Vérifier si on doit rééquilibrer (basé sur la fréquence ou un événement de drawdown)
        days_since_rebalance = (current_date - last_rebalance_date).days
        should_rebalance = days_since_rebalance >= Config.rebalance_freq
        forced_rebalance = False
        
        # Calculer le drawdown actuel
        if len(portfolio_value) > 1:
            portfolio_cumulative_return = portfolio_value[-1] / Config.initial_capital
            portfolio_peak = max(portfolio_value) / Config.initial_capital
            portfolio_drawdown = (portfolio_cumulative_return - portfolio_peak) / portfolio_peak
            drawdown_history.append(portfolio_drawdown)
            
            # Vérifier les conditions de drawdown pour protection
            if in_drawdown_protection and portfolio_drawdown > Config.recovery_threshold:
                should_rebalance = True
                forced_rebalance = True
                in_drawdown_protection = False
                print(f"Sortie de la protection de drawdown à {current_date.strftime('%Y-%m-%d')}")
            
            elif not in_drawdown_protection and portfolio_drawdown < Config.max_drawdown_threshold:
                should_rebalance = True
                forced_rebalance = True
                in_drawdown_protection = True
                print(f"Activation de la protection de drawdown à {current_date.strftime('%Y-%m-%d')}")
        
        # Obtenez le drawdown du benchmark (Bitcoin)
        market_drawdown = 0.0
        if actual_date in returns.index:
            benchmark_returns_to_date = returns[Config.benchmark].loc[:actual_date]
            benchmark_cumulative = (1 + benchmark_returns_to_date).cumprod()
            benchmark_peak = benchmark_cumulative.cummax()
            benchmark_drawdown = (benchmark_cumulative - benchmark_peak) / benchmark_peak
            market_drawdown = benchmark_drawdown.iloc[-1]
        
        # Obtenir la volatilité actuelle pour l'optimisation
        current_volatility = None
        if actual_date in volatility.index:
            current_volatility = volatility.loc[actual_date]
        
        # Vérifier si on a suffisamment de données pour un rebalancement
        if (should_rebalance or forced_rebalance) and len(data_until_current) > Config.horizons['moyen']['window']:
            try:
                # Construire le graphe de dépendance multi-horizon
                G, precision_matrix = build_multi_horizon_dependency_graph(returns, actual_date)
                
                # Pour quelques dates, sauvegarder le graphe pour visualisation
                if len(graph_history) < 5 or forced_rebalance:
                    graph_history.append((actual_date, G, precision_matrix, momentum_scores, volatility))
                
                # Sélectionner les actifs pour le portefeuille
                current_prices = data_until_current.iloc[-1]
                selected_assets = select_portfolio_assets(G, momentum_scores, volatility, actual_date)
                
                # Allouer le portefeuille (poids cibles théoriques)
                target_weights = allocate_portfolio(selected_assets, precision_matrix, returns, 
                                                  momentum_scores, volatility, actual_date, 
                                                  previous_weights=current_portfolio if current_portfolio else None)
                
                # Appliquer le contrôle du drawdown si nécessaire
                target_weights, cash_from_drawdown, is_protected = apply_drawdown_control(
                    portfolio_value[-1], target_weights, portfolio_drawdown
                )
                in_drawdown_protection = is_protected
                
                # Ajuster les poids pour la préservation des gains en stablecoin
                adjusted_weights = gain_preserver.adjust_allocation_weights(
                    target_weights, portfolio_value[-1], actual_date, 
                    market_drawdown, current_volatility, current_portfolio
                )
                
                # Calculer l'allocation totale en cash/stablecoin
                stablecoin_weights = {
                    asset: weight for asset, weight in adjusted_weights.items()
                    if asset in gain_preserver.stablecoin_assets
                }
                stablecoin_allocation = sum(stablecoin_weights.values())
                total_cash_allocation = cash_from_drawdown + stablecoin_allocation
                
                # Estimer l'amélioration attendue du rendement
                expected_returns_estimate = momentum_scores.loc[momentum_scores.index <= actual_date].iloc[-1]
                expected_improvement = transaction_optimizer.estimate_expected_improvement(
                    current_portfolio, adjusted_weights, expected_returns_estimate, 
                    horizon=Config.rebalance_freq
                )
                
                # Décider si le rebalancement doit être effectué et optimiser les transactions
                do_rebalance, optimized_weights, transaction_cost = transaction_optimizer.should_rebalance(
                    current_portfolio, adjusted_weights, portfolio_value[-1],
                    expected_improvement, actual_date, current_volatility
                )
                
                if do_rebalance or forced_rebalance:
                    # Exécuter le rebalancement
                    executed_rebalances += 1
                    transaction_costs.append(transaction_cost)
                    
                    # Mettre à jour le portefeuille avec les poids optimisés
                    current_portfolio = optimized_weights
                    last_rebalance_date = current_date
                    rebalance_dates.append(current_date)
                    
                    # Extraire l'allocation en stablecoin des poids optimisés
                    optimized_stablecoin_weights = {
                        asset: weight for asset, weight in optimized_weights.items()
                        if asset in gain_preserver.stablecoin_assets
                    }
                    stablecoin_allocation = sum(optimized_stablecoin_weights.values())
                    
                    # Enregistrer l'historique
                    weight_history.append((actual_date, optimized_weights, stablecoin_allocation))
                    stablecoin_history.append((actual_date, stablecoin_allocation))
                    
                    if stablecoin_allocation > 0:
                        preservation_events += 1
                    
                    portfolio_history.append({
                        'date': actual_date,
                        'assets': selected_assets,
                        'weights': optimized_weights,
                        'stablecoin_allocation': stablecoin_allocation,
                        'drawdown': portfolio_drawdown
                    })
                    
                    # Afficher le détail du rebalancement
                    assets_str = ', '.join([f'{a}: {w:.2f}' for a, w in optimized_weights.items() 
                                          if a not in gain_preserver.stablecoin_assets])
                    stablecoin_str = ', '.join([f'{a}: {w:.2f}' for a, w in optimized_stablecoin_weights.items()])
                    
                    print(f"Rééquilibrage à {actual_date.strftime('%Y-%m-%d')}:")
                    print(f"  Crypto: {assets_str}")
                    if stablecoin_str:
                        print(f"  Stablecoin: {stablecoin_str} (Total: {stablecoin_allocation:.2f})")
                    print(f"  Coût de transaction: {transaction_cost:.2f} ({transaction_cost/portfolio_value[-1]*100:.2f}%)")
                    
                else:
                    # Rebalancement ignoré
                    skipped_rebalances += 1
                    print(f"Rebalancement ignoré à {actual_date.strftime('%Y-%m-%d')} - Amélioration attendue insuffisante")
                
            except Exception as e:
                print(f"Erreur lors du rééquilibrage à {actual_date}: {e}")
                # Conserver le portefeuille actuel
        
        # Calculer la performance quotidienne
        if current_portfolio and actual_date in returns.index:
            daily_returns_data = returns.loc[actual_date]
            
            # Calculer le rendement pour chaque actif du portefeuille
            portfolio_return = 0.0
            for asset, weight in current_portfolio.items():
                # Les stablecoins ont un rendement de 0 (ou très faible)
                if asset in gain_preserver.stablecoin_assets:
                    # Supposer un rendement annuel de 2% pour les stablecoins
                    daily_stablecoin_return = 0.02 / 365
                    portfolio_return += weight * daily_stablecoin_return
                elif asset in daily_returns_data:
                    portfolio_return += weight * daily_returns_data[asset]
            
            # Mettre à jour la valeur du portefeuille
            new_value = portfolio_value[-1] * (1 + portfolio_return)
            
            # Déduire les frais de transaction si un rebalancement a été effectué ce jour
            if current_date in rebalance_dates:
                # Les frais ont déjà été calculés par transaction_optimizer
                last_transaction_cost = transaction_costs[-1]
                new_value -= last_transaction_cost
            
            portfolio_value.append(max(0, new_value))  # Éviter les valeurs négatives
            
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
    
    # Créer un DataFrame avec les résultats
    # Utiliser uniquement les dates présentes dans l'index de data
    valid_dates = [date for date in data.index if date >= pd.Timestamp(Config.start_date) + timedelta(days=max(Config.horizons['long']['window'], 90))]
    
    if len(valid_dates) == 0:
        print("Pas de dates valides pour le backtest!")
        return pd.DataFrame(), graph_history, weight_history, rebalance_dates, drawdown_history
    
    # Utiliser seulement les dates disponibles
    results_dates = valid_dates[:min(len(valid_dates), len(portfolio_value)-1)]
    
    results = pd.DataFrame(index=results_dates)
    results['Portfolio Value'] = portfolio_value[1:len(results_dates)+1]
    results['Benchmark Value'] = benchmark_value[1:len(results_dates)+1]
    
    if len(drawdown_history) >= len(results_dates):
        results['Drawdown'] = drawdown_history[:len(results_dates)]
    
    # Ajouter les statistiques d'optimisation
    transaction_stats = transaction_optimizer.get_optimization_stats()
    preservation_stats = gain_preserver.generate_report()
    
    # Afficher les statistiques finales
    print("\n=== Statistiques d'Optimisation des Transactions ===")
    print(f"Rebalancements exécutés: {executed_rebalances}")
    print(f"Rebalancements ignorés: {skipped_rebalances}")
    print(f"Taux d'exécution: {executed_rebalances/(executed_rebalances+skipped_rebalances)*100:.1f}%")
    print(f"Coût total des transactions: {sum(transaction_costs):.2f} ({sum(transaction_costs)/Config.initial_capital*100:.2f}% du capital initial)")
    print(f"Coût moyen par transaction: {np.mean(transaction_costs) if transaction_costs else 0:.2f}")
    
    print("\n=== Statistiques de Préservation des Gains ===")
    print(f"Événements de préservation: {preservation_events}")
    print(f"Allocation finale en stablecoin: {stablecoin_allocation*100:.2f}%")
    if 'preserved_capital' in preservation_stats:
        print(f"Capital préservé: {preservation_stats['preserved_capital']:.2f}")
    if 'preservation_ratio' in preservation_stats:
        print(f"Ratio de préservation: {preservation_stats['preservation_ratio']}")
    
    extended_stats = {
        'transaction_stats': transaction_stats,
        'preservation_stats': preservation_stats,
        'executed_rebalances': executed_rebalances,
        'skipped_rebalances': skipped_rebalances,
        'transaction_costs': transaction_costs,
        'stablecoin_history': stablecoin_history
    }
    
    return results, graph_history, weight_history, rebalance_dates, drawdown_history, extended_stats


def run_enhanced_mhgna_analysis():
    """
    Exécute une analyse complète MHGNA avec optimisation des transactions
    et préservation des gains en stablecoin.
    """
    print("\n" + "="*80)
    print("EXÉCUTION DE L'ANALYSE MHGNA AVANCÉE")
    print("="*80)
    
    # Récupérer les données
    print("Chargement des données...")
    data, returns, volatility, momentum_scores = get_data()
    print(f"Données chargées: {len(returns)} jours pour {len(returns.columns)} actifs")
    
    # Exécuter le backtest avancé
    results, graph_history, weight_history, rebalance_dates, drawdown_history, extended_stats = enhanced_mhgna_backtest(
        data, returns, volat