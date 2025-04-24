# TransactionOptimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable


class TransactionOptimizer:
    """
    Module d'optimisation des transactions pour MHGNA.
    
    Ce module évalue les coûts de transaction et optimise le rebalancement
    pour maximiser le rendement net des frais de transaction.
    """
    
    def __init__(
        self,
        fee_structure: Dict[str, float] = None,
        base_fee_rate: float = 0.001,  # 0.1% par défaut (typique des exchanges)
        min_improvement_threshold: float = 0.002,  # 0.2% d'amélioration minimale
        observation_period: int = 5,  # jours d'observation après rebalancement
        adaptive_turnover: bool = True,  # ajustement dynamique du turnover
        market_impact_model: Optional[Callable] = None
    ):
        """
        Initialise le module d'optimisation des transactions.
        
        Parameters:
        -----------
        fee_structure : Dict[str, float], optional
            Structure de frais par actif (si différente selon les actifs)
        base_fee_rate : float, default 0.001
            Taux de frais de base (0.1% par défaut)
        min_improvement_threshold : float, default 0.002
            Seuil minimal d'amélioration attendue pour justifier un rebalancement
        observation_period : int, default 5
            Période d'observation après rebalancement (en jours)
        adaptive_turnover : bool, default True
            Ajuster dynamiquement le turnover maximum selon les conditions
        market_impact_model : Callable, optional
            Fonction pour évaluer l'impact de marché des transactions
        """
        self.fee_structure = fee_structure if fee_structure else {}
        self.base_fee_rate = base_fee_rate
        self.min_improvement_threshold = min_improvement_threshold
        self.observation_period = observation_period
        self.adaptive_turnover = adaptive_turnover
        self.market_impact_model = market_impact_model
        
        # Statistiques internes
        self.last_rebalance_date = None
        self.rebalance_history = []
        self.transaction_costs = []
        self.skipped_rebalances = 0
        self.executed_rebalances = 0
        
        # Paramètres adaptatifs
        self.current_max_turnover = 0.3  # Valeur initiale
        self.turnover_adjustment_factor = 1.0
        self.volatility_scaling = True

    def calculate_transaction_costs(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        volumes: Dict[str, float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcule les coûts de transaction pour un rebalancement donné.
        
        Parameters:
        -----------
        current_weights : Dict[str, float]
            Poids actuels du portefeuille
        target_weights : Dict[str, float]
            Poids cibles du portefeuille
        portfolio_value : float
            Valeur totale du portefeuille
        volumes : Dict[str, float], optional
            Volumes de trading récents pour chaque actif
            
        Returns:
        --------
        float
            Coût total des transactions en valeur absolue
        Dict[str, float]
            Coûts détaillés par actif
        """
        transaction_costs = {}
        total_cost = 0.0
        
        # Calculer le turnover total
        turnover = 0.0
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            weight_change = abs(target_weight - current_weight)
            turnover += weight_change
        
        # Pour chaque actif, calculer le coût de transaction
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)
            weight_change = abs(target_weight - current_weight)
            
            # Montant de la transaction
            transaction_amount = weight_change * portfolio_value
            
            # Taux de frais spécifique à l'actif ou taux de base
            fee_rate = self.fee_structure.get(asset, self.base_fee_rate)
            
            # Coût de base des frais
            base_cost = transaction_amount * fee_rate
            
            # Coût d'impact de marché si un modèle est fourni
            market_impact_cost = 0.0
            if self.market_impact_model and volumes and asset in volumes:
                volume = volumes[asset]
                relative_size = transaction_amount / volume if volume > 0 else 0
                market_impact_cost = self.market_impact_model(relative_size, transaction_amount)
            
            # Coût total pour cet actif
            asset_cost = base_cost + market_impact_cost
            transaction_costs[asset] = asset_cost
            total_cost += asset_cost
        
        return total_cost, transaction_costs
    
    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        expected_improvement: float,
        current_date: pd.Timestamp,
        volatility: Optional[pd.Series] = None
    ) -> Tuple[bool, Dict[str, float], float]:
        """
        Détermine si un rebalancement doit être effectué en fonction des coûts
        et de l'amélioration attendue.
        
        Parameters:
        -----------
        current_weights : Dict[str, float]
            Poids actuels du portefeuille
        target_weights : Dict[str, float]
            Poids cibles du portefeuille
        portfolio_value : float
            Valeur totale du portefeuille
        expected_improvement : float
            Amélioration attendue du rendement (en %)
        current_date : pd.Timestamp
            Date actuelle
        volatility : pd.Series, optional
            Série de volatilité pour les actifs
            
        Returns:
        --------
        bool
            True si le rebalancement est recommandé
        Dict[str, float]
            Poids optimisés (peut être différent des poids cibles)
        float
            Coût estimé du rebalancement
        """
        # Vérifier la période d'observation
        if self.last_rebalance_date:
            days_since_last_rebalance = (current_date - self.last_rebalance_date).days
            if days_since_last_rebalance < self.observation_period:
                self.skipped_rebalances += 1
                return False, current_weights, 0.0
        
        # Calculer les coûts de transaction
        total_cost, _ = self.calculate_transaction_costs(current_weights, target_weights, portfolio_value)
        
        # Coût relatif (en pourcentage du portefeuille)
        relative_cost = total_cost / portfolio_value
        
        # Calculer le turnover
        turnover = sum(abs(target_weights.get(asset, 0.0) - current_weights.get(asset, 0.0)) 
                       for asset in set(current_weights.keys()) | set(target_weights.keys()))
        
        # Ajuster le turnover maximum en fonction de la volatilité si activé
        if self.adaptive_turnover and volatility is not None:
            # Calculer la volatilité moyenne pondérée du portefeuille
            weighted_vol = 0.0
            for asset, weight in current_weights.items():
                if asset in volatility.index:
                    weighted_vol += weight * volatility[asset]
            
            # Ajuster le facteur de turnover en fonction de la volatilité
            # Plus le marché est volatil, plus nous autorisons de turnover
            vol_scaling = min(2.0, max(0.5, weighted_vol / 0.03))  # 3% comme référence
            self.turnover_adjustment_factor = vol_scaling
            self.current_max_turnover = min(0.5, 0.3 * vol_scaling)  # Plafonné à 50%
        
        # Vérifier si le turnover est acceptable
        if turnover > self.current_max_turnover:
            # Optimiser les poids pour respecter la contrainte de turnover
            optimized_weights = self.optimize_weights_with_turnover_constraint(
                current_weights, target_weights, self.current_max_turnover
            )
            
            # Recalculer les coûts avec les poids optimisés
            total_cost, _ = self.calculate_transaction_costs(current_weights, optimized_weights, portfolio_value)
            relative_cost = total_cost / portfolio_value
            
        else:
            optimized_weights = target_weights
        
        # Décider si le rebalancement vaut la peine
        if expected_improvement > (relative_cost + self.min_improvement_threshold):
            self.executed_rebalances += 1
            self.last_rebalance_date = current_date
            self.transaction_costs.append(total_cost)
            return True, optimized_weights, total_cost
        else:
            self.skipped_rebalances += 1
            return False, current_weights, 0.0
    
    def optimize_weights_with_turnover_constraint(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        max_turnover: float
    ) -> Dict[str, float]:
        """
        Optimise les poids du portefeuille en respectant une contrainte de turnover maximum.
        
        Parameters:
        -----------
        current_weights : Dict[str, float]
            Poids actuels du portefeuille
        target_weights : Dict[str, float]
            Poids cibles du portefeuille
        max_turnover : float
            Turnover maximum autorisé
            
        Returns:
        --------
        Dict[str, float]
            Poids optimisés
        """
        all_assets = list(set(current_weights.keys()) | set(target_weights.keys()))
        n_assets = len(all_assets)
        
        # Convertir les dictionnaires en arrays pour l'optimisation
        current_array = np.zeros(n_assets)
        target_array = np.zeros(n_assets)
        
        for i, asset in enumerate(all_assets):
            current_array[i] = current_weights.get(asset, 0.0)
            target_array[i] = target_weights.get(asset, 0.0)
        
        # Fonction objectif: minimiser la distance entre les poids optimisés et les poids cibles
        def objective(weights):
            return np.sum((weights - target_array) ** 2)
        
        # Contrainte: la somme des poids doit être égale à 1
        def sum_constraint(weights):
            return np.sum(weights) - 1.0
        
        # Contrainte: le turnover ne doit pas dépasser le maximum
        def turnover_constraint(weights):
            return max_turnover - np.sum(np.abs(weights - current_array))
        
        # Limites: chaque poids doit être entre 0 et 1
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': sum_constraint},
            {'type': 'ineq', 'fun': turnover_constraint}
        ]
        
        # Optimisation
        result = minimize(
            objective,
            x0=current_array,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        # Convertir le résultat en dictionnaire
        optimized_weights = {}
        for i, asset in enumerate(all_assets):
            # Arrondir à 4 décimales et ignorer les poids très petits
            weight = round(result.x[i], 4)
            if weight > 0.0001:
                optimized_weights[asset] = weight
        
        # Normaliser pour s'assurer que la somme est exactement 1
        weight_sum = sum(optimized_weights.values())
        if weight_sum > 0:
            optimized_weights = {asset: weight / weight_sum for asset, weight in optimized_weights.items()}
        
        return optimized_weights
    
    def estimate_expected_improvement(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        expected_returns: pd.Series,
        horizon: int = 21  # horizon par défaut en jours
    ) -> float:
        """
        Estime l'amélioration attendue du rendement en changeant l'allocation.
        
        Parameters:
        -----------
        current_weights : Dict[str, float]
            Poids actuels du portefeuille
        target_weights : Dict[str, float]
            Poids cibles du portefeuille
        expected_returns : pd.Series
            Rendements attendus pour chaque actif
        horizon : int, default 21
            Horizon de prévision en jours
            
        Returns:
        --------
        float
            Amélioration attendue du rendement (en %)
        """
        # Calculer le rendement attendu pour l'allocation actuelle
        current_return = 0.0
        for asset, weight in current_weights.items():
            if asset in expected_returns.index:
                current_return += weight * expected_returns[asset]
        
        # Calculer le rendement attendu pour l'allocation cible
        target_return = 0.0
        for asset, weight in target_weights.items():
            if asset in expected_returns.index:
                target_return += weight * expected_returns[asset]
        
        # Calculer la différence sur l'horizon
        improvement = (target_return - current_return) * horizon
        
        return improvement
    
    def get_optimization_stats(self) -> Dict[str, float]:
        """
        Retourne des statistiques sur l'optimisation des transactions.
        
        Returns:
        --------
        Dict[str, float]
            Statistiques d'optimisation
        """
        stats = {
            "executed_rebalances": self.executed_rebalances,
            "skipped_rebalances": self.skipped_rebalances,
            "rebalance_ratio": self.executed_rebalances / max(1, self.executed_rebalances + self.skipped_rebalances),
            "avg_transaction_cost": np.mean(self.transaction_costs) if self.transaction_costs else 0.0,
            "total_transaction_cost": sum(self.transaction_costs),
            "current_max_turnover": self.current_max_turnover,
            "turnover_adjustment_factor": self.turnover_adjustment_factor
        }
        
        return stats
    
    def reset_stats(self):
        """
        Réinitialise les statistiques d'optimisation.
        """
        self.last_rebalance_date = None
        self.rebalance_history = []
        self.transaction_costs = []
        self.skipped_rebalances = 0
        self.executed_rebalances = 0
        self.current_max_turnover = 0.3
        self.turnover_adjustment_factor = 1.0


# Exemple de modèle d'impact de marché simple
def simple_market_impact(relative_size, transaction_amount):
    """
    Modèle d'impact de marché simple basé sur la taille relative de la transaction.
    
    Parameters:
    -----------
    relative_size : float
        Taille relative de la transaction par rapport au volume
    transaction_amount : float
        Montant de la transaction
        
    Returns:
    --------
    float
        Coût estimé de l'impact de marché
    """
    # Paramètres de calibration
    base_impact = 0.0001  # 1 point de base d'impact minimal
    nonlinear_factor = 1.5  # Facteur non linéaire pour les grandes transactions
    
    # L'impact augmente de façon non linéaire avec la taille relative
    impact_factor = base_impact * (1.0 + nonlinear_factor * relative_size ** 2)
    
    return transaction_amount * impact_factor