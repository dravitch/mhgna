# PreservationStrategy.py

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
        volatility: float = None,
        current_weights: Dict[str, float] = None
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
        volatility: float = None,
        current_weights: Dict[str, float] = None
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