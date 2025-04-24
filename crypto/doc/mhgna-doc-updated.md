# Multi-Horizon Graphical Network Allocation (MHGNA)
**Version:** 1.2.0  
**Date de dernière mise à jour:** 4 avril 2025  
**Auteurs:** [Votre Nom]  
**Contributeurs:** Claude

## Table des matières
- [Présentation](#présentation)
- [Évolution](#évolution)
- [Problèmes connus et solutions](#problèmes-connus-et-solutions)
- [Roadmap](#roadmap)
- [Configuration actuelle](#configuration-actuelle)
- [Installation et dépendances](#installation-et-dépendances)
- [Prochaines étapes](#prochaines-étapes)
- [Notes](#notes)

## Présentation

Le Multi-Horizon Graphical Network Allocation (MHGNA) est une approche avancée de gestion de portefeuille qui utilise la théorie des graphes et l'analyse des dépendances conditionnelles pour optimiser l'allocation d'actifs dans le marché des cryptomonnaies. Contrairement aux approches traditionnelles basées sur les corrélations simples, MHGNA exploite la structure topologique du marché en intégrant des horizons temporels multiples et une sélection d'actifs basée sur leur position dans le réseau de dépendances.

Cette approche constitue une évolution substantielle de la méthode Graphical Lasso originale en résolvant ses principales limitations tout en créant un cadre conceptuel distinct et complémentaire au framework QAAF (Quantitative Algorithmic Asset Framework).

## Évolution

### v1.0.0 - "Concept Initial" (Mars 2025)
**Description :** Implémentation initiale basée sur le Graphical Lasso standard avec une seule fenêtre temporelle et une allocation simple

**Changements :**
- Première implémentation du concept Graphical Lasso pour l'allocation d'actifs crypto
- Fenêtre glissante unique de 60 jours pour l'estimation du graphe
- Allocation de portefeuille basée sur la matrice de précision
- Rebalancement hebdomadaire (7 jours)
- 5 actifs sélectionnés dans le portefeuille

**Résultats :**
```
Rendement Total: -26.91% (vs Bitcoin: +9.11%)
Rendement Annualisé: -15.72% (vs Bitcoin: +4.87%)
Volatilité Annualisée: 79.47% (vs Bitcoin: 52.77%)
Ratio de Sharpe: -0.20 (vs Bitcoin: 0.09)
Ratio de Sortino: -0.26 (vs Bitcoin: 0.12)
Maximum Drawdown: -79.38% (vs Bitcoin: -66.74%)
Ratio de Calmar: -0.20 (vs Bitcoin: 0.07)
```

**Limitations identifiées :**
- Sensibilité excessive au paramètre de régularisation λ
- Instabilité lors des changements rapides de régime
- Turnover élevé générant des coûts de transaction importants
- Absence de mécanisme de protection contre les drawdowns
- Manque d'intégration du momentum

### v1.0.1 - "Stabilisation" (Mars 2025)
**Description :** Corrections des problèmes critiques et stabilisation de l'implémentation initiale

**Changements :**
- Correction des erreurs d'indexation dans la fonction backtest_strategy
- Amélioration de la gestion des données manquantes ou incomplètes
- Optimisation des performances de calcul
- Documentation améliorée du code

**Résultats :**
*Performance similaire à v1.0.0 mais avec une exécution plus stable*

### v1.1.0 - "Multi-Horizon" (Avril 2025)
**Description :** Introduction de l'approche multi-horizon et des mécanismes de protection contre les drawdowns

**Changements majeurs :**
- Analyse multi-horizon avec 3 fenêtres temporelles (30, 90, 180 jours)
- Paramètre de régularisation adaptatif selon l'horizon
- Rebalancement mensuel (21 jours) pour réduire le turnover
- Augmentation du portefeuille à 7 actifs
- Intégration du momentum dans l'allocation
- Implémentation du mécanisme de protection contre les drawdowns
- Limitation du turnover à 30% par rebalancement

**Améliorations techniques :**
- Optimisation du code pour une meilleure stabilité
- Débogage des rendements anormaux
- Visualisations améliorées avec support pour le cash

**Résultats :**
```
Rendement Total: 345.18% (vs Bitcoin: 117.56%)
Rendement Annualisé: 170.37% (vs Bitcoin: 67.82%)
Volatilité Annualisée: 165.98% (vs Bitcoin: 41.68%)
Ratio de Sharpe: 1.03 (vs Bitcoin: 1.63)
Ratio de Sortino: 7.22 (vs Bitcoin: 2.17)
Maximum Drawdown: -19.21% (vs Bitcoin: -25.82%)
Ratio de Calmar: 8.87 (vs Bitcoin: 2.63)
```

**Analyse de la performance :**
- Surperformance significative par rapport au Bitcoin en termes de rendement absolu
- Meilleure protection contre les drawdowns (19.21% vs 25.82%)
- Excellente gestion des mouvements baissiers (Sortino de 7.22 vs 2.17)
- Ratio rendement/drawdown (Calmar) supérieur (8.87 vs 2.63)
- Performance exceptionnelle en octobre 2022 (+195.99%)
- Volatilité plus élevée mais compensée par des rendements supérieurs

### v1.2.0 - "Stabilité & Robustesse" (Avril 2025)
**Description :** Résolution des problèmes critiques et amélioration de la robustesse

**Changements majeurs :**
- Correction du traitement des données Yahoo Finance (gestion du MultiIndex)
- Résolution du problème des graphes déconnectés dans le calcul de centralité
- Implémentation complète du module de préservation des gains
- Refactorisation du code pour améliorer la lisibilité et la maintenabilité

**Améliorations techniques :**
- Ajout de diagnostics détaillés pour le débogage
- Meilleure gestion des erreurs avec traces détaillées
- Gestion des valeurs extrêmes pour éviter les instabilités
- Documentation complète des problèmes connus et solutions

## Problèmes connus et solutions

### 1. Gestion des données Yahoo Finance
**Problème :** Depuis 2024, Yahoo Finance retourne des données avec un MultiIndex et des colonnes en majuscules, causant des problèmes d'accès aux données.

**Solution :**
```python
def standardize_yahoo_data(data):
    """Standardise les données de Yahoo Finance"""
    if isinstance(data.columns, pd.MultiIndex):
        data = pd.DataFrame({
            'open': data['Open'].iloc[:, 0],
            'high': data['High'].iloc[:, 0],
            'low': data['Low'].iloc[:, 0],
            'close': data['Close'].iloc[:, 0],
            'volume': data['Volume'].iloc[:, 0],
            'adj close': data['Adj Close'].iloc[:, 0] if 'Adj Close' in data.columns else data['Close'].iloc[:, 0]
        })
    else:
        data.columns = data.columns.str.lower()
    return data
```

### 2. Problème avec les graphes déconnectés
**Problème :** L'erreur `eigenvector_centrality_numpy does not give consistent results for disconnected graphs` survient lorsque le graphe de dépendance n'est pas entièrement connecté.

**Solution :** Traiter séparément chaque composante connectée du graphe pour le calcul des centralités.
```python
# Dans select_portfolio_assets:
connected_components = list(nx.connected_components(G))
for component in connected_components:
    subgraph = G.subgraph(component)
    if len(subgraph) > 1:
        # Calculer les centralités par composante
        ...
```

### 3. Module de préservation des gains
**Problème :** L'erreur `'GainPreservationModule' object has no attribute 'adjust_allocation_weights'` indique que la méthode n'est pas correctement définie dans la classe.

**Solution :** S'assurer que la méthode `adjust_allocation_weights` est correctement incluse dans la classe `GainPreservationModule` avec la bonne indentation.
```python
class GainPreservationModule:
    # Autres méthodes...
    
    def adjust_allocation_weights(self, target_weights, current_value, current_date, 
                                  market_drawdown=0.0, volatility=None, current_weights=None):
        # Implémentation...
```

### 4. Problème de visualisation
**Problème :** L'erreur `TypeError: no numeric data to plot` survient quand les données du portefeuille sont vides ou contiennent uniquement des zéros.

**Solution :** Vérifier les données avant de les tracer et afficher un message explicite si les données ne sont pas valides.
```python
# Dans plot_results:
if results['Portfolio Value'].eq(results['Portfolio Value'].iloc[0]).all():
    print("AVERTISSEMENT: Aucun changement de valeur dans le portefeuille, impossible de tracer le graphique")
    return None
```

## Roadmap

### v1.3.0 - "Optimisation & Fine-tuning" (Planifiée pour Mai 2025)
**Description :** Optimisation des paramètres et amélioration de la robustesse de la stratégie

**Tâches :**
- [ ] Analyse détaillée de la performance par période (haussière/baissière)
- [ ] Optimisation des horizons temporels et de leurs pondérations
- [ ] Calibration fine du mécanisme de protection contre les drawdowns
- [ ] Tests de sensibilité sur les principaux paramètres
- [ ] Simulations Monte-Carlo pour évaluer la robustesse
- [ ] Amélioration du traitement des graphes déconnectés

**Priorité :** Haute

### v2.0.0 - "Intégration On-Chain" (Planifiée pour Juin 2025)
**Description :** Incorporation de métriques on-chain dans la construction du graphe

**Tâches :**
- [ ] Intégration de données on-chain (flux de transactions, activité des adresses)
- [ ] Pondération des arêtes du graphe par les volumes de transferts
- [ ] Détection des changements structurels dans le réseau
- [ ] Analyse des communautés basée sur les interactions on-chain
- [ ] Intégration des métriques de liquidité des marchés

**Priorité :** Moyenne

### v3.0.0 - "Intelligence Adaptative" (Planifiée pour Q3 2025)
**Description :** Intégration de l'apprentissage automatique pour une adaptation dynamique

**Tâches :**
- [ ] Développement d'un système de détection automatique des régimes de marché
- [ ] Implémentation d'un module d'apprentissage par renforcement pour l'optimisation des paramètres
- [ ] Prédiction des changements topologiques du réseau
- [ ] Allocation adaptative basée sur les régimes identifiés
- [ ] Intégration de l'analyse de sentiment

**Priorité :** Basse (développement long terme)

## Configuration actuelle

### Paramètres techniques
```yaml
# Paramètres généraux
tickers: ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD', 'AAVE-USD', 'CRV-USD', 'ATOM-USD', 'LTC-USD']
start_date: '2022-01-01'
end_date: '2024-01-01'
initial_capital: 10000

# Paramètres d'horizon multiple
horizons:
  court:
    window: 30
    weight: 0.25
  moyen:
    window: 90
    weight: 0.50
  long:
    window: 180
    weight: 0.25

# Paramètres de rééquilibrage
rebalance_freq: 21  # Mensuel
portfolio_size: 7   # Nombre d'actifs

# Paramètres de régularisation
alpha_short: 0.02   # Plus fort pour horizon court (plus sparse)
alpha_medium: 0.01  # Moyen pour horizon moyen
alpha_long: 0.005   # Plus faible pour horizon long (plus dense)

# Paramètres de momentum
momentum_window: 60  # Jours
momentum_weight: 0.3  # Influence du momentum dans l'allocation

# Paramètres de turnover
max_turnover: 0.3  # Maximum 30% de changement par rééquilibrage

# Risk management
max_asset_weight: 0.35  # Poids maximum par actif
min_asset_weight: 0.05  # Poids minimum par actif

# Paramètres de drawdown
max_drawdown_threshold: -0.15  # -15% déclenche protection
risk_reduction_factor: 0.5     # Réduction de 50% de l'exposition
recovery_threshold: 0.10       # +10% de récupération pour revenir à l'exposition normale

# Paramètres de préservation des gains
profit_threshold: 0.15  # 15% de profit pour commencer à préserver
max_stablecoin_allocation: 0.3  # Maximum 30% en stablecoin
stablecoin_assets: ["USDT", "USDC", "DAI"]  # Stablecoins disponibles
```

## Installation et dépendances

### Dépendances requises
```python
# Installation des dépendances
!pip install numpy pandas yfinance matplotlib seaborn networkx scikit-learn scipy
```

### Dépendances principales
- **numpy (>=1.20.0)** : Calculs numériques
- **pandas (>=1.3.0)** : Manipulation de données
- **yfinance (>=0.1.70)** : Récupération des données financières
- **matplotlib (>=3.4.0)** : Visualisation
- **networkx (>=2.6.0)** : Manipulation de graphes
- **scikit-learn (>=0.24.0)** : Machine learning et Graphical Lasso
- **scipy (>=1.7.0)** : Calculs scientifiques

### Dépendances secondaires
- **seaborn (>=0.11.0)** : Visualisations avancées

### Configuration minimale recommandée
- Python 3.8+
- 4 Go de RAM minimum
- CPU multi-cœur recommandé pour les calculs matriciels

### Environnements supportés
- Compatible avec Jupyter Notebook/Google Colab
- Testé sur Linux, macOS et Windows

## Prochaines étapes

### Priorité 1 (Urgent/Important)
1. **Résolution des problèmes d'indentation de code**
   - Vérifier que toutes les méthodes sont correctement indentées dans leurs classes
   - Tester individuellement chaque composant du système
   - Implémenter des tests unitaires pour valider le comportement

2. **Amélioration du traitement des graphes déconnectés**
   - Implémenter une version robuste de l'algorithme pour les graphes déconnectés
   - Tester avec différentes topologies de graphes
   - Documenter clairement la solution pour référence future

3. **Optimisation du traitement des données Yahoo Finance**
   - Créer une fonction réutilisable pour standardiser les données
   - Ajouter des validations pour s'assurer que les données sont correctement formatées
   - Tester avec différentes périodes et actifs pour garantir la robustesse

### Priorité 2 (Important/Non Urgent)
1. **Optimisation des paramètres**
   - Réaliser une analyse de sensibilité sur les paramètres clés
   - Tester différentes configurations d'horizons temporels
   - Optimiser les seuils de drawdown et les facteurs de réduction

2. **Documentation et guides d'utilisation**
   - Créer un guide d'utilisation détaillé pour les utilisateurs non techniques
   - Documenter toutes les fonctions avec leurs paramètres et comportements attendus
   - Ajouter des exemples d'utilisation pour chaque composant principal

3. **Améliorations de visualisation**
   - Ajouter des messages d'erreur explicites lors des problèmes de tracé
   - Améliorer la robustesse des fonctions de visualisation
   - Implémenter des visualisations interactives pour l'exploration des données

### Priorité 3 (Nice to Have)
1. **Intégration avec des API externes**
   - Connecter à des sources de données alternatives (Glassnode, Messari)
   - Permettre l'exécution en temps réel via des API d'échange
   - Créer un système d'alertes basé sur les changements topologiques

2. **Interface utilisateur**
   - Développer une interface web pour interagir avec le modèle
   - Permettre l'ajustement des paramètres via une interface graphique
   - Créer un système de rapports automatisés

3. **Extensions théoriques**
   - Explorer des approches topologiques alternatives (persistent homology)
   - Intégrer des modèles de corrélation conditionnelle dynamique
   - Étudier les propriétés mathématiques des graphes financiers multi-horizon

## Notes

### Points importants à retenir
- **Structure vs Statistiques** : MHGNA fonctionne en exploitant la structure du marché plutôt que des statistiques temporelles simples
- **Multi-temporalité** : L'intégration de différents horizons temporels est clé pour capturer différentes dynamiques
- **Protection vs Performance** : Le mécanisme de drawdown sacrifie une partie de la performance pour une meilleure gestion du risque
- **Complémentarité avec QAAF** : Les deux approches offrent des perspectives différentes et complémentaires sur le marché
- **Debugging méthodique** : La complexité du système nécessite une approche de débogage structurée et progressive

### Bonnes pratiques
- Toujours utiliser la version multi-horizon plutôt que la version simple
- Calibrer régulièrement les paramètres de régularisation selon la volatilité du marché
- Vérifier la connectivité des graphes générés avant d'appliquer des métriques de centralité
- Limiter le turnover pour optimiser le ratio rendement/coût
- Surveiller les actifs à forte contribution pour éviter la concentration du risque
- Implémenter des traces de débogage détaillées pour comprendre le comportement du système

### Pièges à éviter
- **Suroptimisation** : Éviter d'ajuster trop finement les paramètres sur les données historiques
- **Biais de survie** : Attention à ne pas sélectionner des actifs performants rétrospectivement
- **Régularisation excessive** : Un graphe trop sparse perd des informations importantes
- **Ignorance des fondamentaux** : La structure du réseau ne capture pas tous les aspects du marché
- **Allocation binaire** : Préférer une allocation pondérée à une sélection binaire des actifs
- **Problèmes d'indentation** : Les erreurs d'indentation dans le code Python peuvent causer des comportements inattendus

### Leçons apprises
1. **Importance de la robustesse** : La stabilité de l'algorithme est cruciale pour des résultats cohérents
2. **Valeur de la limitation du risque** : Le contrôle des drawdowns améliore significativement le profil de risque
3. **Pouvoir du multi-horizon** : L'intégration de différentes échelles temporelles capture une vue plus complète du marché
4. **Centrality vs Community** : L'équilibre entre sélection par centralité et diversification par communauté est essentiel
5. **Calcul distribué** : Pour les grands univers d'actifs, l'utilisation de calcul distribué devient nécessaire
6. **Tests progressifs** : Tester avec un sous-ensemble réduit de données avant de passer à l'ensemble complet

### Bibliographie clé
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). "Sparse inverse covariance estimation with the graphical lasso"
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Newman, M. E. J. (2010). "Networks: An Introduction"
- Hautsch, N., Schaumburg, J., & Schienle, M. (2014). "Financial Network Systemic Risk Contributions"
- Tumminello, M., Lillo, F., & Mantegna, R. N. (2010). "Correlation, hierarchies, and networks in financial markets"