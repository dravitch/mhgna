# Multi-Horizon Graphical Network Allocation (MHGNA)
**Version:** 1.1.0  
**Date de dernière mise à jour:** 2 avril 2025  
**Auteurs:** [Votre Nom]  
**Contributeurs:** Claude

## Table des matières
- [Présentation](#présentation)
- [Évolution](#évolution)
- [Roadmap](#roadmap)
- [Configuration actuelle](#configuration-actuelle)
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

## Roadmap

### v1.2.0 - "Optimisation & Robustesse" (Planifiée pour Mai 2025)
**Description :** Optimisation des paramètres et amélioration de la robustesse de la stratégie

**Tâches :**
- [ ] Analyse détaillée de la performance par période (haussière/baissière)
- [ ] Optimisation des horizons temporels et de leurs pondérations
- [ ] Calibration fine du mécanisme de protection contre les drawdowns
- [ ] Tests de sensibilité sur les principaux paramètres
- [ ] Simulations Monte-Carlo pour évaluer la robustesse
- [ ] Amélioration du traitement des graphes déconnectés

**Priorité :** Haute

### v1.3.0 - "Intégration On-Chain" (Planifiée pour Juin 2025)
**Description :** Incorporation de métriques on-chain dans la construction du graphe

**Tâches :**
- [ ] Intégration de données on-chain (flux de transactions, activité des adresses)
- [ ] Pondération des arêtes du graphe par les volumes de transferts
- [ ] Détection des changements structurels dans le réseau
- [ ] Analyse des communautés basée sur les interactions on-chain
- [ ] Intégration des métriques de liquidité des marchés

**Priorité :** Moyenne

### v2.0.0 - "Intelligence Adaptative" (Planifiée pour Q3 2025)
**Description :** Intégration de l'apprentissage automatique pour une adaptation dynamique

**Tâches :**
- [ ] Développement d'un système de détection automatique des régimes de marché
- [ ] Implémentation d'un module d'apprentissage par renforcement pour l'optimisation des paramètres
- [ ] Prédiction des changements topologiques du réseau
- [ ] Allocation adaptative basée sur les régimes identifiés
- [ ] Intégration de l'analyse de sentiment

**Priorité :** Basse (développement long terme)

### v3.0.0 - "Fusion MHGNA-QAAF" (Planifiée pour Q4 2025)
**Description :** Intégration des approches MHGNA et QAAF dans un framework unifié

**Tâches :**
- [ ] Développement d'un framework de sélection d'actifs hybride
- [ ] Création d'un système d'allocation qui combine structure de réseau et facteurs
- [ ] Implémentation d'un mécanisme de gestion du risque multi-dimensionnel
- [ ] Interface unifiée pour l'analyse et la visualisation
- [ ] Comparaison systématique des performances

**Priorité :** Exploratoire

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
```

### Dépendances
- **Principales :**
  - pandas (>=1.3.0)
  - numpy (>=1.20.0)
  - matplotlib (>=3.4.0)
  - networkx (>=2.6.0)
  - scikit-learn (>=0.24.0)
  - yfinance (>=0.1.70)

- **Secondaires :**
  - seaborn (>=0.11.0)
  - scipy (>=1.7.0)

### Environnements supportés
- Python 3.8+
- Compatible avec les environnements Jupyter Notebook
- Testé sur Linux, macOS et Windows

## Prochaines étapes

### Priorité 1 (Urgent/Important)
1. **Analyse du pic d'octobre 2022**
   - Identifier les actifs spécifiques qui ont contribué à la performance de +195.99%
   - Vérifier si cette surperformance est due à un effet de levier ou à une sélection judicieuse
   - Implémenter des garde-fous pour éviter les biais de sélection rétrospectifs

2. **Correction des graphes déconnectés**
   - Résoudre l'erreur `eigenvector_centrality_numpy does not give consistent results for disconnected graphs`
   - Implémentation d'une version robuste de l'algorithme pour les graphes déconnectés
   - Améliorer la détection des communautés pour les graphes partiellement connectés

3. **Documentation des fonctions**
   - Documenter toutes les fonctions principales avec leurs paramètres et valeurs de retour
   - Créer un guide d'utilisation pour les utilisateurs non techniques
   - Ajouter des exemples d'utilisation pour chaque composant principal

### Priorité 2 (Important/Non Urgent)
1. **Optimisation des paramètres**
   - Réaliser une analyse de sensibilité sur les paramètres clés
   - Tester différentes configurations d'horizons temporels
   - Optimiser les seuils de drawdown et les facteurs de réduction

2. **Analyses comparatives**
   - Comparer systématiquement avec d'autres stratégies (QAAF, Markowitz, Equal Weight)
   - Tester sur différentes périodes de marché (bull/bear/sideways)
   - Analyser la performance par classe d'actifs (Layer 1, DeFi, etc.)

3. **Améliorations de visualisation**
   - Créer un dashboard interactif pour le suivi de la stratégie
   - Ajouter des visualisations dynamiques de l'évolution du graphe
   - Développer des représentations des communautés d'actifs

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

### Bonnes pratiques
- Toujours utiliser la version multi-horizon plutôt que la version simple
- Calibrer régulièrement les paramètres de régularisation selon la volatilité du marché
- Vérifier la connectivité des graphes générés avant d'appliquer des métriques de centralité
- Limiter le turnover pour optimiser le ratio rendement/coût
- Surveiller les actifs à forte contribution pour éviter la concentration du risque

### Pièges à éviter
- **Suroptimisation** : Éviter d'ajuster trop finement les paramètres sur les données historiques
- **Biais de survie** : Attention à ne pas sélectionner des actifs performants rétrospectivement
- **Régularisation excessive** : Un graphe trop sparse perd des informations importantes
- **Ignorance des fondamentaux** : La structure du réseau ne capture pas tous les aspects du marché
- **Allocation binaire** : Préférer une allocation pondérée à une sélection binaire des actifs

### Leçons apprises
1. **Importance de la robustesse** : La stabilité de l'algorithme est cruciale pour des résultats cohérents
2. **Valeur de la limitation du risque** : Le contrôle des drawdowns améliore significativement le profil de risque
3. **Pouvoir du multi-horizon** : L'intégration de différentes échelles temporelles capture une vue plus complète du marché
4. **Centrality vs Community** : L'équilibre entre sélection par centralité et diversification par communauté est essentiel
5. **Calcul distribué** : Pour les grands univers d'actifs, l'utilisation de calcul distribué devient nécessaire

### Bibliographie clé
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). "Sparse inverse covariance estimation with the graphical lasso"
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Newman, M. E. J. (2010). "Networks: An Introduction"
- Hautsch, N., Schaumburg, J., & Schienle, M. (2014). "Financial Network Systemic Risk Contributions"
- Tumminello, M., Lillo, F., & Mantegna, R. N. (2010). "Correlation, hierarchies, and networks in financial markets"