# Guide d'utilisation de MHGNA Simplifié
## Version pour le trading manuel de cryptomonnaies

Ce guide vous explique comment utiliser la version simplifiée de MHGNA (Multi-Horizon Graphical Network Allocation) pour le trading manuel de cryptomonnaies.

## Introduction

MHGNA Simplifié est une adaptation de l'algorithme MHGNA complet, spécialement conçue pour les traders manuels souhaitant bénéficier des analyses avancées de réseau sans avoir à gérer la complexité technique du système complet. Cette version vous fournit :

1. **Des recommandations d'actifs claires** basées sur leur position dans le réseau, leur momentum et leur stabilité
2. **Des signaux d'alerte explicites** pour vous aider à gérer le risque
3. **Des visualisations intuitives** du marché et de sa structure

## Installation

Prérequis : Python 3.8+ avec les bibliothèques suivantes :
```bash
pip install numpy pandas yfinance matplotlib seaborn networkx scikit-learn
```

## Utilisation rapide

La façon la plus simple d'utiliser MHGNA Simplifié est d'utiliser la fonction `start_mhgna_analysis` :

```python
from mhgna_simplified import start_mhgna_analysis

# Exécuter l'analyse avec les paramètres par défaut
mhgna, recommendations, report = start_mhgna_analysis()

# Ou avec des paramètres personnalisés
mhgna, recommendations, report = start_mhgna_analysis(
    # Liste personnalisée de cryptos à suivre
    tickers=['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD', 
             'MATIC-USD', 'LINK-USD', 'DOT-USD', 'UNI-USD'],
    # Période d'historique en années
    lookback_period=1,
    # Nombre d'actifs à recommander
    recommended_assets=5,
    # Dossier où sauvegarder les fichiers générés
    output_folder='./mhgna_reports'
)
```

Cette fonction va :
1. Récupérer les données historiques
2. Construire le réseau multi-horizon
3. Générer des recommandations et alertes
4. Créer des visualisations
5. Produire un rapport textuel

## Utilisation détaillée

Pour plus de contrôle, vous pouvez utiliser directement la classe `MHGNASimplified` :

```python
from mhgna_simplified import MHGNASimplified, SimpleConfig

# Configuration personnalisée (optionnelle)
config = SimpleConfig()
config.tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']
config.recommended_assets = 3
config.drawdown_alert_threshold = -0.12  # Plus sensible aux drawdowns

# Créer l'instance MHGNA
mhgna = MHGNASimplified(config)

# Étape 1: Récupérer les données
mhgna.fetch_data()  # Par défaut jusqu'à aujourd'hui
# ou pour une date spécifique
# mhgna.fetch_data(end_date='2025-03-15')

# Étape 2: Construire le réseau
mhgna.build_network()

# Étape 3: Obtenir les recommandations
recommendations = mhgna.recommend_assets()
print(recommendations)

# Étape 4: Vérifier les alertes
alerts = mhgna.check_alerts()
for alert in alerts:
    print(alert['message'])

# Étape 5: Créer les visualisations
# Visualisation du réseau
mhgna.visualize_network(filename='network_graph.png')

# Visualisation des tendances et signaux
mhgna.visualize_market_trends(filename='market_trends.png')

# Étape 6: Générer le rapport complet
report = mhgna.generate_report(output_folder='./mhgna_reports')
print(report)
```

## Comprendre les recommandations

Le DataFrame des recommandations contient plusieurs colonnes :

| Colonne | Description |
|---------|-------------|
| **Rang** | Position dans le classement général |
| **Score Global** | Score combiné (0-1) basé sur toutes les métriques |
| **Centralité** | Position de l'actif dans le réseau (0-1) |
| **Momentum** | Force et direction de la tendance récente |
| **Stabilité** | Inverse de la volatilité normalisée (0-1) |
| **Rend. 30j** | Rendement sur les 30 derniers jours |
| **Rend. 90j** | Rendement sur les 90 derniers jours |

### Comment interpréter ces métriques :

- **Centralité élevée** : Actif fortement connecté au reste du marché, souvent plus stable et influent
- **Momentum élevé** : Tendance haussière forte
- **Stabilité élevée** : Volatilité relativement faible
- **Score Global élevé** : Bon équilibre entre position réseau, tendance et risque

## Comprendre les alertes

Le système génère quatre types d'alertes :

1. **DRAWDOWN** : Un actif a subi une baisse dépassant le seuil d'alerte (par défaut -15%)
2. **VOLATILITY** : La volatilité du marché est anormalement élevée
3. **OVERSOLD** : Un actif central a un momentum très négatif, suggérant une opportunité potentielle
4. **OVERBOUGHT** : Un actif périphérique a un momentum très positif, suggérant un risque de correction

## Comprendre les visualisations

### Visualisation du réseau
![Exemple de réseau](https://i.imgur.com/example_network.jpg)

Dans cette visualisation :
- **Taille des nœuds** : Représente la centralité (plus grand = plus central)
- **Couleur des nœuds** : Représente le momentum (vert = positif, rouge = négatif)
- **Épaisseur des liens** : Représente la force de la relation entre les actifs

### Visualisation des tendances
![Exemple de tendances](https://i.imgur.com/example_trends.jpg)

Cette visualisation contient trois parties :
1. **Courbes de prix normalisées** des actifs recommandés (100 = valeur de départ)
2. **Tableau des signaux d'alerte** actifs pour chaque actif
3. **Graphique des métriques** pour les actifs recommandés

## Bonnes pratiques d'utilisation

- **Fréquence** : Exécutez l'analyse une fois par mois pour le trading à moyen terme
- **Validation** : Utilisez cette analyse comme un complément à votre propre recherche
- **Diversification** : Ne concentrez pas tout votre capital sur un seul actif, même s'il a le meilleur score
- **Gestion du risque** : Prenez en compte les alertes de drawdown et de volatilité pour ajuster votre exposition

## Exemple de workflow mensuel

1. Au début de chaque mois, exécutez l'analyse complète
2. Consultez les 3-5 actifs les mieux classés et les alertes actives
3. Ajustez votre portefeuille en fonction des recommandations
4. Conservez le rapport et les visualisations comme référence
5. Si des évènements majeurs surviennent en cours de mois, vous pouvez exécuter une analyse supplémentaire

## Limites et considérations

- L'analyse est basée uniquement sur les données historiques, sans garantie de résultats futurs
- La qualité des recommandations dépend de la qualité des données récupérées
- Les relations entre actifs peuvent changer rapidement lors d'événements de marché majeurs
- Ce système n'intègre pas les fondamentaux des projets crypto, seulement leur comportement de prix

## Personnalisation avancée

Pour personnaliser davantage le système, vous pouvez modifier les paramètres suivants dans `SimpleConfig` :

```python
config = SimpleConfig()

# Horizon temporel
config.horizons = {
    'court': 20,    # Plus réactif aux mouvements récents
    'moyen': 60,
    'long': 120
}

# Seuils d'alerte
config.drawdown_alert_threshold = -0.10    # Plus sensible
config.volatility_alert_threshold = 0.7    # Plus sensible

# Style des visualisations
config.chart_style = 'whitegrid'
config.network_colors = 'plasma'
config.figsize = (14, 10)  # Plus grand
```

---

Nous espérons que cet outil vous aidera à prendre des décisions de trading plus éclairées grâce à l'analyse avancée des réseaux de cryptomonnaies. Bonne chance dans vos investissements!