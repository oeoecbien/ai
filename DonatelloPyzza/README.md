# DonatelloPyzza - Agent Q-Learning

## Description

DonatelloPyzza est un projet d'apprentissage par renforcement qui utilise l'algorithme Q-Learning pour entraîner une tortue à naviguer dans un labyrinthe et trouver une pizza.

## Fonctionnalités

- Agent Q-Learning avec équation de Bellman
- Stratégie epsilon-greedy adaptative
- Système de récompenses avec bonus d'exploration
- Visualisation en temps réel avec pygame
- Métriques de performance et tests automatisés

## Installation

```bash
# Cloner le repository
git clone <url-du-repo>
cd DonatelloPyzza

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate      # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

```bash
cd examples
python qlearning.py
```

Le script vous guidera à travers la sélection de l'environnement, la configuration de l'entraînement et les tests de validation.

### Exemple d'utilisation programmatique

```python
from donatellopyzza import RLGame, Action, Feedback
from examples.qlearning import QLearningAgent

# Créer l'agent
agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9, epsilon=0.3)

# Entraîner sur un environnement
game = RLGame("maze", gui=True)
turtle = game.start()
reward, steps, success = agent.train_episode(game, turtle, show_gui=True, verbose=True)
```

## Algorithme Q-Learning

L'agent apprend une politique optimale en explorant l'environnement et en mettant à jour une table Q selon l'équation de Bellman :

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

### Hyperparamètres
- **Learning Rate (α)** : 0.1 - Vitesse d'apprentissage
- **Discount Factor (γ)** : 0.9 - Importance des récompenses futures
- **Epsilon (ε)** : 0.3 → 0.01 - Taux d'exploration (décroissant)

### Système de Récompenses
```python
rewards = {
    'pizza_found': 200.0,      # Récompense principale
    'pizza_touched': 100.0,    # Toucher la pizza
    'collision': -15.0,        # Pénalité collision
    'step': -0.5,              # Coût par étape
    'wall_touched': -8.0,      # Pénalité mur
    'new_state': 8.0,          # Bonus exploration
    'proximity_bonus': 3.0,    # Bonus proximité
    'efficiency_bonus': 10.0   # Bonus efficacité
}
```

## Environnements Disponibles

1. **maze** : Labyrinthe standard (6x6)
2. **assessment_maze** : Labyrinthe d'évaluation (8x8)
3. **hard_maze** : Labyrinthe difficile (10x10)
4. **line** : Environnement linéaire simple
5. **test** : Environnement de test

## Résultats Typiques

- **Épisodes d'entraînement** : 50
- **Taux de succès** : 100%
- **Meilleur chemin** : 15-25 étapes
- **États appris** : 200-300
- **Temps d'entraînement** : 2-5 minutes

### Évolution de l'Apprentissage
```
Épisode 1:  Échec | 1000 étapes | ε: 0.300
Épisode 10: Succès | 45 étapes  | ε: 0.285
Épisode 20: Succès | 28 étapes  | ε: 0.270
Épisode 30: Succès | 22 étapes  | ε: 0.255
Épisode 40: Succès | 19 étapes  | ε: 0.240
Épisode 50: Succès | 17 étapes  | ε: 0.225
```

## Dépannage

**Import Error**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Pygame Non Installé**
```bash
pip install pygame
```

**Performance Lente**
```python
agent.train_episode(game, turtle, show_gui=False, verbose=False)
```

## Auteur

Développé dans le cadre d'un projet d'apprentissage par renforcement.
Technologies utilisées : Python, Pygame, Q-Learning, Reinforcement Learning.