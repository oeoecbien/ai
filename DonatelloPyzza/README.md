# DonatelloPyzza - Agent Q-Learning Modulaire

[![Python](https://img.shields.io/badge/Python-3.1%2B-blue.svg)](https://python.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.1.2-green.svg)](https://pygame.org)
[![License](https://img.shields.io/badge/License-Free%20for%20non--commercial%20use-yellow.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Modulaire-purple.svg)](#architecture-modulaire)

## Description

DonatelloPyzza est un environnement d'apprentissage par renforcement éducatif qui implémente l'algorithme Q-Learning avec une **architecture modulaire avancée** pour entraîner un agent à naviguer dans des labyrinthes. Ce projet combine l'apprentissage par renforcement avec une interface graphique interactive, facilitant la compréhension des concepts d'intelligence artificielle tout en respectant les propriétés MDP (Markov Decision Process).

## Objectifs Pédagogiques

- Comprendre les concepts fondamentaux de l'apprentissage par renforcement
- Implémenter et analyser l'algorithme Q-Learning
- Visualiser le processus d'apprentissage d'un agent intelligent
- Expérimenter avec différents environnements et hyperparamètres
- Analyser les performances et la convergence des algorithmes

## Fonctionnalités

### Architecture Modulaire Avancée
- **Composants séparés** : QTable, ExplorationStrategy, RewardSystem, ConvergenceDetector, PerformanceTracker
- **États MDP cohérents** : Représentation markovienne avec clés d'état uniformes
- **Interface propre** : Adaptateur d'environnement pour isolation des appels RLGame
- **Configuration centralisée** : AgentConfig avec dataclass pour paramètres

### Agent Q-Learning Théoriquement Correct
- **Formule TD correcte** : `Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]`
- **Epsilon-greedy propre** : Exploration/exploitation équilibrée avec décroissance
- **Récompenses markoviennes** : Système de récompenses avec bonus d'exploration count-based
- **Convergence robuste** : Détection basée sur critères explicites (epsilon, succès, variance)

### Interface Interactive
- Visualisation en temps réel avec Pygame
- Environnements de complexité variable
- Métriques de performance en direct
- Mode d'entraînement sans interface graphique pour les performances

### Analyse et Monitoring
- **Logging structuré** : Fichiers de log et console avec niveaux
- **Statistiques détaillées** : Taux de succès, étapes moyennes, convergence
- **Détection de boucles** : Timeout intelligent pour éviter les cycles infinis
- **Export des résultats** : Métriques et performances exportables

## Installation

### Prérequis
- Python 3.1+ (recommandé : Python 3.8+)
- pip (gestionnaire de paquets Python)

### Installation

```bash
# 1. Cloner le repository
git clone https://github.com/oeoecbien/ai.git
cd ai/DonatelloPyzza

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv

# 3. Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur Linux/Mac :
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Vérifier l'installation
python -c "import donatellopyzza; print('Installation réussie')"
```

## Utilisation

### Démarrage Rapide

```bash
# Lancer l'exemple Q-Learning interactif
python .\examples\qlearning.py 
```

Le script guide l'utilisateur à travers :
1. Sélection de l'environnement (maze, hard_maze, etc.)
2. Configuration des hyperparamètres (learning rate, epsilon, etc.)
3. Choix du mode d'affichage (avec/sans interface graphique)
4. Lancement de l'entraînement avec monitoring en temps réel

### Utilisation Programmatique

```python
from donatellopyzza import RLGame, Action, Feedback
from examples.qlearning import QLearningAgent, AgentConfig, EnvironmentAdapter

# Configuration modulaire de l'agent
config = AgentConfig(
    learning_rate=0.1,      # Vitesse d'apprentissage
    discount_factor=0.9,    # Importance des récompenses futures
    epsilon=0.3,           # Taux d'exploration initial
    epsilon_decay=0.995,    # Décroissance de l'exploration
    convergence_window=20,  # Fenêtre de convergence
    max_steps=1000         # Limite d'étapes par épisode
)

# Création de l'agent avec architecture modulaire
agent = QLearningAgent(config)

# Création de l'adaptateur d'environnement
env_adapter = EnvironmentAdapter("maze", show_gui=True)

# Entraînement d'un épisode avec composants modulaires
reward, steps, success = agent.train_episode(env_adapter, verbose=True)

print(f"Résultat: {'Succès' if success else 'Échec'}")
print(f"Récompense: {reward:.2f}")
print(f"Étapes: {steps}")

# Accès aux composants modulaires
print(f"Taille Q-table: {agent.q_table.size()}")
print(f"Epsilon actuel: {agent.exploration_strategy.get_epsilon():.3f}")
print(f"Convergence: {agent.check_convergence()}")
```

### Exemples Avancés

```python
# Entraînement complet avec convergence automatique
from examples.qlearning import train_agent

# Entraînement avec architecture modulaire
agent = train_agent(
    environment_name="assessment_maze",
    show_gui=True,
    verbose=True,
    max_episodes=200,
    config=config
)

# Accès aux statistiques détaillées
stats = agent.get_statistics()
convergence_info = agent.get_convergence_info()

print(f"Épisodes: {stats['episode_count']}")
print(f"Meilleur chemin: {stats['best_steps']} étapes")
print(f"Q-table: {stats['q_table_size']} états")
print(f"Convergence: {convergence_info['converged']}")

# Test de performance sur plusieurs environnements
environments = ["maze", "assessment_maze", "hard_maze"]
for env_name in environments:
    env_adapter = EnvironmentAdapter(env_name, show_gui=False)
    reward, steps, success = agent.train_episode(env_adapter, verbose=False)
    print(f"{env_name}: {'Succès' if success else 'Échec'} en {steps} étapes")
```

## Algorithme Q-Learning

### Principe Mathématique

L'agent apprend une politique optimale en mettant à jour une table Q selon l'équation de Bellman :

```
Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]
```

Cette implémentation utilise la forme standard de l'équation de Bellman où :
- **Q(s,a)** : Valeur Q de l'état s et de l'action a
- **α (alpha)** : Taux d'apprentissage (learning rate)
- **r** : Récompense immédiate
- **γ (gamma)** : Facteur d'escompte (discount factor)
- **s'** : État suivant

### Architecture Modulaire

Le système est organisé en composants spécialisés :

- **QTable** : Gestion de la table Q avec clés d'état cohérentes
- **ExplorationStrategy** : Stratégie epsilon-greedy avec décroissance
- **RewardSystem** : Système de récompenses markovien avec bonus d'exploration
- **ConvergenceDetector** : Détection de convergence basée sur critères statistiques
- **PerformanceTracker** : Suivi des performances et statistiques
- **EnvironmentAdapter** : Interface d'abstraction pour l'environnement RLGame

### Hyperparamètres Recommandés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **Learning Rate (α)** | 0.1 | Vitesse d'apprentissage |
| **Discount Factor (γ)** | 0.9 | Importance des récompenses futures |
| **Epsilon (ε)** | 0.3 → 0.01 | Taux d'exploration (décroissant) |
| **Epsilon Decay** | 0.995 | Vitesse de décroissance de l'exploration |

### Système de Récompenses Markovien

Le système de récompenses respecte les propriétés MDP avec des récompenses markoviennes :

```python
class RewardSystem:
    def __init__(self):
        self.step_penalty = -1.0        # Coût temporel par étape
        self.wall_penalty = -5.0        # Pénalité pour collision/mur
        self.pizza_reward = 100.0       # Récompense pour atteindre la pizza
        self.touch_cost = -0.5          # Coût de l'action TOUCH
        self.exploration_bonus = 0.5    # Bonus d'exploration count-based
```

**Caractéristiques** :
- Récompenses basées uniquement sur l'état actuel et l'action
- Bonus d'exploration count-based : `bonus = β / √N(s)`
- Pas de dépendance à l'historique des actions
- Respect des propriétés markoviennes pour garantir la convergence

## Environnements Disponibles

| Environnement | Taille | Difficulté | Description |
|---------------|--------|------------|-------------|
| **maze** | 6×6 | Facile | Labyrinthe standard pour débuter |
| **assessment_maze** | 8×8 | Moyen | Labyrinthe d'évaluation |
| **hard_maze** | 10×10 | Difficile | Labyrinthe complexe |
| **line** | 1×N | Très facile | Environnement linéaire simple |
| **test** | Variable | Variable | Environnement de test personnalisé |

### Créer un Environnement Personnalisé

```python
from donatellopyzza import MazeGenerator

# Génération d'un labyrinthe aléatoire
generator = MazeGenerator(width=8, height=8)
maze_data = generator.generate_maze()

# Sauvegarde pour utilisation future
generator.save_maze("mon_labyrinthe.txt", maze_data)
```

## Résultats et Performances

### Métriques Typiques

- **Taux de succès** : 95-100% après convergence
- **Épisodes d'entraînement** : 30-50 épisodes
- **Meilleur chemin** : 15-25 étapes
- **États appris** : 200-400 états uniques
- **Temps d'entraînement** : 2-5 minutes

### Évolution de l'Apprentissage

```
Épisode 1:  Échec  | 1000 étapes | ε: 0.300 | Q-table: 0 états
Épisode 10: Succès | 45 étapes   | ε: 0.285 | Q-table: 45 états
Épisode 20: Succès | 28 étapes   | ε: 0.270 | Q-table: 78 états
Épisode 30: Succès | 22 étapes   | ε: 0.255 | Q-table: 112 états
Épisode 40: Succès | 19 étapes   | ε: 0.240 | Q-table: 145 états
Épisode 50: Succès | 17 étapes   | ε: 0.225 | Q-table: 178 états
```

## Configuration Avancée

### Personnalisation de l'Agent

```python
# Configuration personnalisée avec architecture modulaire
config = AgentConfig(
    learning_rate=0.15,        # Apprentissage plus rapide
    discount_factor=0.95,     # Plus d'importance aux récompenses futures
    epsilon=0.4,              # Plus d'exploration initiale
    epsilon_decay=0.99,       # Décroissance plus lente
    epsilon_min=0.01,         # Exploration minimale
    max_steps=200,            # Limite d'étapes par épisode
    convergence_window=15,    # Fenêtre de convergence réduite
    convergence_threshold=0.03 # Seuil de convergence plus strict
)

agent = QLearningAgent(config)
```

### Mode Performance

```python
# Entraînement sans interface graphique (plus rapide)
env_adapter = EnvironmentAdapter("maze", show_gui=False)
agent.train_episode(env_adapter, verbose=False)

# Entraînement en lot sur plusieurs environnements
environments = ["maze", "assessment_maze", "hard_maze"]
for env_name in environments:
    env_adapter = EnvironmentAdapter(env_name, show_gui=False)
    reward, steps, success = agent.train_episode(env_adapter, verbose=False)
    print(f"{env_name}: {'Succès' if success else 'Échec'} en {steps} étapes")
```

## Dépannage

### Problèmes Courants

#### Erreur d'Import
```bash
# Solution : Ajouter le répertoire au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Ou sur Windows :
set PYTHONPATH=%PYTHONPATH%;%CD%
```

#### Pygame Non Installé
```bash
# Installation manuelle de Pygame
pip install pygame==2.1.2
# Ou mise à jour complète
pip install --upgrade -r requirements.txt
```

#### Performance Lente
```python
# Désactiver l'interface graphique
agent.train_episode(game, turtle, show_gui=False, verbose=False)

# Réduire la fréquence d'affichage
agent.train_episode(game, turtle, show_gui=True, display_frequency=10)
```

#### Convergence Lente
```python
# Ajuster les hyperparamètres avec architecture modulaire
config = AgentConfig(
    learning_rate=0.2,        # Augmenter le taux d'apprentissage
    epsilon_decay=0.99,       # Décroissance plus lente
    epsilon_min=0.05,         # Exploration minimale plus élevée
    convergence_threshold=0.1  # Seuil de convergence plus permissif
)
agent = QLearningAgent(config)
```

### Logs et Debug

```python
# Activation des logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Affichage des statistiques détaillées avec architecture modulaire
env_adapter = EnvironmentAdapter("maze", show_gui=True)
agent.train_episode(env_adapter, verbose=True)

# Accès aux composants pour debug
print(f"Q-table size: {agent.q_table.size()}")
print(f"Epsilon: {agent.exploration_strategy.get_epsilon():.3f}")
print(f"Convergence: {agent.check_convergence()}")
```

## Exemples et Tutoriels

### Exemples Disponibles

| Fichier | Description | Niveau |
|---------|-------------|--------|
| `basic_example.py` | Introduction de base | Débutant |
| `qlearning.py` | Q-Learning complet | Intermédiaire |
| `algorithm_assessment.py` | Évaluation de performance | Avancé |
| `generate_maze.py` | Génération de labyrinthes | Intermédiaire |
| `changing_color.py` | Personnalisation visuelle | Débutant |

### Tutoriel Pas-à-Pas

1. **Débuter avec DonatelloPyzza**
   ```bash
   python examples/basic_example.py
   ```

2. **Comprendre le Q-Learning**
   ```bash
   python examples/qlearning.py
   ```

3. **Évaluer les performances**
   ```bash
   python examples/algorithm_assessment.py
   ```

4. **Créer des labyrinthes personnalisés**
   ```bash
   python examples/generate_maze.py
   ```

## Contribution

### Comment Contribuer

1. **Fork** le repository
2. **Créer** une branche feature (`git checkout -b feature/amazing-feature`)
3. **Commit** vos changements (`git commit -m 'Add amazing feature'`)
4. **Push** vers la branche (`git push origin feature/amazing-feature`)
5. **Ouvrir** une Pull Request

### Guidelines de Contribution

- Code en français (commentaires et documentation)
- Tests unitaires pour les nouvelles fonctionnalités
- Documentation mise à jour
- Respect du style de code existant
- Architecture modulaire maintenue
- Respect des propriétés MDP pour les algorithmes RL

## Licence

Ce projet est sous licence **Free for non-commercial use**. 

- Utilisation libre pour l'éducation et la recherche
- Modification et distribution autorisées
- Usage commercial non autorisé sans permission

## Auteurs

- **Mickaël Bettinelli** - *Développement principal* - [@MilowB](https://github.com/MilowB)
- **Contributeur** - Melih CETINKAYA - [@oeoecbien](https://github.com/oeoecbien)

---

**DonatelloPyzza** - *Apprentissage par renforcement éducatif avec architecture modulaire*