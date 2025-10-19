# DonatelloPyzza - Agent Q-Learning

[![Python](https://img.shields.io/badge/Python-3.1%2B-blue.svg)](https://python.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.1.2-green.svg)](https://pygame.org)
[![License](https://img.shields.io/badge/License-Free%20for%20non--commercial%20use-yellow.svg)](LICENSE)

## Description

DonatelloPyzza est un environnement d'apprentissage par renforcement éducatif qui implémente l'algorithme Q-Learning pour entraîner un agent à naviguer dans des labyrinthes. Ce projet combine l'apprentissage par renforcement avec une interface graphique interactive, facilitant la compréhension des concepts d'intelligence artificielle.

## Objectifs Pédagogiques

- Comprendre les concepts fondamentaux de l'apprentissage par renforcement
- Implémenter et analyser l'algorithme Q-Learning
- Visualiser le processus d'apprentissage d'un agent intelligent
- Expérimenter avec différents environnements et hyperparamètres
- Analyser les performances et la convergence des algorithmes

## Fonctionnalités

### Agent Q-Learning
- Implémentation de l'équation de Bellman pour la mise à jour des valeurs Q
- Stratégie epsilon-greedy adaptative pour l'équilibre exploration/exploitation
- Système de convergence automatique basé sur la stabilité des performances
- Mécanisme de récompenses sophistiqué pour guider l'apprentissage

### Interface Interactive
- Visualisation en temps réel avec Pygame
- Environnements de complexité variable
- Métriques de performance en direct
- Mode d'entraînement sans interface graphique pour les performances

### Analyse et Monitoring
- Graphiques de performance et évolution de l'apprentissage
- Statistiques détaillées (taux de succès, nombre d'étapes moyennes)
- Export des résultats et métriques
- Tests automatisés pour la validation des performances

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
from examples.qlearning import QLearningAgent

# Configuration de l'agent
agent = QLearningAgent(
    learning_rate=0.1,      # Vitesse d'apprentissage
    discount_factor=0.9,     # Importance des récompenses futures
    epsilon=0.3,            # Taux d'exploration initial
    epsilon_decay=0.995     # Décroissance de l'exploration
)

# Création de l'environnement
game = RLGame("maze", gui=True)  # Interface graphique activée
turtle = game.start()

# Entraînement d'un épisode
reward, steps, success = agent.train_episode(
    game, turtle, 
    show_gui=True,    # Affichage graphique
    verbose=True      # Affichage des détails
)

print(f"Résultat: {'Succès' if success else 'Échec'}")
print(f"Récompense: {reward:.2f}")
print(f"Étapes: {steps}")
```

### Exemples Avancés

```python
# Entraînement complet avec convergence automatique
results = agent.train_until_convergence(
    game, turtle,
    max_episodes=100,
    target_success_rate=0.9,
    patience=10
)

# Test de performance
performance = agent.assess_performance(game, turtle, num_tests=50)
print(f"Taux de succès: {performance['success_rate']:.2%}")
print(f"Étapes moyennes: {performance['avg_steps']:.1f}")
```

## Algorithme Q-Learning

### Principe Mathématique

L'agent apprend une politique optimale en mettant à jour une table Q selon l'équation de Bellman :

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Où :
- **Q(s,a)** : Valeur Q de l'état s et de l'action a
- **α (alpha)** : Taux d'apprentissage (learning rate)
- **r** : Récompense immédiate
- **γ (gamma)** : Facteur d'escompte (discount factor)
- **s'** : État suivant

### Hyperparamètres Recommandés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **Learning Rate (α)** | 0.1 | Vitesse d'apprentissage |
| **Discount Factor (γ)** | 0.9 | Importance des récompenses futures |
| **Epsilon (ε)** | 0.3 → 0.01 | Taux d'exploration (décroissant) |
| **Epsilon Decay** | 0.995 | Vitesse de décroissance de l'exploration |

### Système de Récompenses

```python
REWARD_SYSTEM = {
    'pizza_found': 200.0,      # Objectif principal
    'pizza_touched': 100.0,    # Proximité de l'objectif
    'collision': -15.0,         # Pénalité collision
    'step': -0.5,              # Coût temporel
    'wall_touched': -8.0,      # Pénalité mur
    'new_state': 8.0,          # Bonus exploration
    'proximity_bonus': 3.0,    # Bonus proximité
    'efficiency_bonus': 10.0   # Bonus efficacité
}
```

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
# Agent personnalisé avec paramètres optimisés
agent = QLearningAgent(
    learning_rate=0.15,        # Apprentissage plus rapide
    discount_factor=0.95,      # Plus d'importance aux récompenses futures
    epsilon=0.4,              # Plus d'exploration initiale
    epsilon_decay=0.99,       # Décroissance plus lente
    min_epsilon=0.01,         # Exploration minimale
    max_episodes=200          # Limite d'épisodes
)
```

### Mode Performance

```python
# Entraînement sans interface graphique (plus rapide)
agent.train_episode(game, turtle, show_gui=False, verbose=False)

# Entraînement en lot sur plusieurs environnements
environments = ["maze", "assessment_maze", "hard_maze"]
for env_name in environments:
    game = RLGame(env_name, gui=False)
    turtle = game.start()
    results = agent.train_until_convergence(game, turtle)
    print(f"{env_name}: {results['episodes']} épisodes, {results['success_rate']:.2%} succès")
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
# Ajuster les hyperparamètres
agent = QLearningAgent(
    learning_rate=0.2,        # Augmenter le taux d'apprentissage
    epsilon_decay=0.99,       # Décroissance plus lente
    min_epsilon=0.05          # Exploration minimale plus élevée
)
```

### Logs et Debug

```python
# Activation des logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Affichage des statistiques détaillées
agent.train_episode(game, turtle, verbose=True, debug=True)
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

## Licence

Ce projet est sous licence **Free for non-commercial use**. 

- Utilisation libre pour l'éducation et la recherche
- Modification et distribution autorisées
- Usage commercial non autorisé sans permission

## Auteurs

- **Mickaël Bettinelli** - *Développement principal* - [@MilowB](https://github.com/MilowB)
- **Contributeurs** - Voir [CONTRIBUTORS.md](CONTRIBUTORS.md)

## Remerciements

- **Pygame** pour l'interface graphique
- **Communauté Python** pour les outils et bibliothèques
- **Étudiants et enseignants** pour les retours et suggestions

## Support

- **Issues** : [GitHub Issues](https://github.com/oeoecbien/ai/issues)
- **Discussions** : [GitHub Discussions](https://github.com/oeoecbien/ai/discussions)
- **Email** : mickael.bettinelli@univ-smb.fr

---

**DonatelloPyzza** - *Apprentissage par renforcement éducatif*

[![GitHub stars](https://img.shields.io/github/stars/oeoecbien/ai?style=social)](https://github.com/oeoecbien/ai/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/oeoecbien/ai?style=social)](https://github.com/oeoecbien/ai/network)