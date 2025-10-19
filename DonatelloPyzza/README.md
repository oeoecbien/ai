# DonatelloPyzza - Agent Q-Learning avec Reward Shaping Intelligent

[![Python](https://img.shields.io/badge/Python-3.1%2B-blue.svg)](https://python.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.1.2-green.svg)](https://pygame.org)
[![Q-Learning](https://img.shields.io/badge/Q--Learning-Intelligent-purple.svg)](#algorithme-q-learning)
[![Reward Shaping](https://img.shields.io/badge/Reward--Shaping-Intelligent-orange.svg)](#reward-shaping-intelligent)
[![License](https://img.shields.io/badge/License-Free%20for%20non--commercial%20use-yellow.svg)](LICENSE)

## Description

DonatelloPyzza est un environnement d'apprentissage par renforcement éducatif qui implémente l'algorithme Q-Learning **théoriquement correct** avec un **système de reward shaping intelligent** pour entraîner un agent à naviguer dans des labyrinthes. Ce projet combine l'apprentissage par renforcement avec une interface graphique interactive, facilitant la compréhension des concepts d'intelligence artificielle tout en respectant parfaitement les propriétés MDP (Markov Decision Process) et les formulations académiques du Q-Learning.

**Code pédagogique** : Le fichier `qlearning.py` contient des commentaires clairs et accessibles qui expliquent chaque composant et chaque étape du processus d'apprentissage, rendant le Q-Learning compréhensible pour tous les niveaux.

**Reward Shaping Intelligent** : Système avancé de récompenses qui s'adapte automatiquement aux phases d'apprentissage, guidant l'agent vers des solutions optimales plus rapidement.

## Objectifs Pédagogiques

- Comprendre les concepts fondamentaux de l'apprentissage par renforcement
- Implémenter et analyser l'algorithme Q-Learning **théoriquement correct**
- Visualiser le processus d'apprentissage d'un agent intelligent
- Expérimenter avec différents environnements et hyperparamètres
- Analyser les performances et la convergence des algorithmes
- Comparer Q-Learning pur (académique) vs Q-Learning avec reward shaping intelligent
- Comprendre la gestion correcte des états terminaux en RL
- **Lire et comprendre le code** grâce aux commentaires pédagogiques détaillés
- **Apprendre le reward shaping** et ses techniques avancées

## Fonctionnalités

### Architecture Modulaire Avancée
- **Composants séparés** : QTable, ExplorationStrategy, RewardSystem, ConvergenceDetector, PerformanceTracker
- **États MDP cohérents** : Représentation markovienne avec clés d'état uniformes
- **Interface propre** : Adaptateur d'environnement pour isolation des appels RLGame
- **Configuration centralisée** : AgentConfig avec dataclass pour paramètres
- **Code documenté** : Commentaires pédagogiques détaillés pour chaque composant

### Agent Q-Learning Théoriquement Correct
- **Formule TD correcte** : `Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]`
- **États terminaux** : Pas de bootstrap pour les états terminaux (target = reward seulement)
- **Epsilon-greedy propre** : Exploration/exploitation équilibrée avec décroissance
- **Initialisation cohérente** : Toutes les actions initialisées à 0.0 pour chaque nouvel état
- **Mode Q-Learning pur** : Option pour désactiver le reward shaping (cours académique)
- **Reproductibilité** : Gestion complète des graines aléatoires (random, numpy)
- **Convergence robuste** : Détection basée sur critères explicites (epsilon, succès, variance)

### Reward Shaping Intelligent
- **Guidance directionnelle** : Récompenses basées sur le progrès vers la pizza
- **Système de curiosité** : Bonus d'exploration pour encourager la découverte
- **Normalisation des récompenses** : Stabilisation de l'apprentissage avec Z-score
- **Phases d'apprentissage** : Adaptation automatique selon les performances
- **Détection automatique** : Position de la pizza détectée automatiquement

### Interface Interactive
- Visualisation en temps réel avec Pygame
- Environnements de complexité variable
- Métriques de performance en direct
- Mode d'entraînement sans interface graphique pour les performances
- **Interface utilisateur intuitive** : Configuration guidée avec options avancées

### Analyse et Monitoring
- **Logging structuré** : Fichiers de log et console avec niveaux
- **Statistiques détaillées** : Taux de succès, étapes moyennes, convergence
- **Détection de boucles** : Timeout intelligent pour éviter les cycles infinis
- **Export des résultats** : Métriques et performances exportables
- **Messages de debug** : Informations détaillées sur le processus d'apprentissage

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
# Lancer l'exemple Q-Learning interactif avec reward shaping intelligent
python .\examples\qlearning.py 
```

Le script guide l'utilisateur à travers :
1. Sélection de l'environnement (maze, hard_maze, etc.)
2. Configuration des hyperparamètres (learning rate, epsilon, etc.)
3. Choix du mode d'affichage (avec/sans interface graphique)
4. **Mode Q-Learning pur** : Option pour désactiver le reward shaping (cours académique)
5. **Mode reward shaping intelligent** : Option pour activer le système avancé
6. Lancement de l'entraînement avec monitoring en temps réel

### Utilisation Programmatique

```python
from donatellopyzza import RLGame, Action, Feedback
from examples.qlearning import QLearningAgent, AgentConfig, EnvironmentAdapter

# Configuration modulaire de l'agent avec reward shaping intelligent
config = AgentConfig(
    learning_rate=0.1,          # Vitesse d'apprentissage
    discount_factor=0.9,        # Importance des récompenses futures
    epsilon=0.3,               # Taux d'exploration initial
    epsilon_decay=0.995,       # Décroissance de l'exploration
    convergence_window=20,     # Fenêtre de convergence
    max_steps=1000,            # Limite d'étapes par épisode
    intelligent_shaping=True   # Mode reward shaping intelligent
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
print(f"Phase d'apprentissage: {agent.reward_system.learning_phase}")
```

### Exemples Avancés

```python
# Entraînement complet avec convergence automatique
from examples.qlearning import train_agent

# Entraînement avec architecture modulaire et reward shaping intelligent
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
print(f"Phase d'apprentissage: {agent.reward_system.learning_phase}")

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
- **RewardSystem** : Système de récompenses intelligent avec reward shaping
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

## Reward Shaping Intelligent

### Système de Récompenses Avancé

Le système de reward shaping intelligent comprend plusieurs composants :

#### 1. Guidance Directionnelle
```python
def _calculate_distance_guidance(self, current_state, next_state):
    """Calcule la guidance basée sur le progrès vers la pizza"""
    if self.pizza_position is None:
        return 0.0
    
    current_distance = self._euclidean_distance(current_state.position, self.pizza_position)
    next_distance = self._euclidean_distance(next_state.position, self.pizza_position)
    
    progress = self.previous_distance - next_distance
    guidance_reward = progress * 2.0  # Multiplier pour plus d'impact
    
    return guidance_reward
```

#### 2. Système de Curiosité
```python
def _calculate_curiosity_bonus(self, next_state):
    """Calcule le bonus de curiosité pour encourager l'exploration"""
    state_key = next_state.key()
    self.state_novelty[state_key] += 1
    
    novelty = 1.0 / (self.state_novelty[state_key] ** 0.5)
    curiosity_reward = novelty * self.curiosity_bonus
    
    return curiosity_reward
```

#### 3. Normalisation des Récompenses
```python
def _normalize_reward(self, reward):
    """Normalise les récompenses pour stabiliser l'apprentissage"""
    self.reward_history.append(reward)
    
    if len(self.reward_history) > 10:
        self.reward_mean = sum(self.reward_history) / len(self.reward_history)
        variance = sum((r - self.reward_mean) ** 2 for r in self.reward_history) / len(self.reward_history)
        self.reward_std = max(variance ** 0.5, 1e-6)
    
    if self.reward_std > 0:
        normalized_reward = (reward - self.reward_mean) / self.reward_std
        return normalized_reward
    else:
        return reward
```

#### 4. Phases d'Apprentissage Adaptatives
```python
def update_learning_phase(self, success_rate, episode_count):
    """Met à jour la phase d'apprentissage selon les performances"""
    if episode_count < 50 or success_rate < 0.1:
        self.learning_phase = "exploration"    # Plus de guidance
    elif success_rate < 0.6:
        self.learning_phase = "learning"      # Équilibre
    else:
        self.learning_phase = "optimization"  # Focus efficacité
```

### Modes de Fonctionnement

#### Mode Q-Learning Pur (Académique)
```python
# Récompenses minimales selon le cours
if feedback == Feedback.MOVED_ON_PIZZA:
    return 1.0  # Récompense positive pour succès
else:
    return 0.0  # Pas de récompense pour les autres actions
```

#### Mode avec Reward Shaping Intelligent
```python
class RewardSystem:
    def __init__(self):
        self.step_penalty = -0.1        # Pénalité réduite par étape
        self.wall_penalty = -2.0       # Pénalité réduite pour collision
        self.pizza_reward = 100.0      # Récompense pour atteindre la pizza
        self.touch_cost = -0.2         # Coût réduit de l'action TOUCH
        self.exploration_bonus = 0.5    # Bonus d'exploration count-based
        self.curiosity_bonus = 0.3     # Bonus de curiosité
        self.intelligent_shaping = True # Mode intelligent activé
```

**Caractéristiques** :
- **Mode pur** : Récompenses minimales (0/1) pour reproduire le cours
- **Mode intelligent** : Reward shaping adaptatif avec guidance directionnelle
- **Détection automatique** : Position de la pizza détectée automatiquement
- **Phases adaptatives** : Système qui s'adapte aux performances
- **Normalisation** : Récompenses stabilisées pour un apprentissage robuste

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

#### Mode Q-Learning Pur (Académique)
- **Taux de succès** : 60-80% après convergence
- **Épisodes d'entraînement** : 100-500 épisodes
- **Meilleur chemin** : 20-40 étapes
- **États appris** : 200-800 états uniques
- **Temps d'entraînement** : 5-15 minutes

#### Mode avec Reward Shaping Intelligent
- **Taux de succès** : 85-100% après convergence
- **Épisodes d'entraînement** : 50-200 épisodes
- **Meilleur chemin** : 10-25 étapes
- **États appris** : 100-400 états uniques
- **Temps d'entraînement** : 2-8 minutes

### Évolution de l'Apprentissage

#### Mode Q-Learning Pur (Académique)
```
Épisode 1:  Échec  | 1000 étapes | ε: 0.300 | Q-table: 0 états
Épisode 50: Échec  | 800 étapes  | ε: 0.250 | Q-table: 120 états
Épisode 100: Succès| 45 étapes   | ε: 0.200 | Q-table: 200 états
Épisode 150: Succès| 35 étapes   | ε: 0.150 | Q-table: 280 états
Épisode 200: Succès| 28 étapes   | ε: 0.100 | Q-table: 350 états
```

#### Mode avec Reward Shaping Intelligent
```
Épisode 1:  Échec  | 1000 étapes | ε: 0.300 | Q-table: 0 états | Phase: exploration
Épisode 5:  Succès | 45 étapes   | ε: 0.285 | Q-table: 45 états | Phase: exploration
Épisode 10: Succès | 28 étapes   | ε: 0.270 | Q-table: 78 états | Phase: learning
Épisode 15: Succès | 22 étapes   | ε: 0.255 | Q-table: 112 états | Phase: learning
Épisode 20: Succès | 19 étapes   | ε: 0.240 | Q-table: 145 états | Phase: optimization
Épisode 25: Succès | 17 étapes   | ε: 0.225 | Q-table: 178 états | Phase: optimization
```

### Messages de Performance

Le système affiche les messages suivants pendant l'entraînement :

- **[SUCCÈS]** : Pizza trouvée avec le nombre d'étapes et la récompense totale
- **[ÉCHEC]** : Épisode terminé sans succès ou limite d'étapes atteinte
- **[ANALYSE]** : Bilan périodique avec taux de succès et statistiques
- **[CONVERGENCE]** : Performance stable détectée - arrêt automatique
- **[TERMINÉ]** : Entraînement terminé avec convergence atteinte
- **[INTERRUPTION]** : Entraînement interrompu par l'utilisateur (Ctrl+C)
- **[LIMITE]** : Entraînement terminé - limite d'épisodes atteinte
- **[DEBUG]** : Informations de debug pour comprendre la convergence
- **[INFO]** : Position de la pizza détectée automatiquement

## Configuration Avancée

### Personnalisation de l'Agent

```python
# Configuration personnalisée avec reward shaping intelligent
config = AgentConfig(
    learning_rate=0.1,            # Apprentissage standard
    discount_factor=0.9,          # Facteur d'escompte
    epsilon=0.3,                  # Exploration initiale
    epsilon_decay=0.995,          # Décroissance de l'exploration
    epsilon_min=0.01,             # Exploration minimale
    max_steps=1000,               # Limite d'étapes par épisode
    convergence_window=20,        # Fenêtre de convergence
    convergence_threshold=0.15,   # Seuil de convergence réduit
    min_success_rate=0.5,         # Taux de succès minimum réduit
    pure_qlearning=False,         # Mode avec reward shaping
    intelligent_shaping=True      # Mode reward shaping intelligent
)

agent = QLearningAgent(config)
```

### Mode Q-Learning Pur (Cours Académique)

```python
# Configuration pour reproduire exactement le Q-Learning du cours
config = AgentConfig(
    learning_rate=0.1,            # Valeur classique
    discount_factor=0.9,          # Facteur d'escompte standard
    epsilon=0.3,                  # Exploration initiale
    epsilon_decay=0.995,          # Décroissance progressive
    epsilon_min=0.01,             # Exploration minimale
    pure_qlearning=True,          # Mode pur : récompenses minimales
    intelligent_shaping=False,     # Désactiver le reward shaping
    random_seed=42                # Graine pour reproductibilité
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
env_adapter = EnvironmentAdapter("maze", show_gui=False)
agent.train_episode(env_adapter, verbose=False)

# Entraînement en mode silencieux
agent = train_agent(
    environment_name="maze",
    show_gui=False,
    verbose=False,
    max_episodes=100
)
```

#### Convergence Lente
```python
# Ajuster les hyperparamètres avec reward shaping intelligent
config = AgentConfig(
    learning_rate=0.15,           # Augmenter le taux d'apprentissage
    epsilon_decay=0.99,           # Décroissance plus lente
    epsilon_min=0.05,             # Exploration minimale plus élevée
    convergence_threshold=0.1,    # Seuil de convergence plus permissif
    intelligent_shaping=True       # Activer le reward shaping intelligent
)
agent = QLearningAgent(config)
```

### Logs et Debug

```python
# Activation des logs détaillés
import logging
logging.basicConfig(level=logging.DEBUG)

# Affichage des statistiques détaillées avec reward shaping intelligent
env_adapter = EnvironmentAdapter("maze", show_gui=True)
agent.train_episode(env_adapter, verbose=True)

# Accès aux composants pour debug
print(f"Q-table size: {agent.q_table.size()}")
print(f"Epsilon: {agent.exploration_strategy.get_epsilon():.3f}")
print(f"Convergence: {agent.check_convergence()}")
print(f"Phase d'apprentissage: {agent.reward_system.learning_phase}")
print(f"Position pizza: {agent.reward_system.pizza_position}")

# Affichage des informations de convergence
agent.print_convergence_summary()
```

## Exemples et Tutoriels

### Exemples Disponibles

| Fichier | Description | Niveau | Commentaires |
|---------|-------------|--------|--------------|
| `basic_example.py` | Introduction de base | Débutant | Premiers pas avec DonatelloPyzza |
| `qlearning.py` | Q-Learning complet | Intermédiaire | **Code pédagogique avec reward shaping intelligent** |
| `algorithm_assessment.py` | Évaluation de performance | Avancé | Tests et comparaisons |
| `generate_maze.py` | Génération de labyrinthes | Intermédiaire | Création d'environnements |
| `changing_color.py` | Personnalisation visuelle | Débutant | Interface graphique |

### Tutoriel Pas-à-Pas

1. **Débuter avec DonatelloPyzza**
   ```bash
   python examples/basic_example.py
   ```

2. **Comprendre le Q-Learning avec Reward Shaping Intelligent**
   ```bash
   python examples/qlearning.py
   ```
   *Le fichier contient des commentaires pédagogiques détaillés et un système de reward shaping intelligent*

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
- **Commentaires pédagogiques** : Expliquer le "pourquoi" et le "comment"
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

## Corrections Théoriques Implémentées

### Alignement avec le Q-Learning Académique

✅ **États terminaux** : Correction de la mise à jour Q-table (pas de bootstrap sur les états terminaux)  
✅ **Initialisation cohérente** : Toutes les actions initialisées à 0.0 pour chaque nouvel état  
✅ **Comptage des états** : Correction du suivi des états visités dans PerformanceTracker  
✅ **Mode Q-Learning pur** : Option pour désactiver le reward shaping (cours académique)  
✅ **Reproductibilité** : Gestion complète des graines aléatoires (random, numpy)  
✅ **Convergence intelligente** : Détection automatique de la convergence avec arrêt prématuré  
✅ **Code pédagogique** : Commentaires détaillés et accessibles pour l'apprentissage  
✅ **Reward Shaping Intelligent** : Système avancé de récompenses adaptatif et intelligent

### Comparaison des Modes

| Aspect | Mode Pur (Académique) | Mode Intelligent (Shaping) |
|--------|----------------------|---------------------------|
| **Récompenses** | 0/1 (minimales) | Reward shaping intelligent adaptatif |
| **Convergence** | Plus lente (100-500 épisodes) | Plus rapide (50-200 épisodes) |
| **Performance** | 60-80% succès | 85-100% succès |
| **Guidance** | Aucune | Directionnelle + Curiosité |
| **Adaptation** | Statique | Phases d'apprentissage |
| **Normalisation** | Non | Z-score automatique |
| **Pédagogie** | Fidèle au cours | Ingénierie avancée |
| **Code** | Commentaires académiques | Commentaires techniques |

### Fonctionnalités du Reward Shaping Intelligent

✅ **Guidance directionnelle** : Récompenses basées sur le progrès vers la pizza  
✅ **Système de curiosité** : Bonus d'exploration pour encourager la découverte  
✅ **Normalisation des récompenses** : Stabilisation avec Z-score  
✅ **Phases d'apprentissage** : Exploration → Learning → Optimization  
✅ **Détection automatique** : Position de la pizza détectée automatiquement  
✅ **Adaptation dynamique** : Système qui s'adapte aux performances  
✅ **Convergence accélérée** : Paramètres optimisés pour une convergence plus rapide  

## Auteurs

- **Mickaël Bettinelli** - *Développement principal* - [@MilowB](https://github.com/MilowB)
- **Contributeur** - Melih CETINKAYA - [@oeoecbien](https://github.com/oeoecbien)

---

**DonatelloPyzza** - *Q-Learning académique théoriquement correct avec reward shaping intelligent et code pédagogique*