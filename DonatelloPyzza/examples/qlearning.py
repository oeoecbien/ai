"""
Agent Q-Learning pour DonatelloPyzza

Ce fichier implémente un agent Q-Learning complet pour apprendre à naviguer dans des labyrinthes.
L'agent utilise l'apprentissage par renforcement pour découvrir le chemin optimal vers la pizza.

Composants principaux :
- AgentConfig : Configuration des hyperparamètres
- State : Représentation de l'état de la tortue
- RewardSystem : Calcul des récompenses (mode pur ou avancé)
- QTable : Table d'apprentissage Q(s,a)
- ExplorationStrategy : Équilibre exploration/exploitation
- ConvergenceDetector : Détection de la convergence
- PerformanceTracker : Suivi des performances
- EnvironmentAdapter : Interface avec DonatelloPyzza
- QLearningAgent : Agent principal qui coordonne tout

L'agent peut fonctionner en mode "pur" (académique) ou "avancé" (avec reward shaping).
"""

import sys
import os
import random
import signal
from typing import Dict, Tuple, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Configuration du chemin d'accès au module parent
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from donatellopyzza import RLGame, Action, Feedback

# Variable globale pour gérer l'interruption clavier
interrupted = False

def signal_handler(signum, frame):
    """Gestionnaire pour l'interruption clavier (Ctrl+C)"""
    global interrupted
    print("\n\nInterruption détectée (Ctrl+C)")
    print("Arrêt de l'entraînement...")
    interrupted = True



@dataclass
class AgentConfig:
    """Configuration de l'agent Q-Learning avec reward shaping intelligent"""
    learning_rate: float = 0.1          # Vitesse d'apprentissage (0.1 = 10%)
    discount_factor: float = 0.9        # Importance des récompenses futures (90%)
    epsilon: float = 0.3                # Taux d'exploration initial (30%)
    epsilon_decay: float = 0.995        # Réduction de l'exploration par épisode
    epsilon_min: float = 0.01           # Exploration minimale (1%)
    convergence_window: int = 20         # Nombre d'épisodes pour détecter la convergence
    convergence_threshold: float = 0.15  # Seuil de stabilité réduit pour convergence plus rapide
    max_steps: int = 1000               # Limite d'étapes par épisode
    random_seed: Optional[int] = None   # Graine pour reproductibilité
    min_success_rate: float = 0.5       # Taux de succès minimum réduit (50%)
    q_value_min: float = -1000.0        # Valeur Q minimum
    q_value_max: float = 1000.0          # Valeur Q maximum
    pure_qlearning: bool = False         # Mode Q-Learning pur (cours académique)
    intelligent_shaping: bool = True     # Mode reward shaping intelligent


StateKey = Tuple[Tuple[int, int], int, bool]

@dataclass(frozen=True)
class State:
    """Représente l'état de la tortue dans le labyrinthe"""
    position: Tuple[int, int]           # Position (x, y) de la tortue
    orientation: int                    # Direction (0=Nord, 1=Est, 2=Sud, 3=Ouest)
    has_touched_wall: bool = False     # A-t-elle touché un mur récemment ?
    steps_since_pizza: int = 0         # Nombre d'étapes depuis la dernière pizza
    previous_actions: Tuple[int, ...] = field(default_factory=tuple, compare=False, hash=False)
    
    def key(self) -> StateKey:
        """Identifiant unique de l'état pour la table Q"""
        return (self.position, self.orientation, self.has_touched_wall)


class RewardSystem:
    """Système de récompenses intelligent pour guider l'apprentissage"""
    def __init__(self, step_penalty: float = -0.1, wall_penalty: float = -2.0, 
                 pizza_reward: float = 100.0, touch_cost: float = -0.2, 
                 exploration_bonus: float = 0.5, pure_qlearning: bool = False,
                 intelligent_shaping: bool = True):
        self.step_penalty = step_penalty        # Pénalité réduite pour chaque étape
        self.wall_penalty = wall_penalty        # Pénalité réduite pour toucher un mur
        self.pizza_reward = pizza_reward        # Récompense pour trouver la pizza
        self.touch_cost = touch_cost            # Coût réduit de l'action "toucher"
        self.exploration_bonus = exploration_bonus  # Bonus pour explorer de nouveaux endroits
        self.pure_qlearning = pure_qlearning   # Mode académique (récompenses simples)
        self.intelligent_shaping = intelligent_shaping  # Mode reward shaping intelligent
        self.visit_counts = defaultdict(int)    # Compteur de visites par état
        
        # Système de normalisation des récompenses
        self.reward_history = deque(maxlen=100)  # Historique des récompenses
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Système de curiosité
        self.state_novelty = defaultdict(int)   # Nouveauté des états
        self.curiosity_bonus = 0.3              # Bonus de curiosité
        
        # Système de guidance directionnelle
        self.previous_distance = None           # Distance précédente à la pizza
        self.pizza_position = None              # Position de la pizza (si connue)
        
        # Système de reward shaping progressif
        self.episode_count = 0                  # Compteur d'épisodes
        self.learning_phase = "exploration"     # Phase d'apprentissage
    
    def calculate_reward(self, feedback: Feedback, current_state: State, 
                        next_state: State, action: Action) -> float:
        """Calcule la récompense selon le mode choisi avec reward shaping intelligent"""
        if self.pure_qlearning:
            # Mode académique : récompenses simples (0 ou 1)
            if feedback == Feedback.MOVED_ON_PIZZA or feedback == Feedback.TOUCHED_PIZZA:
                return 1.0  # Récompense pour trouver la pizza
            else:
                return 0.0  # Pas de récompense pour les autres actions
        else:
            # Mode avancé avec reward shaping intelligent
            reward = self._calculate_base_reward(feedback, action)
            
            if self.intelligent_shaping:
                # Guidance directionnelle basée sur la distance
                reward += self._calculate_distance_guidance(current_state, next_state)
                
                # Bonus de curiosité pour l'exploration
                reward += self._calculate_curiosity_bonus(next_state)
                
                # Reward shaping progressif selon la phase
                reward = self._apply_progressive_shaping(reward)
            
            # Normalisation des récompenses
            reward = self._normalize_reward(reward)
            
            return reward
    
    def _calculate_base_reward(self, feedback: Feedback, action: Action) -> float:
        """Calcule la récompense de base"""
        reward = self.step_penalty  # Pénalité réduite pour chaque étape
        
        if feedback == Feedback.COLLISION or feedback == Feedback.TOUCHED_WALL:
            reward += self.wall_penalty  # Pénalité réduite pour toucher un mur
        elif feedback == Feedback.MOVED_ON_PIZZA or feedback == Feedback.TOUCHED_PIZZA:
            reward += self.pizza_reward  # Grosse récompense pour la pizza
        
        if action == Action.TOUCH:
            reward += self.touch_cost  # Coût réduit de l'action "toucher"
        
        return reward
    
    def _calculate_distance_guidance(self, current_state: State, next_state: State) -> float:
        """Calcule la guidance directionnelle basée sur la distance à la pizza"""
        if self.pizza_position is None:
            return 0.0  # Pas de guidance si position pizza inconnue
        
        # Calculer les distances
        current_distance = self._euclidean_distance(current_state.position, self.pizza_position)
        next_distance = self._euclidean_distance(next_state.position, self.pizza_position)
        
        # Guidance basée sur le progrès
        if self.previous_distance is not None:
            progress = self.previous_distance - next_distance
            guidance_reward = progress * 2.0  # Multiplier pour plus d'impact
        else:
            guidance_reward = 0.0
        
        # Mettre à jour la distance précédente
        self.previous_distance = current_distance
        
        return guidance_reward
    
    def _calculate_curiosity_bonus(self, next_state: State) -> float:
        """Calcule le bonus de curiosité pour encourager l'exploration"""
        state_key = next_state.key()
        self.state_novelty[state_key] += 1
        
        # Bonus de curiosité inversement proportionnel à la fréquence de visite
        novelty = 1.0 / (self.state_novelty[state_key] ** 0.5)
        curiosity_reward = novelty * self.curiosity_bonus
        
        return curiosity_reward
    
    def _apply_progressive_shaping(self, reward: float) -> float:
        """Applique le reward shaping progressif selon la phase d'apprentissage"""
        if self.learning_phase == "exploration":
            # Phase d'exploration : plus de guidance, moins de pénalités
            return reward * 1.5
        elif self.learning_phase == "learning":
            # Phase d'apprentissage : équilibre
            return reward
        else:  # "optimization"
            # Phase d'optimisation : focus sur l'efficacité
            return reward * 0.8
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalise les récompenses pour stabiliser l'apprentissage"""
        # Ajouter à l'historique
        self.reward_history.append(reward)
        
        # Mettre à jour les statistiques
        if len(self.reward_history) > 10:
            self.reward_mean = sum(self.reward_history) / len(self.reward_history)
            variance = sum((r - self.reward_mean) ** 2 for r in self.reward_history) / len(self.reward_history)
            self.reward_std = max(variance ** 0.5, 1e-6)  # Éviter division par zéro
        
        # Normalisation Z-score
        if self.reward_std > 0:
            normalized_reward = (reward - self.reward_mean) / self.reward_std
            return normalized_reward
        else:
            return reward
    
    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance euclidienne entre deux positions"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def update_learning_phase(self, success_rate: float, episode_count: int):
        """Met à jour la phase d'apprentissage selon les performances"""
        self.episode_count = episode_count
        
        if episode_count < 50 or success_rate < 0.1:
            self.learning_phase = "exploration"
        elif success_rate < 0.6:
            self.learning_phase = "learning"
        else:
            self.learning_phase = "optimization"
    
    def set_pizza_position(self, position: Tuple[int, int]):
        """Définit la position de la pizza pour la guidance directionnelle"""
        self.pizza_position = position


class QTable:
    """Table Q : stocke les valeurs d'apprentissage pour chaque état-action"""
    def __init__(self, config: AgentConfig):
        self.config = config
        self.table: Dict[StateKey, Dict[int, float]] = defaultdict(dict)
    
    def _ensure_state(self, state: State):
        """Initialise toutes les actions à 0.0 pour un nouvel état"""
        qd = self.table[state.key()]
        if not qd:
            # Pour chaque action possible, initialiser la valeur Q à 0
            for a in [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH]:
                qd[a.value] = 0.0
    
    def get(self, state: State, action: Action) -> float:
        self._ensure_state(state)
        return self.table[state.key()].get(action.value, 0.0)
    
    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        """Met à jour la valeur Q selon la formule de Bellman"""
        q_sa = self.get(state, action)  # Valeur Q actuelle
        
        if done:
            # Si l'épisode est terminé, pas de récompense future
            target = reward
        else:
            # Sinon, ajouter la meilleure récompense future possible
            next_qs = self.table.get(next_state.key(), {})
            max_next = max(next_qs.values()) if next_qs else 0.0
            target = reward + self.config.discount_factor * max_next
        
        # Formule de mise à jour Q-Learning
        new_q = (1 - self.config.learning_rate) * q_sa + self.config.learning_rate * target
        
        # Limiter les valeurs Q pour éviter l'explosion
        new_q = max(self.config.q_value_min, min(self.config.q_value_max, new_q))
        self.table[state.key()][action.value] = new_q
    
    def best_action(self, state: State) -> Action:
        """Trouve la meilleure action selon les valeurs Q apprises"""
        self._ensure_state(state)
        qdict = self.table[state.key()]
        
        # Trouver la valeur Q la plus élevée
        max_q = max(qdict.values())
        # Récupérer toutes les actions avec cette valeur maximale
        best_actions = [Action(k) for k, v in qdict.items() if v == max_q]
        # Choisir aléatoirement parmi les meilleures actions
        return random.choice(best_actions)
    
    def size(self) -> int:
        return len(self.table)


class ExplorationStrategy:
    """Stratégie d'exploration : équilibre entre exploration et exploitation"""
    def __init__(self, config: AgentConfig):
        self.config = config
        self.epsilon = config.epsilon        # Taux d'exploration actuel
        self.epsilon_min = config.epsilon_min # Taux minimum d'exploration
        self.epsilon_decay = config.epsilon_decay # Réduction par épisode
    
    def choose_action(self, state: State, q_table: QTable) -> Action:
        """Choisit une action : exploration aléatoire ou exploitation"""
        if random.random() < self.epsilon:
            # Exploration : action aléatoire pour découvrir de nouveaux chemins
            return random.choice([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH])
        else:
            # Exploitation : utiliser les connaissances apprises
            return q_table.best_action(state)
    
    def decay_epsilon(self):
        """Réduit progressivement l'exploration au fil des épisodes"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self) -> float:
        return self.epsilon


class ConvergenceDetector:
    """Détecte quand l'agent a fini d'apprendre (convergence)"""
    def __init__(self, config: AgentConfig):
        self.config = config
        self.history = deque(maxlen=config.convergence_window)  # Historique des performances
        self.converged = False  # L'agent a-t-il convergé ?
    
    def update(self, steps: int, success: bool, reward: float):
        """Ajoute les résultats d'un épisode à l'historique"""
        self.history.append((steps, success, reward))
    
    def is_converged(self, epsilon: float) -> bool:
        """Vérifie si l'agent a convergé (fini d'apprendre)"""
        if len(self.history) < self.config.convergence_window:
            return False  # Pas assez de données
        
        # Analyser les performances récentes
        steps = [s for s, _, _ in self.history]
        successes = [1 if ok else 0 for _, ok, _ in self.history]
        
        mean_steps = sum(steps) / len(steps)  # Nombre moyen d'étapes
        success_rate = sum(successes) / len(successes)  # Taux de succès
        variance = sum((s - mean_steps) ** 2 for s in steps) / len(steps)
        std_dev = variance ** 0.5
        cv = std_dev / mean_steps if mean_steps > 0 else float('inf')  # Coefficient de variation
        
        # Critères de convergence
        epsilon_low = epsilon <= self.config.epsilon_min + 1e-6  # Exploration minimale
        success_high = success_rate >= self.config.min_success_rate  # Bon taux de succès
        variance_low = cv <= self.config.convergence_threshold  # Performance stable
        
        # Vérifier si la performance s'améliore
        performance_improving = False
        if len(self.history) >= 10:
            recent_steps = steps[-10:]  # 10 derniers épisodes
            older_steps = steps[-20:-10] if len(steps) >= 20 else steps[:-10]
            if older_steps:
                recent_avg = sum(recent_steps) / len(recent_steps)
                older_avg = sum(older_steps) / len(older_steps)
                performance_improving = recent_avg < older_avg * 0.8  # 20% d'amélioration
        
        # Différentes conditions de convergence
        # 1. Performance stable avec bon taux de succès
        performance_stable = variance_low and success_high
        
        # 2. Exploration terminée avec performance stable
        epsilon_converged = epsilon_low and variance_low
        
        # 3. Performance stable depuis longtemps (même avec succès modéré)
        long_term_stable = len(self.history) >= self.config.convergence_window * 2 and variance_low
        
        # 4. Convergence précoce si exploration terminée ET performance s'améliore
        early_convergence = epsilon_low and success_rate >= 0.3 and variance_low
        
        # 5. Convergence si performance s'améliore significativement
        improvement_convergence = performance_improving and success_rate >= 0.4 and variance_low
        
        # L'agent a convergé si une de ces conditions est remplie
        self.converged = performance_stable or epsilon_converged or long_term_stable or early_convergence or improvement_convergence
        
        # Messages de debug pour comprendre pourquoi l'agent n'a pas encore convergé
        if not self.converged:
            if not success_high and not variance_low:
                print(f"[DEBUG] Convergence bloquée: succès={success_rate:.1%} < {self.config.min_success_rate:.1%}, CV={cv:.4f} > {self.config.convergence_threshold}")
            elif not success_high:
                print(f"[DEBUG] Convergence bloquée par succès: {success_rate:.1%} < seuil={self.config.min_success_rate:.1%}")
            elif not variance_low:
                print(f"[DEBUG] Convergence bloquée par stabilité: CV={cv:.4f} > seuil={self.config.convergence_threshold}")
        
        return self.converged
    
    def get_convergence_info(self) -> Dict[str, Any]:
        if len(self.history) < self.config.convergence_window:
            return {
                'converged': False,
                'episodes_needed': self.config.convergence_window - len(self.history),
                'reason': 'Pas assez d\'épisodes'
            }
        
        steps = [s for s, _, _ in self.history]
        successes = [1 if ok else 0 for _, ok, _ in self.history]
        
        mean_steps = sum(steps) / len(steps)
        success_rate = sum(successes) / len(successes)
        variance = sum((s - mean_steps) ** 2 for s in steps) / len(steps)
        std_dev = variance ** 0.5
        cv = std_dev / mean_steps if mean_steps > 0 else float('inf')
        
        return {
            'converged': self.converged,
            'coefficient_variation': cv,
            'threshold': self.config.convergence_threshold,
            'mean_performance': mean_steps,
            'std_performance': std_dev,
            'success_rate': success_rate,
            'episodes_analyzed': len(self.history)
        }


class PerformanceTracker:
    """Suit les performances de l'agent pendant l'entraînement"""
    def __init__(self):
        self.episode_count = 0  # Nombre total d'épisodes
        self.best_steps = float('inf')  # Meilleur nombre d'étapes
        self.episode_results = []  # Résultats de tous les épisodes
        self.visited_states = set()  # États visités
        self.action_history = deque(maxlen=3)  # Historique des actions récentes
        self.position_history = deque(maxlen=20)  # Historique des positions
        self.best_successful_steps = float('inf')  # Meilleur succès
        self.adaptive_steps_history = deque(maxlen=10)  # Historique des succès
    
    def update_episode(self, steps: int, success: bool, reward: float):
        """Met à jour les statistiques après un épisode"""
        self.episode_count += 1
        
        # Mettre à jour le meilleur score
        if success and steps < self.best_steps:
            self.best_steps = steps
        
        # Suivre les succès récents
        if success and steps < self.best_successful_steps:
            self.best_successful_steps = steps
            self.adaptive_steps_history.append(steps)
        
        # Sauvegarder les résultats
        self.episode_results.append((reward, steps, success))
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'episode_count': self.episode_count,
            'best_steps': self.best_steps,
            'best_successful_steps': self.best_successful_steps,
            'visited_states': len(self.visited_states),
            'total_episodes': len(self.episode_results)
        }
    
    def clear_episode_data(self):
        self.action_history.clear()
        self.position_history.clear()


class EnvironmentAdapter:
    """Interface entre l'agent et l'environnement DonatelloPyzza"""
    def __init__(self, environment_name: str, show_gui: bool = True):
        self.environment_name = environment_name  # Nom du labyrinthe
        self.show_gui = show_gui  # Afficher l'interface graphique ?
        self.game = None  # Instance du jeu
        self.turtle = None  # Instance de la tortue
    
    def reset(self) -> State:
        """Remet l'environnement à zéro et retourne l'état initial"""
        self.game = RLGame(self.environment_name, gui=self.show_gui)
        self.turtle = self.game.start()
        
        # Récupérer la position et l'orientation initiales
        position = self.game.getTurtlePosition(self.turtle)
        orientation = self.game.getTurtleOrientation(self.turtle)
        
        return State(position=position, orientation=orientation)
    
    def step(self, action: Action) -> Tuple[State, Feedback, bool]:
        """Exécute une action et retourne le nouvel état, le feedback et si c'est terminé"""
        feedback, _ = self.turtle.execute(action)
        
        # Récupérer le nouvel état après l'action
        new_position = self.game.getTurtlePosition(self.turtle)
        new_orientation = self.game.getTurtleOrientation(self.turtle)
        has_touched_wall = feedback == Feedback.TOUCHED_WALL
        
        next_state = State(
            position=new_position,
            orientation=new_orientation,
            has_touched_wall=has_touched_wall
        )
        
        # Vérifier si l'épisode est terminé (pizza trouvée)
        done = self.game.isWon(prnt=False)
        
        return next_state, feedback, done
    
    def is_won(self) -> bool:
        return self.game.isWon(prnt=False) if self.game else False


class QLearningAgent:
    """Agent Q-Learning principal qui coordonne tous les composants"""
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        # Configuration de la reproductibilité
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            # S'assurer que toutes les sources d'aléa utilisent la même graine
            try:
                import numpy as np
                np.random.seed(self.config.random_seed)
            except ImportError:
                pass
        
        # Initialisation des composants
        self.q_table = QTable(self.config)  # Table d'apprentissage
        self.reward_system = RewardSystem(
            pure_qlearning=self.config.pure_qlearning,
            intelligent_shaping=self.config.intelligent_shaping
        )  # Système de récompenses intelligent
        self.exploration_strategy = ExplorationStrategy(self.config)  # Stratégie d'exploration
        self.convergence_detector = ConvergenceDetector(self.config)  # Détection de convergence
        self.performance_tracker = PerformanceTracker()  # Suivi des performances
        
        self.max_steps = self.config.max_steps  # Limite d'étapes par épisode
    
    def train_episode(self, env_adapter: EnvironmentAdapter, verbose: bool = True) -> Tuple[float, int, bool]:
        """Entraîne l'agent sur un épisode complet"""
        total_reward = 0.0
        steps = 0
        success = False
        
        # Commencer un nouvel épisode
        current_state = env_adapter.reset()
        
        # Détecter automatiquement la position de la pizza si pas encore fait
        if self.reward_system.pizza_position is None:
            self._detect_pizza_position(env_adapter)
        
        # Boucle principale de l'épisode
        while steps < self.max_steps:
            steps += 1
            
            # Choisir une action (exploration ou exploitation)
            action = self.exploration_strategy.choose_action(current_state, self.q_table)
            next_state, feedback, done = env_adapter.step(action)
            
            # Calculer la récompense
            reward = self.reward_system.calculate_reward(feedback, current_state, next_state, action)
            
            # Mettre à jour la table Q
            self.q_table.update(current_state, action, reward, next_state, done)
            
            # Suivre les statistiques
            self.performance_tracker.action_history.append(action)
            self.performance_tracker.position_history.append(next_state.position)
            self.performance_tracker.visited_states.add(current_state.key())
            self.performance_tracker.visited_states.add(next_state.key())
            
            total_reward += reward
            
            if verbose and steps % 10 == 0:
                action_desc_map = {
                    'MOVE_FORWARD': "avance",
                    'TURN_LEFT': "tourne à gauche",
                    'TURN_RIGHT': "tourne à droite",
                    'TOUCH': "touche"
                }
                action_desc = action_desc_map.get(action.name, action.name)
                print(f"  Étape {steps}: {action_desc} -> position {next_state.position}, récompense {reward:.1f}")
            
            if done:
                success = env_adapter.is_won()
                if verbose:
                    if success:
                        print(f"\n[SUCCÈS] Pizza trouvée en {steps} étapes! (récompense totale: {total_reward:.1f})")
                    else:
                        print(f"\n[ÉCHEC] Épisode terminé en {steps} étapes (récompense totale: {total_reward:.1f})")
                break
            
            current_state = next_state
        
        if steps >= self.max_steps and not success:
            if verbose:
                print(f"\n[ÉCHEC] Limite d'étapes atteinte ({self.max_steps}) - l'agent n'a pas trouvé la pizza")
        
        self._update_statistics(total_reward, steps, success)
        
        return total_reward, steps, success
    
    def _detect_pizza_position(self, env_adapter: EnvironmentAdapter):
        """Détecte automatiquement la position de la pizza dans l'environnement"""
        try:
            # Essayer de récupérer la position de la pizza depuis l'environnement
            if hasattr(env_adapter.game, 'getPizzaPosition'):
                pizza_pos = env_adapter.game.getPizzaPosition()
                if pizza_pos:
                    self.reward_system.set_pizza_position(pizza_pos)
                    print(f"[INFO] Position de la pizza détectée: {pizza_pos}")
            else:
                # Fallback: chercher la pizza en explorant
                self._find_pizza_by_exploration(env_adapter)
        except Exception as e:
            print(f"[WARNING] Impossible de détecter la position de la pizza: {e}")
    
    def _find_pizza_by_exploration(self, env_adapter: EnvironmentAdapter):
        """Trouve la pizza en explorant l'environnement"""
        print("[INFO] Exploration pour trouver la pizza...")
        # Cette méthode pourrait être implémentée pour explorer l'environnement
        # et détecter la position de la pizza automatiquement
        pass
    
    def _update_statistics(self, total_reward: float, steps: int, success: bool):
        self.performance_tracker.update_episode(steps, success, total_reward)
        self.convergence_detector.update(steps, success, total_reward)
        self.exploration_strategy.decay_epsilon()
        
        # Mettre à jour la phase d'apprentissage du reward system
        success_rate = self.performance_tracker.episode_count
        if success_rate > 0:
            recent_successes = sum(1 for _, _, s in self.performance_tracker.episode_results[-20:] if s)
            recent_rate = recent_successes / min(20, len(self.performance_tracker.episode_results))
            self.reward_system.update_learning_phase(recent_rate, self.performance_tracker.episode_count)
        
        self.performance_tracker.clear_episode_data()
    
    def check_convergence(self) -> bool:
        return self.convergence_detector.is_converged(self.exploration_strategy.get_epsilon())
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = self.performance_tracker.get_statistics()
        convergence_info = self.convergence_detector.get_convergence_info()
        
        return {
            'q_table_size': self.q_table.size(),
            'epsilon': self.exploration_strategy.get_epsilon(),
            'episode_count': stats['episode_count'],
            'best_steps': stats['best_steps'],
            'converged': convergence_info.get('converged', False),
            'visited_states': stats['visited_states']
        }
    
    def get_convergence_info(self) -> Dict[str, Any]:
        return self.convergence_detector.get_convergence_info()
    
    def print_convergence_summary(self):
        convergence_info = self.get_convergence_info()
        
        print("\n" + "=" * 50)
        print("ANALYSE DE CONVERGENCE")
        print("=" * 50)
        
        if convergence_info.get('converged', False):
            print("[SUCCÈS] CONVERGENCE ATTEINTE!")
            print(f"   - Stabilité: {convergence_info.get('coefficient_variation', 0):.4f}")
            print(f"   - Seuil requis: {convergence_info.get('threshold', 0)}")
            print(f"   - Performance moyenne: {convergence_info.get('mean_performance', 0):.1f} étapes")
            print(f"   - Épisodes analysés: {convergence_info.get('episodes_analyzed', 0)}")
        else:
            print("[APPRENTISSAGE] Convergence en cours...")
            if 'episodes_needed' in convergence_info:
                print(f"   - Épisodes nécessaires: {convergence_info['episodes_needed']}")
            else:
                print(f"   - Stabilité actuelle: {convergence_info.get('coefficient_variation', 0):.4f}")
                print(f"   - Seuil requis: {convergence_info.get('threshold', 0)}")
                print(f"   - Performance moyenne: {convergence_info.get('mean_performance', 0):.1f} étapes")
                print(f"   - Épisodes analysés: {convergence_info.get('episodes_analyzed', 0)}")
        
        print(f"   - Taux d'exploration: {self.exploration_strategy.get_epsilon():.3f}")
        print("=" * 50)


def train_agent(
    environment_name: str = "maze",
    show_gui: bool = True,
    verbose: bool = True,
    max_episodes: int = 2000,
    config: AgentConfig = None
) -> QLearningAgent:
    global interrupted
    interrupted = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    agent = QLearningAgent(config)
    
    print("=" * 60)
    print("ENTRAÎNEMENT Q-LEARNING - DONATELLOPYZZA")
    print("=" * 60)
    print(f"Environnement: {environment_name}")
    print(f"Limite d'épisodes: {max_episodes}")
    print(f"Learning rate: {agent.config.learning_rate}")
    print(f"Discount factor: {agent.config.discount_factor}")
    print(f"Epsilon initial: {agent.config.epsilon}")
    print("=" * 60)
    print("[ASTUCE] Appuyez sur Ctrl+C pour arrêter l'entraînement à tout moment")
    print("=" * 60)

    successful_episodes = 0
    episode = 0
    
    env_adapter = EnvironmentAdapter(environment_name, show_gui)

    while not agent.convergence_detector.converged and not interrupted and episode < max_episodes:
        episode += 1
        print(f"\nÉPISODE {episode}")
        print("-" * 40)

        reward, steps, success = agent.train_episode(env_adapter, verbose)

        if success:
            successful_episodes += 1

        if episode % 10 == 0:
            stats = agent.get_statistics()
            success_rate = successful_episodes / episode
            
            print(f"\n[ANALYSE] BILAN APRÈS {episode} ÉPISODES:")
            print(f"  Taux de succès global: {success_rate:.1%}")
            
            convergence_info = agent.get_convergence_info()
            if 'success_rate' in convergence_info:
                print(f"  Taux de succès récent ({agent.config.convergence_window}): {convergence_info['success_rate']:.1%}")
            
            print(f"  Meilleur chemin trouvé: {stats['best_steps']} étapes")
            print(f"  Epsilon (exploration): {stats['epsilon']:.3f}")
            print(f"  États appris: {stats['q_table_size']}")
            
            # Afficher les informations sur le reward shaping intelligent
            if agent.config.intelligent_shaping and not agent.config.pure_qlearning:
                print(f"  Phase d'apprentissage: {agent.reward_system.learning_phase}")
                print(f"  Position pizza: {agent.reward_system.pizza_position if agent.reward_system.pizza_position else 'Non détectée'}")
                print(f"  Récompenses normalisées: {len(agent.reward_system.reward_history)} échantillons")
            
            # Vérifier la convergence après chaque analyse
            if agent.check_convergence():
                print(f"\n[CONVERGENCE] Performance stable détectée après {episode} épisodes!")
                break
            
            agent.print_convergence_summary()

    print("\n" + "=" * 60)
    if interrupted:
        print("[INTERRUPTION] Entraînement interrompu par l'utilisateur (Ctrl+C)")
    elif episode >= max_episodes:
        print("[LIMITE] Entraînement terminé - limite d'épisodes atteinte!")
    else:
        print("[TERMINÉ] Entraînement terminé - convergence atteinte!")
    print("=" * 60)
    
    final_stats = agent.get_statistics()
    final_success_rate = successful_episodes / episode
    convergence_info = agent.get_convergence_info()
    
    print(f"RÉSULTATS FINAUX:")
    print(f"  Épisodes total: {episode}")
    print(f"  Taux de succès final: {final_success_rate:.1%}")
    print(f"  Meilleur chemin trouvé: {final_stats['best_steps']} étapes")
    print(f"  États appris: {final_stats['q_table_size']}")
    
    if convergence_info.get('converged', False):
        print(f"  [SUCCÈS] Convergence: atteinte!")
        print(f"  Performance stable: {convergence_info.get('mean_performance', 0):.1f} ± {convergence_info.get('std_performance', 0):.1f} étapes")
    else:
        print("  [APPRENTISSAGE] Convergence: non atteinte - l'agent apprend encore...")
    
    
    print("=" * 60)
    
    agent.print_convergence_summary()

    return agent


def get_user_config() -> Dict[str, Any]:
    print("=" * 60)
    print("[IA] Q-LEARNING POUR DONATELLOPYZZA")
    print("=" * 60)
    print("Objectif: Apprendre à naviguer vers la pizza")
    print("Méthode: Apprentissage par renforcement (Q-Learning)")
    print("=" * 60)

    environments = ["maze", "assessment_maze", "hard_maze", "line", "test"]

    print("\nEnvironnements disponibles:")
    for i, env in enumerate(environments, 1):
        print(f"  {i}. {env}")

    while True:
        try:
            choice = int(input("\nChoisissez un environnement (1-5) [défaut: 1]: ") or "1")
            if 1 <= choice <= len(environments):
                environment_name = environments[choice - 1]
                break
            else:
                print("Choix invalide. Veuillez choisir entre 1 et 5.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")

    try:
        show_gui = input("Afficher l'interface graphique ? (o/n) [défaut: o]: ").lower() != 'n'
        verbose = input("Affichage détaillé ? (o/n) [défaut: o]: ").lower() != 'n'
    except ValueError:
        show_gui = True
        verbose = True
    
    try:
        advanced = input("Configuration avancée ? (o/n) [défaut: n]: ").lower() == 'o'
        if advanced:
            learning_rate = float(input("Learning rate [défaut: 0.1]: ") or "0.1")
            epsilon = float(input("Epsilon initial [défaut: 0.3]: ") or "0.3")
            max_episodes = int(input("Max épisodes [défaut: 2000]: ") or "2000")
            pure_mode = input("Mode Q-Learning pur (sans reward shaping) ? (o/n) [défaut: n]: ").lower() == 'o'
            intelligent_mode = input("Mode reward shaping intelligent ? (o/n) [défaut: o]: ").lower() != 'n'
            
            config = AgentConfig(
                learning_rate=learning_rate,
                epsilon=epsilon,
                pure_qlearning=pure_mode,
                intelligent_shaping=intelligent_mode
            )
        else:
            config = AgentConfig()
            max_episodes = 2000
    except ValueError:
        config = AgentConfig()
        max_episodes = 2000

    return {
        'environment_name': environment_name,
        'show_gui': show_gui,
        'verbose': verbose,
        'config': config,
        'max_episodes': max_episodes
    }


def run_training_pipeline(config: Dict[str, Any]) -> QLearningAgent:
    agent = train_agent(
        environment_name=config['environment_name'],
        show_gui=config['show_gui'],
        verbose=config['verbose'],
        max_episodes=config.get('max_episodes', 2000),
        config=config.get('config', None)
    )

    return agent


def main():
    config = get_user_config()
    agent = run_training_pipeline(config)
    
    print("\n[TERMINÉ] Programme terminé!")
    stats = agent.get_statistics()
    print(f"Agent final avec {stats['q_table_size']} états appris")
    print(f"Chemin optimal: {stats['best_steps']} étapes")
    print(f"États visités: {stats['visited_states']}")
    print("L'agent a fini son apprentissage! [APPRENTISSAGE TERMINÉ]")


if __name__ == "__main__":
    main()