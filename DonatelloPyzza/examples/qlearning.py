import sys
import os
import random
import time
import signal
import logging
from typing import Dict, Tuple, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass

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


# ============================================================================
# CLASSES DE CONFIGURATION ET DE DONNÉES
# ============================================================================

@dataclass
class AgentConfig:
    """Configuration de l'agent Q-Learning"""
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.3
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    convergence_window: int = 20
    convergence_threshold: float = 0.05
    max_steps: int = 1000
    random_seed: Optional[int] = None
    systematic_exploration_episodes: int = 50
    min_adaptive_steps: int = 50
    adaptive_margin: float = 0.1
    q_value_min: float = -1000.0
    q_value_max: float = 1000.0
    min_success_rate: float = 0.8


@dataclass
class State:
    """Représentation enrichie d'un état"""
    position: Tuple[int, int]
    orientation: int
    has_touched_wall: bool = False
    steps_since_pizza: int = 0
    previous_actions: List[Action] = None
    
    def __post_init__(self):
        if self.previous_actions is None:
            self.previous_actions = []
    
    def __hash__(self):
        return hash((self.position, self.orientation, self.has_touched_wall))
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (self.position == other.position and 
                self.orientation == other.orientation and
                self.has_touched_wall == other.has_touched_wall)


class RewardSystem:
    """Système de récompenses sophistiqué"""
    
    def __init__(self):
        self.base_rewards = {
            'pizza_found': 100.0,
            'pizza_touched': 50.0,
            'collision': -10.0,
            'step': -2.0,
            'wall_touched': -5.0,
            'new_state': 3.0,
            'redundant_action': -3.0
        }
        self.bonus_multipliers = {
            'efficiency': 1.5,
            'exploration': 2.0,
            'short_path': 2.0
        }
    
    def calculate_reward(self, feedback: Feedback, current_state: State, 
                        next_state: State, action: Action, 
                        visited_states: set, action_history: List[Action]) -> float:
        """Calcule la récompense avec logique sophistiquée"""
        base_reward = 0.0
        
        # Récompense de base selon le feedback
        if feedback == Feedback.MOVED_ON_PIZZA:
            base_reward = self.base_rewards['pizza_found']
        elif feedback == Feedback.TOUCHED_PIZZA:
            base_reward = self.base_rewards['pizza_touched']
        elif feedback == Feedback.COLLISION:
            base_reward = self.base_rewards['collision']
        elif feedback == Feedback.TOUCHED_WALL:
            base_reward = self.base_rewards['wall_touched']
        else:
            base_reward = self.base_rewards['step']
        
        # Bonus pour exploration de nouveaux états
        if next_state not in visited_states:
            base_reward += self.base_rewards['new_state'] * self.bonus_multipliers['exploration']
            visited_states.add(next_state)
        
        # Pénalités pour actions redondantes
        if len(action_history) >= 2:
            if self._detect_redundant_actions(action_history):
                base_reward += self.base_rewards['redundant_action']
        
        # Bonus d'efficacité pour chemins courts
        if hasattr(next_state, 'steps_since_pizza') and next_state.steps_since_pizza < 20:
            base_reward *= self.bonus_multipliers['efficiency']
        
        return base_reward
    
    def _detect_redundant_actions(self, action_history: List[Action]) -> bool:
        """Détecte les actions redondantes"""
        if len(action_history) < 2:
            return False
        
        # Détection de tours alternés (gauche-droite)
        if (action_history[-2] == Action.TURN_RIGHT and action_history[-1] == Action.TURN_LEFT) or \
           (action_history[-2] == Action.TURN_LEFT and action_history[-1] == Action.TURN_RIGHT):
            return True
        
        # Détection de répétitions (3 fois la même action)
        if len(action_history) >= 3:
            if action_history[-3] == action_history[-2] == action_history[-1]:
                return True
        
        return False


class ConvergenceDetector:
    """Détecteur de convergence intelligent"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.performance_history = []
        self.converged = False
    
    def update_performance(self, steps: int, success: bool, reward: float):
        """Met à jour l'historique des performances"""
        self.performance_history.append({
            'steps': steps,
            'success': success,
            'reward': reward
        })
        
        # Garder seulement les dernières performances
        if len(self.performance_history) > self.config.convergence_window * 2:
            self.performance_history = self.performance_history[-self.config.convergence_window:]
    
    def is_converged(self, episode_count: int, epsilon: float, 
                    visited_states: int) -> bool:
        """Vérifie si l'agent a convergé"""
        if len(self.performance_history) < self.config.convergence_window:
            return False
        
        recent_results = self.performance_history[-self.config.convergence_window:]
        
        # Calcul des statistiques
        steps = [r['steps'] for r in recent_results]
        successes = [r['success'] for r in recent_results]
        
        mean_steps = sum(steps) / len(steps)
        success_rate = sum(successes) / len(successes)
        
        # Coefficient de variation
        variance = sum((s - mean_steps) ** 2 for s in steps) / len(steps)
        std_dev = variance ** 0.5
        cv = std_dev / mean_steps if mean_steps > 0 else float('inf')
        
        # Critères de convergence
        exploration_sufficient = visited_states > 50 or episode_count > 300
        
        convergence_criteria = [
            cv < self.config.convergence_threshold,
            epsilon <= self.config.epsilon_min,
            success_rate >= self.config.min_success_rate,
            exploration_sufficient
        ]
        
        self.converged = all(convergence_criteria)
        return self.converged
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Retourne les informations de convergence"""
        if len(self.performance_history) < self.config.convergence_window:
            return {
                'converged': False,
                'episodes_needed': self.config.convergence_window - len(self.performance_history),
                'reason': 'Pas assez d\'épisodes'
            }
        
        recent_results = self.performance_history[-self.config.convergence_window:]
        steps = [r['steps'] for r in recent_results]
        successes = [r['success'] for r in recent_results]
        
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
            'episodes_analyzed': len(recent_results)
        }


class PerformanceTracker:
    """Suivi des performances et statistiques"""
    
    def __init__(self):
        self.episode_count = 0
        self.best_steps = float('inf')
        self.episode_results = []
        self.visited_states = set()
        self.action_history = deque(maxlen=3)
        self.position_history = deque(maxlen=20)
        self.best_successful_steps = float('inf')
        self.adaptive_steps_history = deque(maxlen=10)
    
    def update_episode(self, steps: int, success: bool, reward: float):
        """Met à jour les statistiques d'un épisode"""
        self.episode_count += 1
        
        if success and steps < self.best_steps:
            self.best_steps = steps
        
        if success and steps < self.best_successful_steps:
            self.best_successful_steps = steps
            self.adaptive_steps_history.append(steps)
        
        self.episode_results.append((reward, steps, success))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques actuelles"""
        return {
            'episode_count': self.episode_count,
            'best_steps': self.best_steps,
            'best_successful_steps': self.best_successful_steps,
            'visited_states': len(self.visited_states),
            'total_episodes': len(self.episode_results)
        }
    
    def clear_episode_data(self):
        """Nettoie les données temporaires d'épisode"""
        self.action_history.clear()
        self.position_history.clear()


class QTable:
    """Gestionnaire de la Q-table avec optimisations"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.q_table: Dict[State, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    
    def get_q_value(self, state: State, action: Action) -> float:
        """Récupère la valeur Q pour un état-action"""
        return self.q_table[state][int(action.value)]
    
    def update_q_value(self, state: State, action: Action, reward: float, next_state: State):
        """Met à jour la valeur Q selon la formule TD"""
        action_key = int(action.value)
        current_q = self.q_table[state][action_key]
        
        # Calcul de la valeur Q maximale du prochain état
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0
        
        # Formule TD
        td_target = reward + self.config.discount_factor * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.config.learning_rate * td_error
        
        # Validation et bornage des valeurs
        if not isinstance(new_q, (int, float)) or new_q != new_q:
            new_q = 0.0
        elif new_q == float('inf') or new_q == float('-inf'):
            new_q = 0.0
        else:
            new_q = max(self.config.q_value_min, min(self.config.q_value_max, new_q))
        
        self.q_table[state][action_key] = new_q
    
    def get_best_action(self, state: State) -> Action:
        """Retourne la meilleure action pour un état"""
        if state in self.q_table and self.q_table[state]:
            qdict = self.q_table[state]
            max_q = max(qdict.values())
            best_actions = [k for k, v in qdict.items() if v == max_q]
            return Action(random.choice(best_actions))
        return random.choice([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH])
    
    def get_size(self) -> int:
        """Retourne la taille de la Q-table"""
        return len(self.q_table)


class ExplorationStrategy:
    """Stratégie d'exploration intelligente"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.exploration_sequence = [
            Action.TOUCH, Action.TURN_LEFT, Action.TOUCH, Action.TURN_LEFT,
            Action.TOUCH, Action.TURN_LEFT, Action.TOUCH, Action.TURN_LEFT,
            Action.MOVE_FORWARD
        ]
        self.exploration_index = 0
    
    def choose_action(self, state: State, q_table: QTable, 
                     visited_states: set, episode_count: int) -> Action:
        """Choisit une action selon la stratégie d'exploration"""
        # Exploration systématique en début d'apprentissage
        if episode_count < self.config.systematic_exploration_episodes:
            return self._systematic_exploration(state, visited_states)
        
        # Epsilon-greedy classique
        return self._epsilon_greedy(state, q_table)
    
    def _systematic_exploration(self, state: State, visited_states: set) -> Action:
        """Exploration systématique pour nouveaux états"""
        if state not in visited_states:
            if self.exploration_index < len(self.exploration_sequence):
                action = self.exploration_sequence[self.exploration_index]
                self.exploration_index += 1
                return action
            else:
                self.exploration_index = 0
                return Action.MOVE_FORWARD
        
        return self._epsilon_greedy(state, None)
    
    def _epsilon_greedy(self, state: State, q_table: QTable) -> Action:
        """Sélection epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.choice([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH])
        
        if q_table and state in q_table.q_table and q_table.q_table[state]:
            return q_table.get_best_action(state)
        
        return random.choice([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH])
    
    def decay_epsilon(self):
        """Réduit le taux d'exploration"""
        if self.epsilon > self.config.epsilon_min:
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
    
    def get_epsilon(self) -> float:
        """Retourne la valeur actuelle d'epsilon"""
        return self.epsilon


class QLearningAgent:
    """Agent Q-Learning refactorisé pour DonatelloPyzza"""
    
    def __init__(self, config: AgentConfig = None):
        """Initialise l'agent Q-Learning avec la nouvelle architecture modulaire"""
        self.config = config or AgentConfig()
        
        # Initialisation du générateur aléatoire
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        # Composants modulaires
        self.q_table = QTable(self.config)
        self.reward_system = RewardSystem()
        self.exploration_strategy = ExplorationStrategy(self.config)
        self.convergence_detector = ConvergenceDetector(self.config)
        self.performance_tracker = PerformanceTracker()
        
        # État adaptatif
        self.max_steps = self.config.max_steps
        self.original_max_steps = self.config.max_steps
        
        # Configuration du logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le système de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('qlearning.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_state(self, position: Tuple[int, int], orientation: int, 
                  has_touched_wall: bool = False, steps_since_pizza: int = 0) -> State:
        """Génère un état enrichi"""
        return State(
            position=position,
            orientation=orientation,
            has_touched_wall=has_touched_wall,
            steps_since_pizza=steps_since_pizza,
            previous_actions=list(self.performance_tracker.action_history)
        )

    def calculate_reward(self, feedback: Feedback, current_state: State, 
                        next_state: State, action: Action) -> float:
        """Calcule la récompense avec le système sophistiqué"""
        return self.reward_system.calculate_reward(
            feedback, current_state, next_state, action,
            self.performance_tracker.visited_states,
            list(self.performance_tracker.action_history)
        )

    def detect_loop(self, position: Tuple[int, int]) -> bool:
        """Détecte si l'agent tourne en rond"""
        self.performance_tracker.position_history.append(position)
        
        if len(self.performance_tracker.position_history) >= 10:
            recent_positions = list(self.performance_tracker.position_history)[-10:]
            position_counts = {}
            for pos in recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            return max(position_counts.values()) > 3
        
        return False

    def adjust_max_steps_for_loops(self, current_steps: int) -> int:
        """Ajuste la limite d'étapes si l'agent tourne en rond"""
        if current_steps > 50 and self.detect_loop(self.performance_tracker.position_history[-1] if self.performance_tracker.position_history else (0, 0)):
            return min(current_steps + 20, self.max_steps)
        return self.max_steps

    def choose_action(self, state: State) -> Action:
        """Sélection d'action intelligente avec les composants modulaires"""
        return self.exploration_strategy.choose_action(
            state, self.q_table, 
            self.performance_tracker.visited_states,
            self.performance_tracker.episode_count
        )

    def update_q_value(self, state: State, action: Action, reward: float, next_state: State):
        """Met à jour la Q-table avec le composant modulaire"""
        self.q_table.update_q_value(state, action, reward, next_state)

    def update_adaptive_episode_length(self, steps: int, success: bool):
        """Met à jour la limite d'épisodes avec logique adaptative"""
        if success:
            if steps < self.performance_tracker.best_successful_steps:
                self.performance_tracker.best_successful_steps = steps
                new_max_steps = max(int(steps * 2.0), self.config.min_adaptive_steps)
                if self.performance_tracker.episode_count > 100:
                    self.max_steps = new_max_steps
            
            self.performance_tracker.adaptive_steps_history.append(steps)
            
            if len(self.performance_tracker.adaptive_steps_history) >= 10 and self.performance_tracker.episode_count > 200:
                recent_avg = sum(list(self.performance_tracker.adaptive_steps_history)[-10:]) / 10
                if recent_avg < self.max_steps * 0.6:
                    new_limit = max(int(recent_avg * 1.5), self.config.min_adaptive_steps)
                    if new_limit < self.max_steps:
                        self.max_steps = new_limit
        else:
            if self.performance_tracker.episode_count < 50:
                self.max_steps = min(self.max_steps * 1.1, 2000)
            elif self.performance_tracker.episode_count < 200:
                self.max_steps = min(self.max_steps * 1.05, 1500)

    def decay_epsilon(self):
        """Réduit le taux d'exploration"""
        self.exploration_strategy.decay_epsilon()

    def check_convergence(self) -> bool:
        """Vérifie la convergence avec le détecteur modulaire"""
        return self.convergence_detector.is_converged(
            self.performance_tracker.episode_count,
            self.exploration_strategy.get_epsilon(),
            len(self.performance_tracker.visited_states)
        )

    def _execute_step(self, game: RLGame, turtle, current_state: State, steps: int, verbose: bool) -> Tuple[float, State, bool]:
        """Exécute un pas d'apprentissage avec les composants modulaires"""
        action = self.choose_action(current_state)
        feedback, _ = turtle.execute(action)

        new_position = game.getTurtlePosition(turtle)
        new_orientation = game.getTurtleOrientation(turtle)
        
        # Mise à jour de l'historique des actions
        self.performance_tracker.action_history.append(action)
        
        # Création du nouvel état enrichi
        has_touched_wall = feedback == Feedback.TOUCHED_WALL
        next_state = self.get_state(new_position, new_orientation, has_touched_wall, steps)

        reward = self.calculate_reward(feedback, current_state, next_state, action)
        
        # Détection de boucles pour timeout intelligent
        is_looping = self.detect_loop(new_position)
        if is_looping:
            reward += self.reward_system.base_rewards['redundant_action'] * 2

        self.update_q_value(current_state, action, reward, next_state)

        if verbose:
            action_desc_map = {
                'MOVE_FORWARD': "avance",
                'TURN_LEFT': "tourne à gauche", 
                'TURN_RIGHT': "tourne à droite",
                'TOUCH': "touche"
            }
            feedback_desc_map = {
                'MOVED_ON_PIZZA': "PIZZA TROUVÉE!",
                'TOUCHED_PIZZA': "a touché la pizza",
                'COLLISION': "collision avec un mur",
                'TOUCHED_WALL': "a touché un mur",
                'MOVED': "s'est déplacé"
            }
            action_desc = action_desc_map.get(action.name, action.name)
            feedback_desc = feedback_desc_map.get(feedback.name, feedback.name)
            
            print(f"Episode {self.performance_tracker.episode_count + 1}, étape {steps}: {action_desc} -> "
                  f"position {new_position}, récompense {reward:.1f}, {feedback_desc}")

        return reward, next_state, game.isWon(prnt=False)

    def _update_statistics(self, total_reward: float, steps: int, success: bool):
        """Met à jour les statistiques avec les composants modulaires"""
        # Mise à jour du tracker de performance
        self.performance_tracker.update_episode(steps, success, total_reward)
        
        # Mise à jour du détecteur de convergence
        self.convergence_detector.update_performance(steps, success, total_reward)
        
        # Ajustement adaptatif de la longueur d'épisode
        self.update_adaptive_episode_length(steps, success)
        
        # Décroissance d'epsilon
        self.decay_epsilon()
        
        # Nettoyage des données temporaires
        self.performance_tracker.clear_episode_data()

    def get_convergence_info(self) -> Dict[str, Any]:
        """Retourne les informations de convergence avec le détecteur modulaire"""
        return self.convergence_detector.get_convergence_info()

    def train_episode(self, game: RLGame, turtle, show_gui: bool = True, verbose: bool = True, training_mode: bool = True) -> Tuple[float, int, bool]:
        """Exécute un épisode avec l'architecture modulaire"""
        total_reward = 0.0
        steps = 0

        current_position = game.getTurtlePosition(turtle)
        current_orientation = game.getTurtleOrientation(turtle)
        current_state = self.get_state(current_position, current_orientation)

        while steps < self.max_steps:
            steps += 1

            # Ajuster la limite d'étapes si l'agent tourne en rond
            if steps > 50:
                adjusted_max_steps = self.adjust_max_steps_for_loops(steps)
                if steps >= adjusted_max_steps:
                    if verbose:
                        print(f"\n[TIMEOUT INTELLIGENT] Arrêt après {steps} étapes - détection de boucle")
                    break

            reward, next_state, success = self._execute_step(game, turtle, current_state, steps, verbose)
            total_reward += reward

            if show_gui:
                time.sleep(0.01)

            if success:
                if verbose:
                    print(f"\n[SUCCÈS] Pizza trouvée en {steps} étapes! (récompense totale: {total_reward:.1f})")
                    if steps <= 20:
                        print("  -> Excellent chemin trouvé!")
                    elif steps <= 50:
                        print("  -> Bon chemin!")
                    else:
                        print("  -> Chemin trouvé!")
                break

            current_state = next_state
        
        if steps >= self.max_steps and not success:
            if verbose:
                print(f"\n[ÉCHEC] Limite d'étapes atteinte ({self.max_steps}) - l'agent n'a pas trouvé la pizza")
                print(f"  -> Récompense totale: {total_reward:.1f}")
                print("  -> L'agent doit encore apprendre...")

        if training_mode:
            self._update_statistics(total_reward, steps, success)
            converged = self.check_convergence()
        else:
            converged = False

        if verbose:
            stats = self.performance_tracker.get_statistics()
            convergence_status = "convergé" if converged else "en cours"
            
            status_emoji = "✓" if success else "✗"
            status_text = "Réussi!" if success else "Échoué"
            
            print(f"\nEpisode {stats['episode_count']}: {status_emoji} {status_text} en {steps} étapes")
            print(f"  -> Récompense: {total_reward:.1f}")
            print(f"  -> Epsilon (exploration): {self.exploration_strategy.get_epsilon():.3f}")
            print(f"  -> Nouveaux états découverts: {stats['visited_states']}")
            print(f"  -> Convergence: {convergence_status}")

        return total_reward, steps, success


    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques avec les composants modulaires"""
        stats = self.performance_tracker.get_statistics()
        convergence_info = self.get_convergence_info()
        
        return {
            'q_table_size': self.q_table.get_size(),
            'epsilon': self.exploration_strategy.get_epsilon(),
            'episode_count': stats['episode_count'],
            'best_steps': stats['best_steps'],
            'converged': convergence_info.get('converged', False),
            'visited_states': stats['visited_states']
        }

    def print_convergence_summary(self):
        """Affiche le résumé de convergence avec les composants modulaires"""
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
    """Entraîne l'agent Q-Learning avec l'architecture modulaire"""
    global interrupted
    interrupted = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Création de l'agent avec configuration
    agent = QLearningAgent(config)

    print("=" * 60)
    print("ENTRAÎNEMENT Q-LEARNING - DONATELLOPYZZA (ARCHITECTURE MODULAIRE)")
    print("=" * 60)
    print(f"Environnement: {environment_name}")
    print(f"Configuration de l'agent:")
    print(f"  - Fenêtre de convergence: {agent.config.convergence_window} épisodes")
    print(f"  - Seuil de convergence: {agent.config.convergence_threshold}")
    print(f"  - Limite d'étapes initiale: {agent.config.max_steps} (s'adapte automatiquement)")
    print(f"  - Limite d'épisodes: {max_episodes}")
    print(f"  - Learning rate: {agent.config.learning_rate}")
    print(f"  - Discount factor: {agent.config.discount_factor}")
    print(f"  - Epsilon initial: {agent.config.epsilon}")
    print(f"  - Exploration systématique: {agent.config.systematic_exploration_episodes} épisodes")
    print(f"  - Limite minimale: {agent.config.min_adaptive_steps} étapes")
    print("=" * 60)
    print("Fonctionnalités avancées:")
    print("  [SMART EXPLORER] Exploration systématique")
    print("  [ADAPTIVE LENGTH] Ajustement automatique des limites")
    print("  [CONVERGENCE] Optimisation basée sur les performances")
    print("  [MODULAR ARCHITECTURE] Composants séparés et maintenables")
    print("=" * 60)
    print("[ASTUCE] Appuyez sur Ctrl+C pour arrêter l'entraînement à tout moment")
    print("=" * 60)

    successful_episodes = 0
    episode = 0

    while not agent.convergence_detector.converged and not interrupted and episode < max_episodes:
        episode += 1
        print(f"\nÉPISODE {episode}")
        print("-" * 40)

        game = RLGame(environment_name, gui=show_gui)
        turtle = game.start()

        reward, steps, success = agent.train_episode(game, turtle, show_gui, verbose)

        if success:
            successful_episodes += 1

        if episode % 10 == 0:
            stats = agent.get_statistics()
            success_rate = successful_episodes / episode
            
            print(f"\n[ANALYSE] BILAN APRÈS {episode} ÉPISODES:")
            print(f"  Taux de succès global: {success_rate:.1%}")
            
            # Utilisation des composants modulaires pour les statistiques récentes
            convergence_info = agent.get_convergence_info()
            if 'success_rate' in convergence_info:
                print(f"  Taux de succès récent ({agent.config.convergence_window}): {convergence_info['success_rate']:.1%}")
            
            print(f"  Meilleur chemin trouvé: {stats['best_steps']} étapes")
            print(f"  Epsilon (exploration): {stats['epsilon']:.3f}")
            print(f"  États appris: {stats['q_table_size']}")
            
            if stats['best_steps'] < 50:
                print(f"  [EXCELLENT] Performance: excellente!")
            elif stats['best_steps'] < 100:
                print(f"  [BIEN] Performance: bonne!")
            else:
                print(f"  [APPRENTISSAGE] Performance: en cours...")
            
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
    
    if final_stats['best_steps'] <= 20:
        print("  [EXCELLENT] Performance exceptionnelle!")
    elif final_stats['best_steps'] <= 30:
        print("  [TRÈS BIEN] Performance excellente!")
    elif final_stats['best_steps'] <= 50:
        print("  [BIEN] Performance très bonne!")
    elif final_stats['best_steps'] <= 100:
        print("  [CORRECT] Performance bonne!")
    else:
        print("  [EN COURS] Performance en cours d'amélioration...")
    
    print("=" * 60)
    
    agent.print_convergence_summary()

    return agent




def get_user_config() -> Dict[str, Any]:
    """Collecte la configuration utilisateur avec architecture modulaire"""
    print("=" * 60)
    print("[IA] Q-LEARNING MODULAIRE POUR DONATELLOPYZZA")
    print("=" * 60)
    print("Objectif: Apprendre à naviguer vers la pizza")
    print("Méthode: Apprentissage par renforcement (Q-Learning)")
    print("Architecture: Composants modulaires et maintenables")
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
    
    # Configuration avancée optionnelle
    try:
        advanced = input("Configuration avancée ? (o/n) [défaut: n]: ").lower() == 'o'
        if advanced:
            learning_rate = float(input("Learning rate [défaut: 0.1]: ") or "0.1")
            epsilon = float(input("Epsilon initial [défaut: 0.3]: ") or "0.3")
            max_episodes = int(input("Max épisodes [défaut: 2000]: ") or "2000")
            
            config = AgentConfig(
                learning_rate=learning_rate,
                epsilon=epsilon
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
    """Exécute le pipeline d'entraînement avec architecture modulaire"""
    agent = train_agent(
        environment_name=config['environment_name'],
        show_gui=config['show_gui'],
        verbose=config['verbose'],
        max_episodes=config.get('max_episodes', 2000),
        config=config.get('config', None)
    )

    return agent

def main():
    """Point d'entrée principal avec architecture modulaire"""
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
