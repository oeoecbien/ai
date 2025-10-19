import sys
import os
import random
import time
import signal
import logging
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
    min_success_rate: float = 0.8
    q_value_min: float = -1000.0
    q_value_max: float = 1000.0


# Clé d'état MDP utilisée pour la Q-table
StateKey = Tuple[Tuple[int, int], int, bool]

@dataclass(frozen=True)
class State:
    """État MDP immuable pour la Q-table"""
    position: Tuple[int, int]
    orientation: int
    has_touched_wall: bool = False
    
    # Contexte non-markovien (non utilisé pour la Q-table)
    steps_since_pizza: int = 0
    previous_actions: Tuple[int, ...] = field(default_factory=tuple, compare=False, hash=False)
    
    def key(self) -> StateKey:
        """Retourne la clé d'état utilisée pour la Q-table"""
        return (self.position, self.orientation, self.has_touched_wall)


class RewardSystem:
    """Système de récompenses markovien avec bonus d'exploration"""
    
    def __init__(self, step_penalty: float = -1.0, wall_penalty: float = -5.0, 
                 pizza_reward: float = 100.0, touch_cost: float = -0.5, 
                 exploration_bonus: float = 0.5):
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.pizza_reward = pizza_reward
        self.touch_cost = touch_cost
        self.exploration_bonus = exploration_bonus
        self.visit_counts = defaultdict(int)
    
    def calculate_reward(self, feedback: Feedback, current_state: State, 
                        next_state: State, action: Action) -> float:
        """Calcule la récompense markovienne avec bonus d'exploration"""
        # Récompense de base markovienne
        reward = self.step_penalty
        
        if feedback == Feedback.COLLISION or feedback == Feedback.TOUCHED_WALL:
            reward += self.wall_penalty
        elif feedback == Feedback.MOVED_ON_PIZZA or feedback == Feedback.TOUCHED_PIZZA:
            reward += self.pizza_reward
        
        if action == Action.TOUCH:
            reward += self.touch_cost
        
        # Bonus d'exploration basé sur le comptage (count-based)
        state_key = next_state.key()
        self.visit_counts[state_key] += 1
        reward += self.exploration_bonus / (self.visit_counts[state_key] ** 0.5)
        
        return reward


class QTable:
    """Q-table avec clés d'état cohérentes"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.table: Dict[StateKey, Dict[int, float]] = defaultdict(dict)
    
    def get(self, state: State, action: Action) -> float:
        """Récupère la valeur Q pour un état-action"""
        return self.table[state.key()].get(action.value, 0.0)
    
    def update(self, state: State, action: Action, reward: float, next_state: State):
        """Met à jour la valeur Q selon la formule TD correcte"""
        q_sa = self.get(state, action)
        next_qs = self.table.get(next_state.key(), {})
        max_next = max(next_qs.values()) if next_qs else 0.0
        
        # Formule TD: Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]
        target = reward + self.config.discount_factor * max_next
        new_q = (1 - self.config.learning_rate) * q_sa + self.config.learning_rate * target
        
        # Clipping pour éviter les valeurs aberrantes
        new_q = max(self.config.q_value_min, min(self.config.q_value_max, new_q))
        self.table[state.key()][action.value] = new_q
    
    def best_action(self, state: State) -> Action:
        """Retourne la meilleure action pour un état"""
        qdict = self.table.get(state.key(), {})
        if not qdict:
            return random.choice([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH])
        
        max_q = max(qdict.values())
        best_actions = [Action(k) for k, v in qdict.items() if v == max_q]
        return random.choice(best_actions)
    
    def size(self) -> int:
        """Retourne la taille de la Q-table"""
        return len(self.table)


class ExplorationStrategy:
    """Stratégie d'exploration epsilon-greedy"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
    
    def choose_action(self, state: State, q_table: QTable) -> Action:
        """Choisit une action selon la stratégie epsilon-greedy"""
        if random.random() < self.epsilon:
            # Exploration: action aléatoire
            return random.choice([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH])
        else:
            # Exploitation: meilleure action selon Q-table
            return q_table.best_action(state)
    
    def decay_epsilon(self):
        """Réduit le taux d'exploration"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self) -> float:
        """Retourne la valeur actuelle d'epsilon"""
        return self.epsilon


class ConvergenceDetector:
    """Détecteur de convergence avec API claire"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.history = deque(maxlen=config.convergence_window)
        self.converged = False
    
    def update(self, steps: int, success: bool, reward: float):
        """Met à jour l'historique des performances"""
        self.history.append((steps, success, reward))
    
    def is_converged(self, epsilon: float) -> bool:
        """Vérifie si l'agent a convergé avec critères explicites"""
        if len(self.history) < self.config.convergence_window:
            return False
        
        # Extraction des données
        steps = [s for s, _, _ in self.history]
        successes = [1 if ok else 0 for _, ok, _ in self.history]
        
        # Calcul des statistiques
        mean_steps = sum(steps) / len(steps)
        success_rate = sum(successes) / len(successes)
        variance = sum((s - mean_steps) ** 2 for s in steps) / len(steps)
        std_dev = variance ** 0.5
        cv = std_dev / mean_steps if mean_steps > 0 else float('inf')
        
        # Critères de convergence explicites
        epsilon_low = epsilon <= self.config.epsilon_min + 1e-6
        success_high = success_rate >= self.config.min_success_rate
        variance_low = cv <= self.config.convergence_threshold
        
        self.converged = epsilon_low and success_high and variance_low
        return self.converged
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Retourne les informations de convergence"""
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


class EnvironmentAdapter:
    """Adaptateur pour isoler les appels à RLGame"""
    
    def __init__(self, environment_name: str, show_gui: bool = True):
        self.environment_name = environment_name
        self.show_gui = show_gui
        self.game = None
        self.turtle = None
    
    def reset(self) -> State:
        """Réinitialise l'environnement et retourne l'état initial"""
        self.game = RLGame(self.environment_name, gui=self.show_gui)
        self.turtle = self.game.start()
        
        # TODO: Adapter selon les vraies méthodes de RLGame
        position = self.game.getTurtlePosition(self.turtle)
        orientation = self.game.getTurtleOrientation(self.turtle)
        
        return State(position=position, orientation=orientation)
    
    def step(self, action: Action) -> Tuple[State, Feedback, bool]:
        """Exécute une action et retourne le nouvel état, feedback et done"""
        # TODO: Adapter selon les vraies méthodes de RLGame
        feedback, _ = self.turtle.execute(action)
        
        new_position = self.game.getTurtlePosition(self.turtle)
        new_orientation = self.game.getTurtleOrientation(self.turtle)
        has_touched_wall = feedback == Feedback.TOUCHED_WALL
        
        next_state = State(
            position=new_position,
            orientation=new_orientation,
            has_touched_wall=has_touched_wall
        )
        
        # TODO: Adapter selon les vraies méthodes de RLGame
        done = self.game.isWon(prnt=False)
        
        return next_state, feedback, done
    
    def is_won(self) -> bool:
        """Vérifie si l'objectif est atteint"""
        # TODO: Adapter selon les vraies méthodes de RLGame
        return self.game.isWon(prnt=False) if self.game else False


class QLearningAgent:
    """Agent Q-Learning avec architecture modulaire"""
    
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
    
    def train_episode(self, env_adapter: EnvironmentAdapter, verbose: bool = True) -> Tuple[float, int, bool]:
        """Exécute un épisode d'entraînement"""
        total_reward = 0.0
        steps = 0
        success = False
        
        # État initial
        current_state = env_adapter.reset()
        
        while steps < self.max_steps:
            steps += 1
            
            # Sélection d'action
            action = self.exploration_strategy.choose_action(current_state, self.q_table)
            
            # Exécution de l'action
            next_state, feedback, done = env_adapter.step(action)
            
            # Calcul de la récompense
            reward = self.reward_system.calculate_reward(feedback, current_state, next_state, action)
            
            # Mise à jour de la Q-table
            self.q_table.update(current_state, action, reward, next_state)
            
            # Mise à jour des statistiques
            self.performance_tracker.action_history.append(action)
            self.performance_tracker.position_history.append(next_state.position)
            
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
        
        # Mise à jour des statistiques finales
        self._update_statistics(total_reward, steps, success)
        
        return total_reward, steps, success
    
    def _update_statistics(self, total_reward: float, steps: int, success: bool):
        """Met à jour les statistiques avec les composants modulaires"""
        # Mise à jour du tracker de performance
        self.performance_tracker.update_episode(steps, success, total_reward)
        
        # Mise à jour du détecteur de convergence
        self.convergence_detector.update(steps, success, total_reward)
        
        # Décroissance d'epsilon
        self.exploration_strategy.decay_epsilon()
        
        # Nettoyage des données temporaires
        self.performance_tracker.clear_episode_data()
    
    def check_convergence(self) -> bool:
        """Vérifie la convergence avec le détecteur modulaire"""
        return self.convergence_detector.is_converged(self.exploration_strategy.get_epsilon())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques avec les composants modulaires"""
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
        """Retourne les informations de convergence avec le détecteur modulaire"""
        return self.convergence_detector.get_convergence_info()
    
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
    print("=" * 60)
    print("Fonctionnalités avancées:")
    print("  [SMART EXPLORER] Exploration epsilon-greedy")
    print("  [ADAPTIVE LENGTH] Ajustement automatique des limites")
    print("  [CONVERGENCE] Optimisation basée sur les performances")
    print("  [MODULAR ARCHITECTURE] Composants séparés et maintenables")
    print("=" * 60)
    print("[ASTUCE] Appuyez sur Ctrl+C pour arrêter l'entraînement à tout moment")
    print("=" * 60)

    successful_episodes = 0
    episode = 0
    
    # Création de l'adaptateur d'environnement
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