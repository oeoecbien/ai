import sys
import os
import random
import time
import signal
from typing import Dict, Tuple, List, Any
from collections import defaultdict

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


class QLearningAgent:
    """Agent Q-Learning pour DonatelloPyzza"""
    
    # Constantes pour les valeurs magiques
    MIN_SUCCESS_RATE = 0.8  # Taux de succès minimum pour convergence
    Q_VALUE_MIN = -1000.0   # Valeur Q minimale
    Q_VALUE_MAX = 1000.0    # Valeur Q maximale

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        convergence_window: int = 20,
        convergence_threshold: float = 0.05,
        max_steps: int = 1000,
        random_seed: int = None,
        # Smart Explorer parameters
        systematic_exploration_episodes: int = 50,
        min_adaptive_steps: int = 50,
        adaptive_margin: float = 0.1
    ):
        """Initialise l'agent Q-Learning"""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.max_steps = max_steps
        self.original_max_steps = max_steps  # Sauvegarde de la valeur initiale

        self.systematic_exploration_episodes = systematic_exploration_episodes
        self.min_adaptive_steps = min_adaptive_steps
        self.adaptive_margin = adaptive_margin
        
        self.best_successful_steps = float('inf')
        self.adaptive_steps_history = []
        self.adaptive_window = 10

        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        self.actions = [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH]

        self.rewards = {
            'pizza_found': 100.0,
            'pizza_touched': 50.0,
            'collision': -10.0,
            'step': -2.0,
            'wall_touched': -5.0,
            'new_state': 3.0,
            'redundant_action': -3.0
        }

        self.episode_count = 0
        self.best_steps = float('inf')
        self.episode_results = []
        self.converged = False
        
        self.visited_states = set()
        
        self.action_history = []
        self.max_history = 3
        
        # Détection de boucles pour timeout intelligent
        self.position_history = []
        self.max_position_history = 20
        
        if random_seed is not None:
            random.seed(random_seed)

    def get_state(self, position: Tuple[int, int], orientation: int) -> Tuple[int, int, int]:
        """Génère un état unique"""
        return (position[0], position[1], orientation)

    def calculate_reward(self, feedback: Feedback, current_state: Tuple, next_state: Tuple, action: Action = None) -> float:
        """Calcule la récompense"""
        base_reward = 0.0
        
        if feedback == Feedback.MOVED_ON_PIZZA:
            base_reward = self.rewards['pizza_found']
        elif feedback == Feedback.TOUCHED_PIZZA:
            base_reward = self.rewards['pizza_touched']
        elif feedback == Feedback.COLLISION:
            base_reward = self.rewards['collision']
        elif feedback == Feedback.TOUCHED_WALL:
            base_reward = self.rewards['wall_touched']
        else:
            base_reward = self.rewards['step']
        
        # Bonus exploration sur l'état atteint (next_state)
        if next_state not in self.visited_states:
            base_reward += self.rewards['new_state']
            self.visited_states.add(next_state)
        
        if action is not None:
            self.action_history.append(action)
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)
            
            if len(self.action_history) >= 2:
                if (self.action_history[-2] == Action.TURN_RIGHT and self.action_history[-1] == Action.TURN_LEFT) or \
                   (self.action_history[-2] == Action.TURN_LEFT and self.action_history[-1] == Action.TURN_RIGHT):
                    base_reward += self.rewards['redundant_action']
                
                if len(self.action_history) >= 3 and \
                   self.action_history[-3] == self.action_history[-2] == self.action_history[-1] and \
                   self.action_history[-1] in [Action.TURN_LEFT, Action.TURN_RIGHT]:
                    base_reward += self.rewards['redundant_action']
        
        return base_reward

    def detect_loop(self, position: Tuple[int, int]) -> bool:
        """Détecte si l'agent tourne en rond"""
        self.position_history.append(position)
        if len(self.position_history) > self.max_position_history:
            self.position_history.pop(0)
        
        # Si on a assez d'historique, vérifier les répétitions
        if len(self.position_history) >= 10:
            # Chercher des patterns répétitifs dans les 10 dernières positions
            recent_positions = self.position_history[-10:]
            # Si la même position apparaît plus de 3 fois dans les 10 dernières
            position_counts = {}
            for pos in recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Si une position apparaît plus de 3 fois, c'est probablement une boucle
            return max(position_counts.values()) > 3
        
        return False

    def adjust_max_steps_for_loops(self, current_steps: int) -> int:
        """Ajuste la limite d'étapes si l'agent tourne en rond"""
        if current_steps > 50 and self.detect_loop(self.position_history[-1] if self.position_history else (0, 0)):
            # Si l'agent tourne en rond après 50 étapes, réduire la limite
            return min(current_steps + 20, self.max_steps)  # Donner encore 20 étapes max
        return self.max_steps

    def systematic_exploration(self, state: Tuple) -> Action:
        """Exploration systématique"""
        if state not in self.visited_states:
            if not hasattr(self, '_exploration_sequence'):
                self._exploration_sequence = [
                    Action.TOUCH,
                    Action.TURN_LEFT,
                    Action.TOUCH,
                    Action.TURN_LEFT,
                    Action.TOUCH,
                    Action.TURN_LEFT,
                    Action.TOUCH,
                    Action.TURN_LEFT,
                    Action.MOVE_FORWARD
                ]
                self._exploration_index = 0
            
            if self._exploration_index < len(self._exploration_sequence):
                action = self._exploration_sequence[self._exploration_index]
                self._exploration_index += 1
                return action
            else:
                self._exploration_index = 0
                return Action.MOVE_FORWARD
        
        return self.choose_action_classic(state)

    def choose_action_classic(self, state: Tuple) -> Action:
        """Sélection d'action classique"""
        exploration_rate = self.epsilon
        if self.best_steps < 20 and self.episode_count > 50:
            exploration_rate *= 0.5
        
        if random.random() < exploration_rate:
            return random.choice(self.actions)

        if state in self.q_table and self.q_table[state]:
            # Tirage aléatoire parmi les meilleures actions
            qdict = self.q_table[state]
            max_q = max(qdict.values())
            best_keys = [k for k, v in qdict.items() if v == max_q]
            return Action(random.choice(best_keys))

        return random.choice(self.actions)

    def choose_action(self, state: Tuple) -> Action:
        """Sélection d'action intelligente"""
        # Si on veut une exploitation stricte (tests): epsilon à 0 court-circuite toute exploration
        if self.epsilon <= 0.0:
            return self.choose_action_classic(state)  # avec epsilon==0, ce sera glouton

        if self.episode_count < self.systematic_exploration_episodes:
            return self.systematic_exploration(state)
        
        return self.choose_action_classic(state)

    def update_q_value(self, state: Tuple, action: Action, reward: float, next_state: Tuple):
        """Met à jour la Q-table"""
        action_key = int(action.value)
        current_q = self.q_table[state][action_key]

        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0

        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error

        if not isinstance(new_q, (int, float)) or new_q != new_q:
            new_q = 0.0
        elif new_q == float('inf') or new_q == float('-inf'):
            new_q = 0.0
        else:
            new_q = max(self.Q_VALUE_MIN, min(self.Q_VALUE_MAX, new_q))

        self.q_table[state][action_key] = new_q

    def update_adaptive_episode_length(self, steps: int, success: bool):
        """Met à jour la limite d'épisodes"""
        if success:
            if steps < self.best_successful_steps:
                self.best_successful_steps = steps
                # Marge plus généreuse pour permettre l'exploration
                new_max_steps = max(
                    int(steps * 2.0),  # Double du meilleur score pour exploration
                    self.min_adaptive_steps
                )
                # Ne réduire max_steps que si on a assez d'épisodes d'exploration
                if self.episode_count > 100:  # Après 100 épisodes d'exploration
                    self.max_steps = new_max_steps
            
            self.adaptive_steps_history.append(steps)
            if len(self.adaptive_steps_history) > self.adaptive_window:
                self.adaptive_steps_history.pop(0)
            
            # Ajustement plus conservateur
            if len(self.adaptive_steps_history) >= 10 and self.episode_count > 200:
                recent_avg = sum(self.adaptive_steps_history[-10:]) / 10
                if recent_avg < self.max_steps * 0.6:  # Seuil plus strict
                    new_limit = max(int(recent_avg * 1.5), self.min_adaptive_steps)
                    if new_limit < self.max_steps:
                        self.max_steps = new_limit
        else:
            # Si échec, augmenter progressivement la limite pour permettre plus d'exploration
            if self.episode_count < 50:  # Phase d'exploration intensive
                self.max_steps = min(self.max_steps * 1.1, 2000)  # Augmenter jusqu'à 2000 max
            elif self.episode_count < 200:  # Phase d'apprentissage
                self.max_steps = min(self.max_steps * 1.05, 1500)  # Augmenter jusqu'à 1500 max

    def decay_epsilon(self):
        """Réduit le taux d'exploration"""
        # Garder un epsilon plus élevé plus longtemps pour l'exploration
        if self.episode_count < 200:  # Phase d'exploration intensive
            self.epsilon = max(0.1, self.epsilon * 0.999)  # Décroissance très lente
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def check_convergence(self) -> bool:
        """Vérifie la convergence"""
        if len(self.episode_results) < self.convergence_window:
            return False
        
        recent_results = self.episode_results[-self.convergence_window:]
        
        stats = self._calculate_performance_stats(recent_results)
        
        recent_successes = sum(1 for result in recent_results if result[2])
        success_rate = recent_successes / len(recent_results)
        
        # Vérifier si l'exploration est suffisante
        exploration_sufficient = len(self.visited_states) > 50 or self.episode_count > 300
        
        convergence_criteria = [
            stats['cv'] < self.convergence_threshold,
            self.epsilon <= self.epsilon_min,
            len(self.episode_results) >= self.convergence_window,
            success_rate >= self.MIN_SUCCESS_RATE,
            exploration_sufficient  # Ajouter le critère d'exploration
        ]
        
        converged = all(convergence_criteria)
        
        if converged:
            self.converged = True
            
        return converged

    def _execute_step(self, game: RLGame, turtle, current_state: Tuple, steps: int, verbose: bool) -> Tuple[float, Tuple, bool]:
        """Exécute un pas d'apprentissage"""
        action = self.choose_action(current_state)
        feedback, _ = turtle.execute(action)

        new_position = game.getTurtlePosition(turtle)
        new_orientation = game.getTurtleOrientation(turtle)
        next_state = self.get_state(new_position, new_orientation)

        reward = self.calculate_reward(feedback, current_state, next_state, action)
        
        # Détection de boucles pour timeout intelligent
        is_looping = self.detect_loop(new_position)
        if is_looping:
            # Pénalité pour les boucles
            reward += self.rewards['redundant_action'] * 2

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
            
            print(f"Episode {self.episode_count + 1}, étape {steps}: {action_desc} -> "
                  f"position {new_position}, récompense {reward:.1f}, {feedback_desc}")

        return reward, next_state, game.isWon(prnt=False)

    def _update_statistics(self, total_reward: float, steps: int, success: bool):
        """Met à jour les statistiques"""
        if success and steps < self.best_steps:
            self.best_steps = steps

        self.episode_results.append((total_reward, steps, success))
        
        self.update_adaptive_episode_length(steps, success)
        
        self.decay_epsilon()
        
        self.action_history.clear()
        self.position_history.clear()
        

    def _calculate_performance_stats(self, results: List[Tuple]) -> Dict[str, float]:
        """Calcule les statistiques"""
        scores = [result[1] for result in results]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        coefficient_variation = std_dev / mean_score if mean_score > 0 else float('inf')
        
        return {
            'mean': mean_score,
            'std_dev': std_dev,
            'variance': variance,
            'cv': coefficient_variation
        }

    def get_convergence_info(self) -> Dict[str, Any]:
        """Retourne les informations de convergence"""
        if len(self.episode_results) < self.convergence_window:
            return {
                'converged': False,
                'episodes_needed': self.convergence_window - len(self.episode_results),
                'reason': 'Pas assez d\'épisodes',
                'epsilon': self.epsilon
            }
        
        recent_results = self.episode_results[-self.convergence_window:]
        stats = self._calculate_performance_stats(recent_results)
        
        return {
            'converged': self.converged,
            'coefficient_variation': stats['cv'],
            'threshold': self.convergence_threshold,
            'mean_performance': stats['mean'],
            'std_performance': stats['std_dev'],
            'epsilon': self.epsilon,
            'episodes_analyzed': len(recent_results)
        }

    def train_episode(self, game: RLGame, turtle, show_gui: bool = True, verbose: bool = True, training_mode: bool = True) -> Tuple[float, int, bool]:
        """Exécute un épisode"""
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
            self.episode_count += 1
            
            converged = self.check_convergence()
        else:
            converged = False

        if verbose:
            new_states_count = len(self.visited_states)
            convergence_status = "convergé" if converged else "en cours"
            
            status_emoji = "✓" if success else "✗"
            status_text = "Réussi!" if success else "Échoué"
            
            print(f"\nEpisode {self.episode_count}: {status_emoji} {status_text} en {steps} étapes")
            print(f"  -> Récompense: {total_reward:.1f}")
            print(f"  -> Epsilon (exploration): {self.epsilon:.3f}")
            print(f"  -> Nouveaux états découverts: {new_states_count}")
            print(f"  -> Convergence: {convergence_status}")

        return total_reward, steps, success


    def get_statistics(self) -> Dict[str, any]:
        """Retourne les statistiques"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'best_steps': self.best_steps,
            'converged': self.converged
        }

    def print_convergence_summary(self):
        """Affiche le résumé de convergence"""
        convergence_info = self.get_convergence_info()
        
        print("\n" + "=" * 50)
        print("ANALYSE DE CONVERGENCE")
        print("=" * 50)
        
        if convergence_info['converged']:
            print("[SUCCÈS] CONVERGENCE ATTEINTE!")
            print(f"   - Stabilité: {convergence_info['coefficient_variation']:.4f}")
            print(f"   - Seuil requis: {convergence_info['threshold']}")
            print(f"   - Performance moyenne: {convergence_info['mean_performance']:.1f} étapes")
            print(f"   - Épisodes analysés: {convergence_info['episodes_analyzed']}")
        else:
            print("[APPRENTISSAGE] Convergence en cours...")
            if 'episodes_needed' in convergence_info:
                print(f"   - Épisodes nécessaires: {convergence_info['episodes_needed']}")
            else:
                print(f"   - Stabilité actuelle: {convergence_info['coefficient_variation']:.4f}")
                print(f"   - Seuil requis: {convergence_info['threshold']}")
                print(f"   - Performance moyenne: {convergence_info['mean_performance']:.1f} étapes")
                print(f"   - Épisodes analysés: {convergence_info['episodes_analyzed']}")
        
        print(f"   - Taux d'exploration: {convergence_info['epsilon']:.3f}")
        print("=" * 50)


def train_agent(
    environment_name: str = "maze",
    show_gui: bool = True,
    verbose: bool = True,
    max_episodes: int = 2000
) -> QLearningAgent:
    """Entraîne l'agent Q-Learning"""
    global interrupted
    interrupted = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    agent = QLearningAgent()
    
    agent._verbose_adaptive = verbose

    print("=" * 60)
    print("ENTRAÎNEMENT Q-LEARNING - DONATELLOPYZZA")
    print("=" * 60)
    print(f"Environnement: {environment_name}")
    print(f"Configuration de l'agent:")
    print(f"  - Fenêtre de convergence: {agent.convergence_window} épisodes")
    print(f"  - Seuil de convergence: {agent.convergence_threshold}")
    print(f"  - Limite d'étapes initiale: {agent.max_steps} (s'adapte automatiquement)")
    print(f"  - Limite d'épisodes: {max_episodes}")
    print(f"  - Learning rate: {agent.learning_rate}")
    print(f"  - Discount factor: {agent.discount_factor}")
    print(f"  - Epsilon initial: {agent.epsilon}")
    print(f"  - Exploration systématique: {agent.systematic_exploration_episodes} épisodes")
    print(f"  - Limite minimale: {agent.min_adaptive_steps} étapes")
    print("=" * 60)
    print("Fonctionnalités avancées:")
    print("  [SMART EXPLORER] Exploration systématique")
    print("  [ADAPTIVE LENGTH] Ajustement automatique des limites")
    print("  [CONVERGENCE] Optimisation basée sur les performances")
    print("=" * 60)
    print("[ASTUCE] Appuyez sur Ctrl+C pour arrêter l'entraînement à tout moment")
    print("=" * 60)

    successful_episodes = 0
    episode = 0

    while not agent.converged and not interrupted and episode < max_episodes:
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
            
            if len(agent.episode_results) >= agent.convergence_window:
                recent_results = agent.episode_results[-agent.convergence_window:]
                recent_successes = sum(1 for result in recent_results if result[2])
                recent_success_rate = recent_successes / len(recent_results)
                print(f"  Taux de succès récent ({agent.convergence_window}): {recent_success_rate:.1%}")
            
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
    
    if convergence_info['converged']:
        print(f"  [SUCCÈS] Convergence: atteinte!")
        print(f"  Performance stable: {convergence_info['mean_performance']:.1f} ± {convergence_info['std_performance']:.1f} étapes")
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
    """Collecte la configuration utilisateur"""
    print("=" * 60)
    print("[IA] Q-LEARNING SIMPLIFIÉ POUR DONATELLOPYZZA")
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
    
    convergence_window = 20
    convergence_threshold = 0.05

    return {
        'environment_name': environment_name,
        'show_gui': show_gui,
        'verbose': verbose
    }

def run_training_pipeline(config: Dict[str, Any]) -> QLearningAgent:
    """Exécute le pipeline d'entraînement"""
    agent = train_agent(
        environment_name=config['environment_name'],
        show_gui=config['show_gui'],
        verbose=config['verbose']
    )

    return agent

def main():
    """Point d'entrée principal"""
    config = get_user_config()
    agent = run_training_pipeline(config)
    
    print("\n[TERMINÉ] Programme terminé!")
    print(f"Agent final avec {agent.get_statistics()['q_table_size']} états appris")
    print(f"Chemin optimal: {agent.get_statistics()['best_steps']} étapes")
    print("L'agent a fini son apprentissage! [APPRENTISSAGE TERMINÉ]")


if __name__ == "__main__":
    main()
