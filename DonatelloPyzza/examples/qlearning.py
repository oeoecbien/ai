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
    """
    Agent Q-Learning simplifié pour DonatelloPyzza avec convergence
    
    Fonctionnalités:
        - Apprentissage par renforcement avec Q-table
        - Équation de Bellman pour mise à jour des valeurs Q
        - Exploration vs exploitation (epsilon-greedy)
        - Système de récompenses simplifié
        - Convergence automatique basée sur la stabilité des performances
    """
    
    # Constantes pour les valeurs magiques
    MIN_SUCCESS_RATE = 0.8  # Taux de succès minimum pour convergence
    Q_VALUE_MIN = -1000.0   # Valeur Q minimale
    Q_VALUE_MAX = 1000.0    # Valeur Q maximale
    EFFICIENT_STEPS_THRESHOLD = 30  # Seuil pour considérer un chemin efficace

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
        """
        Initialise l'agent Q-Learning avec critères de convergence et Smart Explorer
        
        Args:
            learning_rate: Taux d'apprentissage (α)
            discount_factor: Facteur de réduction (γ)
            epsilon: Taux d'exploration initial
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
            convergence_window: Nombre d'épisodes pour évaluer la convergence
            convergence_threshold: Seuil de variation pour considérer la convergence
            max_steps: Nombre maximum d'étapes par épisode pour éviter les boucles infinies
            systematic_exploration_episodes: Nombre d'épisodes pour exploration systématique
            min_adaptive_steps: Nombre minimum d'étapes pour l'adaptation
            adaptive_margin: Marge pour accepter une légère augmentation temporaire
        """
        # Hyperparamètres Q-Learning
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Paramètres de convergence
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.max_steps = max_steps
        self.original_max_steps = max_steps  # Sauvegarde de la valeur initiale

        # Smart Explorer parameters
        self.systematic_exploration_episodes = systematic_exploration_episodes
        self.min_adaptive_steps = min_adaptive_steps
        self.adaptive_margin = adaptive_margin
        
        # Adaptive Episode Length
        self.best_successful_steps = float('inf')
        self.adaptive_steps_history = []
        self.adaptive_window = 10  # Nombre d'épisodes pour évaluer l'adaptation

        # Q-table: état -> {action: valeur_Q}
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Actions disponibles
        self.actions = [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH]

        # Configuration des récompenses normalisées
        self.rewards = {
            'pizza_found': 100.0,      # Récompense principale
            'pizza_touched': 50.0,     # Récompense pour toucher la pizza
            'collision': -10.0,         # Pénalité pour collision
            'step': -2.0,               # Coût par étape (augmenté pour décourager les mouvements inutiles)
            'wall_touched': -5.0,      # Pénalité pour toucher un mur
            'new_state': 3.0,          # Bonus pour découvrir un nouvel état (réduit)
            'redundant_action': -3.0   # Pénalité pour actions redondantes
        }

        # Statistiques et suivi de convergence
        self.episode_count = 0
        self.best_steps = float('inf')
        self.episode_results = []  # Historique des résultats pour convergence
        self.converged = False
        
        # Suivi des états visités pour bonus d'exploration
        self.visited_states = set()
        
        # Position de la pizza (sera détectée dynamiquement)
        self.pizza_position = None
        
        # Historique des actions pour détecter les mouvements redondants
        self.action_history = []
        self.max_history = 3  # Garder les 3 dernières actions
        
        # Initialisation du seed aléatoire pour reproductibilité
        if random_seed is not None:
            random.seed(random_seed)

    def get_state(self, position: Tuple[int, int], orientation: int, feedback: Feedback = None) -> Tuple[int, int, int]:
        """
        Génère une représentation d'état unique sous forme de tuple hashable
        
        Args:
            position: Position actuelle (x, y)
            orientation: Orientation actuelle (0-3)
            feedback: Dernier feedback reçu (non utilisé pour simplifier l'espace d'états)
            
        Returns:
            Tuple représentant l'état (x, y, orientation)
        """
        # État simplifié sans feedback pour réduire l'espace d'états
        # Le feedback est une conséquence de l'action, pas une propriété de l'état
        return (position[0], position[1], orientation)

    def calculate_reward(self, feedback: Feedback, state: Tuple, position: Tuple[int, int], action: Action = None) -> float:
        """
        Calcule la récompense basée sur le feedback avec bonus d'exploration et détection de redondance
        
        Args:
            feedback: Feedback reçu de l'environnement
            state: État actuel
            position: Position actuelle
            action: Action effectuée
            
        Returns:
            Valeur de la récompense
        """
        base_reward = 0.0
        
        # Récompenses de base
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
        
        # Bonus d'exploration pour nouvel état
        if state not in self.visited_states:
            base_reward += self.rewards['new_state']
            self.visited_states.add(state)
        
        # Détection d'actions redondantes
        if action is not None:
            self.action_history.append(action)
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)
            
            # Détecter les patterns redondants
            if len(self.action_history) >= 2:
                # TURN_RIGHT suivi de TURN_LEFT (ou vice versa) = mouvement inutile
                if (self.action_history[-2] == Action.TURN_RIGHT and self.action_history[-1] == Action.TURN_LEFT) or \
                   (self.action_history[-2] == Action.TURN_LEFT and self.action_history[-1] == Action.TURN_RIGHT):
                    base_reward += self.rewards['redundant_action']
                
                # Même action répétée plusieurs fois sans mouvement
                if len(self.action_history) >= 3 and \
                   self.action_history[-3] == self.action_history[-2] == self.action_history[-1] and \
                   self.action_history[-1] in [Action.TURN_LEFT, Action.TURN_RIGHT]:
                    base_reward += self.rewards['redundant_action']
        
        return base_reward

    def systematic_exploration(self, state: Tuple) -> Action:
        """
        Exploration systématique : tour d'horizon complet avant de se déplacer
        
        Args:
            state: État actuel (position, orientation)
            
        Returns:
            Action d'exploration systématique
        """
        # Si nouvel état, faire un tour d'horizon complet
        if state not in self.visited_states:
            # Séquence d'exploration : TOUCH pour détecter les obstacles
            # puis tourner pour explorer toutes les directions
            if not hasattr(self, '_exploration_sequence'):
                self._exploration_sequence = [
                    Action.TOUCH,      # Détecter obstacle devant
                    Action.TURN_LEFT,  # Regarder à gauche
                    Action.TOUCH,      # Détecter obstacle à gauche
                    Action.TURN_LEFT,  # Regarder en arrière
                    Action.TOUCH,      # Détecter obstacle derrière
                    Action.TURN_LEFT,   # Regarder à droite
                    Action.TOUCH,      # Détecter obstacle à droite
                    Action.TURN_LEFT,  # Retourner face avant
                    Action.MOVE_FORWARD # Avancer d'un pas
                ]
                self._exploration_index = 0
            
            # Exécuter la séquence d'exploration
            if self._exploration_index < len(self._exploration_sequence):
                action = self._exploration_sequence[self._exploration_index]
                self._exploration_index += 1
                return action
            else:
                # Séquence terminée, réinitialiser pour le prochain état
                self._exploration_index = 0
                return Action.MOVE_FORWARD
        
        # Si état déjà visité, utiliser Q-Learning classique
        return self.choose_action_classic(state)

    def choose_action_classic(self, state: Tuple) -> Action:
        """
        Sélection d'action classique (Q-Learning standard)
        
        Args:
            state: État actuel
            
        Returns:
            Action à exécuter
        """
        # Exploration adaptative: moins d'exploration si l'agent performe bien
        exploration_rate = self.epsilon
        if self.best_steps < 20 and self.episode_count > 50:  # Si l'agent a déjà trouvé un bon chemin
            exploration_rate *= 0.5  # Réduire l'exploration de moitié
        
        # Exploration: action aléatoire
        if random.random() < exploration_rate:
            return random.choice(self.actions)

        # Exploitation: meilleure action connue
        if state in self.q_table and self.q_table[state]:
            best_action_key = max(self.q_table[state], key=self.q_table[state].get)
            # Conversion directe de la clé vers l'Action
            return Action(best_action_key)

        # Si aucune information, action aléatoire
        return random.choice(self.actions)

    def choose_action(self, state: Tuple) -> Action:
        """
        Smart Explorer : Sélection d'action intelligente avec exploration systématique
        
        Args:
            state: État actuel
            
        Returns:
            Action à exécuter
        """
        # Phase 1: Exploration systématique pour les premiers épisodes
        if self.episode_count < self.systematic_exploration_episodes:
            return self.systematic_exploration(state)
        
        # Phase 2: Q-Learning classique avec exploration adaptative
        else:
            return self.choose_action_classic(state)

    def update_q_value(self, state: Tuple, action: Action, reward: float, next_state: Tuple):
        """
        Met à jour la Q-table selon l'équation de Bellman avec validation
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: Nouvel état
        """
        action_key = int(action.value)
        current_q = self.q_table[state][action_key]

        # Calcul de la valeur maximale du prochain état
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0

        # Équation de Bellman
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error

        # Validation et clipping des valeurs Q
        if not isinstance(new_q, (int, float)) or new_q != new_q:  # NaN check
            new_q = 0.0
        elif new_q == float('inf') or new_q == float('-inf'):
            new_q = 0.0
        else:
            # Clipping dans une plage raisonnable
            new_q = max(self.Q_VALUE_MIN, min(self.Q_VALUE_MAX, new_q))

        self.q_table[state][action_key] = new_q

    def update_adaptive_episode_length(self, steps: int, success: bool):
        """
        Met à jour la limite d'épisodes adaptative basée sur les performances
        
        Args:
            steps: Nombre d'étapes de l'épisode
            success: Si l'épisode a réussi
        """
        if success:
            # Mettre à jour le meilleur score réussi
            if steps < self.best_successful_steps:
                self.best_successful_steps = steps
                # Ajuster max_steps avec une marge de sécurité
                new_max_steps = max(
                    int(steps * (1 + self.adaptive_margin)),  # Marge de 10% par défaut
                    self.min_adaptive_steps  # Minimum absolu
                )
                self.max_steps = new_max_steps
                
                if hasattr(self, '_verbose_adaptive') and self._verbose_adaptive:
                    print(f"  🔄 Adaptive Episode Length: max_steps ajusté à {self.max_steps} "
                          f"(meilleur score: {steps} étapes)")
            
            # Enregistrer dans l'historique pour analyse
            self.adaptive_steps_history.append(steps)
            if len(self.adaptive_steps_history) > self.adaptive_window:
                self.adaptive_steps_history.pop(0)
            
            # Vérifier si on peut optimiser davantage
            if len(self.adaptive_steps_history) >= 5:
                recent_avg = sum(self.adaptive_steps_history[-5:]) / 5
                if recent_avg < self.max_steps * 0.8:  # Si on fait 20% mieux que la limite
                    new_limit = max(int(recent_avg * 1.1), self.min_adaptive_steps)
                    if new_limit < self.max_steps:
                        self.max_steps = new_limit
                        if hasattr(self, '_verbose_adaptive') and self._verbose_adaptive:
                            print(f"  📈 Optimisation: max_steps réduit à {self.max_steps} "
                                  f"(moyenne récente: {recent_avg:.1f})")

    def decay_epsilon(self):
        """Réduit le taux d'exploration de manière standard"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def check_convergence(self) -> bool:
        """
        Vérifie si l'agent a convergé basé sur la stabilité des performances
        
        Returns:
            True si l'agent a convergé, False sinon
        """
        # Pas assez de données pour évaluer la convergence
        if len(self.episode_results) < self.convergence_window:
            return False
        
        # Récupère les derniers résultats
        recent_results = self.episode_results[-self.convergence_window:]
        
        # Calcule les statistiques de performance
        stats = self._calculate_performance_stats(recent_results)
        
        # Calcul du taux de succès récent
        recent_successes = sum(1 for result in recent_results if result[2])  # result[2] = success
        success_rate = recent_successes / len(recent_results)
        
        # Critères de convergence améliorés
        convergence_criteria = [
            stats['cv'] < self.convergence_threshold,  # Faible variation des performances
            self.epsilon <= self.epsilon_min,  # Exploration minimale atteinte
            len(self.episode_results) >= self.convergence_window,  # Suffisamment d'épisodes
            success_rate >= self.MIN_SUCCESS_RATE  # Taux de succès minimum
        ]
        
        # Convergence si tous les critères sont remplis
        converged = all(convergence_criteria)
        
        if converged:
            self.converged = True
            
        return converged

    def _execute_step(self, game: RLGame, turtle, current_state: Tuple, steps: int, verbose: bool) -> Tuple[float, Tuple, bool]:
        """
        Exécute un pas d'apprentissage
        
        Args:
            game: Instance du jeu
            turtle: Tortue à contrôler
            current_state: État actuel
            steps: Numéro de l'étape actuelle (pour l'affichage)
            verbose: Affichage détaillé
            
        Returns:
            Tuple (récompense, nouvel état, succès)
        """
        # Sélection et exécution de l'action
        action = self.choose_action(current_state)
        feedback, _ = turtle.execute(action)

        # Nouvel état
        new_position = game.getTurtlePosition(turtle)
        new_orientation = game.getTurtleOrientation(turtle)
        next_state = self.get_state(new_position, new_orientation, feedback)

        # Calcul de la récompense
        reward = self.calculate_reward(feedback, current_state, new_position, action)
        
        # Détection de la position de la pizza
        if feedback == Feedback.MOVED_ON_PIZZA and self.pizza_position is None:
            self.pizza_position = new_position

        # Mise à jour de la Q-table
        self.update_q_value(current_state, action, reward, next_state)

        # Affichage
        if verbose:
            print(f"Episode {self.episode_count + 1}, étape {steps}: {str(action)} -> "
                  f"pos {new_position}, récompense {reward:.1f}, {str(feedback)}")

        return reward, next_state, game.isWon(prnt=False)

    def _update_statistics(self, total_reward: float, steps: int, success: bool):
        """
        Met à jour les statistiques de l'agent avec Adaptive Episode Length
        
        Args:
            total_reward: Récompense totale de l'épisode
            steps: Nombre d'étapes
            success: Succès de l'épisode
        """
        # Mise à jour des meilleures performances
        if success and steps < self.best_steps:
            self.best_steps = steps

        # Enregistrement des résultats pour l'analyse de convergence
        self.episode_results.append((total_reward, steps, success))
        
        # Adaptive Episode Length: ajuster max_steps basé sur les performances
        self.update_adaptive_episode_length(steps, success)
        
        # Décroissance de l'exploration
        self.decay_epsilon()
        
        # Réinitialisation de l'historique des actions pour le prochain épisode
        self.action_history.clear()
        
        # visited_states n'est plus réinitialisé pour récompenser la vraie découverte

    def _calculate_performance_stats(self, results: List[Tuple]) -> Dict[str, float]:
        """
        Calcule les statistiques de performance
        
        Args:
            results: Liste de tuples (reward, steps, success)
            
        Returns:
            Dictionnaire avec mean, std_dev, variance, cv
        """
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
        """
        Retourne les informations sur l'état de convergence
        
        Returns:
            Dictionnaire avec les informations de convergence
        """
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
        """
        Exécute un épisode (entraînement ou test selon training_mode)
        
        Args:
            game: Instance du jeu
            turtle: Tortue à contrôler
            show_gui: Afficher l'interface graphique
            verbose: Afficher les détails
            training_mode: Si True, met à jour les statistiques et vérifie la convergence
            
        Returns:
            Tuple (récompense totale, nombre d'étapes, succès)
        """
        total_reward = 0.0
        steps = 0

        # État initial
        current_position = game.getTurtlePosition(turtle)
        current_orientation = game.getTurtleOrientation(turtle)
        current_state = self.get_state(current_position, current_orientation)

        # Boucle principale - Continue jusqu'à la victoire ou limite d'étapes
        while steps < self.max_steps:
            steps += 1

            # Exécution d'un pas d'apprentissage
            reward, next_state, success = self._execute_step(game, turtle, current_state, steps, verbose)
            total_reward += reward

            # Délai pour la visualisation
            if show_gui:
                time.sleep(0.01)

            # Vérification de la victoire
            if success:
                if verbose:
                    print(f"\nVictoire! Pizza trouvée en {steps} étapes (récompense: {total_reward:.1f})")
                break

            # Transition vers l'état suivant
            current_state = next_state
        
        # Vérification si l'épisode a été interrompu par la limite d'étapes
        if steps >= self.max_steps and not success:
            if verbose:
                print(f"\nLimite d'étapes atteinte ({self.max_steps}), épisode interrompu (récompense: {total_reward:.1f})")

        # Mise à jour des statistiques seulement en mode entraînement
        if training_mode:
            self._update_statistics(total_reward, steps, success)
            self.episode_count += 1
            
            # Vérification de la convergence
            converged = self.check_convergence()
        else:
            converged = False

        if verbose:
            new_states_count = len(self.visited_states)
            efficiency = "efficace" if success and steps <= self.EFFICIENT_STEPS_THRESHOLD else "normal"
            convergence_status = "convergé" if converged else "en cours"
            
            # Affichage des informations adaptatives
            adaptive_info = ""
            if hasattr(self, 'best_successful_steps') and self.best_successful_steps != float('inf'):
                adaptive_info = f", limite adaptative: {self.max_steps} (meilleur: {self.best_successful_steps})"
            
            print(f"\nEpisode {self.episode_count}: "
                  f"{'Succès' if success else 'Échec'} en {steps} étapes, "
                  f"récompense {total_reward:.1f}, "
                  f"epsilon {self.epsilon:.3f}, "
                  f"{new_states_count} nouveaux états, "
                  f"performance {efficiency}, "
                  f"convergence {convergence_status}{adaptive_info}")

        return total_reward, steps, success


    def get_statistics(self) -> Dict[str, any]:
        """Retourne les statistiques de l'agent"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'best_steps': self.best_steps,
            'converged': self.converged
        }

    def print_convergence_summary(self):
        """Affiche un résumé détaillé de l'état de convergence"""
        convergence_info = self.get_convergence_info()
        
        print("\n" + "=" * 50)
        print("RÉSUMÉ DE CONVERGENCE")
        print("=" * 50)
        
        if convergence_info['converged']:
            print("CONVERGENCE ATTEINTE!")
            print(f"   - Coefficient de variation: {convergence_info['coefficient_variation']:.4f}")
            print(f"   - Seuil requis: {convergence_info['threshold']}")
            print(f"   - Performance moyenne: {convergence_info['mean_performance']:.1f} étapes")
            print(f"   - Écart-type: {convergence_info['std_performance']:.1f} étapes")
            print(f"   - Épisodes analysés: {convergence_info['episodes_analyzed']}")
        else:
            print("Convergence en cours...")
            if 'episodes_needed' in convergence_info:
                print(f"   - Épisodes nécessaires: {convergence_info['episodes_needed']}")
            else:
                print(f"   - CV actuel: {convergence_info['coefficient_variation']:.4f}")
                print(f"   - Seuil requis: {convergence_info['threshold']}")
                print(f"   - Performance moyenne: {convergence_info['mean_performance']:.1f} étapes")
                print(f"   - Épisodes analysés: {convergence_info['episodes_analyzed']}")
        
        print(f"   - Epsilon actuel: {convergence_info['epsilon']:.3f}")
        print("=" * 50)


def train_agent(
    environment_name: str = "maze",
    show_gui: bool = True,
    verbose: bool = True
) -> QLearningAgent:
    """
    Entraîne l'agent Q-Learning jusqu'à convergence
    
    Les paramètres de convergence sont gérés automatiquement par l'agent.
    L'agent s'arrête uniquement sur convergence, sans limite d'épisodes.
    Supporte l'interruption clavier (Ctrl+C) pour arrêt propre.
    
    Args:
        environment_name: Nom de l'environnement
        show_gui: Afficher l'interface graphique
        verbose: Afficher les détails
        
    Returns:
        Agent entraîné
    """
    global interrupted
    interrupted = False
    
    # Configuration du gestionnaire de signal pour Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    agent = QLearningAgent()
    
    # Activer le mode verbose pour l'adaptation
    agent._verbose_adaptive = verbose

    print("=" * 60)
    print("ENTRAÎNEMENT Q-LEARNING AVEC SMART EXPLORER - DONATELLOPYZZA")
    print("=" * 60)
    print(f"Environnement: {environment_name}")
    print(f"Fenêtre de convergence: {agent.convergence_window} épisodes (automatique)")
    print(f"Seuil de convergence: {agent.convergence_threshold} (automatique)")
    print(f"Limite d'étapes initiale: {agent.max_steps} (adaptative)")
    print(f"Learning rate: {agent.learning_rate}")
    print(f"Discount factor: {agent.discount_factor}")
    print(f"Epsilon initial: {agent.epsilon}")
    print(f"Smart Explorer: {agent.systematic_exploration_episodes} épisodes d'exploration systématique")
    print(f"Adaptive Episode Length: limite minimale {agent.min_adaptive_steps} étapes")
    print("=" * 60)
    print("Nouvelles fonctionnalités:")
    print("  🔍 Smart Explorer: Exploration systématique pour nouveaux états")
    print("  📏 Adaptive Episode Length: Ajustement automatique des limites")
    print("  🎯 Convergence intelligente: Optimisation basée sur les performances")
    print("=" * 60)
    print("Astuce: Appuyez sur Ctrl+C pour arrêter l'entraînement à tout moment")
    print("=" * 60)

    successful_episodes = 0
    episode = 0

    # Boucle d'entraînement basée sur la convergence uniquement
    while not agent.converged and not interrupted:
        episode += 1
        print(f"\nÉPISODE {episode}")
        print("-" * 40)

        game = RLGame(environment_name, gui=show_gui)
        turtle = game.start()

        reward, steps, success = agent.train_episode(game, turtle, show_gui, verbose)

        if success:
            successful_episodes += 1

        # Affichage des statistiques périodiques améliorées
        if episode % 10 == 0:
            stats = agent.get_statistics()
            success_rate = successful_episodes / episode
            
            print(f"\nStatistiques après {episode} épisodes:")
            print(f"  Taux de succès global: {success_rate:.1%}")
            
            # Ajout du taux de succès récent
            if len(agent.episode_results) >= agent.convergence_window:
                recent_results = agent.episode_results[-agent.convergence_window:]
                recent_successes = sum(1 for result in recent_results if result[2])
                recent_success_rate = recent_successes / len(recent_results)
                print(f"  Taux de succès récent ({agent.convergence_window}): {recent_success_rate:.1%}")
            
            print(f"  Meilleur chemin: {stats['best_steps']} étapes")
            print(f"  Epsilon: {stats['epsilon']:.3f}")
            print(f"  Q-table: {stats['q_table_size']} états")
            
            if stats['best_steps'] < 50:
                print(f"  Performance: excellente (≤50 étapes)")
            elif stats['best_steps'] < 100:
                print(f"  Performance: bonne (≤100 étapes)")
            else:
                print(f"  Performance: en cours d'apprentissage")
            
            # Affichage du résumé de convergence
            agent.print_convergence_summary()

    # Résultats finaux
    print("\n" + "=" * 60)
    if interrupted:
        print("Entraînement interrompu par l'utilisateur (Ctrl+C)")
    else:
        print("Entraînement terminé - convergence atteinte")
    print("=" * 60)
    
    final_stats = agent.get_statistics()
    final_success_rate = successful_episodes / episode
    convergence_info = agent.get_convergence_info()
    
    print(f"Épisodes total: {episode}")
    print(f"Taux de succès final: {final_success_rate:.1%}")
    print(f"Meilleur chemin trouvé: {final_stats['best_steps']} étapes")
    print(f"États appris: {final_stats['q_table_size']}")
    
    if convergence_info['converged']:
        print(f"Convergence: atteinte (CV: {convergence_info['coefficient_variation']:.4f})")
        print(f"Performance stable: {convergence_info['mean_performance']:.1f} ± {convergence_info['std_performance']:.1f} étapes")
    else:
        print("Convergence: non atteinte")
    
    # Évaluation de la performance finale
    if final_stats['best_steps'] <= 20:
        print("Performance exceptionnelle!")
    elif final_stats['best_steps'] <= 30:
        print("Performance excellente!")
    elif final_stats['best_steps'] <= 50:
        print("Performance très bonne!")
    elif final_stats['best_steps'] <= 100:
        print("Performance bonne!")
    else:
        print("Performance en cours d'amélioration!")
    
    print("=" * 60)
    
    # Affichage du résumé final de convergence
    agent.print_convergence_summary()

    return agent


def test_agent(
    agent: QLearningAgent,
    environment_name: str = "maze",
    num_tests: int = 5,
    show_gui: bool = True,
    verbose: bool = True
) -> Tuple[float, List[int]]:
    """
    Teste l'agent entraîné
    
    Args:
        agent: Agent à tester
        environment_name: Nom de l'environnement
        num_tests: Nombre de tests
        show_gui: Afficher l'interface graphique
        verbose: Afficher les détails
        
    Returns:
        Tuple (taux de succès, liste des scores)
    """
    print("\n" + "=" * 60)
    print("PHASE DE TEST")
    print("=" * 60)

    # Sauvegarde de l'epsilon original
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pas d'exploration pendant les tests

    test_results = []
    success_count = 0

    for test_num in range(num_tests):
        print(f"\nTEST {test_num + 1}/{num_tests}")
        print("-" * 30)

        game = RLGame(environment_name, gui=show_gui)
        turtle = game.start()

        # Utilisation du mode test (training_mode=False)
        reward, steps, success = agent.train_episode(game, turtle, show_gui, verbose, training_mode=False)
        test_results.append(steps)

        if success:
            success_count += 1

        status = "Succès" if success else "Échec"
        print(f"Résultat: {status} | {steps} étapes")

    # Restauration de l'epsilon
    agent.epsilon = original_epsilon

    success_rate = success_count / num_tests

    print("\n" + "=" * 60)
    print("RÉSULTATS DES TESTS")
    print("=" * 60)
    print(f"Taux de succès: {success_rate:.1%} ({success_count}/{num_tests})")
    if success_count > 0:
        avg_steps = sum(test_results) / len(test_results)
        best_score = min(test_results)
        print(f"Score moyen: {avg_steps:.1f} étapes")
        print(f"Meilleur chemin: {best_score} étapes")
        
        # Évaluation de la performance des tests
        if best_score <= 15:
            print("Tests: performance exceptionnelle!")
        elif best_score <= 25:
            print("Tests: performance excellente!")
        elif best_score <= 40:
            print("Tests: performance très bonne!")
        else:
            print("Tests: performance bonne!")
    print("=" * 60)

    return success_rate, test_results


def get_user_config() -> Dict[str, Any]:
    """
    Collecte la configuration utilisateur
    
    Returns:
        Dictionnaire avec la configuration
    """
    print("=" * 60)
    print("Q-LEARNING SIMPLIFIÉ POUR DONATELLOPYZZA")
    print("=" * 60)
    print("Objectif: Apprendre à naviguer vers la pizza")
    print("Méthode: Apprentissage par renforcement (Q-Learning)")
    print("=" * 60)

    environments = ["maze", "assessment_maze", "hard_maze", "line", "test"]

    print("\nEnvironnements disponibles:")
    for i, env in enumerate(environments, 1):
        print(f"  {i}. {env}")

    # Validation de l'environnement avec boucle
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
    
    # Paramètres de convergence fixes (gérés automatiquement par l'agent)
    convergence_window = 20
    convergence_threshold = 0.05

    return {
        'environment_name': environment_name,
        'show_gui': show_gui,
        'verbose': verbose
    }

def run_training_pipeline(config: Dict[str, Any]) -> QLearningAgent:
    """
    Exécute le pipeline d'entraînement complet
    
    Args:
        config: Configuration utilisateur
        
    Returns:
        Agent entraîné
    """
    # Entraînement avec convergence
    agent = train_agent(
        environment_name=config['environment_name'],
        show_gui=config['show_gui'],
        verbose=config['verbose']
    )

    # Tests (seulement si l'entraînement n'a pas été interrompu)
    if not interrupted:
        test_choice = input("\nEffectuer des tests ? (o/n) [défaut: o]: ").lower() != 'n'
        if test_choice:
            num_tests = int(input("Nombre de tests [défaut: 3]: ") or "3")
            test_verbose = input("Affichage détaillé pour les tests ? (o/n) [défaut: n]: ").lower() == 'o'

            success_rate, test_results = test_agent(
                agent=agent,
                environment_name=config['environment_name'],
                num_tests=num_tests,
                show_gui=config['show_gui'],
                verbose=test_verbose
            )

    return agent

def main():
    """Point d'entrée principal"""
    config = get_user_config()
    agent = run_training_pipeline(config)
    
    print("\nProgramme terminé!")
    print(f"Agent final avec {agent.get_statistics()['q_table_size']} états appris")
    print(f"Chemin optimal: {agent.get_statistics()['best_steps']} étapes")


if __name__ == "__main__":
    main()
