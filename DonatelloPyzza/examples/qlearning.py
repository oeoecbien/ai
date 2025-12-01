"""
Agent Q-Learning pour DonatelloPyzza

Implémentation générique de l'algorithme Q-Learning basée sur l'équation de Bellman.
L'agent apprend par essais-erreurs à trouver le chemin optimal vers la pizza.

Algorithme Q-Learning (équation de Bellman) :
Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]

Où :
- Q(s,a) : Valeur estimée de l'action a dans l'état s
- α (alpha) : Taux d'apprentissage (learning rate)
- r : Récompense immédiate
- γ (gamma) : Facteur d'actualisation (discount factor)
- s' : État suivant
- max Q(s',a') : Meilleure valeur Q possible dans l'état suivant

Stratégie ε-greedy :
- Avec probabilité ε : exploration (action aléatoire)
- Avec probabilité 1-ε : exploitation (meilleure action selon Q-table)
"""

import sys
import os
import random
from typing import Dict, Tuple, Optional
from collections import defaultdict, deque

# Configuration du chemin d'accès au module parent
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from donatellopyzza import RLGame, Action, Feedback


class QLearningAgent:
    """
    Agent Q-Learning générique basé sur l'équation de Bellman.
    
    L'agent utilise l'équation de Bellman pour mettre à jour une table Q
    qui stocke la valeur estimée de chaque couple (état, action).
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        max_steps_per_episode: int = 1000,
        pure_mode: bool = False
    ):
        """
        Initialise l'agent Q-Learning.
        
        Args:
            learning_rate: Taux d'apprentissage α (0.1 = 10% de mise à jour)
            discount_factor: Facteur d'actualisation γ (0.9 = 90% de valeur future)
            epsilon: Taux d'exploration initial (0.3 = 30% d'actions aléatoires)
            epsilon_decay: Réduction de l'exploration par épisode (0.995 = -0.5% par épisode)
            epsilon_min: Exploration minimale (0.01 = 1% minimum)
            max_steps_per_episode: Nombre maximum d'étapes par épisode
            pure_mode: Si True, récompenses simples (0/1), sinon reward shaping
        """
        self.learning_rate = learning_rate  # α (alpha)
        self.discount_factor = discount_factor  # γ (gamma)
        self.epsilon = epsilon  # ε (epsilon)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.max_steps_per_episode = max_steps_per_episode
        self.pure_mode = pure_mode
        
        # Table Q : Q[état][action] = valeur estimée
        # Structure : Dict[Tuple[position, orientation], Dict[action_value, q_value]]
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # Statistiques pour la détection de convergence
        self.episode_count = 0
        self.best_steps = float('inf')
        self.success_count = 0
        self.episode_results = deque(maxlen=50)  # Historique des 50 derniers épisodes
    
    def _get_state_key(self, position: Tuple[int, int], orientation: int) -> Tuple:
        """
        Crée une clé unique pour représenter l'état markovien.
        
        L'état est défini par la position et l'orientation (propriété markovienne).
        
        Args:
            position: Position (x, y) de la tortue
            orientation: Orientation de la tortue (0=Nord, 1=Est, 2=Sud, 3=Ouest)
        
        Returns:
            Clé d'état (tuple)
        """
        return (position, orientation)
    
    def _calculate_reward(self, feedback: Feedback) -> float:
        """
        Calcule la récompense selon le feedback reçu.
        
        Args:
            feedback: Retour de l'action exécutée
        
        Returns:
            Récompense (float)
        """
        if self.pure_mode:
            # Mode Q-Learning pur (académique) : récompenses simples
            # Conforme à la théorie du Q-Learning standard
            if feedback == Feedback.MOVED_ON_PIZZA or feedback == Feedback.TOUCHED_PIZZA:
                return 1.0  # Récompense positive pour trouver la pizza
            return 0.0  # Pas de récompense pour les autres actions
        else:
            # Mode avec reward shaping : récompenses détaillées pour accélérer l'apprentissage
            if feedback == Feedback.MOVED_ON_PIZZA or feedback == Feedback.TOUCHED_PIZZA:
                return 100.0  # Grosse récompense pour la pizza
            elif feedback == Feedback.COLLISION or feedback == Feedback.TOUCHED_WALL:
                return -5.0  # Pénalité pour toucher un mur
            elif feedback == Feedback.MOVED:
                return -1.0  # Coût temporel pour chaque déplacement
            else:
                return -0.5  # Petit coût pour les autres actions
    
    def _choose_action(self, state_key: Tuple) -> Action:
        """
        Choisit une action selon la stratégie ε-greedy.
        
        Stratégie ε-greedy :
        - Avec probabilité ε : exploration (action aléatoire)
        - Avec probabilité 1-ε : exploitation (meilleure action selon Q-table)
        
        Args:
            state_key: Clé de l'état actuel
        
        Returns:
            Action choisie
        """
        # Exploration : action aléatoire avec probabilité ε
        if random.random() < self.epsilon:
            return random.choice([
                Action.MOVE_FORWARD,
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
                Action.TOUCH
            ])
        
        # Exploitation : meilleure action selon Q-table
        q_values = self.q_table[state_key]
        if not q_values:
            # Si l'état n'existe pas encore dans la Q-table, action aléatoire
            return random.choice([
                Action.MOVE_FORWARD,
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
                Action.TOUCH
            ])
        
        # Trouver l'action avec la valeur Q maximale
        best_value = max(q_values.values())
        best_actions = [Action(a) for a, v in q_values.items() if v == best_value]
        # En cas d'égalité, choisir aléatoirement parmi les meilleures
        return random.choice(best_actions)
    
    def _update_q_table(
        self,
        state: Tuple,
        action: Action,
        reward: float,
        next_state: Tuple,
        done: bool
    ):
        """
        Met à jour la table Q selon l'équation de Bellman.
        
        Équation de Bellman pour Q-Learning :
        Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]
        
        Où :
        - Q(s,a) : Valeur Q actuelle de l'état s et de l'action a
        - α : Taux d'apprentissage (learning_rate)
        - r : Récompense immédiate
        - γ : Facteur d'actualisation (discount_factor)
        - s' : État suivant (next_state)
        - max Q(s',a') : Meilleure valeur Q possible dans l'état suivant
        
        Si l'épisode est terminé (done=True), on n'ajoute pas de récompense future :
        Q(s,a) ← (1-α)Q(s,a) + α[r]
        
        Args:
            state: État actuel s
            action: Action exécutée a
            reward: Récompense immédiate r
            next_state: État suivant s'
            done: True si l'épisode est terminé
        """
        # Valeur Q actuelle : Q(s,a)
        current_q = self.q_table[state][action.value]
        
        if done:
            # Si l'épisode est terminé, pas de récompense future
            # Q(s,a) ← (1-α)Q(s,a) + α[r]
            target = reward
        else:
            # Sinon, utiliser l'équation de Bellman complète
            # Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            target = reward + self.discount_factor * max_next_q
        
        # Mise à jour selon l'équation de Bellman
        # Q(s,a) ← (1-α)Q(s,a) + α[target]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * target
        self.q_table[state][action.value] = new_q
    
    def train_episode(self, game: RLGame, verbose: bool = False) -> Tuple[float, int, bool]:
        """
        Entraîne l'agent sur un épisode complet.
        
        Algorithme Q-Learning pour un épisode :
        1. Initialiser l'état s
        2. Répéter jusqu'à la fin de l'épisode :
           a. Choisir une action a selon la stratégie ε-greedy
           b. Exécuter l'action a, observer la récompense r et l'état suivant s'
           c. Mettre à jour Q(s,a) selon l'équation de Bellman
           d. s ← s'
        
        Args:
            game: Instance du jeu RLGame
            verbose: Afficher les détails de l'épisode
        
        Returns:
            Tuple (récompense totale, nombre d'étapes, succès)
        """
        # 1. Initialiser l'épisode
        turtle = game.start()
        total_reward = 0.0
        steps = 0
        success = False
        
        # Obtenir l'état initial s
        position = game.getTurtlePosition(turtle)
        orientation = game.getTurtleOrientation(turtle)
        state = self._get_state_key(position, orientation)
        
        # 2. Boucle principale de l'épisode
        while steps < self.max_steps_per_episode:
            steps += 1
            
            # a. Choisir une action a selon la stratégie ε-greedy
            action = self._choose_action(state)
            
            # b. Exécuter l'action a, observer la récompense r et l'état suivant s'
            feedback, _ = turtle.execute(action)
            
            # Obtenir le nouvel état s'
            new_position = game.getTurtlePosition(turtle)
            new_orientation = game.getTurtleOrientation(turtle)
            next_state = self._get_state_key(new_position, new_orientation)
            
            # Calculer la récompense r
            reward = self._calculate_reward(feedback)
            total_reward += reward
            
            # Vérifier si l'épisode est terminé
            done = game.isWon(prnt=False)
            
            # c. Mettre à jour Q(s,a) selon l'équation de Bellman
            self._update_q_table(state, action, reward, next_state, done)
            
            if verbose and steps % 10 == 0:
                print(f"  Étape {steps}: position {new_position}, récompense {reward:.1f}")
            
            if done:
                success = True
                if verbose:
                    print(f"\n[SUCCÈS] Pizza trouvée en {steps} étapes!")
                break
            
            # d. s ← s' (passer à l'état suivant)
            state = next_state
        
        if not success and steps >= self.max_steps_per_episode:
            if verbose:
                print(f"\n[ÉCHEC] Limite d'étapes atteinte ({self.max_steps_per_episode})")
        
        # Mettre à jour les statistiques
        self.episode_count += 1
        if success:
            self.success_count += 1
            if steps < self.best_steps:
                self.best_steps = steps
        
        # Enregistrer les résultats pour la détection de convergence
        self.episode_results.append((steps, success, total_reward))
        
        # Réduire l'exploration (décroissance de ε)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_reward, steps, success
    
    def _check_convergence(self, min_episodes: int = 50, window_size: int = 30) -> bool:
        """
        Vérifie si l'agent a convergé (fini d'apprendre).
        
        Critères de convergence :
        1. Au moins min_episodes épisodes effectués
        2. Taux de succès élevé et stable sur les window_size derniers épisodes
        3. Performance stable (faible variance dans le nombre d'étapes)
        4. Epsilon proche du minimum (exploration minimale)
        
        Args:
            min_episodes: Nombre minimum d'épisodes avant de vérifier la convergence
            window_size: Taille de la fenêtre d'analyse des performances récentes
        
        Returns:
            True si l'agent a convergé, False sinon
        """
        if self.episode_count < min_episodes:
            return False
        
        if len(self.episode_results) < window_size:
            return False
        
        # Analyser les window_size derniers épisodes
        recent_results = list(self.episode_results)[-window_size:]
        recent_steps = [r[0] for r in recent_results]
        recent_successes = [r[1] for r in recent_results]
        
        # Calculer les métriques
        success_rate = sum(recent_successes) / len(recent_successes)
        mean_steps = sum(recent_steps) / len(recent_steps) if recent_steps else 0
        
        # Calculer la variance pour mesurer la stabilité
        if len(recent_steps) > 1:
            variance = sum((s - mean_steps) ** 2 for s in recent_steps) / len(recent_steps)
            std_dev = variance ** 0.5
            coefficient_variation = std_dev / mean_steps if mean_steps > 0 else float('inf')
        else:
            coefficient_variation = float('inf')
        
        # Critères de convergence
        high_success_rate = success_rate >= 0.85  # Au moins 85% de succès
        stable_performance = coefficient_variation <= 0.3  # Faible variance (30% max)
        low_exploration = self.epsilon <= self.epsilon_min * 2  # Exploration proche du minimum
        
        # L'agent a convergé si tous les critères sont remplis
        converged = high_success_rate and stable_performance and low_exploration
        
        return converged
    
    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques de l'agent.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        success_rate = self.success_count / self.episode_count if self.episode_count > 0 else 0.0
        
        # Statistiques sur les derniers épisodes
        if len(self.episode_results) > 0:
            recent_results = list(self.episode_results)[-30:]
            recent_steps = [r[0] for r in recent_results]
            recent_successes = [r[1] for r in recent_results]
            recent_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
            mean_recent_steps = sum(recent_steps) / len(recent_steps) if recent_steps else 0
        else:
            recent_success_rate = 0.0
            mean_recent_steps = 0
        
        return {
            'episode_count': self.episode_count,
            'success_count': self.success_count,
            'success_rate': success_rate,
            'recent_success_rate': recent_success_rate,
            'best_steps': self.best_steps if self.best_steps != float('inf') else 0,
            'mean_recent_steps': mean_recent_steps,
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'converged': self._check_convergence()
        }


def train_agent(
    environment_name: str = "maze",
    show_gui: bool = True,
    verbose: bool = True,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    epsilon: float = 0.3,
    pure_mode: bool = False,
    max_episodes: Optional[int] = None
) -> QLearningAgent:
    """
    Entraîne un agent Q-Learning jusqu'à convergence.
    
    L'entraînement s'arrête automatiquement lorsque l'agent a convergé,
    ou après max_episodes si spécifié (par défaut, pas de limite).
    
    Args:
        environment_name: Nom de l'environnement (maze, assessment_maze, hard_maze, etc.)
        show_gui: Afficher l'interface graphique
        verbose: Afficher les détails
        learning_rate: Taux d'apprentissage α
        discount_factor: Facteur d'actualisation γ
        epsilon: Taux d'exploration initial ε
        pure_mode: Mode Q-Learning pur (récompenses simples)
        max_episodes: Nombre maximum d'épisodes (None = pas de limite, arrêt à convergence)
    
    Returns:
        Agent entraîné
    """
    agent = QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        pure_mode=pure_mode
    )
    
    print("=" * 60)
    print("ENTRAÎNEMENT Q-LEARNING - DONATELLOPYZZA")
    print("=" * 60)
    print(f"Environnement: {environment_name}")
    print(f"Mode: {'Q-Learning pur (académique)' if pure_mode else 'Q-Learning avec reward shaping'}")
    print(f"Learning rate (α): {learning_rate}")
    print(f"Discount factor (γ): {discount_factor}")
    print(f"Epsilon initial (ε): {epsilon}")
    if max_episodes:
        print(f"Max épisodes: {max_episodes}")
    else:
        print("Max épisodes: Illimité (arrêt à convergence)")
    print("=" * 60)
    
    episode = 0
    
    # Boucle d'entraînement jusqu'à convergence ou limite d'épisodes
    while True:
        episode += 1
        
        # Vérifier la limite d'épisodes
        if max_episodes and episode > max_episodes:
            if verbose:
                print(f"\n[LIMITE] Nombre maximum d'épisodes ({max_episodes}) atteint")
            break
        
        game = RLGame(environment_name, show_gui)
        
        if verbose:
            print(f"\nÉPISODE {episode}")
            print("-" * 40)
        
        reward, steps, success = agent.train_episode(game, verbose)
        
        # Vérifier la convergence tous les 10 épisodes (après au moins 50 épisodes)
        if episode % 10 == 0:
            stats = agent.get_statistics()
            print(f"\n[ANALYSE] Bilan après {episode} épisodes:")
            print(f"  Taux de succès global: {stats['success_rate']:.1%}")
            print(f"  Taux de succès récent (30): {stats['recent_success_rate']:.1%}")
            print(f"  Meilleur chemin: {stats['best_steps']} étapes")
            print(f"  Étapes moyennes récentes: {stats['mean_recent_steps']:.1f}")
            print(f"  États appris: {stats['q_table_size']}")
            print(f"  Epsilon (exploration): {stats['epsilon']:.3f}")
            
            # Vérifier la convergence
            if stats['converged']:
                print(f"\n[CONVERGENCE] L'agent a convergé après {episode} épisodes!")
                print(f"  - Taux de succès récent: {stats['recent_success_rate']:.1%}")
                print(f"  - Performance stable détectée")
                break
    
    # Statistiques finales
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)
    final_stats = agent.get_statistics()
    print(f"Épisodes total: {final_stats['episode_count']}")
    print(f"Taux de succès final: {final_stats['success_rate']:.1%}")
    print(f"Taux de succès récent: {final_stats['recent_success_rate']:.1%}")
    print(f"Meilleur chemin trouvé: {final_stats['best_steps']} étapes")
    print(f"Étapes moyennes récentes: {final_stats['mean_recent_steps']:.1f}")
    print(f"États appris: {final_stats['q_table_size']}")
    print(f"Convergence: {'Oui' if final_stats['converged'] else 'Non'}")
    print("=" * 60)
    
    return agent


def main():
    """
    Fonction principale avec interface utilisateur interactive.
    """
    print("=" * 60)
    print("Q-LEARNING POUR DONATELLOPYZZA")
    print("=" * 60)
    print("Objectif: Apprendre à naviguer vers la pizza")
    print("Méthode: Apprentissage par renforcement (Q-Learning)")
    print("Algorithme: Équation de Bellman")
    print("=" * 60)
    
    # Choix de l'environnement
    environments = ["maze", "assessment_maze", "hard_maze", "line", "test"]
    print("\nEnvironnements disponibles:")
    for i, env in enumerate(environments, 1):
        print(f"  {i}. {env}")
    
    try:
        choice = int(input("\nChoisissez un environnement (1-5) [défaut: 1]: ") or "1")
        if 1 <= choice <= len(environments):
            environment_name = environments[choice - 1]
        else:
            environment_name = "maze"
    except ValueError:
        environment_name = "maze"
    
    # Options d'affichage
    try:
        show_gui = input("Afficher l'interface graphique ? (o/n) [défaut: o]: ").lower() != 'n'
        verbose = input("Affichage détaillé ? (o/n) [défaut: o]: ").lower() != 'n'
    except:
        show_gui = True
        verbose = True
    
    # Configuration avancée
    try:
        advanced = input("Configuration avancée ? (o/n) [défaut: n]: ").lower() == 'o'
        if advanced:
            learning_rate = float(input("Learning rate (α) [défaut: 0.1]: ") or "0.1")
            epsilon = float(input("Epsilon initial (ε) [défaut: 0.3]: ") or "0.3")
            max_episodes_input = input("Max épisodes (laisser vide pour illimité) [défaut: illimité]: ").strip()
            max_episodes = int(max_episodes_input) if max_episodes_input else None
            pure_mode = input("Mode Q-Learning pur (sans reward shaping) ? (o/n) [défaut: n]: ").lower() == 'o'
        else:
            learning_rate = 0.1
            epsilon = 0.3
            max_episodes = None  # Pas de limite, arrêt à convergence
            pure_mode = False
    except:
        learning_rate = 0.1
        epsilon = 0.3
        max_episodes = None
        pure_mode = False
    
    # Lancement de l'entraînement
    agent = train_agent(
        environment_name=environment_name,
        show_gui=show_gui,
        verbose=verbose,
        learning_rate=learning_rate,
        epsilon=epsilon,
        pure_mode=pure_mode,
        max_episodes=max_episodes
    )
    
    print("\n[TERMINÉ] Programme terminé!")
    stats = agent.get_statistics()
    print(f"Agent final avec {stats['q_table_size']} états appris")
    print(f"Chemin optimal: {stats['best_steps']} étapes")
    if stats['converged']:
        print("✓ L'agent a convergé et a fini d'apprendre")


if __name__ == "__main__":
    main()
