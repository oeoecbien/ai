import sys
import os
import random
import time
import pygame
from typing import Dict, Tuple, List
from collections import defaultdict

# Configuration du chemin d'accès au module parent
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from donatellopyzza import RLGame, Action, Feedback


class QLearningAgent:
    """
    Agent Q-Learning simplifié pour DonatelloPyzza
    
    Fonctionnalités:
        - Apprentissage par renforcement avec Q-table
        - Équation de Bellman pour mise à jour des valeurs Q
        - Exploration vs exploitation (epsilon-greedy)
        - Système de récompenses simplifié
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialise l'agent Q-Learning
        
        Args:
            learning_rate: Taux d'apprentissage (α)
            discount_factor: Facteur de réduction (γ)
            epsilon: Taux d'exploration initial
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        # Hyperparamètres Q-Learning
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: état -> {action: valeur_Q}
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Actions disponibles
        self.actions = [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.TOUCH]

        # Configuration des récompenses optimisée
        self.rewards = {
            'pizza_found': 200.0,      # Récompense principale (augmentée)
            'pizza_touched': 100.0,    # Récompense pour toucher la pizza (augmentée)
            'collision': -15.0,        # Pénalité pour collision (augmentée)
            'step': -0.5,              # Coût par étape (réduit)
            'wall_touched': -8.0,      # Pénalité pour toucher un mur (augmentée)
            'new_state': 8.0,          # Bonus pour découvrir un nouvel état (augmenté)
            'proximity_bonus': 3.0,    # Bonus de proximité vers la pizza (augmenté)
            'efficiency_bonus': 10.0   # Bonus pour chemin court
        }

        # Statistiques simples
        self.episode_count = 0
        self.best_steps = float('inf')
        
        # Suivi des états visités pour bonus d'exploration
        self.visited_states = set()
        
        # Position de la pizza (sera détectée dynamiquement)
        self.pizza_position = None

    def get_state(self, position: Tuple[int, int], orientation: int, feedback: Feedback = None) -> str:
        """
        Génère une représentation d'état unique
        
        Args:
            position: Position actuelle (x, y)
            orientation: Orientation actuelle (0-3)
            feedback: Dernier feedback reçu
            
        Returns:
            Chaîne représentant l'état
        """
        state = f"pos_{position[0]}_{position[1]}_ori_{orientation}"
        
        if feedback == Feedback.TOUCHED_WALL:
            state += "_wall"
        elif feedback == Feedback.TOUCHED_PIZZA:
            state += "_pizza"
        elif feedback == Feedback.TOUCHED_NOTHING:
            state += "_empty"
            
        return state

    def calculate_reward(self, feedback: Feedback, state: str, position: Tuple[int, int]) -> float:
        """
        Calcule la récompense basée sur le feedback avec bonus d'exploration et proximité
        
        Args:
            feedback: Feedback reçu de l'environnement
            state: État actuel
            position: Position actuelle
            
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
        
        # Bonus de proximité si on connaît la position de la pizza
        if self.pizza_position:
            distance = abs(position[0] - self.pizza_position[0]) + abs(position[1] - self.pizza_position[1])
            if distance <= 2:  # Proximité de 2 cases ou moins
                base_reward += self.rewards['proximity_bonus']
        
        # Bonus d'efficacité pour les chemins courts
        if feedback == Feedback.MOVED_ON_PIZZA and self.best_steps != float('inf'):
            if self.episode_count > 0:  # Pas au premier épisode
                base_reward += self.rewards['efficiency_bonus']
        
        return base_reward

    def choose_action(self, state: str) -> Action:
        """
        Sélectionne une action selon la politique epsilon-greedy améliorée
        
        Args:
            state: État actuel
            
        Returns:
            Action à exécuter
        """
        # Stratégie d'exploration adaptative améliorée
        exploration_rate = self.epsilon
        
        # Augmenter l'exploration pour les nouveaux états
        if state not in self.visited_states:
            exploration_rate = min(0.9, self.epsilon * 3)  # Plus d'exploration pour nouveaux états
        
        # Réduire l'exploration si on a déjà un bon score
        if self.best_steps < 50 and self.episode_count > 20:
            exploration_rate *= 0.5  # Moins d'exploration si on a déjà un bon chemin
        
        # Exploration: action aléatoire
        if random.random() < exploration_rate:
            return random.choice(self.actions)

        # Exploitation: meilleure action connue
        if state in self.q_table and self.q_table[state]:
            best_action_value = max(self.q_table[state].items(), key=lambda x: x[1])
            action_index = best_action_value[0]
            
            for action in self.actions:
                if int(action.value) == action_index:
                    return action

        # Si aucune information, action aléatoire
        return random.choice(self.actions)

    def update_q_value(self, state: str, action: Action, reward: float, next_state: str):
        """
        Met à jour la Q-table selon l'équation de Bellman
        
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

        self.q_table[state][action_key] = new_q

    def decay_epsilon(self):
        """Réduit le taux d'exploration de manière adaptative"""
        # Décroissance plus rapide si on a déjà un bon score
        if self.best_steps < 30 and self.episode_count > 10:
            decay_rate = 0.99  # Décroissance plus rapide
        else:
            decay_rate = self.epsilon_decay  # Décroissance normale
        
        self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)

    def train_episode(self, game: RLGame, turtle, show_gui: bool = True, verbose: bool = True) -> Tuple[float, int, bool]:
        """
        Entraîne l'agent sur un épisode
        
        Args:
            game: Instance du jeu
            turtle: Tortue à contrôler
            show_gui: Afficher l'interface graphique
            verbose: Afficher les détails
            
        Returns:
            Tuple (récompense totale, nombre d'étapes, succès)
        """
        total_reward = 0.0
        steps = 0

        # État initial
        current_position = game.getTurtlePosition(turtle)
        current_orientation = game.getTurtleOrientation(turtle)
        current_state = self.get_state(current_position, current_orientation)

        # Boucle principale - Continue jusqu'à la victoire
        while True:
            steps += 1

            # Sélection et exécution de l'action
            action = self.choose_action(current_state)
            feedback, _ = turtle.execute(action)

            # Nouvel état
            new_position = game.getTurtlePosition(turtle)
            new_orientation = game.getTurtleOrientation(turtle)
            next_state = self.get_state(new_position, new_orientation, feedback)

            # Calcul de la récompense avec bonus d'exploration et proximité
            reward = self.calculate_reward(feedback, current_state, new_position)
            total_reward += reward
            
            # Détection de la position de la pizza
            if feedback == Feedback.MOVED_ON_PIZZA and self.pizza_position is None:
                self.pizza_position = new_position

            # Mise à jour de la Q-table
            self.update_q_value(current_state, action, reward, next_state)

            # Affichage
            if verbose:
                print(f"[Ep {self.episode_count + 1:2d}] Étape {steps:3d}: {str(action):15s} | "
                      f"Pos: {new_position} | "
                      f"Réc: {reward:6.1f} | "
                      f"{str(feedback)}")

            # Délai pour la visualisation
            if show_gui:
                time.sleep(0.01)

            # Vérification de la victoire
            if game.isWon(prnt=False):
                if verbose:
                    print(f"\n[Ep {self.episode_count + 1:2d}] PIZZA TROUVÉE EN {steps} ÉTAPES!")
                    print(f"[Ep {self.episode_count + 1:2d}] Récompense totale: {total_reward:.1f}")
                break

            # Transition vers l'état suivant
            current_state = next_state

        # Décroissance de l'exploration
        self.decay_epsilon()
        
        # Réinitialisation des états visités pour le prochain épisode
        self.visited_states.clear()

        success = game.isWon(prnt=False)
        self.episode_count += 1

        # Mise à jour des meilleures performances
        if success and steps < self.best_steps:
            self.best_steps = steps

        if verbose:
            new_states_count = len(self.visited_states)
            efficiency = "Efficace" if success and steps <= 30 else "Normal"
            print(f"\n[Ep {self.episode_count:2d}] Résumé: "
                  f"{'Succès' if success else 'Échec'} | "
                  f"{steps} étapes | "
                  f"Réc: {total_reward:.1f} | "
                  f"ε: {self.epsilon:.3f} | "
                  f"Nouveaux états: {new_states_count} | "
                  f"{efficiency}")

        return total_reward, steps, success


    def get_statistics(self) -> Dict[str, any]:
        """Retourne les statistiques de l'agent"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'best_steps': self.best_steps
        }


def train_agent(
    environment_name: str = "maze",
    max_episodes: int = 100,
    show_gui: bool = True,
    verbose: bool = True
) -> QLearningAgent:
    """
    Entraîne l'agent Q-Learning
    
    Args:
        environment_name: Nom de l'environnement
        max_episodes: Nombre maximum d'épisodes
        show_gui: Afficher l'interface graphique
        verbose: Afficher les détails
        
    Returns:
        Agent entraîné
    """
    agent = QLearningAgent()

    print("=" * 60)
    print("ENTRAÎNEMENT Q-LEARNING - DONATELLOPYZZA")
    print("=" * 60)
    print(f"Environnement: {environment_name}")
    print(f"Épisodes max: {max_episodes}")
    print(f"Learning rate: {agent.learning_rate}")
    print(f"Discount factor: {agent.discount_factor}")
    print(f"Epsilon initial: {agent.epsilon}")
    print("=" * 60)

    successful_episodes = 0

    for episode in range(max_episodes):
        print(f"\nÉPISODE {episode + 1}/{max_episodes}")
        print("-" * 40)

        game = RLGame(environment_name, gui=show_gui)
        turtle = game.start()

        reward, steps, success = agent.train_episode(game, turtle, show_gui, verbose)

        if success:
            successful_episodes += 1

        # Affichage des statistiques périodiques améliorées
        if (episode + 1) % 10 == 0:
            stats = agent.get_statistics()
            success_rate = successful_episodes / (episode + 1)
            efficiency_rate = 0
            if episode >= 9:  # Calculer l'efficacité sur les 10 derniers épisodes
                recent_episodes = max(1, min(10, episode + 1))
                efficiency_rate = sum(1 for i in range(max(0, episode - 9), episode + 1) 
                                    if i < len([success]) and success) / recent_episodes
            
            print(f"\nStatistiques après {episode + 1} épisodes:")
            print(f"  - Taux de succès: {success_rate:.1%}")
            print(f"  - Meilleur chemin: {stats['best_steps']} étapes")
            print(f"  - Epsilon: {stats['epsilon']:.3f}")
            print(f"  - Q-table: {stats['q_table_size']} états")
            if stats['best_steps'] < 50:
                print(f"  - Performance: EXCELLENTE (≤50 étapes)")
            elif stats['best_steps'] < 100:
                print(f"  - Performance: BONNE (≤100 étapes)")
            else:
                print(f"  - Performance: EN COURS D'APPRENTISSAGE")

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    final_stats = agent.get_statistics()
    final_success_rate = successful_episodes / max_episodes
    print(f"Taux de succès final: {final_success_rate:.1%}")
    print(f"Meilleur chemin trouvé: {final_stats['best_steps']} étapes")
    print(f"États appris: {final_stats['q_table_size']}")
    
    # Évaluation de la performance finale
    if final_stats['best_steps'] <= 20:
        print("PERFORMANCE EXCEPTIONNELLE!")
    elif final_stats['best_steps'] <= 30:
        print("PERFORMANCE EXCELLENTE!")
    elif final_stats['best_steps'] <= 50:
        print("PERFORMANCE TRÈS BONNE!")
    elif final_stats['best_steps'] <= 100:
        print("PERFORMANCE BONNE!")
    else:
        print("PERFORMANCE EN COURS D'AMÉLIORATION!")
    
    print("=" * 60)

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

        reward, steps, success = agent.train_episode(game, turtle, show_gui, verbose)
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
            print("TESTS: PERFORMANCE EXCEPTIONNELLE!")
        elif best_score <= 25:
            print("TESTS: PERFORMANCE EXCELLENTE!")
        elif best_score <= 40:
            print("TESTS: PERFORMANCE TRÈS BONNE!")
        else:
            print("TESTS: PERFORMANCE BONNE!")
    print("=" * 60)

    return success_rate, test_results


def main():
    """Point d'entrée principal"""
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

    try:
        choice = int(input("\nChoisissez un environnement (1-5) [défaut: 1]: ") or "1")
        environment_name = environments[choice - 1] if 1 <= choice <= len(environments) else "maze"
    except (ValueError, IndexError):
        environment_name = "maze"

    try:
        max_episodes = int(input("Nombre d'épisodes d'entraînement [défaut: 50]: ") or "50")
        show_gui = input("Afficher l'interface graphique ? (o/n) [défaut: o]: ").lower() != 'n'
        verbose = input("Affichage détaillé ? (o/n) [défaut: o]: ").lower() != 'n'
    except ValueError:
        max_episodes = 50
        show_gui = True
        verbose = True

    # Entraînement
    agent = train_agent(
        environment_name=environment_name,
        max_episodes=max_episodes,
        show_gui=show_gui,
        verbose=verbose
    )

    # Tests
    test_choice = input("\nEffectuer des tests ? (o/n) [défaut: o]: ").lower() != 'n'
    if test_choice:
        num_tests = int(input("Nombre de tests [défaut: 3]: ") or "3")
        test_verbose = input("Affichage détaillé pour les tests ? (o/n) [défaut: n]: ").lower() == 'o'

        success_rate, test_results = test_agent(
            agent=agent,
            environment_name=environment_name,
            num_tests=num_tests,
            show_gui=show_gui,
            verbose=test_verbose
        )

    print("\nProgramme terminé!")
    print(f"Agent final avec {agent.get_statistics()['q_table_size']} états appris")
    print(f"Chemin optimal: {agent.get_statistics()['best_steps']} étapes")


if __name__ == "__main__":
    main()
