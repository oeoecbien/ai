"""
Q-Learning Optimisé pour DonatelloPyzza
Entraînement continu jusqu'à l'optimisation parfaite (moins de cases possible)
"""

import sys
import os
import random
import time
import pickle
from typing import Dict, Tuple, List, Set
from collections import defaultdict

# Configuration du chemin d'accès au module parent
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from donatellopyzza import RLGame, Action, Feedback


class OptimizedQLearningAgent:
    """
    Agent Q-Learning optimisé pour trouver le chemin le plus court vers la pizza
    
    Fonctionnalités:
        - Entraînement continu jusqu'à la découverte de la pizza
        - Détection du chemin optimal
        - Affichage en temps réel des performances
        - Arrêt automatique quand l'optimal est atteint
        - Visualisation du chemin parcouru
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialise l'agent Q-Learning optimisé
        
        Args:
            learning_rate: Taux d'apprentissage (alpha)
            discount_factor: Facteur de réduction (gamma)
            epsilon: Taux d'exploration initial
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        # Hyperparamètres
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionnaire d'états vers dictionnaire d'actions-valeurs
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # Actions disponibles
        self.actions: List[Action] = [
            Action.MOVE_FORWARD,
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
            Action.TOUCH
        ]
        
        # Configuration des récompenses optimisées
        self.rewards_config = {
            'pizza_found': 1000.0,       # Récompense très élevée pour la pizza
            'pizza_touched': 100.0,      # Récompense pour toucher la pizza
            'collision': -50.0,          # Pénalité forte pour collision
            'step_cost': -1.0,           # Coût par étape (encourage l'efficacité)
            'new_cell': 10.0,            # Bonus pour exploration
            'revisit_penalty': -5.0,     # Pénalité pour revisites
            'efficiency_bonus': 50.0     # Bonus pour chemin court
        }
        
        # Statistiques d'optimisation
        self.best_steps = float('inf')
        self.best_reward = float('-inf')
        self.consecutive_optimal = 0
        self.optimal_threshold = 5  # Nombre de succès consécutifs pour considérer optimal
        
        # Historique des performances
        self.performance_history = []
        self.episode_count = 0
        
        # Statistiques d'entraînement
        self.reset_episode_stats()
        
    def reset_episode_stats(self):
        """Réinitialise les statistiques de l'épisode en cours"""
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.cell_visit_count: Dict[Tuple[int, int], int] = defaultdict(int)
        self.episode_path: List[Tuple[int, int]] = []
        
    def get_state_representation(
        self,
        position: Tuple[int, int],
        orientation: int,
        last_feedback: Feedback = None
    ) -> str:
        """
        Génère une représentation d'état unique
        
        Args:
            position: Position actuelle (x, y)
            orientation: Orientation actuelle (0-3)
            last_feedback: Dernier feedback reçu
            
        Returns:
            Chaîne de caractères représentant l'état
        """
        state = f"pos_{position[0]}_{position[1]}_ori_{orientation}"
        
        if last_feedback == Feedback.TOUCHED_WALL:
            state += "_wall"
        elif last_feedback == Feedback.TOUCHED_PIZZA:
            state += "_pizza"
        elif last_feedback == Feedback.TOUCHED_NOTHING:
            state += "_empty"
            
        return state
    
    def calculate_reward(
        self,
        feedback: Feedback,
        current_position: Tuple[int, int],
        steps: int
    ) -> float:
        """
        Calcule la récompense optimisée pour l'efficacité
        
        Args:
            feedback: Feedback reçu de l'environnement
            current_position: Position actuelle
            steps: Nombre d'étapes effectuées
            
        Returns:
            Valeur de la récompense
        """
        reward = 0.0
        
        # Récompenses basées sur le feedback
        if feedback == Feedback.MOVED_ON_PIZZA:
            reward = self.rewards_config['pizza_found']
            # Bonus d'efficacité si le chemin est court
            if steps < self.best_steps:
                reward += self.rewards_config['efficiency_bonus']
        elif feedback == Feedback.TOUCHED_PIZZA:
            reward = self.rewards_config['pizza_touched']
        elif feedback == Feedback.COLLISION:
            reward = self.rewards_config['collision']
        else:
            reward = self.rewards_config['step_cost']
        
        # Gestion de l'exploration des cellules
        if current_position not in self.visited_cells:
            reward += self.rewards_config['new_cell']
            self.visited_cells.add(current_position)
            self.cell_visit_count[current_position] = 1
        else:
            self.cell_visit_count[current_position] += 1
            visits = self.cell_visit_count[current_position]
            if visits > 1:
                revisit_penalty = self.rewards_config['revisit_penalty'] * (visits - 1)
                reward += revisit_penalty
        
        return reward
    
    def choose_action(self, state: str) -> Action:
        """
        Sélectionne une action selon la politique epsilon-greedy
        
        Args:
            state: État actuel
            
        Returns:
            Action à exécuter
        """
        # Exploration: action aléatoire
        if random.random() < self.epsilon:
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
    
    def update_q_value(
        self,
        state: str,
        action: Action,
        reward: float,
        next_state: str
    ):
        """
        Met à jour la Q-table selon l'équation de Bellman
        
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
        
        # Équation de Bellman: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error
        
        self.q_table[state][action_key] = new_q
    
    def decay_exploration(self):
        """Réduit le taux d'exploration epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def is_optimal_performance(self, steps: int, reward: float) -> bool:
        """
        Détermine si la performance est optimale
        
        Args:
            steps: Nombre d'étapes effectuées
            reward: Récompense totale obtenue
            
        Returns:
            True si la performance est optimale
        """
        # Mise à jour des meilleures performances
        if steps < self.best_steps:
            self.best_steps = steps
            self.consecutive_optimal = 0
        
        if reward > self.best_reward:
            self.best_reward = reward
        
        # Vérification si la performance est optimale
        if steps <= self.best_steps and reward >= self.best_reward * 0.9:
            self.consecutive_optimal += 1
            return self.consecutive_optimal >= self.optimal_threshold
        else:
            self.consecutive_optimal = 0
            return False
    
    def visualize_path(self, game: RLGame, path: List[Tuple[int, int]]):
        """
        Visualise le chemin parcouru par la tortue
        
        Args:
            game: Instance du jeu
            path: Liste des positions parcourues
        """
        try:
            # Marque les cellules visitées
            for position in path:
                visit_count = self.cell_visit_count[position]
                # Couleur en fonction du nombre de visites
                if visit_count == 1:
                    color = "lightblue"  # Première visite
                elif visit_count == 2:
                    color = "yellow"     # Deuxième visite
                else:
                    color = "orange"     # Visites multiples
                
                # Marque la cellule (si la méthode existe)
                if hasattr(game, 'mark_cell'):
                    game.mark_cell(position, color)
        except Exception as e:
            # Ignore les erreurs de visualisation
            pass
    
    def train_episode_optimized(
        self,
        game: RLGame,
        turtle,
        show_gui: bool = True,
        verbose: bool = True
    ) -> Tuple[float, int, bool, bool]:
        """
        Entraîne l'agent sur un épisode avec optimisation
        L'épisode continue jusqu'à ce que la pizza soit trouvée
        
        Args:
            game: Instance du jeu
            turtle: Tortue à contrôler
            show_gui: Afficher l'interface graphique
            verbose: Afficher les détails de chaque étape
            
        Returns:
            Tuple (récompense totale, nombre d'étapes, succès, performance optimale)
        """
        self.reset_episode_stats()
        
        total_reward = 0.0
        steps = 0
        step_delay = 0.01  # Délai fixe entre les étapes
        
        # État initial
        current_position = game.getTurtlePosition(turtle)
        current_orientation = game.getTurtleOrientation(turtle)
        current_state = self.get_state_representation(current_position, current_orientation)
        
        # Enregistrement du chemin
        self.episode_path.append(current_position)
        
        # Noms pour l'affichage
        action_names = {
            "MOVE_FORWARD": "AVANCER",
            "TURN_LEFT": "TOURNER_GAUCHE", 
            "TURN_RIGHT": "TOURNER_DROITE",
            "TOUCH": "TOUCHER"
        }
        
        feedback_names = {
            "MOVED_ON_PIZZA": "PIZZA TROUVÉE!",
            "TOUCHED_PIZZA": "PIZZA TOUCHÉE!",
            "COLLISION": "COLLISION!",
            "TOUCHED_WALL": "MUR TOUCHÉ",
            "TOUCHED_NOTHING": "RIEN TOUCHÉ",
            "MOVED": "DÉPLACEMENT OK"
        }
        
        # Boucle principale: continue jusqu'à trouver la pizza
        while True:
            steps += 1
            
            # Sélection et exécution de l'action
            action = self.choose_action(current_state)
            feedback, _ = turtle.execute(action)
            
            # Calcul de la récompense
            new_position = game.getTurtlePosition(turtle)
            reward = self.calculate_reward(feedback, new_position, steps)
            total_reward += reward
            
            # Enregistrement du chemin
            if new_position != current_position:
                self.episode_path.append(new_position)
            
            # Affichage des informations de l'étape
            if verbose:
                feedback_str = feedback_names.get(str(feedback), str(feedback))
                print(f"Étape {steps:4d}: {action_names.get(str(action), str(action)):16s} | "
                      f"Pos: {new_position} | "
                      f"Réc.: {reward:7.1f} | "
                      f"{feedback_str}")
            
            # Nouvel état
            new_orientation = game.getTurtleOrientation(turtle)
            next_state = self.get_state_representation(
                new_position,
                new_orientation,
                feedback
            )
            
            # Mise à jour de la Q-table (apprentissage par renforcement)
            self.update_q_value(current_state, action, reward, next_state)
            
            # Délai pour la visualisation
            if show_gui and step_delay > 0:
                time.sleep(step_delay)
            
            # Vérification de la condition de victoire
            if game.isWon(prnt=False):
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"🍕 PIZZA TROUVÉE EN {steps} ÉTAPES!")
                    print(f"Récompense totale: {total_reward:.1f}")
                    print(f"Cases visitées: {len(self.visited_cells)}")
                    print(f"Cases parcourues: {len(self.episode_path)}")
                
                # Visualisation du chemin
                if show_gui:
                    self.visualize_path(game, self.episode_path)
                
                # Vérification de la performance optimale
                is_optimal = self.is_optimal_performance(steps, total_reward)
                if is_optimal and verbose:
                    print(f"🏆 PERFORMANCE OPTIMALE ATTEINTE! ({steps} étapes)")
                
                print(f"{'='*70}")
                break
            
            # Transition vers l'état suivant
            current_state = next_state
        
        # Décroissance de l'exploration
        self.decay_exploration()
        
        success = True  # Toujours vrai car on continue jusqu'à la victoire
        is_optimal = self.is_optimal_performance(steps, total_reward)
        
        # Enregistrement des performances
        self.performance_history.append({
            'episode': self.episode_count,
            'steps': steps,
            'reward': total_reward,
            'success': success,
            'optimal': is_optimal,
            'epsilon': self.epsilon,
            'path_length': len(self.episode_path),
            'unique_cells': len(self.visited_cells)
        })
        
        self.episode_count += 1
        
        if verbose:
            print(f"\n📊 Résumé de l'épisode {self.episode_count}:")
            print(f"  - Succès: Oui (pizza trouvée)")
            print(f"  - Étapes: {steps}")
            print(f"  - Récompense totale: {total_reward:.1f}")
            print(f"  - Performance optimale: {'Oui' if is_optimal else 'Non'}")
            print(f"  - Epsilon: {self.epsilon:.4f}")
            print(f"  - Meilleur score: {self.best_steps} étapes")
            print(f"  - Cases uniques visitées: {len(self.visited_cells)}")
            print(f"  - Longueur du chemin: {len(self.episode_path)}")
            print("=" * 70)
        
        return total_reward, steps, success, is_optimal
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Retourne les statistiques de l'agent
        
        Returns:
            Dictionnaire contenant les statistiques
        """
        successful_episodes = [p for p in self.performance_history if p['success']]
        optimal_episodes = [p for p in self.performance_history if p['optimal']]
        
        return {
            'q_table_size': len(self.q_table),
            'total_state_actions': sum(len(actions) for actions in self.q_table.values()),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'episode_count': self.episode_count,
            'success_rate': len(successful_episodes) / max(self.episode_count, 1),
            'optimal_rate': len(optimal_episodes) / max(self.episode_count, 1),
            'best_steps': self.best_steps,
            'best_reward': self.best_reward,
            'consecutive_optimal': self.consecutive_optimal
        }
    
    def save(self, filename: str):
        """
        Sauvegarde l'agent dans un fichier
        
        Args:
            filename: Nom du fichier de sauvegarde
        """
        agent_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'best_steps': self.best_steps,
            'best_reward': self.best_reward,
            'episode_count': self.episode_count,
            'performance_history': self.performance_history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"\n💾 Agent sauvegardé dans '{filename}'")
    
    @classmethod
    def load(cls, filename: str) -> 'OptimizedQLearningAgent':
        """
        Charge un agent depuis un fichier
        
        Args:
            filename: Nom du fichier de sauvegarde
            
        Returns:
            Agent chargé
        """
        with open(filename, 'rb') as f:
            agent_data = pickle.load(f)
        
        agent = cls(
            learning_rate=agent_data['learning_rate'],
            discount_factor=agent_data['discount_factor'],
            epsilon=agent_data['epsilon'],
            epsilon_decay=agent_data['epsilon_decay'],
            epsilon_min=agent_data['epsilon_min']
        )
        
        agent.q_table = defaultdict(lambda: defaultdict(float), agent_data['q_table'])
        agent.best_steps = agent_data['best_steps']
        agent.best_reward = agent_data['best_reward']
        agent.episode_count = agent_data['episode_count']
        agent.performance_history = agent_data['performance_history']
        
        print(f"\n📂 Agent chargé depuis '{filename}'")
        
        return agent


def train_until_optimal(
    environment_name: str = "maze",
    max_episodes: int = 1000,
    show_gui: bool = True,
    verbose: bool = True,
    target_optimal_episodes: int = 10,
    save_agent: bool = False
) -> OptimizedQLearningAgent:
    """
    Entraîne l'agent jusqu'à atteindre une performance optimale
    
    Args:
        environment_name: Nom de l'environnement
        max_episodes: Nombre maximum d'épisodes
        show_gui: Afficher l'interface graphique
        verbose: Afficher les détails de chaque étape
        target_optimal_episodes: Nombre d'épisodes optimaux consécutifs requis
        save_agent: Sauvegarder l'agent après l'entraînement
        
    Returns:
        Agent entraîné
    """
    agent = OptimizedQLearningAgent()
    agent.optimal_threshold = target_optimal_episodes
    
    print("=" * 80)
    print(f"ENTRAÎNEMENT Q-LEARNING OPTIMISÉ - DONATELLOPYZZA")
    print("=" * 80)
    print(f"Environnement: {environment_name}")
    print(f"Objectif: Trouver le chemin optimal vers la pizza")
    print(f"\nHyperparamètres:")
    print(f"  - Learning rate (α): {agent.learning_rate}")
    print(f"  - Discount factor (γ): {agent.discount_factor}")
    print(f"  - Epsilon initial (ε): {agent.epsilon}")
    print(f"  - Épisodes max: {max_episodes}")
    print(f"  - Interface graphique: {'Oui' if show_gui else 'Non'}")
    print(f"  - Affichage verbose: {'Oui' if verbose else 'Non'}")
    print(f"  - Délai entre étapes: 0.01s (fixe)")
    print(f"  - Épisodes optimaux requis: {target_optimal_episodes}")
    print(f"  - Limite d'étapes: Aucune (continue jusqu'à la pizza)")
    print("=" * 80)
    
    start_time = time.time()
    optimal_episodes = 0
    consecutive_optimal = 0
    
    for episode in range(max_episodes):
        print(f"\n🚀 ÉPISODE {episode + 1}/{max_episodes}")
        print("=" * 70)
        
        game = RLGame(environment_name, gui=show_gui)
        turtle = game.start()
        
        reward, steps, success, is_optimal = agent.train_episode_optimized(
            game, turtle, show_gui, verbose
        )
        
        # Gestion des épisodes optimaux
        if is_optimal:
            optimal_episodes += 1
            consecutive_optimal += 1
            print(f"🏆 ÉPISODE OPTIMAL #{optimal_episodes} (consécutif #{consecutive_optimal})")
        else:
            consecutive_optimal = 0
        
        # Vérification de la condition d'arrêt
        if consecutive_optimal >= target_optimal_episodes:
            print(f"\n{'='*80}")
            print(f"🎯 OBJECTIF ATTEINT!")
            print(f"L'agent a atteint une performance optimale pendant")
            print(f"{consecutive_optimal} épisodes consécutifs!")
            print(f"{'='*80}")
            break
        
        # Affichage des statistiques périodiques
        if (episode + 1) % 10 == 0:
            stats = agent.get_statistics()
            print(f"\n📈 Statistiques après {episode + 1} épisodes:")
            print(f"  - Taux de succès: {stats['success_rate']:.1%}")
            print(f"  - Taux d'optimal: {stats['optimal_rate']:.1%}")
            print(f"  - Meilleur score: {stats['best_steps']} étapes")
            print(f"  - Épisodes optimaux: {optimal_episodes}")
            print(f"  - Epsilon actuel: {stats['epsilon']:.4f}")
            print(f"  - Taille Q-table: {stats['q_table_size']} états")
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    print(f"Durée: {training_time:.1f}s")
    print(f"Épisodes total: {agent.episode_count}")
    print(f"Épisodes optimaux: {optimal_episodes}")
    
    stats = agent.get_statistics()
    print(f"\nStatistiques finales:")
    print(f"  - Taux de succès: {stats['success_rate']:.1%}")
    print(f"  - Taux d'optimal: {stats['optimal_rate']:.1%}")
    print(f"  - Meilleur score: {stats['best_steps']} étapes")
    print(f"  - Taille Q-table: {stats['q_table_size']} états")
    print(f"  - Epsilon final: {stats['epsilon']:.4f}")
    
    if consecutive_optimal >= target_optimal_episodes:
        print(f"\n🎉 SUCCÈS: Agent optimisé en {agent.episode_count} épisodes!")
    else:
        print(f"\n⚠️  ATTENTION: Objectif non atteint après {max_episodes} épisodes")
        print(f"   Épisodes optimaux consécutifs: {consecutive_optimal}/{target_optimal_episodes}")
    
    print("=" * 80)
    
    # Sauvegarde de l'agent si demandé
    if save_agent:
        filename = f"qlearning_agent_{environment_name}_{int(time.time())}.pkl"
        agent.save(filename)
    
    return agent


def test_optimized_agent(
    agent: OptimizedQLearningAgent,
    environment_name: str = "maze",
    num_tests: int = 5,
    show_gui: bool = True,
    verbose: bool = True
) -> Tuple[float, List[Tuple[float, int, bool]]]:
    """
    Teste l'agent optimisé
    
    Args:
        agent: Agent à tester
        environment_name: Nom de l'environnement
        num_tests: Nombre de tests à effectuer
        show_gui: Afficher l'interface graphique
        verbose: Afficher les détails de chaque étape
        
    Returns:
        Tuple (taux de succès, résultats des tests)
    """
    test_results: List[Tuple[float, int, bool]] = []
    success_count = 0
    
    print("\n" + "=" * 80)
    print(f"PHASE DE TEST DE L'AGENT OPTIMISÉ")
    print("=" * 80)
    
    # Sauvegarde de l'epsilon original
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pas d'exploration pendant les tests
    
    for test_num in range(num_tests):
        print(f"\n🧪 TEST {test_num + 1}/{num_tests}")
        print("=" * 70)
        
        game = RLGame(environment_name, gui=show_gui)
        turtle = game.start()
        
        reward, steps, success, is_optimal = agent.train_episode_optimized(
            game, turtle, show_gui, verbose
        )
        test_results.append((reward, steps, success))
        
        if success:
            success_count += 1
        
        status = "✅ Succès" if success else "❌ Échec"
        optimal_status = "🏆 Optimal" if is_optimal else "📊 Standard"
        print(f"\nRésultat du test {test_num + 1}: {status} | {optimal_status}")
        print(f"  - Étapes: {steps}")
        print(f"  - Récompense: {reward:.1f}")
    
    # Restauration de l'epsilon
    agent.epsilon = original_epsilon
    
    success_rate = success_count / num_tests
    
    print("\n" + "=" * 80)
    print(f"RÉSULTATS DES TESTS")
    print("=" * 80)
    print(f"Taux de succès: {success_rate:.1%} ({success_count}/{num_tests})")
    if success_count > 0:
        avg_steps = sum(r[1] for r in test_results if r[2]) / success_count
        print(f"Étapes moyennes (succès): {avg_steps:.1f}")
        print(f"Meilleur score: {min(r[1] for r in test_results if r[2])} étapes")
    print("=" * 80)
    
    return success_rate, test_results


def main():
    """Point d'entrée principal du programme"""
    print("=" * 80)
    print("Q-LEARNING OPTIMISÉ POUR DONATELLOPYZZA")
    print("=" * 80)
    print("Objectif: Entraîner la tortue jusqu'à trouver le chemin optimal")
    print("(le moins de cases possible vers la pizza)")
    
    environments = ["maze", "assessment_maze", "hard_maze", "line", "test"]
    
    print("\nEnvironnements disponibles:")
    for i, env in enumerate(environments, 1):
        print(f"  {i}. {env}")
    
    try:
        choice = int(input("\nChoisissez un environnement (1-5) [défaut: 1]: ") or "1")
        environment_name = environments[choice - 1] if 1 <= choice <= len(environments) else "maze"
    except (ValueError, IndexError):
        environment_name = "maze"
        print(f"Utilisation de l'environnement par défaut: {environment_name}")
    
    try:
        max_episodes = int(input("Nombre maximum d'épisodes [défaut: 500]: ") or "500")
        target_optimal = int(input("Épisodes optimaux consécutifs requis [défaut: 5]: ") or "5")
        show_gui = input("Afficher l'interface graphique ? (o/n) [défaut: o]: ").lower() != 'n'
        verbose = input("Affichage verbose (détails de chaque étape) ? (o/n) [défaut: o]: ").lower() != 'n'
        save_agent = input("Sauvegarder l'agent après l'entraînement ? (o/n) [défaut: n]: ").lower() == 'o'
    except ValueError:
        max_episodes = 500
        target_optimal = 5
        show_gui = True
        verbose = True
        save_agent = False
    
    print(f"\n⚙️  Configuration:")
    print(f"  - Environnement: {environment_name}")
    print(f"  - Épisodes max: {max_episodes}")
    print(f"  - Épisodes optimaux requis: {target_optimal}")
    print(f"  - Interface graphique: {'Oui' if show_gui else 'Non'}")
    print(f"  - Mode verbose: {'Oui' if verbose else 'Non'}")
    print(f"  - Sauvegarde: {'Oui' if save_agent else 'Non'}")
    print(f"  - Délai entre étapes: 0.01s (fixe)")
    print(f"  - Limite d'étapes: Aucune (continue jusqu'à la pizza)")
    
    # Entraînement optimisé
    agent = train_until_optimal(
        environment_name=environment_name,
        max_episodes=max_episodes,
        show_gui=show_gui,
        verbose=verbose,
        target_optimal_episodes=target_optimal,
        save_agent=save_agent
    )
    
    # Tests de l'agent optimisé
    test_choice = input("\nEffectuer des tests de l'agent ? (o/n) [défaut: o]: ").lower() != 'n'
    
    if test_choice:
        num_tests = int(input("Nombre de tests [défaut: 3]: ") or "3")
        test_verbose = input("Affichage verbose pour les tests ? (o/n) [défaut: n]: ").lower() == 'o'
        
        success_rate, test_results = test_optimized_agent(
            agent=agent,
            environment_name=environment_name,
            num_tests=num_tests,
            show_gui=show_gui,
            verbose=test_verbose
        )
    
    print("\n🎉 Programme terminé!")
    print(f"Agent final avec {agent.get_statistics()['q_table_size']} états appris")
    print(f"Meilleur score atteint: {agent.get_statistics()['best_steps']} étapes")


if __name__ == "__main__":
    main()
