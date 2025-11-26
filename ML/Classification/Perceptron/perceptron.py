import numpy as np
import matplotlib.pyplot as plt
import sys
import io
import os

# Configuration de l'encodage UTF-8 pour Windows (pour éviter les erreurs d'affichage dans la console)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class Perceptron:
    """
    Implémentation d'un perceptron simple (Classifieur Linéaire)
    
    Structure :
    - Net input = x₁w₁ + x₂w₂ + ... + xₘwₘ + b
    - Output = f(net input) où f est une fonction d'activation (step function)
    """
    
    def __init__(self, n_features, learning_rate=0.1, max_iter=100, random_state=None):
        """
        Initialise le perceptron
        
        Parameters:
        -----------
        n_features : int
            Nombre de features (entrées)
        learning_rate : float
            Taux d'apprentissage α (par défaut 0.1)
        max_iter : int
            Nombre maximum d'itérations d'entraînement
        random_state : int, optional
            Graine pour la génération aléatoire (reproductibilité)
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        # Initialisation aléatoire contrôlée
        if random_state:
            np.random.seed(random_state)
            
        # Initialiser les poids w et le biais b (petites valeurs proches de 0)
        self.w = np.random.uniform(-0.5, 0.5, n_features)
        self.b = np.random.uniform(-0.5, 0.5)
        
        # Historique pour visualisation
        self.errors_history = []
        
    def net_input(self, X):
        """Calcule le net input : z = w·x + b"""
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        """
        Prédit la sortie ŷ en utilisant une fonction d'activation step.
        Retourne 1 si z >= 0, sinon 0.
        """
        net = self.net_input(X)
        return np.where(net >= 0, 1, 0)
    
    def fit(self, X, y):
        """
        Entraîne le perceptron (Algorithme de Rosenblatt)
        
        Mise à jour :
        w ← w + α(y - ŷ)x
        b ← b + α(y - ŷ)
        """
        X = np.array(X)
        y = np.array(y)
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Dim incorrecte. Attendu: {self.n_features}, Reçu: {X.shape[1]}")
        
        self.errors_history = []
        
        for iteration in range(self.max_iter):
            errors = 0
            
            # Apprentissage stochastique (exemple par exemple)
            for i in range(len(X)):
                # 1. Prédire
                y_pred = self.predict(X[i])
                
                # 2. Calculer l'erreur
                error = y[i] - y_pred
                
                # 3. Mettre à jour si erreur
                if error != 0:
                    self.w += self.learning_rate * error * X[i]
                    self.b += self.learning_rate * error
                    errors += 1
            
            self.errors_history.append(errors)
            
            # Critère d'arrêt : convergence (0 erreur)
            if errors == 0:
                print(f"  -> Convergence atteinte après {iteration + 1} époques.")
                break
                
        return self
    
    def score(self, X, y):
        """Calcule l'accuracy (taux de réussite)"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# --- Fonctions Utilitaires ---

def create_logic_gate_dataset(gate_type):
    """Génère les données pour AND, OR, XOR"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    gate_map = {
        'AND': np.array([0, 0, 0, 1]),
        'OR':  np.array([0, 1, 1, 1]),
        'XOR': np.array([0, 1, 1, 0])
    }
    
    if gate_type.upper() not in gate_map:
        raise ValueError(f"Porte inconnue : {gate_type}")
        
    return X, gate_map[gate_type.upper()]

def visualize_results(X, y, perceptron, gate_type, save_path=None):
    """Affiche la frontière de décision et la courbe d'apprentissage"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # -- Graphique 1 : Frontière de décision --
    ax1 = axes[0]
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    resolution = 0.02
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Prédire sur toute la grille
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Affichage des zones
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap='bwr') # Blue-White-Red
    ax1.contour(xx, yy, Z, colors='k', linewidths=0.5) # Ligne de séparation
    
    # Affichage des points réels
    # 0 = Bleu, 1 = Rouge
    colors = ['blue' if label == 0 else 'red' for label in y]
    ax1.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolors='k')
    
    ax1.set_title(f'Frontière de décision ({gate_type})')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    
    # -- Graphique 2 : Historique des erreurs --
    ax2 = axes[1]
    ax2.plot(range(1, len(perceptron.errors_history) + 1), perceptron.errors_history, marker='o')
    ax2.set_title('Convergence (Erreurs par époque)')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Nb Erreurs')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"  [Info] Graphique sauvegardé sous : {save_path}")
    
    # Commenter cette ligne si vous lancez le script sur un serveur sans écran
    # plt.show() 
    plt.close()

def run_test(gate_type):
    print(f"\n{'='*50}")
    print(f"TEST : PORTE {gate_type}")
    print(f"{'='*50}")
    
    # 1. Données
    X, y = create_logic_gate_dataset(gate_type)
    
    # 2. Modèle (seed fixée pour avoir toujours le même résultat pédagogique)
    ppn = Perceptron(n_features=2, learning_rate=0.1, max_iter=20, random_state=1)
    
    # 3. Entraînement
    print(f"Entraînement en cours...")
    ppn.fit(X, y)
    
    # 4. Évaluation
    acc = ppn.score(X, y)
    print(f"Précision finale : {acc * 100:.1f}%")
    print(f"Poids finaux : w={ppn.w}, b={ppn.b}")
    
    # 5. Visualisation
    visualize_results(X, y, ppn, gate_type, save_path=f"result_{gate_type}.png")
    
    return acc

def main():
    print("--- DÉMARRAGE DU PERCEPTRON ---")
    
    # Test AND (Séparable linéairement)
    run_test('AND')
    
    # Test OR (Séparable linéairement)
    run_test('OR')
    
    # Test XOR (Non séparable linéairement)
    print("\n--- ATTENTION : LE CAS DU XOR ---")
    acc_xor = run_test('XOR')
    
    if acc_xor < 1.0:
        print("\n>>> CONCLUSION SUR XOR :")
        print("Comme prévu, le perceptron simple ÉCHOUE sur le XOR (Précision < 100%).")
        print("Raison : Le XOR n'est pas linéairement séparable (on ne peut pas tracer une droite unique).")
        print("Solution : Utiliser un réseau de neurones multicouches (MLP).")

if __name__ == "__main__":
    main()