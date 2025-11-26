import matplotlib.pyplot as plt
import numpy as np

# 1. Créer le dataset
# X: Les entrées (input)
# Y: Les sorties attendues (labels) - Classes attribuées manuellement

X = [
    # Points classiques
    (0, 0),      # Point 1
    (0, 1),      # Point 2
    (1, 0),      # Point 3
    (1, 1),      # Point 4
    # Points supplémentaires avec valeurs décimales
    (0.2, 0.1),  # Point 5
    (0.1, 0.2),  # Point 6
    (0.3, 0.2),  # Point 7
    (0.2, 0.3),  # Point 8
    (0.8, 0.7),  # Point 9
    (0.7, 0.8),  # Point 10
    (0.9, 0.9),  # Point 11
    (0.6, 0.6),  # Point 12
    (0.4, 0.5),  # Point 13
    (0.5, 0.4),  # Point 14
    # Points avec valeurs négatives
    (-0.2, 0.3), # Point 15
    (0.3, -0.2), # Point 16
    (-0.1, -0.1), # Point 17
    # Points avec valeurs > 1
    (1.2, 0.8),  # Point 18
    (0.8, 1.2),  # Point 19
    (1.3, 1.1),  # Point 20
    # Points intermédiaires
    (0.25, 0.75), # Point 21
    (0.75, 0.25), # Point 22
    (0.15, 0.85), # Point 23
    (0.85, 0.15), # Point 24
]

# Classes attribuées manuellement (en dur) pour chaque point
# Le perceptron devra apprendre à classifier ces points
Y = [
    0,  # (0, 0) -> Classe 0
    0,  # (0, 1) -> Classe 0
    0,  # (1, 0) -> Classe 0
    1,  # (1, 1) -> Classe 1
    # Points supplémentaires
    0,  # (0.2, 0.1) -> Classe 0
    0,  # (0.1, 0.2) -> Classe 0
    0,  # (0.3, 0.2) -> Classe 0
    0,  # (0.2, 0.3) -> Classe 0
    1,  # (0.8, 0.7) -> Classe 1
    1,  # (0.7, 0.8) -> Classe 1
    1,  # (0.9, 0.9) -> Classe 1
    1,  # (0.6, 0.6) -> Classe 1
    0,  # (0.4, 0.5) -> Classe 0
    0,  # (0.5, 0.4) -> Classe 0
    # Points avec valeurs négatives
    0,  # (-0.2, 0.3) -> Classe 0
    0,  # (0.3, -0.2) -> Classe 0
    0,  # (-0.1, -0.1) -> Classe 0
    # Points avec valeurs > 1
    1,  # (1.2, 0.8) -> Classe 1
    1,  # (0.8, 1.2) -> Classe 1
    1,  # (1.3, 1.1) -> Classe 1
    # Points intermédiaires
    0,  # (0.25, 0.75) -> Classe 0
    0,  # (0.75, 0.25) -> Classe 0
    0,  # (0.15, 0.85) -> Classe 0
    0,  # (0.85, 0.15) -> Classe 0
]

# Note: Les classes sont attribuées manuellement
# Le perceptron devra apprendre à séparer ces classes
print(f"Dataset: {len(X)} points avec classes attribuées manuellement\n")

# Hyperparamètres
TAUX_APPRENTISSAGE = 0.1
EPOQUES = 10  # Nombre de cycles d'entraînement

# 2. Initialiser un paramètre w et un biais b
# w1 et w2 correspondent aux deux entrées (x1 et x2)
w1, w2 = 0.0, 0.0
b = 0.0  # Biais

# Fonction d'activation (Step Function ou Fonction Échelon)
# Le perceptron utilise cette fonction : 1 si la somme est >= 0, sinon 0.
def activation(somme):
    return 1 if somme >= 0 else 0

print("--- Début de l'entraînement ---")

# Boucle d'entraînement
for epoque in range(EPOQUES):
    erreurs = 0
    
    # Parcourir chaque exemple du dataset
    for (x1, x2), y_reel in zip(X, Y):
        
        # 3. Faire prédire un y_hat à votre neurone
        # Calcul de la somme pondérée (dot product + biais)
        somme_ponderee = (x1 * w1) + (x2 * w2) + b
        y_predit = activation(somme_ponderee)
        
        # 4. Calculer l'erreur
        erreur = y_reel - y_predit
        
        # Si la prédiction est fausse, mettre à jour les poids
        if erreur != 0:
            erreurs += 1
            
            # 5. Mettre à jour w et b (Règle d'apprentissage du Perceptron)
            # Mise à jour des poids: w_nouveau = w_ancien + alpha * erreur * entrée
            w1 = w1 + TAUX_APPRENTISSAGE * erreur * x1
            w2 = w2 + TAUX_APPRENTISSAGE * erreur * x2
            
            # Mise à jour du biais: b_nouveau = b_ancien + alpha * erreur
            b = b + TAUX_APPRENTISSAGE * erreur
    
    print(f"Époque {epoque+1}/{EPOQUES}: {erreurs} erreurs. Poids (w1, w2): ({round(w1, 2)}, {round(w2, 2)}), Biais (b): {round(b, 2)}")
    
    # Si aucune erreur n'est faite, le perceptron a appris, on arrête l'entraînement.
    if erreurs == 0:
        print("L'entraînement a convergé.")
        break

print("--- Fin de l'entraînement ---")

# --- Test du Perceptron entraîné ---
print("\n--- Test final du Perceptron ---")
for (x1, x2), y_reel in zip(X, Y):
    somme_ponderee = (x1 * w1) + (x2 * w2) + b
    y_predit = activation(somme_ponderee)
    print(f"Entrée ({x1}, {x2}) | Prédit: {y_predit} | Réel: {y_reel}")

# --- Visualisation graphique ---
print("\n--- Création du graphique ---")

# Créer la figure simple
fig, ax = plt.subplots(figsize=(8, 8))

# Tracer les points
points_classe_0 = [(x1, x2) for (x1, x2), y in zip(X, Y) if y == 0]
points_classe_1 = [(x1, x2) for (x1, x2), y in zip(X, Y) if y == 1]

# Points de classe 0
if points_classe_0:
    x1_0, x2_0 = zip(*points_classe_0)
    ax.scatter(x1_0, x2_0, c='red', s=100, label='Classe 0', marker='o')

# Points de classe 1
if points_classe_1:
    x1_1, x2_1 = zip(*points_classe_1)
    ax.scatter(x1_1, x2_1, c='green', s=100, label='Classe 1', marker='o')

# Tracer la frontière de décision
# La frontière est définie par: w1*x1 + w2*x2 + b = 0
x_min_plot = min(x[0] for x in X) - 0.2
x_max_plot = max(x[0] for x in X) + 0.2
if w2 != 0:
    x1_line = np.linspace(x_min_plot, x_max_plot, 100)
    x2_line = -(w1 * x1_line + b) / w2
    ax.plot(x1_line, x2_line, color='blue', linestyle='--', 
            linewidth=2, label='Frontiere de decision')
elif w1 != 0:
    x1_line = -b / w1
    ax.axvline(x=x1_line, color='blue', linestyle='--', 
               linewidth=2, label='Frontiere de decision')

# Configuration du graphique - Ajuster les limites pour inclure tous les points
x_min = min(x[0] for x in X) - 0.2
x_max = max(x[0] for x in X) + 0.2
y_min = min(x[1] for x in X) - 0.2
y_max = max(x[1] for x in X) + 0.2
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_title('Perceptron - Classification avec classes attribuées manuellement', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
ax.set_aspect('equal')

# Ajouter les informations sur les poids finaux
info_text = f'Poids finaux:\nw1 = {w1:.1f}\nw2 = {w2:.1f}\nb = {b:.1f}'
ax.text(0.98, 0.98, info_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Sauvegarder le graphique
import os
os.makedirs('Images', exist_ok=True)
chemin_image = os.path.join('Images', 'perceptron_graph_classes_manuelles.png')
plt.savefig(chemin_image, dpi=150, bbox_inches='tight')
print(f"Graphique sauvegardé sous '{chemin_image}'")

# Afficher le graphique
plt.show()

print("\n--- Fin du script ---")

