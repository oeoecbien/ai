# Apprentissage supervisé, réseaux de neurones et renforcement

**Version enrichie avec illustrations conceptuelles (ASCII avancées)**

---

# 0. Introduction à l'intelligence artificielle

## 0.1 Définir l'intelligence artificielle

L'intelligence artificielle (IA) évolue depuis les années 1950 :

```
1950 : Artificial Intelligence
       └─> Démonstration du comportement humain par la machine

1980 : Machine Learning (ML) / Apprentissage machine
       └─> Apprentissage à partir de données

2010 : Deep Learning (DL)
       └─> Modèles de ML qui imitent le fonctionnement du cerveau

2020 : Generative AI
       └─> Modèles de DL qui créent du contenu original
```

## 0.2 Les grands types d'IA

```
┌─────────────────────────────────────────────────────────┐
│ 1. Le supervisé                                         │
│    ├─> La régression (prédire une valeur continue)      │
│    └─> La classification (prédire une classe)           │
│                                                         │
│ 2. Le non-supervisé                                     │
│    ├─> Clustering                                       │
│    ├─> Réduction de dimensions                          │
│    └─> Apprentissage de représentations                 │
│                                                         │
│ 3. Auto-supervisé                                       │
│    └─> Modélisation du langage                          │
│                                                         │
│ 4. Le renforcement                                      │
│    └─> Prise de décision                                │
└─────────────────────────────────────────────────────────┘
```

---

# 1. Apprentissage supervisé

## 1.1 Principe général  
Objectif : apprendre une fonction \(f : X \rightarrow Y\) à partir d'exemples étiquetés \((x^{(i)}, y^{(i)})\).

```
Données annotées
┌───────────────────────────┐
│ (x1,y1), (x2,y2), ...     │
└───────────────────────────┘
│
▼
┌─────────────────┐
│   Modèle hθ(x)  │
└─────────────────┘
│
▼
┌──────────────────────────┐
│ Prédiction ŷ = hθ(x)     │
└──────────────────────────┘
```

---

## 1.2 La pipeline d'un projet ML

Étapes essentielles d'un projet d'apprentissage automatique :

```
┌──────────────────────────────────────────────────────────┐
│ 1. Préparer les données (70% du travail)                 │
│    ├─> Homogénéiser leur format (même unité)             │
│    ├─> Supprimer les données incomplètes                 │
│    ├─> Supprimer les anomalies                           │
│    └─> Éventuellement homogénéiser leur distribution     │
│                                                          │
│ 2. Entraînement (30% du travail)                         │
│    ├─> Définir le modèle                                 │
│    │   (nb entrées / sorties / couches cachées)          │
│    └─> Choisir les hyperparamètres                       │
│        (e.g., taux d'apprentissage)                      │
│                                                          │
│ 3. Évaluation                                            │
│    ├─> Usage d'un jeu de validation de manière itérative │
│    ├─> Ajuster les choix des hyperparamètres             │
│    └─> Comparer plusieurs modèles                        │
│                                                          │
│ 4. Test                                                  │
│    └─> Usage d'un jeu de test                            │
│        Teste une unique fois le modèle final             │
│        sur de nouvelles données                          │
└──────────────────────────────────────────────────────────┘
```

### 1.2.1 Pipeline de développement ML détaillé

```
┌─────────────────────────────────────────────────────────────┐
│ 1. COMPRÉHENSION DU PROBLÈME                                │
│    ├─> Définir l'objectif métier                            │
│    ├─> Identifier les métriques de succès                   │
│    └─> Estimer la faisabilité                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. COLLECTE ET EXPLORATION DES DONNÉES                      │
│    ├─> Identifier les sources de données                    │
│    ├─> Analyse exploratoire (EDA)                           │
│    └─> Vérifier la qualité des données                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. PRÉPARATION DES DONNÉES (70% du temps)                   │
│    ├─> Nettoyage (valeurs manquantes, outliers)             │
│    ├─> Feature engineering                                  │
│    ├─> Normalisation/Standardisation                        │
│    └─> Split train/validation/test                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. MODÉLISATION                                             │
│    ├─> Commencer simple (baseline)                          │
│    ├─> Itérer avec des modèles plus complexes               │
│    └─> Validation croisée                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ÉVALUATION                                               │
│    ├─> Métriques appropriées au problème                    │
│    ├─> Analyse des erreurs                                  │
│    └─> Test sur le jeu de test final                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. DÉPLOIEMENT ET MONITORING                                │
│    ├─> Mise en production                                   │
│    ├─> Surveillance des performances                        │
│    └─> Réentraînement périodique                            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2.2 Checklist avant entraînement

- [ ] Les données sont-elles représentatives du problème ?
- [ ] Y a-t-il des valeurs manquantes ou aberrantes ?
- [ ] Les features sont-elles normalisées/standardisées ?
- [ ] Le dataset est-il divisé correctement (train/val/test) ?
- [ ] Les classes sont-elles équilibrées (classification) ?
- [ ] Un modèle baseline a-t-il été établi ?
- [ ] Les hyperparamètres ont-ils été choisis judicieusement ?

### 1.2.3 Overfitting vs Underfitting

```
                Performance
                    ↑
                    |     ┌─── Overfitting
                    |    /│    (trop complexe)
                    |   / │
   Bon équilibre ───┼──●  │
                    | /   │
                    |/    └─── Underfitting
                    |          (trop simple)
                    └──────────────────────→
                        Complexité du modèle
```

**Solutions :**
- **Overfitting** → Régularisation, plus de données, dropout
- **Underfitting** → Modèle plus complexe, plus de features

## 1.3 Régression

Prédire une valeur continue.

**Exemples d'applications :**
- Prédire le prix d'un appartement
- Prédire le nombre d'appels hebdomadaires pour le support utilisateur
- Prédire les pannes des produits installés chez les clients

**Fonction hypothèse (régression linéaire) :**

La fonction approximée se nomme *h* (hypothèse) :

$$f(X) = aX + b \quad \Leftrightarrow \quad h(X) = \Theta_0 + \Theta_1 X$$

où :
- \(\Theta_0\) et \(\Theta_1\) sont les paramètres qu'on doit déterminer
- \(X\) est l'ensemble de nos entrées (abscisse)
- \(Y\) est l'ensemble de nos sorties (ordonnée)

**Formalisation mathématique :**

- \(\Theta\) : paramètres
- \(m\) : le nombre de données d'entraînement
- \(X\) : les features (entrées)
- \(Y\) : les targets (sorties)
- \((X, Y)\) : une donnée d'entraînement

Si on a \(n\) entrées telles que \(X=\left\{X_{0}, X_{1}, \ldots, X_{n}\right\}\) alors :

$$h(X)=\Theta_{0}+\Theta_{1} X_{1}+\Theta_{2} X_{2}+\ldots+\Theta_{n} X_{n} \approx Y$$

Version compactée : \(h(X)=\sum_{j=0}^{|X|} \Theta_{j} X_{j} \quad\) avec \(X_{0}=1\)

```
Entrées : x = [surface, quartier, étage]
          │
          ▼
┌─────────────────────────┐
│ hθ(x) = θ₀ + θ₁x₁ + …   │
└─────────────────────────┘
          │
          ▼
      Sortie : y = prix
```

### 1.3.1 Implémentation pratique : Régression linéaire

**Exemple complet avec dataset Salaire/Expérience :**

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger les données
data = np.array([
    [1.1, 39343], [1.3, 46205], [1.5, 37731],
    [2.0, 43525], [2.2, 39891], [2.9, 56642],
    [3.0, 60150], [3.2, 54445], [3.2, 64445],
    [3.7, 57189], [3.9, 63218], [4.0, 55794]
])

X = data[:, 0]  # Années d'expérience
Y = data[:, 1]  # Salaire

# 2. Normalisation (optionnelle mais recommandée)
X_mean, X_std = X.mean(), X.std()
Y_mean, Y_std = Y.mean(), Y.std()
X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# 3. Initialisation des paramètres
theta_0 = 0.0  # biais
theta_1 = 0.0  # poids
alpha = 0.01   # taux d'apprentissage
iterations = 1000

# 4. Fonction hypothèse
def h(x, theta_0, theta_1):
    return theta_0 + theta_1 * x

# 5. Fonction de coût
def cost_function(X, Y, theta_0, theta_1):
    m = len(X)
    predictions = h(X, theta_0, theta_1)
    return (1/(2*m)) * np.sum((predictions - Y)**2)

# 6. Descente de gradient
costs = []
for i in range(iterations):
    m = len(X_norm)
    predictions = h(X_norm, theta_0, theta_1)
    error = predictions - Y_norm
    
    # Mise à jour des paramètres
    theta_0 -= alpha * (1/m) * np.sum(error)
    theta_1 -= alpha * (1/m) * np.sum(error * X_norm)
    
    # Enregistrer le coût
    costs.append(cost_function(X_norm, Y_norm, theta_0, theta_1))

# 7. Visualisation
plt.figure(figsize=(12, 4))

# Graphique 1 : Données et régression
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', label='Données')
X_line = np.linspace(X.min(), X.max(), 100)
X_line_norm = (X_line - X_mean) / X_std
Y_line_norm = h(X_line_norm, theta_0, theta_1)
Y_line = Y_line_norm * Y_std + Y_mean
plt.plot(X_line, Y_line, color='red', label='Régression')
plt.xlabel('Années d\'expérience')
plt.ylabel('Salaire')
plt.legend()
plt.title('Régression linéaire')

# Graphique 2 : Évolution du coût
plt.subplot(1, 2, 2)
plt.plot(costs)
plt.xlabel('Itérations')
plt.ylabel('Coût J(θ)')
plt.title('Convergence de la descente de gradient')
plt.tight_layout()
plt.show()

print(f"Paramètres finaux : θ₀ = {theta_0:.4f}, θ₁ = {theta_1:.4f}")
```

### 1.3.2 Cas d'usage : Prédiction de séries temporelles

```python
import numpy as np
import matplotlib.pyplot as plt

# Génération de données de ventes mensuelles
np.random.seed(42)
months = np.arange(1, 25)
trend = 100 + 5 * months
seasonality = 20 * np.sin(2 * np.pi * months / 12)
noise = np.random.randn(24) * 5
sales = trend + seasonality + noise

# Régression linéaire simple pour la tendance
X = months.reshape(-1, 1)
y = sales

# Calcul des paramètres
X_mean = X.mean()
y_mean = y.mean()
theta_1 = np.sum((X.flatten() - X_mean) * (y - y_mean)) / np.sum((X.flatten() - X_mean)**2)
theta_0 = y_mean - theta_1 * X_mean

# Prédiction
y_pred = theta_0 + theta_1 * X.flatten()

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(months, sales, 'o', label='Données réelles')
plt.plot(months, y_pred, '-', label='Tendance linéaire')
plt.xlabel('Mois')
plt.ylabel('Ventes')
plt.title('Prédiction de ventes avec régression linéaire')
plt.legend()
plt.grid(True)
plt.show()

print(f"Équation : ventes = {theta_0:.2f} + {theta_1:.2f} × mois")
```

### 1.3.3 Métriques d'évaluation pour la régression

```python
import numpy as np

def mse(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """Coefficient de détermination R²"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)
```

---

## 1.6 Classification

Prédire une **classe** (exemple : reconnaître un chiffre MNIST).

**Exemples d'applications :**
- Trier des légumes automatiquement
- Prédiction des chiffres lus sur des images

```
Image 28×28 pixels
       │
       ▼
┌───────────────────┐
│ Réseau de neurones│
└───────────────────┘
       │
       ▼
[P(0), P(1), ..., P(9)]
       │
       ▼
    argmax
       │
       ▼
   {0, 1, ..., 9}
```

### 1.6.1 Cas d'usage : Classification d'images (MNIST simplifié)

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Chargement des données
digits = load_digits()
X, y = digits.data, digits.target

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Forme des données d'entraînement : {X_train.shape}")
print(f"Forme des données de test : {X_test.shape}")
print(f"Nombre de classes : {len(np.unique(y))}")
```

### 1.6.2 Métriques d'évaluation pour la classification

```python
def accuracy(y_true, y_pred):
    """Exactitude (accuracy)"""
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, n_classes):
    """Matrice de confusion"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm

def precision_recall(cm, class_idx):
    """Précision et rappel pour une classe"""
    tp = cm[class_idx, class_idx]
    fp = cm[:, class_idx].sum() - tp
    fn = cm[class_idx, :].sum() - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall
```

## 1.7 Le perceptron

### 1.7.1 Structure du perceptron

Le perceptron est un classifieur linéaire, unité de base des réseaux de neurones.

**Calcul :**

```
Entrées : x₁, x₂, ..., xₘ
    │      │         │
    ▼      ▼         ▼
   w₁     w₂        wₘ  (poids)
    \      |         /
     \     |        /
      \    |       /
       \   |      /
        \  |     /
         \ |    /
      ┌──────────────┐
      │ Σ wᵢxᵢ + b   │  (net input)
      └──────────────┘
            │
            ▼
        ┌───────┐
        │ f(·)  │  (fonction d'activation)
        └───────┘
            │
            ▼
         Output
```

**Formule :**

$$\text{Net input} = x_1w_1 + x_2w_2 + \ldots + x_mw_m + b$$

$$\text{Output} = f(\text{net input})$$

### 1.7.2 Exemple : Porte logique AND

| \(x_1\) | \(x_2\) | \(y\) |
|---------|---------|-------|
| 0       | 0       | 0     |
| 0       | 1       | 0     |
| 1       | 0       | 0     |
| 1       | 1       | 1     |

**Solution :** \(w_1 = w_2 = 1\) et \(b = -1.5\)

**Exemple de calcul :**
- Pour \(x_1 = 1\) et \(x_2 = 0\) :
  - Net input = \(1 \times 1 + 0 \times 1 - 1.5 = -0.5\)
  - \(f(-0.5) = 0\)

### 1.7.3 Exemple : Porte logique OR

| \(x_1\) | \(x_2\) | \(y\) |
|---------|---------|-------|
| 0       | 0       | 0     |
| 0       | 1       | 1     |
| 1       | 0       | 1     |
| 1       | 1       | 1     |

**Exercice :** Proposer des poids \(w_1, w_2\) et un biais \(b\) pour que le perceptron retourne la bonne réponse.

### 1.7.4 Apprentissage du perceptron

Ajustement des poids et du biais :

$$
\begin{aligned}
w_{i} &\leftarrow w_{i}+\alpha(y-\hat{y}) x_{i} \\
b &\leftarrow b+\alpha(y-\hat{y})
\end{aligned}
$$

où :
- \(\alpha\) est le taux d'apprentissage
- \(y\) est la sortie attendue
- \(\hat{y}\) est la sortie prédite

**Processus d'apprentissage :**

```
1. Initialiser w et b
       │
       ▼
2. Pour chaque exemple (x, y) :
       │
       ├─> Calculer ŷ = f(Σwᵢxᵢ + b)
       │
       ├─> Calculer l'erreur : y - ŷ
       │
       └─> Mettre à jour :
           wᵢ ← wᵢ + α(y - ŷ)xᵢ
           b ← b + α(y - ŷ)
       │
       ▼
3. Répéter jusqu'à convergence
```

### 1.7.5 Limites du perceptron

**Exemple : Porte logique XOR**

| \(x_1\) | \(x_2\) | \(y\) |
|---------|---------|-------|
| 0       | 0       | 0     |
| 0       | 1       | 1     |
| 1       | 0       | 1     |
| 1       | 1       | 0     |

**Problème :** Le perceptron ne peut pas apprendre le XOR car il s'agit d'un problème non linéairement séparable.

**Solution :** Utiliser plusieurs couches de perceptrons (réseau multicouche).

### 1.7.6 Implémentation complète du perceptron

```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        """
        Initialise le perceptron
        n_inputs: nombre d'entrées
        learning_rate: taux d'apprentissage α
        """
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
        self.alpha = learning_rate
    
    def activation(self, x):
        """Fonction d'activation en escalier"""
        return 1 if x > 0 else 0
    
    def predict(self, inputs):
        """Prédiction pour un ensemble d'entrées"""
        net_input = np.dot(inputs, self.weights) + self.bias
        return self.activation(net_input)
    
    def train(self, X, y, epochs=100):
        """
        Entraînement du perceptron
        X: matrice des entrées (n_samples, n_features)
        y: vecteur des sorties attendues
        epochs: nombre d'itérations sur le dataset
        """
        errors_history = []
        
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                # Prédiction
                prediction = self.predict(inputs)
                
                # Calcul de l'erreur
                error = target - prediction
                total_error += abs(error)
                
                # Mise à jour des poids et biais
                self.weights += self.alpha * error * inputs
                self.bias += self.alpha * error
            
            errors_history.append(total_error)
            
            # Arrêt si convergence
            if total_error == 0:
                print(f"Convergence atteinte à l'époque {epoch}")
                break
        
        return errors_history

# Test sur la porte AND
print("=== Test porte AND ===")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron_and = Perceptron(n_inputs=2, learning_rate=0.1)
errors = perceptron_and.train(X_and, y_and, epochs=100)

print("\nRésultats pour AND:")
for inputs, target in zip(X_and, y_and):
    pred = perceptron_and.predict(inputs)
    print(f"  {inputs} → {pred} (attendu: {target})")

# Test sur la porte OR
print("\n=== Test porte OR ===")
y_or = np.array([0, 1, 1, 1])

perceptron_or = Perceptron(n_inputs=2, learning_rate=0.1)
errors = perceptron_or.train(X_and, y_or, epochs=100)

print("\nRésultats pour OR:")
for inputs, target in zip(X_and, y_or):
    pred = perceptron_or.predict(inputs)
    print(f"  {inputs} → {pred} (attendu: {target})")

# Test sur la porte XOR (échec attendu)
print("\n=== Test porte XOR (non linéairement séparable) ===")
y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(n_inputs=2, learning_rate=0.1)
errors = perceptron_xor.train(X_and, y_xor, epochs=100)

print("\nRésultats pour XOR (échec attendu):")
for inputs, target in zip(X_and, y_xor):
    pred = perceptron_xor.predict(inputs)
    status = "✓" if pred == target else "✗"
    print(f"  {inputs} → {pred} (attendu: {target}) {status}")
```

---

## 1.4 Fonction de coût

Mesure l'écart entre prédictions et cibles.

**Objectif :** Minimiser la fonction \(J(\Theta)\) en trouvant les bons paramètres \(\Theta\)

**Fonction de coût (erreur quadratique) :**

$$
J(\Theta)=\frac{1}{2} \sum_{i=1}^{m}\left(h_{\Theta}\left(X^{i}\right)-y^{i}\right)^{2}
$$

où :
- \(m\) est le nombre d'exemples d'entraînement
- Le facteur \(\frac{1}{2}\) simplifie le calcul du gradient

**Illustration :**

```
Erreur pour chaque point
    ▲
    │     ● (x², y²)
    │    /
    │   /  erreur = h(x²) - y²
    │  /
    │ ● (x¹, y¹)
    │/
    └─────────────────────►
         h(X) = Θ₀ + Θ₁X
```

---

## 1.5 Descente de gradient

Algorithme itératif pour minimiser la fonction de coût \(J(\Theta)\).

**Principe :**
1. Initialiser \(\Theta\)
2. Modifier \(\Theta\) de façon à réduire \(J(\Theta)\)

**Algorithme :**

```
1. Initialiser Θ (poids aléatoires)
       │
       ▼
2. Calculer ∇J(Θ) (gradient)
       │
       ▼
3. Mettre à jour : Θⱼ ← Θⱼ - α Σᵢ ∂J(Θ)/∂Θⱼ
       │          pour j = 0, 1, ..., |X|
       │          (α = taux d'apprentissage ~ 10⁻³)
       ▼
4. Répéter jusqu'à convergence
```

**Formule de mise à jour :**

$$\Theta_{j}=\Theta_{j}-\alpha \sum_{i=1}^{m} \frac{\partial}{\partial \Theta_{j}} J(\Theta) \quad \text{pour } j=0,1, \ldots,|X|$$

où \(\alpha\) est une constante choisie au préalable (\(\sim 10^{-3}\)).

**Illustration du paysage de perte :**

```
Perte J(Θ)
    ▲
    │           o
    │        o
    │     o
    │  o
    └─────────────────────► Θ
         minimum global
```

### 1.5.1 Batch vs Mini-batch vs Stochastic

| Type | Taille batch | Avantages | Inconvénients |
|------|-------------|-----------|---------------|
| **Batch** | Tout le dataset | Convergence stable | Lent, beaucoup de mémoire |
| **Stochastic** | 1 exemple | Rapide, peu de mémoire | Convergence bruitée |
| **Mini-batch** | 32-256 exemples | Bon compromis | Nécessite tuning |

### 1.5.2 Régularisation

**Objectif :** Éviter le surapprentissage (overfitting)

#### L2 Regularization (Ridge)

```
Fonction de coût modifiée :
J(θ) = 1/2m Σᵢ(h(xⁱ) - yⁱ)² + λ/(2m) Σⱼθⱼ²

où λ est le paramètre de régularisation
```

#### L1 Regularization (Lasso)

```
J(θ) = 1/2m Σᵢ(h(xⁱ) - yⁱ)² + λ/m Σⱼ|θⱼ|
```

---

# 2. Réseaux de neurones

## 2.1 Neurone artificiel

Unité de calcul de base :

```
Entrées : x₁, x₂, x₃
    │      │      │
    ▼      ▼      ▼
   w₁     w₂     w₃  (poids)
    \      |      /
     \     |     /
      \    |    /
       \   |   /
        \  |  /
         \ | /
      ┌──────────┐
      │ Σ wᵢxᵢ + b│  (somme pondérée + biais)
      └──────────┘
            │
            ▼
        ┌───────┐
        │ f(·)  │  (fonction d'activation)
        └───────┘
            │
            ▼
         Sortie
```

### 2.1.1 Fonctions d'activation

Comparaison des principales fonctions d'activation :

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# 1. Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# 2. Tanh
tanh = np.tanh(x)

# 3. ReLU
relu = np.maximum(0, x)

# 4. Leaky ReLU
leaky_relu = np.where(x > 0, x, 0.01 * x)

# Visualisation
plt.figure(figsize=(12, 3))

plt.subplot(1, 4, 1)
plt.plot(x, sigmoid)
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(1, 4, 2)
plt.plot(x, tanh)
plt.title('Tanh')
plt.grid(True)

plt.subplot(1, 4, 3)
plt.plot(x, relu)
plt.title('ReLU')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x, leaky_relu)
plt.title('Leaky ReLU')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 2.2 Réseau multicouche (MLP)

Architecture en couches :

```
                    Architecture d'un réseau multicouche (MLP)

Couche d'entrée          Couche cachée 1        Couche cachée 2        Couche de sortie
┌─────────────┐         ┌─────────────┐         ┌─────────────┐         ┌──────────┐
│     x₁      │────────▶│    h₁¹      │───────▶│    h₁²      │───────▶│          │
│             │    ╱    │             │    ╱    │             │    ╱    │    ŷ     │
│     x₂      │───╱     │    h₂¹      │───╱     │    h₂²      │───╱     │          │
│             │  ╱      │             │  ╱      │             │  ╱      │          │
│     x₃      │─╱       │    h₃¹      │─╱       │    h₃²      │─╱       │          │
│             │         │             │         │             │         │          │
│     ...     │────────▶│    ...      │───────▶│    ...      │───────▶│          │
└─────────────┘         └─────────────┘         └─────────────┘         └──────────┘
       │                       │                      │                      │
       │                       │                      │                      │
       └───────────────────────┴──────────────────────┴──────────────────────┘
                    Chaque neurone est connecté à tous les neurones
                    de la couche suivante (fully connected)
                    avec des poids wᵢⱼ et des biais b
```

Chaque connexion a un poids associé.

---

## 2.3 Rétropropagation

Algorithme en deux phases :

**Phase 1 : Forward (propagation avant)**

```
x → couche 1 → couche 2 → ... → couche L → ŷ
```

**Phase 2 : Backward (propagation arrière)**

```
ŷ → calcul erreur → gradients → mise à jour θ
```

**Cycle complet :**

```
┌─────────────────────────────────────────┐
│ FORWARD PASS                            │
│ x ────────────────────────────────▶ ŷ  │
│                                         │
│         Calcul perte L(ŷ, y)            │
│                                         │
│ BACKWARD PASS                           │
│ ◀──────────────────────────────────────│
│ gradients propagés vers l'arrière       │
└─────────────────────────────────────────┘
```

---

# 3. Apprentissage par renforcement (RL)

## 3.1 Concept central

Boucle d'interaction agent-environnement :

```
┌──────────────────┐
│  Environnement   │
└───────▲──────────┘
        │ état s_t
        │ récompense r_t
        │
┌───────┴────────┐
│     Agent      │
│  (politique π) │
└───────┬────────┘
        │ action a_t = π(s_t)
        ▼
```

**Cycle :** \(s_t \rightarrow a_t \rightarrow r_t, s_{t+1} \rightarrow a_{t+1} \rightarrow ...\)

---

## 3.2 Processus de décision markovien (MDP)

Modèle formel défini par le tuple \((S, A, T, R, \gamma)\) :

```
S = États (espace d'états)
A = Actions (espace d'actions)
T = P(s' | s, a)  (probabilité de transition)
R = Récompense (fonction de récompense)
γ = facteur d'actualisation (0 ≤ γ ≤ 1)
```

**Exemple de transition :**

```
État s₁ ──a₁──▶ État s₂ ──a₂──▶ État s₃
  │ r=+1          │ r=-1
  ▼               ▼
amélioration    pénalité
```

---

## 3.3 Politique

La politique \(\pi\) définit le comportement de l'agent :

```
π(a|s) = probabilité de choisir l'action a dans l'état s
```

**Types de politiques :**
- **Déterministe :** \(\pi(s) = a\) (une action par état)
- **Stochastique :** \(\pi(a|s)\) (distribution de probabilités)

**Chaîne de décision complète :**

```
s₀ ──π──▶ a₀ ──▶ r₀, s₁ ──π──▶ a₁ ──▶ r₁, s₂ ──π──▶ ...
```

Objectif : trouver la politique optimale \(\pi^*\) qui maximise le retour cumulé.

---

## 3.4 Valeur d'état et équation de Bellman

$$
V^\pi(s)=E[r+\gamma V^\pi(s')].
$$

Schéma :

```
   future
┌──────────────┐
│ γ V(s')      │
└──────────────┘
       ▲
       │
┌──────────────┐
│ récompense r │
└──────────────┘
       ▲
       │
état courant s
```

---

# 4. Q-Learning

## 4.1 Q-table

La **Q-table** (table Q) stocke la valeur Q(s,a) pour chaque paire (état, action), représentant la **valeur attendue** du retour cumulé en choisissant l'action \(a\) dans l'état \(s\) et en suivant ensuite la politique optimale.

**Définition formelle :**
$$
Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s',a') \mid s, a\right]
$$

**Structure de la Q-table :**

```
                    Actions disponibles
            ┌────────────────────────────────────┐
            │   a₁      a₂      a₃      ...      │
            ├────────────────────────────────────┤
            │                                    │
États    s₁ │  Q(s₁,a₁) Q(s₁,a₂) Q(s₁,a₃) ...    │
            │                                    │
         s₂ │  Q(s₂,a₁) Q(s₂,a₂) Q(s₂,a₃) ...    │
            │                                    │
         s₃ │  Q(s₃,a₁) Q(s₃,a₂) Q(s₃,a₃) ...    │
            │                                    │
  =     ... │   ...      ...      ...     ...    │
            └────────────────────────────────────┘
```

**Interprétation :**
- Chaque cellule \(Q(s,a)\) représente la **qualité** de l'action \(a\) dans l'état \(s\)
- Plus la valeur est élevée, meilleure est l'action dans cet état
- La politique optimale : \(\pi^*(s) = \arg\max_a Q(s,a)\) (choisir l'action avec la plus grande valeur Q)

**Exemple concret (grille 3×3) :**

```
Actions: ↑ (haut), ↓ (bas), ← (gauche), → (droite)
État cible: s₉ (coin inférieur droit, récompense +10)

        Actions
        ↑    ↓    ←    →
┌─────────────────────────────┐
s₁ │  2.5  1.0  1.0  3.0      │  (état initial)
s₂ │  3.5  2.0  2.5  4.0      │
s₃ │  5.0  3.5  4.0  6.0      │
s₄ │  4.0  2.5  3.0  5.0      │
s₅ │  6.0  4.0  5.0  7.0      │
s₆ │  8.0  6.0  7.0  9.0      │
s₇ │  7.0  5.0  6.0  8.0      │
s₈ │  9.0  7.0  8.0  9.5      │
s₉ │ 10.0  8.0  9.0 10.0      │  (état terminal)
└─────────────────────────────┘
```

**Limitations :**
- Nécessite un espace d'états **discret et fini**
- Ne peut pas généraliser à des états non vus
- Devient impraticable pour de grands espaces d'états (curse of dimensionality)

---

## 4.2 Mise à jour Bellman

**Équation de mise à jour Q-learning :**

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

où :
- \(\alpha\) = taux d'apprentissage
- \(\gamma\) = facteur d'actualisation
- \(r + \gamma \max_{a'} Q(s',a')\) = cible (target)
- \(Q(s,a)\) = valeur actuelle

**Processus :**

```
Observation : (s, a, r, s')
       │
       ▼
Calcul cible : r + γ max Q(s',a')
       │
       ▼
Mise à jour : Q(s,a) ← Q(s,a) + α[cible - Q(s,a)]
```

---

# 5. Deep Q-Network (DQN)

## 5.1 Approximation par réseau profond

Remplace la Q-table par un réseau neuronal qui généralise :

```
État s (ex: image, vecteur)
    │
    ▼
┌─────────────────┐
│ CNN ou MLP      │
│ (réseau profond)│
└─────────────────┘
    │
    ▼
[Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)]
```

**Avantages :**
- Généralisation à des états non vus
- Traitement d'espaces d'états continus ou très grands
- Apprentissage de représentations utiles

---

## 5.2 Experience Replay

Mécanisme pour améliorer la stabilité de l'apprentissage :

```
Buffer d'expériences (mémoire)
┌──────────────────────────────────────────────┐
│ (s₁,a₁,r₁,s₁'), (s₂,a₂,r₂,s₂'), ...          │
│ (s₃,a₃,r₃,s₃'), (s₄,a₄,r₄,s₄'), ...          │
│ ...                                          │
└──────────────────────────────────────────────┘                                                                                                                
         │
         ▼
    Échantillonnage
    aléatoire (batch)
         │
         ▼
    Entraînement du réseau
```

**Bénéfices :**
- Réutilisation des expériences passées
- Réduction de la corrélation entre échantillons
- Apprentissage plus stable

---

## 5.3 Réseau cible

Technique pour stabiliser l'apprentissage :

```
Q_online (réseau principal)
    │
    │ mise à jour continue
    │
    │ copie périodique
    │ (tous les N pas)
    ▼
Q_target (réseau cible)
```

**Principe :**
- \(Q_{online}\) : utilisé pour choisir les actions et être mis à jour
- \(Q_{target}\) : utilisé pour calculer les cibles (targets) stables
- Copie périodique : \(Q_{target} \leftarrow Q_{online}\) tous les N pas

---

# 6. Policy Gradient

Méthode qui optimise directement la politique paramétrée.

**Gradient de la fonction objectif :**

$$
\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t\right]
$$

où \(G_t\) est le retour cumulé (discounted return).

**Processus :**

```
Politique π_θ
    │
    ▼
Actions a ~ π_θ(·|s)
    │
    ▼
Récompenses r
    │
    ▼
Calcul retour G_t
    │
    ▼
Gradient ∇_θ J
    │
    ▼
Mise à jour θ ← θ + α∇_θ J
```

**Avantages :**
- Fonctionne avec actions continues
- Politiques stochastiques naturelles
- Pas besoin d'estimer Q(s,a)

---

# 7. Synthèse

## 7.1 Comparaison des approches

| Aspect              | Apprentissage supervisé | Apprentissage par renforcement |
|---------------------|-------------------------|--------------------------------|
| **Données**         | Labels fournis          | Récompenses (sparses)          |
| **Dataset**         | Fixe, statique          | Dynamique, généré              |
| **Objectif**        | Minimiser erreur        | Maximiser retour cumulé        |
| **Feedback**        | Immédiat, dense         | Retardé, souvent sparse        |
| **Exploration**     | Non nécessaire          | Essentielle                    |

## 7.2 Rôle des réseaux de neurones

```
Apprentissage supervisé :
  └─> Réseaux pour apprendre f: X → Y

Renforcement :
  └─> Réseaux pour approximer Q(s,a) ou π(a|s)
      └─> Permet de scaler à de grands espaces d'états
```

## 7.3 Points clés

**Supervisé :**
- Données étiquetées requises
- Objectif : minimiser la perte sur les données d'entraînement
- Généralisation à de nouvelles données similaires

**Renforcement :**
- Interaction agent/environnement
- Objectif : maximiser la récompense cumulée
- Apprentissage par essais-erreurs
- Compromis exploration/exploitation

---

## 7.4 Ressources complémentaires

### 7.4.1 Datasets pour pratiquer

1. **Régression**
   - Boston Housing (prix immobiliers)
   - California Housing
   - Bike Sharing Dataset

2. **Classification**
   - MNIST (chiffres manuscrits)
   - CIFAR-10 (images)
   - Iris (classification simple)

3. **Séries temporelles**
   - Stock prices
   - Weather data
   - Sales forecasting

### 7.4.2 Outils et bibliothèques

```python
# Installation des bibliothèques essentielles
"""
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tensorflow  # ou pytorch
"""

# Exemple d'utilisation de scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# Régression linéaire avec sklearn
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Perceptron multicouche avec sklearn
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), 
                    max_iter=1000,
                    random_state=42)
mlp.fit(X_train, y_train)
```

### 7.4.3 Exercices supplémentaires

**Exercice 1 : Régression polynomiale**
Modifier le code de régression linéaire pour inclure des termes polynomiaux (x², x³).

**Exercice 2 : Perceptron multicouche**
Implémenter un réseau à 2 couches pour résoudre le problème XOR.

**Exercice 3 : Validation croisée**
Implémenter la validation croisée k-fold pour évaluer un modèle.

**Exercice 4 : Early stopping**
Ajouter un mécanisme d'arrêt anticipé basé sur la performance de validation.

---

## 📝 Notes finales

Pour approfondir l'apprentissage automatique :

1. **Pratiquez régulièrement** avec des datasets variés
2. **Visualisez** vos données et résultats
3. **Comprenez** les limitations de chaque méthode
4. **Itérez** : commencez simple, puis complexifiez
5. **Documentez** vos expériences et résultats

**Bon apprentissage ! 🚀**