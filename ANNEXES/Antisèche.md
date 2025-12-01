# ANTISÈCHE - APPRENTISSAGE MACHINE
## Guide complet avec explications détaillées, illustrations et exemples

---

## TABLE DES MATIÈRES

1. [Introduction à l'Intelligence Artificielle](#1-introduction-à-lintelligence-artificielle)
2. [Apprentissage Supervisé - Fondamentaux](#2-apprentissage-supervisé---fondamentaux)
3. [Régression Linéaire](#3-régression-linéaire)
4. [Perceptron et Classification](#4-perceptron-et-classification)
5. [Réseaux de Neurones et Deep Learning](#5-réseaux-de-neurones-et-deep-learning)
6. [Apprentissage par Renforcement](#6-apprentissage-par-renforcement)

---

## 1. INTRODUCTION À L'INTELLIGENCE ARTIFICIELLE

### 1.1 Qu'est-ce que l'Intelligence Artificielle ?

**Définition :** L'Intelligence Artificielle (IA) est la capacité d'une machine à simuler l'intelligence humaine, notamment la capacité d'apprendre, de raisonner et de résoudre des problèmes.

**Grands noms du domaine :**
- **Alan Turing (1912-1954)** : Père de l'informatique moderne, test de Turing
- **Marvin Minsky (1927-2016)** : Co-fondateur du MIT AI Lab
- **Yann LeCun (1960-)** : Pionnier des réseaux de neurones convolutifs

### 1.2 Les Grands Types d'IA

#### A. Apprentissage Supervisé
L'agent apprend à partir d'exemples étiquetés (on lui donne les bonnes réponses).

**Sous-catégories :**
- **Régression** : Prédire une valeur continue (ex: prix d'un appartement)
- **Classification** : Prédire une catégorie (ex: reconnaître un chiffre sur une image)

**Illustration :**
```
Régression : Prix = f(superficie, quartier, étage)
             → Sortie : 250 000€

Classification : Image → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                  → Sortie : "C'est un 3"
```

#### B. Apprentissage Non-Supervisé
L'agent apprend sans exemples étiquetés, découvre des patterns par lui-même.

**Sous-catégories :**
- **Clustering** : Regrouper des données similaires
- **Réduction de dimensions** : Simplifier les données
- **Apprentissage de représentations** : Extraire des features

#### C. Apprentissage Auto-Supervisé
L'agent génère ses propres labels à partir des données (ex: modélisation du langage)

#### D. Apprentissage par Renforcement
L'agent apprend par essais-erreurs, reçoit des récompenses/pénalités.

**Utilisation :** Prise de décision, contrôle de robots, jeux vidéo

### 1.3 Pipeline d'un Projet Machine Learning

```
┌─────────────────────────────────────────────────────────┐
│ 1. PRÉPARATION DES DONNÉES (70% du travail)             │
│    - Homogénéiser le format                             │
│    - Supprimer données incomplètes                      │
│    - Supprimer anomalies                                │
│    - Normaliser les distributions                       │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. ENTRAÎNEMENT (30% du travail)                        │
│    - Définir le modèle                                  │
│    - Choisir hyperparamètres                            │
│    - Entraîner sur données d'entraînement               │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. ÉVALUATION (Itérative)                               │
│    - Utiliser jeu de validation                         │
│    - Ajuster hyperparamètres                            │
│    - Comparer plusieurs modèles                         │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 4. TEST (Une seule fois)                                │
│    - Tester sur jeu de test                             │
│    - Évaluer performance finale                         │
└─────────────────────────────────────────────────────────┘
```

**Répartition des données :**
- **Données d'entraînement** : ~70% - Pour apprendre
- **Données de validation** : ~15% - Pour ajuster
- **Données de test** : ~15% - Pour tester finalement

---

## 2. APPRENTISSAGE SUPERVISÉ - FONDAMENTAUX

### 2.1 Définition

**Apprentissage Supervisé :** Type d'apprentissage où l'on dispose d'un ensemble de données avec des **entrées (X)** et des **sorties attendues (Y)**. Le modèle apprend à prédire Y à partir de X.

**Exemple concret :**
```
Entrée (X) : Photo d'un chat
Sortie (Y) : "chat"

Le modèle apprend : Photo → "chat"
```

### 2.2 Cas d'Application

#### Classification
- **Trier des légumes automatiquement** : Image → Type de légume
- **Reconnaissance de chiffres** : Image → Chiffre (0-9)
- **Détection de spam** : Email → Spam ou Non-spam

#### Régression
- **Prédire le prix d'un appartement** : (superficie, quartier, étage) → Prix
- **Prédire le nombre d'appels support** : Historique → Nombre d'appels
- **Prédire les pannes** : Données capteurs → Probabilité de panne

---

## 3. RÉGRESSION LINÉAIRE

### 3.1 Concept Fondamental

**Objectif :** Trouver une droite (ou hyperplan) qui passe "au mieux" à travers des points de données.

**Illustration :**
```
Prix (Y)
  ↑
  |     ●
  |   ●   ●
  | ●       ●
  |●
  └─────────────→ Superficie (X)
```

### 3.2 Formalisation Mathématique

**Fonction hypothèse :**
```
h(X) = Θ₀ + Θ₁X₁ + Θ₂X₂ + ... + ΘₙXₙ
```

**Version compacte :**
```
h(X) = Σ(Θⱼ × Xⱼ)  pour j = 0 à n
       avec X₀ = 1
```

**Où :**
- **Θ** (thêta) : Paramètres à déterminer (pente, ordonnée à l'origine)
- **X** : Features (entrées) - ex: superficie, quartier
- **Y** : Target (sortie) - ex: prix
- **m** : Nombre de données d'entraînement
- **n** : Nombre de features

**Exemple simple (1 dimension) :**
```
h(X) = Θ₀ + Θ₁X

Si Θ₀ = 100  et  Θ₁ = 50
Alors h(X) = 100 + 50X

Pour X = 2 : h(2) = 100 + 50×2 = 200
```

### 3.3 Fonction de Coût (Cost Function)

**Objectif :** Mesurer l'erreur entre les prédictions et les valeurs réelles.

**Formule :**
```
J(Θ) = ½ × Σ(hΘ(Xᵢ) - yᵢ)²
       pour i = 1 à m
```

**Explication :**
- Pour chaque donnée i, on calcule : (prédiction - vraie valeur)²
- On fait la somme de toutes ces erreurs
- Le facteur ½ simplifie les calculs de dérivée

**Illustration :**
```
Erreur pour chaque point :
  |     ●
  |   ●   ●  ← Distance verticale = erreur
  | ●       ●
  |●
  └─────────────→
```

**Objectif :** Minimiser J(Θ) pour avoir la meilleure droite possible.

### 3.4 Descente de Gradient (Gradient Descent)

**Principe :** Algorithme pour trouver les paramètres Θ qui minimisent J(Θ).

**Méthode :**
1. Initialiser Θ (valeurs aléatoires ou à 0)
2. Répéter jusqu'à convergence :
   - Calculer la dérivée de J(Θ)
   - Modifier Θ dans le sens qui réduit J(Θ)

**Formule de mise à jour :**
```
Θⱼ = Θⱼ - α × ∂J(Θ)/∂Θⱼ
```

**Où :**
- **α (alpha)** : Taux d'apprentissage (~0.001 à 0.01)
  - Trop petit : Apprentissage lent
  - Trop grand : Risque de divergence

**Illustration du processus :**
```
Itération 1: J(Θ) = 1000  →  Θ = [0, 0]
Itération 2: J(Θ) = 800   →  Θ = [10, 5]
Itération 3: J(Θ) = 500   →  Θ = [20, 10]
...
Itération N: J(Θ) = 0.5   →  Θ = [100, 50]  ← Convergence
```

**Algorithme complet :**
```
1. Initialiser Θ₀, Θ₁, ..., Θₙ
2. Répéter pour chaque itération :
   Pour j = 0 à n :
     Θⱼ = Θⱼ - α × Σ(hΘ(Xᵢ) - yᵢ) × Xᵢⱼ
3. Arrêter quand J(Θ) ne diminue plus
```

### 3.5 Exemple Pratique : Prédiction de Salaire

**Données :**
```
Expérience (années) | Salaire (€)
-------------------|------------
1                  | 30000
2                  | 35000
3                  | 40000
4                  | 45000
```

**Étapes :**
1. Initialiser : Θ₀ = 0, Θ₁ = 0
2. Calculer h(X) = Θ₀ + Θ₁X pour chaque point
3. Calculer J(Θ) = ½ × Σ(erreurs)²
4. Mettre à jour Θ avec descente de gradient
5. Répéter jusqu'à convergence

**Résultat final :**
```
h(X) = 25000 + 5000X

Pour 5 ans d'expérience : h(5) = 25000 + 5000×5 = 50000€
```

---

## 4. PERCEPTRON ET CLASSIFICATION

### 4.1 Qu'est-ce qu'un Perceptron ?

**Définition :** Un perceptron est un neurone artificiel simple inventé par Frank Rosenblatt en 1957. C'est l'unité de base des réseaux de neurones qui effectue une classification binaire (0 ou 1) en séparant les données par une frontière linéaire.

**Contexte historique :**
- Premier modèle d'apprentissage automatique inspiré du neurone biologique
- Fondement des réseaux de neurones modernes
- Limitation : Ne peut résoudre que des problèmes linéairement séparables

**Structure et composants :**
```
              PERCEPTRON
    ┌───────────────────────────────────┐
    │                                   │
    │  x₁ ──[w₁]──┐                     │
    │  x₂ ──[w₂]──┤                     │
    │  x₃ ──[w₃]──┤                     │
    │  ...        │                     │
    │  xₙ  ──[wₙ]──┤                      │
    │             │                     │
    │             ├─→ Σ(wᵢ × xᵢ) + b    │
    │             │                     │
    │             ↓                     │
    │         f(net_input)              │
    │             │                     │
    │             ↓                     │
    │            ŷ ∈ {0, 1}             │
    │                                   │
    │  b (biais) ─┘                     │
    │                                   │
    └───────────────────────────────────┘

Flux : Entrées → Multiplication par poids → Somme + Biais → Activation → Sortie
```

**Composants détaillés :**
- **Entrées (x₁, x₂, ..., xₙ)** : Les features/attributs des données
- **Poids (w₁, w₂, ..., wₙ)** : Paramètres à apprendre, déterminent l'importance de chaque entrée
- **Biais (b)** : Paramètre qui permet de décaler la frontière de décision
- **Fonction d'activation f(·)** : Fonction seuil (step function) qui transforme la somme pondérée en 0 ou 1
- **Sortie (ŷ)** : Prédiction binaire (classe 0 ou classe 1)

**Principe de fonctionnement :**
Le perceptron calcule une combinaison linéaire des entrées pondérées, ajoute un biais, puis applique une fonction d'activation pour produire une sortie binaire. Il apprend en ajustant les poids et le biais pour minimiser les erreurs de classification.

### 4.2 Fonctionnement Détaillé

**Étape 1 : Calcul du Net Input**
```
net_input = x₁w₁ + x₂w₂ + ... + xₙwₙ + b
         = Σ(wᵢ × xᵢ) + b
```

**Étape 2 : Fonction d'Activation (Seuil)**
```
f(net_input) = {
    0  si net_input ≤ 0
    1  si net_input > 0
}
```

**Étape 3 : Sortie**
```
ŷ = f(net_input)
```

### 4.3 Exemple : Porte Logique AND

**Table de vérité :**
```
x₁  x₂  |  y
--------|----
0   0   |  0
0   1   |  0
1   0   |  0
1   1   |  1
```

**Paramètres :**
- w₁ = 1
- w₂ = 1
- b = -1.5

**Calcul pour (x₁=1, x₂=0) :**
```
net_input = 1×1 + 0×1 - 1.5 = 1 - 1.5 = -0.5
f(-0.5) = 0  (car -0.5 ≤ 0)
ŷ = 0  ✓ Correct !
```

**Calcul pour (x₁=1, x₂=1) :**
```
net_input = 1×1 + 1×1 - 1.5 = 2 - 1.5 = 0.5
f(0.5) = 1  (car 0.5 > 0)
ŷ = 1  ✓ Correct !
```

**Représentation graphique :**
```
x₂
 ↑
1|     ● (1,1) → y=1
 |  ────────────  Ligne de séparation
0|●     ●       (0,0) et (0,1) et (1,0) → y=0
 └─────────────→ x₁
  0     1
```

### 4.4 Exemple : Porte Logique OR

**Table de vérité :**
```
x₁  x₂  |  y
--------|----
0   0   |  0
0   1   |  1
1   0   |  1
1   1   |  1
```

**Solution possible :**
- w₁ = 1
- w₂ = 1
- b = -0.5

**Vérification :**
```
(0,0): 0×1 + 0×1 - 0.5 = -0.5 → 0 ✓
(0,1): 0×1 + 1×1 - 0.5 = 0.5  → 1 ✓
(1,0): 1×1 + 0×1 - 0.5 = 0.5  → 1 ✓
(1,1): 1×1 + 1×1 - 0.5 = 1.5  → 1 ✓
```

### 4.5 Apprentissage du Perceptron

**Objectif :** Ajuster automatiquement les poids w et le biais b.

**Règle de mise à jour :**
```
wᵢ ← wᵢ + α × (y - ŷ) × xᵢ
b  ← b  + α × (y - ŷ)
```

**Où :**
- **α** : Taux d'apprentissage (ex: 0.1)
- **y** : Sortie attendue (0 ou 1)
- **ŷ** : Sortie prédite (0 ou 1)
- **xᵢ** : Entrée i

**Algorithme :**
```
1. Initialiser w et b (valeurs aléatoires ou 0)
2. Pour chaque exemple (x, y) :
   a. Calculer ŷ = f(Σ(wᵢ×xᵢ) + b)
   b. Calculer erreur = y - ŷ
   c. Si erreur ≠ 0 :
      wᵢ ← wᵢ + α × erreur × xᵢ
      b  ← b  + α × erreur
3. Répéter jusqu'à ce que tous les exemples soient corrects
```

**Exemple d'apprentissage :**
```
Itération 1:
  Exemple (1,1) → y=1
  ŷ = 0 (incorrect)
  w₁ ← 0 + 0.1×(1-0)×1 = 0.1
  w₂ ← 0 + 0.1×(1-0)×1 = 0.1
  b  ← 0 + 0.1×(1-0)   = 0.1

Itération 2:
  Exemple (1,1) → y=1
  ŷ = f(0.1×1 + 0.1×1 + 0.1) = f(0.3) = 1 ✓
  Pas de mise à jour nécessaire
  ...
```

### 4.6 Limitation : Le Problème XOR

**Table de vérité XOR :**
```
x₁  x₂  |  y
--------|----
0   0   |  0
0   1   |  1
1   0   |  1
1   1   |  0
```

**Problème :** Un seul perceptron ne peut pas apprendre XOR car les points ne sont pas linéairement séparables.

**Représentation graphique :**
```
x₂
 ↑
1|●     ○  (0,1)→1  (1,1)→0
 |   ✗     Impossible de tracer une ligne
0|○     ●  (0,0)→0  (1,0)→1
 └─────────────→ x₁
  0     1
```

**Solution :** Utiliser plusieurs perceptrons (réseau de neurones) avec des couches cachées.

---

## 5. RÉSEAUX DE NEURONES ET DEEP LEARNING

### 5.1 Vue d'Ensemble

**Définition :** Un réseau de neurones est un approximateur de fonctions universel composé de plusieurs neurones interconnectés.

**Utilisations :**
1. **Régression** : Prédire des valeurs continues
2. **Classification** : Prédire des catégories

**Exemple applicatif : Conduite autonome**
```
Entrée : Photo de la route (1280×720 pixels = 921 600 inputs)
         ↓
Réseau de neurones
         ↓
Sortie : Décision du conducteur (tourner, freiner, accélérer)
```

### 5.2 Structure d'un Réseau de Neurones

**Architecture :**
```
┌──────────────────────────────────────────┐
│  COUCHE D'ENTRÉE                         │
│  (1 neurone par pixel/feature)           │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  COUCHES CACHÉES                         │
│  (n-2 couches avec neurones)             │
│  - Couche 1: 13 neurones                 │
│  - Couche 2: 13 neurones                 │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│  COUCHE DE SORTIE                        │
│  (1 neurone par classe)                  │
│  Ex: 10 neurones pour chiffres 0-9       │
└──────────────────────────────────────────┘
```

**Exemple concret : Reconnaissance de chiffres**
```
Image 28×28 pixels = 784 pixels
         ↓
Couche 1: 784 → 13 neurones
         ↓
Couche 2: 13 → 13 neurones
         ↓
Couche 3: 13 → 10 neurones (0-9)
         ↓
Sortie: [0.02, 0.01, 0.05, 0.98, 0.01, ...]
        → Le chiffre 3 a la probabilité la plus élevée (0.98)
```

### 5.3 Fonctionnement d'un Neurone

**Calcul dans un neurone :**
```
1. Réception des entrées : a₁, a₂, ..., aₙ
2. Calcul pondéré : net = w₁a₁ + w₂a₂ + ... + wₙaₙ + b
3. Activation : output = σ(net)
4. Envoi vers neurones suivants
```

**Illustration :**
```
Neurone précédent 1 ──w₁──┐
Neurone précédent 2 ──w₂──┤
...                       ├─→ Σ(wᵢ×aᵢ) + b ─→ σ(·) ─→ Sortie
Neurone précédent n ──wₙ──┘
                         ↑
                        b (biais)
```

### 5.4 Fonctions d'Activation

**Rôle :** Normaliser les sorties des neurones dans une plage spécifique.

**Fonction Sigmoïde :**
```
σ(x) = 1 / (1 + e^(-x))

Plage : [0, 1]
Utilisation : Couche de sortie pour classification binaire
```

**Fonction ReLU (Rectified Linear Unit) :**
```
ReLU(x) = max(0, x) = {
    x   si x > 0
    0   si x ≤ 0
}

Plage : [0, +∞[
Utilisation : Couches cachées (très populaire)
```

**Fonction Softmax :**
```
softmax(xᵢ) = e^(xᵢ) / Σ(e^(xⱼ))

Plage : [0, 1] avec Σ = 1
Utilisation : Couche de sortie pour classification multi-classes
```

**Comparaison visuelle :**
```
Sigmoïde:        ReLU:           Softmax:
    1                |             1
    |               /|             |
  0.5              / |             |
    |             /  |             |
    0            /   └─────────────┘
    └───────────┘    0
```

### 5.5 Les Biais

**Définition :** Le biais (b) est une constante ajoutée au calcul du neurone.

**Rôle :** Permet de décaler la fonction d'activation.

**Exemple :**
```
Sans biais :  σ(w₁a₁ + w₂a₂)
Avec biais :  σ(w₁a₁ + w₂a₂ + b)

Le biais permet d'ajuster le seuil d'activation
```

**Illustration :**
```
Sans biais : La fonction passe toujours par (0, 0.5)
Avec biais : On peut décaler la courbe horizontalement
```

### 5.6 Calcul du Nombre de Paramètres

**Formule :**
```
Pour une couche :
- Poids : (nombre_entrées × nombre_sorties)
- Biais : nombre_sorties
- Total : (nombre_entrées × nombre_sorties) + nombre_sorties
```

**Exemple : Réseau pour reconnaissance de chiffres**
```
Couche 1: 784 → 13
  Poids : 784 × 13 = 10 192
  Biais : 13
  Total : 10 205

Couche 2: 13 → 13
  Poids : 13 × 13 = 169
  Biais : 13
  Total : 182

Couche 3: 13 → 10
  Poids : 13 × 10 = 130
  Biais : 10
  Total : 140

TOTAL : 10 205 + 182 + 140 = 10 527 paramètres
```

### 5.7 Représentations Apprises

**Principe :** Chaque couche apprend des représentations de plus en plus abstraites.

**Exemple pour une image :**
```
Couche 1 (bas niveau) :
  - Détecte des bords, lignes, courbes simples

Couche 2 (niveau moyen) :
  - Détecte des formes (cercles, rectangles)

Couche 3 (haut niveau) :
  - Détecte des objets complets (yeux, nez, bouche)

Couche de sortie :
  - Combine tout pour reconnaître le chiffre
```

**Illustration :**
```
Image originale
     ↓
[Pixels bruts]
     ↓
[Bords et lignes] ← Couche 1
     ↓
[Formes géométriques] ← Couche 2
     ↓
[Caractéristiques complexes] ← Couche 3
     ↓
[Classification finale] ← Sortie
```

### 5.8 Apprentissage : Rétropropagation du Gradient

**Objectif :** Ajuster tous les poids et biais pour minimiser l'erreur.

**Processus en 3 étapes :**

#### Étape 1 : Propagation Avant (Forward Pass)
```
Entrée → Réseau → Sortie prédite (ŷ)
```

#### Étape 2 : Calcul de l'Erreur
```
Fonction de coût : J = ½ × Σ(y - ŷ)²

Exemple pour classification :
  Attendu : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (chiffre 3)
  Prédit :  [0.02, 0.03, 0.21, 0.01, 0.98, 0.05, ...]
  
  Erreur = ½ × [(0-0.02)² + (0-0.03)² + ... + (1-0.01)² + ...]
         = 1.85
```

#### Étape 3 : Rétropropagation (Backward Pass)
```
1. Calculer l'erreur à la sortie
2. Propager l'erreur vers l'arrière
3. Ajuster les poids en fonction de leur contribution à l'erreur
```

**Formule de mise à jour :**
```
wᵢⱼ ← wᵢⱼ - α × ∂J/∂wᵢⱼ
bⱼ  ← bⱼ  - α × ∂J/∂bⱼ
```

**Illustration :**
```
Erreur à la sortie : 1.85
         ↓
Propagation vers l'arrière
         ↓
Ajustement des poids de la couche 3
         ↓
Ajustement des poids de la couche 2
         ↓
Ajustement des poids de la couche 1
```

**Algorithme complet :**
```
1. Initialiser tous les poids et biais (petites valeurs aléatoires)
2. Pour chaque exemple d'entraînement :
   a. Propagation avant : calculer ŷ
   b. Calculer l'erreur J
   c. Rétropropagation : calculer ∂J/∂w pour chaque poids
   d. Mettre à jour : w ← w - α × ∂J/∂w
3. Répéter pour plusieurs époques (passages complets sur les données)
```

### 5.9 Descente de Gradient

**Principe :** Trouver le minimum de la fonction de coût en suivant la pente descendante.

**Illustration 2D :**
```
J(Θ)
 ↑
 |     ●
 |   ●   ●
 | ●       ●
 |●
 └─────────────→ Θ

On part d'un point aléatoire et on descend vers le minimum
```

**Types de descente :**

1. **Batch Gradient Descent** : Utilise toutes les données à chaque itération
   - Lent mais stable

2. **Stochastic Gradient Descent (SGD)** : Utilise 1 donnée à la fois
   - Rapide mais bruyant

3. **Mini-batch Gradient Descent** : Utilise un petit groupe de données
   - Compromis entre vitesse et stabilité (le plus utilisé)

### 5.10 Hyperparamètres Importants

**Taux d'apprentissage (α) :**
- Trop petit : Apprentissage très lent
- Trop grand : Risque de divergence
- Valeur typique : 0.001 à 0.01

**Nombre d'époques :**
- Nombre de fois qu'on passe toutes les données
- Trop peu : Sous-apprentissage
- Trop beaucoup : Sur-apprentissage

**Taille du batch :**
- Nombre d'exemples utilisés par itération
- Typique : 32, 64, 128, 256

**Régularisation :**
- Technique pour éviter le sur-apprentissage
- Ex: Dropout, L1/L2 regularization

---

## 6. APPRENTISSAGE PAR RENFORCEMENT

### 6.1 Introduction

**Définition :** L'apprentissage par renforcement (Reinforcement Learning - RL) est un type d'apprentissage où un agent apprend à prendre des décisions optimales en interagissant avec un environnement, recevant des récompenses ou des pénalités.

**Analogie humaine :** Apprendre par essais-erreurs, comme un enfant qui apprend à marcher.

**Différence avec l'apprentissage supervisé :**
```
Apprentissage Supervisé :
  - Données étiquetées disponibles
  - Exemples : (entrée, sortie) fournis
  - Objectif : Classification ou régression

Apprentissage par Renforcement :
  - Pas de données pré-collectées
  - Pas de supervision directe
  - Objectif : Contrôle et prise de décision
  - Apprentissage par interaction
```

### 6.2 Glossaire Fondamental

**Agent :**
- Entité qui perçoit et interagit avec l'environnement
- Exemple : Robot, joueur de jeu vidéo, système de trading

**État (State - s) :**
- Situation actuelle de l'environnement et de l'agent
- Exemple : Position dans un labyrinthe, configuration d'un jeu

**Action (a) :**
- Décision prise par l'agent
- Exemple : Avancer, reculer, tourner à gauche

**Récompense (Reward - r) :**
- Feedback numérique reçu après une action
- Positive : Action bénéfique (ex: +1 pour atteindre un objectif)
- Négative : Action pénalisante (ex: -1 pour tomber dans un piège)

**Politique (Policy - π) :**
- Stratégie de l'agent : quelle action prendre dans chaque état
- π(s) = a : "Dans l'état s, faire l'action a"

**Politique optimale (π*) :**
- Politique qui maximise les récompenses à long terme
- C'est ce qu'on cherche à apprendre !

### 6.3 Exemple Concret : La Tortue dans le Labyrinthe

**Scénario :** Une tortue cherche sa pizza dans un labyrinthe.

**Modélisation :**
```
État 1 ──avancer──→ État 2 ──avancer──→ État 3 (PIZZA!)
  ↑                  ↑                    ↑
  │                  │                    │
reculer          reculer              reculer
```

**États :**
- État 1 : Position initiale
- État 2 : Milieu du chemin
- État 3 : Position de la pizza (récompense !)

**Actions possibles :**
- Avancer
- Reculer

**Récompenses :**
- Atteindre la pizza : +1
- Autres actions : 0

**Objectif :** Apprendre à atteindre la pizza le plus rapidement possible.

### 6.4 Processus de Décision Markovien (MDP)

**Définition :** Un MDP est un modèle mathématique pour décrire un problème de prise de décision séquentielle.

**Composants d'un MDP : {S, A, T, R}**

- **S (States)** : Ensemble des états possibles
- **A (Actions)** : Ensemble des actions possibles
- **T (Transitions)** : Probabilité de transition P(s'|s,a)
  - "Probabilité d'arriver à l'état s' depuis l'état s en faisant l'action a"
- **R (Rewards)** : Fonction de récompense R(s,a,s')

**Caractéristiques :**
- **Discret** : États et actions sont discrets (pas continus)
- **Stochastique** : Transitions peuvent être probabilistes
- **Markovien** : L'état futur ne dépend que de l'état actuel et de l'action (pas de l'historique)

**Représentation graphique :**
```
État 1 ──[action, proba, récompense]──→ État 2
  │                                        │
  └────────────────────────────────────────┘
```

**Représentation tabulaire (Q-Table) :**
```
        Action: avancer  |  Action: reculer
État 1  P(→s2|s1,a1)    |  P(→s1|s1,a2)
État 2  P(→s3|s2,a1)    |  P(→s1|s2,a2)
État 3  P(→s3|s3,a1)    |  P(→s2|s3,a2)
```

### 6.5 Équation de Bellman

**Objectif :** Propager la valeur des récompenses à travers les états.

**Formule :**
```
V(s) = R(s) + γ × max[Σ P(s'|s,a) × V(s')]
                    a   s'
```

**Où :**
- **V(s)** : Valeur de l'état s
- **R(s)** : Récompense immédiate
- **γ (gamma)** : Facteur d'actualisation (discount factor)
  - 0 ≤ γ ≤ 1
  - Plus γ est proche de 1, plus on valorise les récompenses futures
  - Plus γ est proche de 0, plus on se concentre sur l'immédiat
- **P(s'|s,a)** : Probabilité de transition

**Version simplifiée (déterministe) :**
```
V(s) = R(s) + γ × max[V(s')]
                    a
```

**Signification :** La valeur d'un état = récompense immédiate + valeur future actualisée.

### 6.6 Propagation des Valeurs

**Exemple : Labyrinthe 3×3**

**Règles :**
- Case pizza (s3) : récompense = +1
- γ = 0.9 (facteur d'actualisation)
- α = 1.0 (taux d'apprentissage)
- Transitions déterministes

**Processus :**

**Itération 0 (initialisation) :**
```
Q-Table (valeurs initiales à 0) :
        Gauche  Droite  Haut  Bas
s1      0       0       0     0
s2      0       0       0     0
s3      0       0       0     0
```

**Itération 1 :**
```
s3 (pizza) : V(s3) = 1 (récompense immédiate)
s2 (voisin de s3) : V(s2) = 0 + 0.9 × 1 = 0.9
s1 (voisin de s2) : V(s1) = 0 + 0.9 × 0.9 = 0.81

Q-Table mise à jour :
        Gauche  Droite  Haut  Bas
s1      0       0.9     0     0
s2      0.81    1.0     0     0
s3      0.9     0       0     0.9
```

**Itération 2 :**
```
Les valeurs continuent de se propager...
s1 : V(s1) = 0 + 0.9 × max(0.9, 0.81, ...) = 0.81
```

**Illustration :**
```
┌─────┬─────┬─────┐
│ 0.81│ 0.9 │ 1.0 │ ← Valeurs propagées
│ s1  │ s2  │ s3  │
└─────┴─────┴─────┘
```

### 6.7 Q-Learning

**Définition :** Algorithme pour apprendre la valeur Q(s,a) = "valeur de faire l'action a dans l'état s".

**Q-Table :** Tableau qui stocke Q(s,a) pour chaque couple (état, action).

**Formule de mise à jour (Équation de Bellman pour Q-Learning) :**
```
Q(s,a) ← Q(s,a) + α × [R + γ × max Q(s',a') - Q(s,a)]
                              a'
```

**Décomposition :**
- **Q(s,a)** : Valeur actuelle
- **R** : Récompense immédiate
- **γ × max Q(s',a')** : Meilleure valeur future
- **α** : Taux d'apprentissage (0 < α ≤ 1)

**Algorithme Q-Learning :**
```
1. Initialiser Q-Table (valeurs à 0 ou aléatoires)
2. Pour chaque épisode :
   a. Initialiser l'état s
   b. Répéter jusqu'à la fin de l'épisode :
      - Choisir une action a (ε-greedy)
      - Exécuter l'action a, observer r et s'
      - Mettre à jour : Q(s,a) ← Q(s,a) + α[r + γ×max Q(s',a') - Q(s,a)]
      - s ← s'
3. Répéter pour plusieurs épisodes
```

### 6.8 Exploration vs Exploitation (ε-greedy)

**Dilemme :**
- **Exploitation** : Utiliser ce qu'on sait déjà (choisir la meilleure action connue)
- **Exploration** : Essayer de nouvelles actions pour découvrir de meilleures stratégies

**Stratégie ε-greedy :**
```
Avec probabilité ε :  Choisir une action aléatoire (exploration)
Avec probabilité 1-ε : Choisir la meilleure action (exploitation)
```

**Exemple :**
```
Si ε = 0.1 (10%) :
  - 10% du temps : action aléatoire
  - 90% du temps : meilleure action selon Q-Table
```

**Évolution de ε :**
```
Début d'apprentissage : ε = 1.0 (100% exploration)
  ↓
Progressivement : ε diminue (ex: ε = 0.9, 0.8, ..., 0.1)
  ↓
Fin d'apprentissage : ε = 0.1 (10% exploration, 90% exploitation)
```

**Illustration :**
```
Épisode 1-100 :   ε = 1.0  → Beaucoup d'exploration
Épisode 101-500 : ε = 0.5  → Équilibre
Épisode 501+ :    ε = 0.1  → Principalement exploitation
```

### 6.9 Limites du Q-Learning

**Problème 1 : Passage à l'échelle**
- Q-Table devient énorme avec beaucoup d'états
- Exemple : Jeu avec 10^6 états possibles → Tableau gigantesque
- Solution : Utiliser des réseaux de neurones (Deep Q-Network)

**Problème 2 : Pas de généralisation**
- Q-Learning mémorise chaque état individuellement
- Ne peut pas généraliser à des états similaires non vus
- Solution : Deep Q-Network apprend des représentations

**Illustration du problème :**
```
États vus :     s1, s2, s3
États similaires: s1', s2', s3' (non vus mais similaires)

Q-Learning classique : Ne peut pas utiliser s1 pour prédire s1'
Deep Q-Learning : Peut généraliser grâce aux réseaux de neurones
```

### 6.10 Deep Q-Network (DQN)

**Principe :** Remplacer la Q-Table par un réseau de neurones qui apprend Q(s,a).

**Architecture :**
```
État (s) → Réseau de Neurones → Q(s, a₁), Q(s, a₂), ..., Q(s, aₙ)
```

**Avantages :**
- Généralisation : Peut prédire pour des états non vus
- Passage à l'échelle : Gère des espaces d'états énormes
- Représentations apprises : Extrait automatiquement des features

**Exemple : Jeu vidéo**
```
Entrée : Image de l'écran (pixels)
         ↓
Réseau de neurones convolutifs
         ↓
Sortie : Q(s, gauche), Q(s, droite), Q(s, saut), Q(s, tir)
```

**Techniques importantes :**
1. **Experience Replay** : Stocker et réutiliser des expériences passées
2. **Target Network** : Réseau séparé pour la stabilité
3. **Frame Stacking** : Utiliser plusieurs frames pour capturer le mouvement

### 6.11 Policy Gradient

**Concept :** Au lieu d'apprendre Q(s,a), apprendre directement la politique π(s) = probabilité de chaque action.

**Différence avec Q-Learning :**
```
Q-Learning :  Apprend Q(s,a) → Choisit action avec max Q
Policy Gradient : Apprend directement π(s) → Probabilité de chaque action
```

**Avantages :**
- Actions stochastiques (probabilistes)
- Meilleur pour espaces d'actions continues
- Convergence souvent plus stable

**Formule (REINFORCE) :**
```
∇J(θ) = E[∇log π(a|s) × R]
```

**Où :**
- **π(a|s)** : Probabilité de l'action a dans l'état s
- **R** : Récompense totale de l'épisode
- **θ** : Paramètres du réseau de neurones

**Algorithme simplifié :**
```
1. Exécuter une politique π pour collecter un épisode
2. Calculer les récompenses R
3. Mettre à jour : θ ← θ + α × ∇log π(a|s) × R
4. Répéter
```

### 6.12 Synthèse : Comparaison des Algorithmes

**Q-Learning :**
- ✅ Simple à comprendre
- ✅ Efficace pour petits espaces d'états
- ❌ Ne passe pas à l'échelle
- ❌ Pas de généralisation

**Deep Q-Network (DQN) :**
- ✅ Passe à l'échelle
- ✅ Généralisation
- ✅ Bon pour espaces d'états complexes
- ❌ Plus complexe à implémenter

**Policy Gradient :**
- ✅ Actions stochastiques
- ✅ Bon pour actions continues
- ✅ Convergence souvent meilleure
- ❌ Variance élevée
- ❌ Plus lent à converger

---

## RÉCAPITULATIF VISUEL

### Comparaison des Types d'Apprentissage

```
┌─────────────────────────────────────────────────────────┐
│ APPRENTISSAGE SUPERVISÉ                                 │
│                                                         │
│ Données : (X, Y) étiquetées                             │
│ Objectif : Prédire Y à partir de X                      │
│ Exemples : Régression, Classification                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ APPRENTISSAGE PAR RENFORCEMENT                          │
│                                                         │
│ Données : Interactions (s, a, r, s')                    │
│ Objectif : Maximiser récompenses cumulées               │
│ Exemples : Jeux, Robots, Trading                        │
└─────────────────────────────────────────────────────────┘
```

### Évolution des Techniques

```
Régression Linéaire (simple)
         ↓
Perceptron (classification binaire)
         ↓
Réseaux de Neurones (multi-couches)
         ↓
Deep Learning (réseaux profonds)
         ↓
Reinforcement Learning (prise de décision)
```

---

## POINTS CLÉS À RETENIR

### Apprentissage Supervisé
1. **Régression** : Prédire des valeurs continues (prix, température)
2. **Classification** : Prédire des catégories (chiffres, objets)
3. **Descente de gradient** : Méthode pour minimiser l'erreur
4. **Perceptron** : Neurone simple pour classification binaire
5. **Réseaux de neurones** : Approximateurs de fonctions universels
6. **Rétropropagation** : Algorithme pour entraîner les réseaux

### Apprentissage par Renforcement
1. **MDP** : Modèle mathématique (États, Actions, Transitions, Récompenses)
2. **Équation de Bellman** : Propage les valeurs des récompenses
3. **Q-Learning** : Apprend Q(s,a) avec une table
4. **Deep Q-Network** : Q-Learning avec réseaux de neurones
5. **Policy Gradient** : Apprend directement la politique
6. **ε-greedy** : Équilibre exploration/exploitation

### Formules Importantes

**Régression Linéaire :**
- Hypothèse : `h(X) = Σ(Θⱼ × Xⱼ)`
- Coût : `J(Θ) = ½ × Σ(h(Xᵢ) - yᵢ)²`
- Mise à jour : `Θⱼ = Θⱼ - α × ∂J/∂Θⱼ`

**Perceptron :**
- Net input : `Σ(wᵢ × xᵢ) + b`
- Activation : `f(x) = 1 si x > 0, sinon 0`
- Mise à jour : `wᵢ ← wᵢ + α × (y - ŷ) × xᵢ`

**Q-Learning :**
- Mise à jour : `Q(s,a) ← Q(s,a) + α[r + γ×max Q(s',a') - Q(s,a)]`

---

## RESSOURCES ET EXERCICES

### Exercices Recommandés

1. **Régression Linéaire** : Prédire le salaire selon l'expérience
2. **Perceptron** : Implémenter les portes logiques (AND, OR, XOR)
3. **Réseaux de Neurones** : Reconnaître les chiffres manuscrits (MNIST)
4. **Q-Learning** : Tortue dans le labyrinthe
5. **Deep Q-Network** : Entraîner un agent sur un jeu simple

### Outils Utiles

- **TensorFlow Playground** : https://playground.tensorflow.org/
  - Visualiser l'apprentissage des réseaux de neurones
  - Expérimenter avec différents paramètres

- **Kaggle** : https://www.kaggle.com/
  - Datasets pour s'entraîner
  - Compétitions pour progresser

---

## GLOSSAIRE COMPLET

**Agent** : Entité qui apprend et prend des décisions

**Biais (Bias)** : Constante ajoutée dans un neurone

**Classification** : Prédire une catégorie parmi plusieurs

**Deep Learning** : Réseaux de neurones avec plusieurs couches cachées

**Descente de Gradient** : Algorithme d'optimisation pour minimiser une fonction

**Épisode** : Séquence complète d'interactions agent-environnement

**État (State)** : Situation actuelle dans un environnement

**Exploration** : Essayer de nouvelles actions

**Exploitation** : Utiliser les connaissances actuelles

**Fonction d'Activation** : Fonction qui normalise la sortie d'un neurone

**Hyperparamètre** : Paramètre fixé avant l'apprentissage (ex: taux d'apprentissage)

**MDP** : Processus de Décision Markovien

**Neurone** : Unité de calcul dans un réseau de neurones

**Politique (Policy)** : Stratégie de l'agent (quelle action dans quel état)

**Q-Learning** : Algorithme pour apprendre les valeurs Q(s,a)

**Q-Table** : Tableau stockant les valeurs Q(s,a)

**Régression** : Prédire une valeur continue

**Récompense (Reward)** : Feedback numérique après une action

**Rétropropagation** : Algorithme pour entraîner les réseaux de neurones

**Réseau de Neurones** : Ensemble de neurones interconnectés

**Sigmoïde** : Fonction d'activation S-shaped, plage [0,1]

**Supervisé** : Apprentissage avec données étiquetées

**Target** : Valeur attendue (Y) dans l'apprentissage supervisé

---

**Fin de l'antisèche - Bon apprentissage !**

