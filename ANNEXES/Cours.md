# Apprentissage supervisé, réseaux de neurones et renforcement

**Résumé structuré des concepts fondamentaux**

---

## 1. Apprentissage supervisé

### 1.1 Principe général

**But :** Apprendre une fonction qui associe des entrées (X) à des sorties (Y) à partir d'exemples (données étiquetées).

### 1.2 Régression

Prédire une valeur continue.

**Exemple :** Prix d'un appartement à partir de caractéristiques (surface, quartier, étage, etc.)

### 1.3 Classification

Prédire une étiquette discrète.

**Exemple :** Reconnaître quel chiffre est présent sur une image (classification MNIST)

### 1.4 Fonction hypothèse

Modèle paramétré par des poids \(\Theta\) qui approxime \(f(X)\).

Notation : \(h_\theta(x)\)

### 1.5 Fonction de coût

Mesure l'erreur entre les prédictions et les cibles. On la minimise pour entraîner le modèle.

**Exemples :**
- Erreur quadratique moyenne (MSE) pour la régression
- Entropie croisée pour la classification

### 1.6 Descente de gradient

Méthode itérative pour ajuster les paramètres et réduire le coût.

**Algorithme :**
1. Initialiser les paramètres \(\theta\)
2. Calculer le gradient \(\nabla J(\theta)\)
3. Mettre à jour : \(\theta \leftarrow \theta - \alpha \nabla J(\theta)\)
4. Répéter jusqu'à convergence

où \(\alpha\) est le taux d'apprentissage.

---

## 2. Réseaux de neurones (Deep Learning)

### 2.1 Neurone artificiel

Unité de calcul de base qui :
- Calcule une somme pondérée des entrées plus un biais
- Applique une fonction d'activation (non linéaire)

**Fonction :** \(y = f(\sum w_i x_i + b)\)

### 2.2 Couche

Groupe de neurones. Plusieurs couches apprennent des représentations de plus en plus abstraites.

**Architecture typique :**
- Couche d'entrée
- Couches cachées (hidden layers)
- Couche de sortie

### 2.3 Fonction d'activation

Fonction non linéaire qui permet au réseau d'apprendre des relations non linéaires.

**Exemples :**
- **ReLU :** \(f(x) = \max(0, x)\)
- **Sigmoid :** \(f(x) = \frac{1}{1 + e^{-x}}\)
- **Tanh :** \(f(x) = \tanh(x)\)

### 2.4 Rétropropagation

Calcul du gradient de la perte par rapport aux poids, utilisé par la descente de gradient pour mettre à jour les poids.

**Processus :**
1. **Forward pass :** Propagation des données de l'entrée vers la sortie
2. **Backward pass :** Propagation du gradient de la sortie vers l'entrée
3. **Mise à jour :** Ajustement des poids selon le gradient calculé

---

## 3. Apprentissage par renforcement (RL)

### 3.1 Principe général

**But :** Apprendre une politique qui choisit des actions pour maximiser les récompenses cumulées en interagissant avec un environnement.

### 3.2 Vocabulaire de base

- **Agent :** Entité qui prend des décisions
- **Environnement :** Contexte dans lequel l'agent évolue
- **État (s) :** Situation actuelle de l'environnement
- **Action (a) :** Décision prise par l'agent
- **Récompense (r) :** Signal de retour (positif ou négatif)

**Cycle :** L'agent observe un état, prend une action, reçoit une récompense et passe à un nouvel état.

### 3.3 Processus de décision markovien (MDP)

Modèle formel décrivant le problème de décision séquentielle.

**Composants :**
- \(S\) : Espace d'états
- \(A\) : Espace d'actions
- \(T\) : Probabilité de transition \(P(s' | s, a)\)
- \(R\) : Fonction de récompense
- \(\gamma\) : Facteur d'actualisation (0 ≤ γ ≤ 1)

### 3.4 Politique

Règle (déterministe ou stochastique) qui associe un état à une action.

**Notation :** \(\pi(a|s)\) = probabilité de choisir l'action \(a\) dans l'état \(s\)

**Objectif :** Trouver la politique optimale \(\pi^*\) qui maximise le retour cumulé.

---

## 4. Algorithmes clés en RL

### 4.1 Q-Learning

**Principe :** Stocke une Q-table \(Q(s,a)\) et met à jour les valeurs avec l'équation de Bellman.

**Équation de mise à jour :**

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
\]

**Limitation :** Ne scale pas quand les états sont nombreux (problème de dimensionnalité).

### 4.2 Équation de Bellman

Relation qui propage la valeur des récompenses futures vers les états précédents. Base de nombreux algorithmes de RL.

**Forme générale :**

\[
V^\pi(s) = \mathbb{E}\left[r + \gamma V^\pi(s')\right]
\]

### 4.3 Exploration vs Exploitation

Compromis fondamental en RL :

- **Exploration :** Tester des actions nouvelles pour découvrir de meilleures stratégies
  - Paramètre : \(\epsilon\) (epsilon-greedy)
- **Exploitation :** Utiliser la meilleure action connue pour maximiser les récompenses immédiates

**Stratégies :**
- \(\epsilon\)-greedy : Exploration aléatoire avec probabilité \(\epsilon\)
- UCB (Upper Confidence Bound)
- Thompson Sampling

### 4.4 DQN (Deep Q-Network)

Remplace la Q-table par un réseau neuronal qui approxime \(Q(s,a)\).

**Avantages :**
- Permet de généraliser à des états non vus
- Traite de grands espaces d'états (voire continus)
- Apprend des représentations utiles

**Techniques clés :**
- **Experience Replay :** Réutilisation d'expériences passées pour stabiliser l'apprentissage
- **Réseau cible :** Copie périodique du réseau principal pour des cibles stables

### 4.5 Policy Gradient

Méthode qui optimise directement la politique paramétrée (plutôt que d'estimer \(Q\)).

**Gradient de la fonction objectif :**

\[
\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t\right]
\]

**Avantages :**
- Utile pour actions continues
- Politiques stochastiques naturelles
- Pas besoin d'estimer \(Q(s,a)\)

**Variantes :**
- REINFORCE
- Actor-Critic
- PPO (Proximal Policy Optimization)

---

## 5. Remarques pratiques

### 5.1 Différences fondamentales

| Aspect | Apprentissage supervisé | Apprentissage par renforcement |
|--------|-------------------------|-------------------------------|
| **Labels** | Fournis explicitement | Récompenses (souvent sparses) |
| **Données** | Dataset fixe | Environnement dynamique |
| **Feedback** | Immédiat et dense | Retardé et souvent sparse |
| **Apprentissage** | À partir d'exemples | Par essais-erreurs |

### 5.2 Rôle des réseaux de neurones

**Apprentissage supervisé :**
- Réseaux pour apprendre \(f: X \rightarrow Y\)

**Renforcement :**
- Réseaux pour approximer \(Q(s,a)\) ou \(\pi(a|s)\)
- Résout la limite d'échelle des tables (Q-table) en apprenant des représentations
- Permet de traiter des espaces d'états continus ou très grands

### 5.3 Points clés à retenir

**Supervisé :**
- Exige des labels fournis
- Objectif : minimiser l'erreur sur les données d'entraînement
- Généralisation à de nouvelles données similaires

**Renforcement :**
- Apprend par essais-erreurs sans labels pré-collectés
- Objectif : maximiser la récompense cumulée
- Interaction continue avec l'environnement
- Compromis exploration/exploitation essentiel

---

**Fin du document**

