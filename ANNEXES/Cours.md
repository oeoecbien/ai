# Apprentissage supervisé, réseaux de neurones et renforcement

**Résumé structuré des concepts fondamentaux**

---

## 0. Introduction à l'intelligence artificielle

### 0.1 Définir l'intelligence artificielle

Évolution de l'IA depuis les années 1950 :
- **1950 :** Artificial Intelligence - Démonstration du comportement humain par la machine
- **1980 :** Machine Learning (ML) - Apprentissage machine à partir de données
- **2010 :** Deep Learning (DL) - Modèles de ML qui imitent le fonctionnement du cerveau
- **2020 :** Generative AI - Modèles de DL qui créent du contenu original

### 0.2 Les grands types d'IA

1. **Le supervisé**
   - La régression (prédire une valeur continue)
   - La classification (prédire une classe)

2. **Le non-supervisé**
   - Clustering
   - Réduction de dimensions
   - Apprentissage de représentations

3. **Auto-supervisé**
   - Modélisation du langage

4. **Le renforcement**
   - Prise de décision

---

## 1. Apprentissage supervisé

### 1.1 Principe général

**But :** Apprendre une fonction qui associe des entrées (X) à des sorties (Y) à partir d'exemples (données étiquetées).

### 1.2 La pipeline d'un projet ML

Étapes essentielles :

1. **Préparer les données (70% du travail)**
   - Homogénéiser leur format (même unité)
   - Supprimer les données incomplètes
   - Supprimer les anomalies
   - Éventuellement homogénéiser leur distribution

2. **Entraînement (30% du travail)**
   - Définir le modèle (nb entrées / sorties / couches cachées)
   - Choisir les hyperparamètres (e.g., taux d'apprentissage)

3. **Évaluation**
   - Usage d'un jeu de validation de manière itérative
   - Ajuster les choix des hyperparamètres
   - Comparer plusieurs modèles

4. **Test**
   - Usage d'un jeu de test
   - Teste une unique fois le modèle final sur de nouvelles données

### 1.3 Régression

Prédire une valeur continue.

**Exemples d'applications :**
- Prix d'un appartement à partir de caractéristiques (surface, quartier, étage, etc.)
- Prédire le nombre d'appels hebdomadaires pour le support utilisateur
- Prédire les pannes des produits installés chez les clients

### 1.4 Classification

Prédire une étiquette discrète.

**Exemples d'applications :**
- Reconnaître quel chiffre est présent sur une image (classification MNIST)
- Trier des légumes automatiquement

### 1.5 Fonction hypothèse (régression linéaire)

La fonction approximée se nomme *h* (hypothèse) :

$$f(X) = aX + b \quad \Leftrightarrow \quad h(X) = \Theta_0 + \Theta_1 X$$

où \(\Theta_0\) et \(\Theta_1\) sont les paramètres qu'on doit déterminer.

**Formalisation mathématique :**

- \(\Theta\) : paramètres
- \(m\) : le nombre de données d'entraînement
- \(X\) : les features (entrées)
- \(Y\) : les targets (sorties)
- \((X, Y)\) : une donnée d'entraînement

Si on a \(n\) entrées telles que \(X=\left\{X_{0}, X_{1}, \ldots, X_{n}\right\}\) alors :

$$h(X)=\Theta_{0}+\Theta_{1} X_{1}+\Theta_{2} X_{2}+\ldots+\Theta_{n} X_{n} \approx Y$$

Version compactée : \(h(X)=\sum_{j=0}^{|X|} \Theta_{j} X_{j} \quad\) avec \(X_{0}=1\)

Notation : \(h_\theta(x)\)

### 1.6 Fonction de coût

**Objectif :** Minimiser la fonction \(J(\Theta)\) en trouvant les bons paramètres \(\Theta\).

Mesure l'erreur entre les prédictions et les cibles. On la minimise pour entraîner le modèle.

**Fonction de coût (erreur quadratique) :**

$$
J(\Theta)=\frac{1}{2} \sum_{i=1}^{m}\left(h_{\Theta}\left(X^{i}\right)-y^{i}\right)^{2}
$$

où \(m\) est le nombre d'exemples d'entraînement. Le facteur \(\frac{1}{2}\) simplifie le calcul du gradient.

**Autres exemples :**
- Erreur quadratique moyenne (MSE) pour la régression
- Entropie croisée pour la classification

### 1.7 Descente de gradient

Méthode itérative pour ajuster les paramètres et réduire le coût \(J(\Theta)\).

**Principe :**
1. Initialiser \(\Theta\)
2. Modifier \(\Theta\) de façon à réduire \(J(\Theta)\)

**Algorithme :**
1. Initialiser les paramètres \(\Theta\)
2. Calculer le gradient \(\nabla J(\Theta)\)
3. Mettre à jour : 
   $$\Theta_{j}=\Theta_{j}-\alpha \sum_{i=1}^{m} \frac{\partial}{\partial \Theta_{j}} J(\Theta) \quad \text{pour } j=0,1, \ldots,|X|$$
4. Répéter jusqu'à convergence

où \(\alpha\) est le taux d'apprentissage (choisi au préalable, \(\sim 10^{-3}\)).

### 1.8 Le perceptron

#### 1.8.1 Structure du perceptron

Le perceptron est un classifieur linéaire, unité de base des réseaux de neurones.

**Calcul :**

$$\text{Net input} = x_1w_1 + x_2w_2 + \ldots + x_mw_m + b$$

$$\text{Output} = f(\text{net input})$$

où :
- \(x_i\) : entrées
- \(w_i\) : poids
- \(b\) : biais
- \(f\) : fonction d'activation

#### 1.8.2 Exemples de portes logiques

**Porte AND :**

| \(x_1\) | \(x_2\) | \(y\) |
|---------|---------|-------|
| 0       | 0       | 0     |
| 0       | 1       | 0     |
| 1       | 0       | 0     |
| 1       | 1       | 1     |

**Solution :** \(w_1 = w_2 = 1\) et \(b = -1.5\)

**Porte OR :**

| \(x_1\) | \(x_2\) | \(y\) |
|---------|---------|-------|
| 0       | 0       | 0     |
| 0       | 1       | 1     |
| 1       | 0       | 1     |
| 1       | 1       | 1     |

#### 1.8.3 Apprentissage du perceptron

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

#### 1.8.4 Limites du perceptron

**Porte XOR (non linéairement séparable) :**

| \(x_1\) | \(x_2\) | \(y\) |
|---------|---------|-------|
| 0       | 0       | 0     |
| 0       | 1       | 1     |
| 1       | 0       | 1     |
| 1       | 1       | 0     |

**Problème :** Le perceptron ne peut pas apprendre le XOR car il s'agit d'un problème non linéairement séparable.

**Solution :** Utiliser plusieurs couches de perceptrons (réseau multicouche).

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

