# Apprentissage supervisé, réseaux de neurones et renforcement

**Version enrichie avec illustrations conceptuelles (ASCII avancées)**

---

# 1. Apprentissage supervisé

## 1.1 Principe général  
Objectif : apprendre une fonction \(f : X \rightarrow Y\) à partir d’exemples étiquetés \((x^{(i)}, y^{(i)})\).

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

## 1.2 Régression

Prédire une valeur continue.

**Exemple :** prédiction du prix d'un appartement

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

---

## 1.3 Classification

Prédire une **classe** (exemple : reconnaître un chiffre MNIST).

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

---

## 1.4 Fonction de coût

Mesure l'écart entre prédictions et cibles.

**Exemple : Erreur quadratique moyenne (MSE)**

\[
J(\theta) = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2
\]

où \(m\) est le nombre d'exemples d'entraînement.

---

## 1.5 Descente de gradient

Algorithme itératif pour minimiser la fonction de coût :

```
1. Initialiser θ (poids aléatoires)
       │
       ▼
2. Calculer ∇J(θ) (gradient)
       │
       ▼
3. Mettre à jour : θ ← θ − α∇J(θ)
       │          (α = taux d'apprentissage)
       ▼
4. Répéter jusqu'à convergence
```

**Illustration du paysage de perte :**

```
Perte J(θ)
    ▲
    │           o
    │        o
    │     o
    │  o
    └─────────────────────► θ
         minimum global
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

\[
V^\pi(s)=E[r+\gamma V^\pi(s')].
\]

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

Table de valeurs Q pour chaque paire (état, action) :

```
        Actions
        a1      a2      a3
┌─────────────────────────────┐
s1  │  Q11    Q12    Q13      │
s2  │  Q21    Q22    Q23      │
s3  │  Q31    Q32    Q33      │
└─────────────────────────────┘
États
```

---

## 4.2 Mise à jour Bellman

**Équation de mise à jour Q-learning :**

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
\]

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

\[
\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t\right]
\]

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

**Fin du document**