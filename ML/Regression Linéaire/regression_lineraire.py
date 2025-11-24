"""
Régression linéaire avec SciKit-Learn
Jeu de données : Salary vs YearsExperience
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuration de l'encodage UTF-8 pour Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Chargement des données
print("=" * 50)
print("ÉTAPE 1 : PRÉPARATION DES DONNÉES")
print("=" * 50)

# Charger le dataset
df = pd.read_csv('Salary_dataset.csv')

# Afficher les premières lignes
print("\n1. Aperçu des données :")
print(df.head(10))

# Informations sur le dataset
print("\n2. Informations sur le dataset :")
print(f"Nombre de lignes : {len(df)}")
print(f"Nombre de colonnes : {len(df.columns)}")
print(f"\nColonnes : {df.columns.tolist()}")

# Vérifier les valeurs manquantes
print("\n3. Vérification des valeurs manquantes :")
print(df.isnull().sum())

# Statistiques descriptives
print("\n4. Statistiques descriptives :")
print(df.describe())

# 2. Nettoyage des données
print("\n5. Nettoyage des données :")

# Supprimer la colonne d'index si elle existe (première colonne sans nom)
if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
    df = df.drop(df.columns[0], axis=1)

# Vérifier et supprimer les doublons
print(f"Nombre de doublons : {df.duplicated().sum()}")
df = df.drop_duplicates()

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

print(f"Nombre de lignes après nettoyage : {len(df)}")

# 3. Préparation des features et target
print("\n6. Préparation des features (X) et target (y) :")

# X = YearsExperience (entrée)
# y = Salary (sortie)
X = df[['YearsExperience']].values
y = df['Salary'].values

print(f"Shape de X : {X.shape}")
print(f"Shape de y : {y.shape}")
print(f"\nPremières valeurs de X (YearsExperience) : {X[:5].flatten()}")
print(f"Premières valeurs de y (Salary) : {y[:5]}")

# 4. Visualisation des données
print("\n7. Visualisation des données :")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6)
plt.xlabel('Années d\'expérience (YearsExperience)')
plt.ylabel('Salaire (Salary)')
plt.title('Relation entre l\'expérience et le salaire')
plt.grid(True, alpha=0.3)
plt.savefig('visualisation_donnees.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé dans 'visualisation_donnees.png'")
plt.close()

# 5. Division train/test
print("\n8. Division des données en train/test (80/20) :")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Taille du set d'entraînement : {X_train.shape[0]} échantillons")
print(f"Taille du set de test : {X_test.shape[0]} échantillons")

# 6. Entraînement du modèle de régression linéaire
print("\n" + "=" * 50)
print("ÉTAPE 2 : ENTRÂINEMENT DU MODÈLE")
print("=" * 50)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

print("\n9. Modèle entraîné avec succès !")

# Afficher les paramètres du modèle
print("\n10. Paramètres du modèle :")
print(f"   Coefficient (Θ1) : {model.coef_[0]:.2f}")
print(f"   Intercept (Θ0) : {model.intercept_:.2f}")
print(f"\n   Équation de la droite : y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")

# 7. Prédictions
print("\n11. Prédictions sur le set de test :")
y_pred = model.predict(X_test)

# Afficher quelques prédictions
print("\n   Comparaison prédictions vs valeurs réelles :")
for i in range(min(6, len(X_test))):
    print(f"   Expérience: {X_test[i][0]:.1f} ans -> Prédit: {y_pred[i]:.2f} € | Réel: {y_test[i]:.2f} €")

# 8. Évaluation du modèle
print("\n12. Évaluation du modèle :")

# Métriques sur le set d'entraînement
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# Métriques sur le set de test
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

print("\n   Sur le set d'entraînement :")
print(f"   - MSE (Mean Squared Error) : {train_mse:.2f}")
print(f"   - RMSE (Root Mean Squared Error) : {train_rmse:.2f}")
print(f"   - R² Score : {train_r2:.4f}")

print("\n   Sur le set de test :")
print(f"   - MSE (Mean Squared Error) : {test_mse:.2f}")
print(f"   - RMSE (Root Mean Squared Error) : {test_rmse:.2f}")
print(f"   - R² Score : {test_r2:.4f}")

# 9. Visualisation de la régression
print("\n13. Visualisation de la régression linéaire :")

plt.figure(figsize=(12, 6))

# Graphique 1 : Données et droite de régression
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Données d\'entraînement')
plt.scatter(X_test, y_test, color='red', alpha=0.6, label='Données de test')
plt.plot(X, model.predict(X), color='green', linewidth=2, label='Régression linéaire')
plt.xlabel('Années d\'expérience (YearsExperience)')
plt.ylabel('Salaire (Salary)')
plt.title('Régression linéaire : Salaire vs Expérience')
plt.legend()
plt.grid(True, alpha=0.3)

# Graphique 2 : Prédictions vs Valeurs réelles
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Ligne parfaite')
plt.xlabel('Valeurs réelles (Salary)')
plt.ylabel('Prédictions (Salary)')
plt.title(f'Prédictions vs Réalité (R² = {test_r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_lineaire.png', dpi=300, bbox_inches='tight')
print("   Graphique sauvegardé dans 'regression_lineaire.png'")
plt.close()

# 10. Prédictions pour de nouvelles valeurs
print("\n14. Exemples de prédictions pour de nouvelles valeurs :")
nouvelles_experiences = np.array([[0.5], [2.5], [5.0], [8.0], [12.0]])
predictions = model.predict(nouvelles_experiences)
for exp, sal in zip(nouvelles_experiences, predictions):
    print(f"   {exp[0]:.1f} ans d'expérience -> Salaire prédit : {sal:.2f} €")

# 11. Résumé final
print("\n" + "=" * 50)
print("RÉSUMÉ FINAL")
print("=" * 50)
print("[OK] Donnees chargees et nettoyees")
print("[OK] Modele de regression lineaire entraine")
print("[OK] Parametres du modele calcules")
print("[OK] Predictions effectuees")
print("[OK] Modele evalue (R² = {:.4f} sur le test)".format(test_r2))
print("[OK] Visualisations creees")
print("\nLa regression lineaire est complete !")
print("=" * 50)

