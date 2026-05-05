from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from perceptron import perceptron 

# ==============================================================================
# 1. GÉNÉRATION DES DONNÉES
# ==============================================================================
# On génère un dataset synthétique avec 2 clusters (blobs)
# n_samples : nombre de points total
# centers : coordonnées des centres des deux classes
# cluster_std : écart-type (dispersion) des points autour des centres
X, y = make_blobs(
    n_samples=100,
    centers=[[2, 2], [6, 6]],
    cluster_std=1.0,
    random_state=42
)

# ==============================================================================
# 2. EXPLORATION ET STATISTIQUES DES DONNÉES
# ==============================================================================
print("*******************************************************")
print("Shape X (caractéristiques) :", X.shape) # Doit être (100, 2)
print("Shape y (cibles) :", y.shape)           # Doit être (100,)

print("*******************************************************")
print("Nombre de points Classe 0 :", len(y[y == 0]))
print("Nombre de points Classe 1 :", len(y[y == 1]))

# Analyse des caractéristiques par classe (Moyenne et Écart-type)
print("*******************************************************")
print("Moyenne classe 0 :", np.mean(X[y == 0]))
print("Écart-type classe 0 :", np.std(X[y == 0]))
print("Moyenne classe 1 :", np.mean(X[y == 1]))
print("Écart-type classe 1 :", np.std(X[y == 1]))
print("\n")

# ==============================================================================
# 3. ENTRAÎNEMENT DU PERCEPTRON (Sur toutes les données)
# ==============================================================================
print("*******************************************************")
print("Entraînement du Perceptron")
print("*******************************************************")
model = perceptron()
model.fit(X, y)

# Prédiction sur le même ensemble pour vérifier l'apprentissage
predictions = model.predict(X)

# Calcul du score de précision (Accuracy)
score = np.mean(predictions == y)
print("*******************************************************")
print("Predictions:", predictions)
print("Actual Labels:", y)
print("Accuracy globale:", score)
print("*******************************************************")

# ==============================================================================
# 4. ÉVALUATION : TRAIN / TEST SPLIT
# ==============================================================================
# On divise les données : 80% pour l'entraînement, 20% pour le test
X_train, y_train = X[:80], y[:80]
X_test, y_test = X[80:], y[80:]

model_split = perceptron()
model_split.fit(X_train, y_train)

# Test sur les données jamais vues par le modèle
predictions_test = model_split.predict(X_test)
score_test = np.mean(predictions_test == y_test)
print("*******************************************************")
print("Après division Train/Test :")
print("Accuracy sur le Test Set :", score_test)
print("*******************************************************")

# Affichage des paramètres finaux du modèle
print("\n")
print("Paramètres appris : w =", model_split.w, "et b =", model_split.b)
print("\n*******************************************************")

# ==============================================================================
# 5. VISUALISATIONS GRAPHIQUES
# ==============================================================================

# --- Graphique 1 : Visualisation des données brutes ---
print("\nVisualisation des données...")
plt.figure(figsize=(6, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='green', label='Sain (classe 0)', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='A risque (classe 1)', alpha=0.6)
plt.xlabel('Indicateur biologique 1 (x1)')
plt.ylabel('Indicateur biologique 2 (x2)')
plt.title('Données patients - Répartition des classes')
plt.legend()
plt.grid(True)
plt.show()

# --- Graphique 2 : Courbe de convergence (Erreurs) ---
print("Affichage de la convergence...")
plt.figure(figsize=(6, 5))
plt.plot(np.arange(len(model_split.errors)), model_split.errors, color='blue', label='Erreur')
plt.xlabel('Nombre de mises à jour (échantillons vus)')
plt.ylabel('Valeur de l\'erreur (y - y_pred)')
plt.title('Convergence du perceptron (Historique des erreurs)')
plt.legend()
plt.grid(True)
plt.show()

# --- Graphique 3 : Droite de séparation ---
# L'équation de la droite est : w1*x1 + w2*x2 + b = 0
# D'où : x2 = -(w1*x1 + b) / w2
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlGn')
x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
y_vals = -(model_split.w[0] * x_vals + model_split.b) / model_split.w[1]
plt.plot(x_vals, y_vals, color='blue', linestyle='-', label='Frontière de décision')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Droite de séparation apprise')
plt.legend()
plt.grid(True)
plt.show()

# --- Graphique 4 : Zone de décision (Contour) ---
print("Zone de décision...")
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Création d'une grille de points (Meshgrid)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
# Prédiction pour chaque point de la grille
Z = model_split.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn') # Zones colorées
plt.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5]) # Ligne de démarcation
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k') # Points originaux
plt.title('Zone de décision colorée')
plt.show()

# ==============================================================================
# 6. EXPÉRIMENTATION : NOMBRE D'EPOCHS RÉDUIT
# ==============================================================================
print("\n****************************************************************")
print("4. Entraînement avec seulement 5 epochs")
print("****************************************************************")

model_5 = perceptron(max_iterations=5)
model_5.fit(X_train, y_train)

predictions_5 = model_5.predict(X_test)
score_5 = np.mean(predictions_5 == y_test)

print("Score (accuracy) avec 5 epochs :", score_5)
print("Poids appris (w) :", model_5.w)
print("Biais appris (b) :", model_5.b)

# Visualisation de la zone de décision pour le modèle à 5 epochs
Z_5 = model_5.predict(np.c_[xx.ravel(), yy.ravel()])
Z_5 = Z_5.reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z_5, alpha=0.3, cmap='RdYlGn')
plt.contour(xx, yy, Z_5, colors='black', linewidths=2, levels=[0.5])
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('Zone de décision - 5 epochs seulement')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()