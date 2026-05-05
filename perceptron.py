import numpy as np
import matplotlib.pyplot as plt

# Classe implémentant l'algorithme du Perceptron Simple
class perceptron:
    """
    Algorithme du Perceptron pour la classification binaire.
    Attributs:
        lr (float): Taux d'apprentissage (Learning Rate).
        max_iter (int): Nombre maximum d'itérations (époques).
        w (np.array): Vecteur des poids synaptiques.
        b (float): Biais (seuil d'activation).
        errors (list): Historique des erreurs pour chaque mise à jour.
    """
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        # Initialisation des hyperparamètres
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.w = None
        self.b = None
        self.errors = []

    def activation(self, z):
        """Fonction d'activation de type Heaviside (échelon)."""
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        """
        Entraîne le modèle sur le jeu de données (X, y).
        
        Paramètres:
            X (np.array): Matrice des caractéristiques (samples x features).
            y (np.array): Vecteur des étiquettes cibles (0 ou 1).
        """
        # Initialisation des poids à zéro (taille égale au nombre de colonnes de X)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        # Boucle d'entraînement (époques)
        for epoch in range(self.max_iter):
            for i in range(len(X)):
                # Calcul de la somme pondérée (z = W.X + b)
                z = np.dot(X[i], self.w) + self.b
                # Prédiction via la fonction d'activation
                y_pred = self.activation(z)
                # Calcul de l'erreur (différence entre valeur réelle et prédite)
                error = y[i] - y_pred
                
                # Mise à jour des poids et du biais si une erreur est commise
                # Règle de mise à jour : W = W + lr * error * X
                self.w += self.lr * error * X[i]
                self.b += self.lr * error
                
                # Sauvegarde de l'erreur pour analyse de convergence
                self.errors.append(error)
        return self

    def predict(self, X):
        """
        Effectue des prédictions sur de nouvelles données X.
        """
        # Calcul de la sortie linéaire pour tout l'ensemble de données
        z = np.dot(X, self.w) + self.b
        # Retourne les classes prédites (0 ou 1)
        return self.activation(z)

# Code de test commenté (utilisé pour le débogage initial)
# X = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([0,0,0,1])
# model = perceptron()
# model.fit(X, y)
# print("Predictions:", model.predict(X))
# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()