# TP2: Implementation and Analysis of the Perceptron Algorithm

This document contains the complete source code for **TP2**, along with a detailed line-by-line explanation of the concepts, functions, and logic used.

---

## 1. The Perceptron Model (`perceptron.py`)

The Perceptron is one of the simplest forms of an artificial neural network. it is a linear classifier used for binary classification.

### Source Code: `perceptron.py`

```python
import numpy as np
import matplotlib.pyplot as plt

class perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.w = None
        self.b = None
        self.errors = []

    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        for epoch in range(self.max_iter):
            for i in range(len(X)):
                z = np.dot(X[i], self.w) + self.b
                y_pred = self.activation(z)
                error = y[i] - y_pred
                
                self.w += self.lr * error * X[i]
                self.b += self.lr * error
                self.errors.append(error)
        return self

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return self.activation(z)
```

### Detailed Explanation of `perceptron.py`

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| 1 | `import numpy as np` | Imports NumPy for numerical operations like matrix multiplication and array handling. |
| 6 | `class perceptron:` | Defines the Perceptron class which encapsulates the model's logic. |
| 7 | `def __init__(self, learning_rate=0.01, max_iterations=1000):` | **Constructor**: Initializes the model parameters. |
| 8 | `self.lr = learning_rate` | Sets the **Learning Rate**, which controls how much we adjust the weights during each update step. |
| 9 | `self.max_iter = max_iterations` | Sets the number of **Epochs** (how many times the model sees the entire dataset). |
| 10-11 | `self.w = None`, `self.b = None` | Placeholders for **Weights** ($w$) and **Bias** ($b$). |
| 12 | `self.errors = []` | A list to store error values during training to track convergence. |
| 13-14 | `def activation(self, z):` | **Activation Function**: This is a Heaviside step function. If the input $z \ge 0$, it returns 1; otherwise, it returns 0. |
| 15 | `def fit(self, X, y):` | **Training Function**: This is where the model learns from the data. |
| 16 | `self.w = np.zeros(X.shape[1])` | Initializes weights to zero. The size corresponds to the number of features in $X$. |
| 17 | `self.b = 0` | Initializes the bias to zero. |
| 19 | `for epoch in range(self.max_iter):` | Loops through the dataset multiple times (epochs). |
| 20 | `for i in range(len(X)):` | Loops through each individual data point in the dataset. |
| 22 | `z = np.dot(X[i], self.w) + self.b` | Calculates the linear combination: $z = w \cdot x + b$. |
| 23 | `y_pred = self.activation(z)` | Passes the result through the activation function to get a prediction (0 or 1). |
| 24 | `error = y[i] - y_pred` | Calculates the difference between the actual label ($y$) and the prediction. |
| 25 | `self.w += self.lr * error * X[i]` | **Weight Update Rule**: Adjusts weights based on the error. If error is 0, no change occurs. |
| 26 | `self.b += self.lr * error` | **Bias Update Rule**: Adjusts the bias similarly to the weights. |
| 27 | `self.errors.append(error)` | Stores the error for later visualization. |
| 29-31 | `def predict(self, X):` | **Inference Function**: Used to predict classes for new data using the learned $w$ and $b$. |

---

## 2. Data Generation and Evaluation (`data.py`)

This script handles data creation, model training, performance evaluation, and visualization.

### Source Code: `data.py`

```python
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from perceptron import perceptron 

# 1. Génération des données
X, y = make_blobs(
    n_samples=100,
    centers=[[2, 2], [6, 6]],
    cluster_std=1.0,
    random_state=42
)

# 2. Perceptron Training
model = perceptron()
model.fit(X, y)

# 3. Splitting data
X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

model = perceptron()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = np.mean(predictions == y_test)

# 4. Visualisation
plt.figure(figsize=(6, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='green', label='Sain (0)')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='A risque (1)')
# ... (see full code for plot details)
plt.show()
```

### Detailed Explanation of `data.py`

| Line(s) | Code | Explanation |
| :--- | :--- | :--- |
| 1-4 | `import ...` | Imports necessary tools: Scikit-learn for data, NumPy for math, Matplotlib for plots, and our custom `perceptron` class. |
| 6-11 | `make_blobs(...)` | Generates a synthetic dataset of 100 points centered around (2,2) and (6,6). This creates two distinct groups (clusters) that the perceptron will try to separate. |
| 14-28 | `print(...)` | Prints basic statistics (Shape, mean, standard deviation) to understand the data distribution. This is essential for verifying that the classes are somewhat separable. |
| 35-36 | `model = perceptron(); model.fit(X, y)` | Creates an instance of our Perceptron and trains it on the entire dataset. |
| 41 | `score = np.mean(predictions == y)` | Calculates **Accuracy**: the percentage of correct predictions. |
| 48-51 | `X_train = X[:80] ...` | **Data Splitting**: We take the first 80 points for training and keep the last 20 for testing. This is crucial to evaluate how well the model generalizes to unseen data. |
| 64 | `print("w =", model.w, "b=", model.b)` | Displays the final learned parameters. These parameters define the equation of the line that separates the two classes. |
| 73-81 | `plt.scatter(...)` | Plots the data points. Green for "Healthy" (Class 0) and Red for "At Risk" (Class 1). |
| 97-103 | `plt.plot(model.errors)` | Plots the **Convergence Curve**. If the error drops to zero, it means the model has successfully separated the classes. |
| 115-121 | `plt.plot(X[:,0], -(model.w[0]*X[:,0]+b)/model.w[1])` | **Decision Boundary**: Plots the actual line $w_1x_1 + w_2x_2 + b = 0$. We solve for $x_2$ to plot it: $x_2 = -(w_1x_1 + b) / w_2$. |
| 133-145 | `np.meshgrid(...)` | Creates a **Decision Zone**. It colors the background based on what the model predicts for every point in the plane, showing exactly where the "boundary" lies. |
| 154 | `perceptron(max_iterations=5)` | Demonstrates what happens with **Underfitting**. Training for only 5 epochs might not be enough for the line to find the perfect separation. |

---

## 3. Why did we create each function?

### In `perceptron.py`:
1.  **`__init__`**: To set the "rules" of learning (Learning Rate) and the "patience" of the model (Max Iterations).
2.  **`activation`**: To transform a continuous number into a binary decision (0 or 1). This mimics the firing of a biological neuron.
3.  **`fit`**: This is the heart of the algorithm. It implements the **Perceptron Learning Rule**. Without this, the model cannot learn from its mistakes.
4.  **`predict`**: To use the "knowledge" (weights and bias) stored in the model to classify new, unknown data points.

### In `data.py`:
1.  **`make_blobs`**: To create a controlled environment where we know the classes are separable, allowing us to test if our algorithm actually works.
2.  **Visualizations (`plt.scatter`, `plt.plot`)**: Because deep learning is often a "black box," visualization helps us see the decision boundary and understand if the model has converged.
3.  **Train/Test Split**: To ensure the model isn't just "memorizing" the data (overfitting) but actually understanding the underlying pattern.

---

## Summary of the Logic
1.  **Initialize** weights and bias to 0.
2.  **Predict** the class of a point.
3.  **Compare** the prediction with the real label.
4.  **Adjust** the weights if there is an error.
5.  **Repeat** until the model stops making mistakes or reaches the maximum number of iterations.
