# TP2: Simple Perceptron Implementation and Evaluation

This project focuses on the implementation of the **Simple Perceptron** algorithm for binary classification. It is part of the Deep Learning (S2) course curriculum.

## 📌 Project Description

The goal of this practical work (TP) is to understand the internal mechanics of an artificial neuron (Perceptron) by implementing it "from scratch" using NumPy and evaluating it on synthetic datasets.

The project covers:
1.  Implementation of the Perceptron learning rule.
2.  Generation of linearly separable data.
3.  Convergence analysis (error history).
4.  Decision boundary visualization.
5.  Performance evaluation (Accuracy, Train/Test Split).

## 📂 File Structure

-   `perceptron.py`: Contains the `perceptron` class with `fit` (training) and `predict` (inference) methods.
-   `data.py`: Main script that generates data, trains the model, and produces visualizations.
-   `TP2_Complete_Guide.md`: Comprehensive guide and detailed instructions for the TP (in French).
-   `.gitignore`: Files and folders to be ignored by Git (e.g., `__pycache__`).

## 🛠️ Installation

Ensure you have Python installed, along with the following libraries:

```bash
pip install numpy matplotlib scikit-learn
```

## 🚀 Usage

To run the full analysis and visualize the results, simply execute:

```bash
python data.py
```

## 📊 Expected Results

The `data.py` script generates several visualizations:
-   **Class Distribution**: Scatter plot of the initial raw data.
-   **Convergence Curve**: Evolution of the error over weight updates.
-   **Decision Boundary**: Visualization of the line learned by the model to separate the two classes.
-   **Decision Zones**: Colored visualization showing how the model classifies the feature space.

## 🧠 Key Concepts
-   **Learning Rate**: Controls the step size of weight updates.
-   **Epochs (Iterations)**: Number of passes over the dataset.
-   **Activation Function**: Heaviside step function (outputs 0 or 1).
-   **Update Rule**: $W = W + \eta \times (y - \hat{y}) \times X$.
