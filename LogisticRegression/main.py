## Logistic Regression for binary classification, multiclass coming soon

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

key = jax.random.key(42)

## 1. Taking input and dividing into arrays
iris = load_iris()
X, y = iris.data, iris.target

# Select only two classes for binary classification (class 0 and class 1)
X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

#written by nvim
def hello_world(inp):
    return "Hello world from nvim"

#written by nvim at temrinal
def hello_world_from_terminal(inp):
    return "Hello world from nvim terminal lkol"


# Sigmoid function
def sigmoid(inp):
    return 1.0 / (1.0 + jnp.exp(-inp))

# Forward pass function
def forward(inputs, weights, bias):
    forward_output = jnp.dot(inputs, weights) + bias
    final_output = sigmoid(forward_output)
    return final_output

# Initialize parameters (weights and bias)
def init_params(key, input_dim, output_dim):
    W = jax.random.normal(key, shape=(input_dim, output_dim))
    b = jax.random.normal(key, shape=(output_dim,))
    return W, b

# Binary cross entropy loss
def binary_cross_entropy(predictions, targets):
    epsilon = 1e-8  # Small value to avoid log(0)
    predictions = jnp.clip(predictions, epsilon, 1 - epsilon)  # Clip predictions for stability
    return -jnp.mean(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions))

# Compute gradients of the loss with respect to weights and bias
def compute_loss_and_grads(params, X_train, y_train):
    predictions = forward(X_train, params[0], params[1])
    loss = binary_cross_entropy(predictions, y_train)
    return loss

if __name__ == "__main__":

    key = jax.random.PRNGKey(42)
    epochs=100
    lr=0.01
    losses = []
    weights, bias = init_params(key, X_train.shape[1], 1)

    for i in range(epochs):
        output = forward(X_train, weights, bias)
        loss = binary_cross_entropy(output, y_train)

        # Compute gradients using jax.grad
        gradients = jax.grad(compute_loss_and_grads)([weights, bias], X_train, y_train)
        weights = weights - lr * gradients[0]
        losses.append(loss)

        print("Loss:", loss)

plt.figure(figsize=(10, 6))
plt.plot(losses, color='blue', label='Training Loss', marker='o', markersize=3, linestyle='-', linewidth=2)

# Add labels and title
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss over Epochs', fontsize=16)

# Add a grid
plt.grid(True)

# Optionally, add a rolling average curve
window_size = 10  # Example window size for smoothing
rolling_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
plt.plot(np.arange(window_size-1, epochs), rolling_avg, color='red', label='Rolling Average', linestyle='--')

plt.legend()
plt.show()
