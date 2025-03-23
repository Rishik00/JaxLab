import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Forward pass function
def linear_forward(inputs, weights, bias):
    return jnp.dot(inputs, weights) + bias

# Initialize parameters (weights and bias)
def init_params(key, input_dim, output_dim):
    W = jax.random.normal(key, shape=(input_dim, output_dim))
    b = jax.random.normal(key, shape=(output_dim,))
    return W, b

# Mean Squared Error (MSE) loss function
def mse_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

# Linear regression loss function (used for gradient computation)
def linear_regression(params, X_train, y_train):
    predictions = linear_forward(X_train, params[0], params[1])
    loss = mse_loss(predictions, y_train)
    return loss

if __name__ == "__main__":
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Select only one target for regression (e.g., sepal length)
    y = X[:, 0]  # Assume we're predicting sepal length
    X = X[:, 1:]  # Use other features as inputs

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to JAX arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train).reshape(-1, 1)  # Reshape to column vector
    X_test = jnp.array(X_test)
    y_test = jnp.array(y_test).reshape(-1, 1)

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    input_dim = X_train.shape[1]
    output_dim = 1
    weights, bias = init_params(key, input_dim, output_dim)

    # Training parameters
    epochs = 100
    lr = 0.01
    losses = []

    for epoch in range(epochs):
        # Compute gradients
        grads = jax.grad(linear_regression)([weights, bias], X_train, y_train)

        # Update weights and bias
        weights -= lr * grads[0]
        bias -= lr * grads[1]

        # Compute loss for monitoring
        loss = linear_regression([weights, bias], X_train, y_train)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()
