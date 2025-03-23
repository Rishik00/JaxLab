## Basic version of K Nearest Neighbors algorithm in JAX
    ## The three main steps that I have broken down into are:
        ## Define a distance metric
        ## Nearest neighbors algortihm (brute-force method), which calculates the distance between the given test_row and the training data
        ## Get the class labels by sorting the neighbor indices from y_train

    ## What should be done: 
        ## 1. Optimise implementation
        ## 2. Docs
        ## 3. Implement K dimensional Tree and Ball trees 

import jax
import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# 1. Set the random key for JAX
key = jax.random.key(42)

# 2. Load the Iris dataset and convert it to JAX arrays
iris = load_iris()
X, y = jnp.array(iris.data), jnp.array(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Euclidean distance metric (alternative to jnp.linalg.norm)
def euclidean_distance_metric(x1, x2, ctype='norm'):
    if ctype == 'norm':
        return jnp.linalg.norm(x1 - x2)
    else:
        diff = (x1 - x2) ** 2
        return jnp.sqrt(jnp.sum(diff, axis=1))

# 4. Nearest neighbors search (find k nearest neighbors)
def nearest_neighbors_search(train, test_row, k=3, distance='euclidean'):
    distances = jnp.array([euclidean_distance_metric(test_row, train_row) for train_row in train])
    sorted_indices = jnp.argsort(distances)[:k]  # Indices of the k smallest distances
    nearest_neighbors = train[sorted_indices]
    return sorted_indices, nearest_neighbors, distances

# 5. Majority voting classification for KNN
def predict_classification(train, test_row, y_train, k=3, distance_metric=euclidean_distance_metric):
    sorted_indices, nearest_neighbors, distances = nearest_neighbors_search(train, test_row, k, distance_metric)
    neighbor_labels = y_train[sorted_indices]
    
    # Use jnp.bincount to count occurrences of each label (assuming labels are integers)
    label_counts = jnp.bincount(neighbor_labels)
    
    # Find the label with the highest count (most common)
    prediction = jnp.argmax(label_counts)
    return prediction

# 6. Inference function to test KNN classification on the whole test set
def inference(X_train, X_test, y_train, y_test, k=3, distance_metric=euclidean_distance_metric):
    correct_predictions = 0
    for i in range(len(X_test)):
        test_row = X_test[i]
        true_label = y_test[i]
        
        # Predict the label for the test row using KNN
        predicted_label = predict_classification(X_train, test_row, y_train, k, distance_metric)
   
        # Compare with the true label
        if predicted_label == true_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(y_test)
    return accuracy

# 1. Set the random key for JAX
key = jax.random.key(42)

if __name__ == "__main__":
    # 2. Load the Iris dataset and convert it to JAX arrays
    iris = load_iris()
    X, y = jnp.array(iris.data), jnp.array(iris.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Testing the KNN accuracy on the Iris dataset
    k = 5
    accuracy = inference(X_train, X_test, y_train, y_test, k)
    print(f"KNN Accuracy on test set with k={k}: {accuracy:.4f}")
