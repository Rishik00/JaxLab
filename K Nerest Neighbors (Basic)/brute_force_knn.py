import jax
import numpy as np
import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

class GeneralizedDistanceMetrics:
    def __init__(self, x1, x2, norm='eu'):
       self.x1 = x1
       self.x2 = x2
       self.norm = norm

    def get_distance(self):
       if self.norm == 'eu':
           return self.euclidean_distance()
       elif self.norm == 'ma':
           return self.manhattan_distance()
       else:
           raise ValueError("Sorry bruv, you're cancelled.")

    def manhattan_distance(self):
        diff = jnp.abs(self.x1 - self.x2)
        print('diff shape: ', diff.shape)
        return jnp.sum(jnp.abs(self.x1-self.x2), axis=1)
    

    def euclidean_distance(self):
        ## Only add axis=1 when reshaping the input to 2D
        diff = (self.x1-self.x2) ** 2
        return jnp.sqrt(jnp.sum(diff), axis=1)

def load_dataset():
    iris = load_iris()
    X, y = jnp.array(iris.data), jnp.array(iris.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


def nearest_neighbor_search(top_k: int, X_train, test_row, y_train, distance_metric:str='ma'):
    
    distances = []
    norm = GeneralizedDistanceMetrics(X_train, test_row, norm=distance_metric)
    distances = norm.get_distance()

    sorted_indices = jnp.argsort(jnp.array(distances))
    top_k_indices = sorted_indices[:top_k]

    ## get top_k rows from training set top_k_rows = sorted_distances_jnp[:top_k]
    k_nearest_neighbors = X_train[top_k_indices]
    neighbor_labels = y_train[top_k_indices]
    
    ## What is this?
    label_counts = jnp.bincount(neighbor_labels)
    
    ## Get the best label
    prediction = jnp.argmax(label_counts)
    
    return sorted_indices, k_nearest_neighbors, distances, prediction


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset()
    # X_train = X_train.reshape(X_train.shape[1], X_train.shape[0])

    print("Loaded data")
    for i in range(2):
        test_row = X_test[i]
        indices, knn, distances, prediction = nearest_neighbor_search(3, X_train, test_row, y_train)
        print("-------Nearest Neighbors-------")
        for neighbor in knn:
            print(neighbor.shape)

        print("-------Prediction-------")
        print(prediction)



    
