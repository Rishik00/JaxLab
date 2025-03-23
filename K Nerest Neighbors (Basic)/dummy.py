import jax.numpy as jnp
import jax

arr1 = jnp.array([[1,2,3], [1,2,4]])
arr2 = jnp.array([1])

print(arr1.shape, arr2.shape)
print(arr1 + arr2)



