#%%
import numpy as np
from jax import jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
from time import time
# %%
def impure_print_side_effect(x):
  print("Executing function")  # This is a side-effect
  return x

# The side-effects appear during the first run
print ("First call: ", jit(impure_print_side_effect)(4.))

# Subsequent runs with parameters of same type and shape may not show the side-effect
# This is because JAX now invokes a cached compilation of the function
print ("Second call: ", jit(impure_print_side_effect)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
print ("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))
# %%

# List available devices
print("JAX devices:", jax.devices())

# Create a large array on GPU
x = jnp.ones((10000, 10000))

jax.device_put(x)

# Perform a computation
start = time()
y = jnp.dot(x, x.T).block_until_ready()  # Ensure computation completes
end = time()

print(f"1st run : Computation finished in {end - start:.3f} seconds")

# %%
start = time()
y = jnp.dot(x, x.T).block_until_ready()  # Ensure computation completes
end = time()

print(f"2nd run : Computation finished in {end - start:.3f} seconds")

# %%
