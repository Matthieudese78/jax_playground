# Jax playground / Quickstart with Jax

- Jax.numpy as jnp
- Jax arrays are immutable
- Jax arrays have a devices attribute to keep track of where the array is stored for parallel computing.  
- Good Jax jupyter [notebook](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)

- jax.jit : using a decorator.
  - enables to compile sequences of operations can be optimized together and run at once. 
  - not for the whole code can be JIT as the array shapes have to be known in advance : static array shapes.    
