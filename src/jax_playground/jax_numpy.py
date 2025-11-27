"""Getting to know Jax elementary command."""

# %%
import jax
import jax.numpy as jnp

# %%
x = jnp.arange(5)
# Way of changing otherwise immutable arrays
x.at[3].set(7)
print(x)
print(f"Is x a jax.Array ? {isinstance(x, jax.Array)}")
print(f"Devices {x.devices()}")

# %%
