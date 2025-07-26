import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.lax.control_flow.loops import _interleave


def associative_reduce(fn: Callable[[Array, Array], Array], elems: Array) -> Array:
    if len(elems) < 2:
        return elems

    result = fn(elems[::2], elems[1::2])
    return associative_reduce(fn, result)


def associative_scan(fn: Callable[[Array, Array], Array], elems: Array) -> Array:
    if len(elems) < 2:
        return elems

    reduced_elems = fn(elems[::2], elems[1::2])
    odd_elems = associative_scan(fn, reduced_elems)
    even_elems = fn(odd_elems[:-1], elems[2::2])
    even_elems = jnp.concat((elems[:1], even_elems))
    return _interleave(even_elems, odd_elems, axis=0)


@jax.jit
def discounted_returns(rewards: Array, gamma: float) -> Array:
    """Compute the discounted sum of rewards using an associative scan.

    ---
    Args:
        rewards: Rewards of a single rollout.
        gamma: Discount factor.

    ---
    Returns:
        The discounted sum of rewards.
    """

    def associative_fn(
        a: tuple[Array, Array], b: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        """`a` and `b` are tuples of (partial result, list index)."""
        value_a, index_a = a
        value_b, index_b = b
        power = index_a - index_b
        return value_b + gamma**power * value_a, index_b

    returns, _ = jax.lax.associative_scan(
        associative_fn,
        (rewards, jnp.arange(len(rewards))),
        reverse=True,
        axis=0,
    )
    return returns


@jax.jit
def discounted_returns_scan(rewards: Array, gamma: float) -> Array:
    def scan_fn(carry: Array, reward: Array) -> tuple[Array, Array]:
        carry = reward + gamma * carry
        return carry, carry

    _, returns = jax.lax.scan(scan_fn, 0.0, rewards, reverse=True)
    return returns


a = jnp.arange(8)

print(jnp.sum(a))
print(associative_reduce(lambda a, b: a + b, a))

print(jnp.cumsum(a))
print(associative_scan(lambda a, b: a + b, a))

print(discounted_returns(a.astype(float), gamma=0.9))
print(discounted_returns_scan(a.astype(float), gamma=0.9))

rewards = jax.random.normal(jax.random.PRNGKey(0), 2**16)
gamma = 0.9

# Compile first.
discounted_returns_scan(rewards, gamma)
discounted_returns(rewards, gamma)

start = time.time()
for _ in range(100):
    discounted_returns_scan(rewards, gamma).block_until_ready()
time_scan = (time.time() - start) / 100

start = time.time()
for _ in range(100):
    discounted_returns(rewards, gamma).block_until_ready()
time_associative = (time.time() - start) / 100

print(f"Time associative: {time_associative * 1000:.2f}ms")
print(f"Time scan: {time_scan * 1000:.2f}ms")
