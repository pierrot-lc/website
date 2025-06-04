---
title: How to Build a Graph Convolutional Network with JAX and Equinox!
description: >-
  JAX implementation of graph neural networks. Uses Equinox and shows two different ways of
  implementing GNNs.
illustration: illustration.png

tags:
  - 2024-09-02
---

I've been learning [JAX][jax-docs] for the past few months and for a recent project I needed to
train a basic Graph Neural Network. Surprisingly, I couldn't find good libraries to easily build
GNNs. Most of them are [outdated](https://github.com/google-deepmind/jraph) and don't use my DL
framework of choice ([Equinox!][equinox-docs]). This was the perfect excuse to implement my own GNNs
from scratch.

> I love the JAX/Equinox combination. In my opinion it is perfect for DL research. It gives the user
> an approachable low-level API that they can play with and that scales well thanks to its powerful
> JIT compiler. If you're not already using JAX, have a look at [long live JAX][long-live-jax].

So let's dive in. We'll see the simplest implementation first using the adjacency matrix of the
graph. Then, we'll continue with the more complex but modular approach using the edge list
representation.

## With the Adjacency Matrix

The adjacency formulation is pretty straighforward:

```python
import equinox as eqx
import equinox.nn as nn
import jax.experimental.sparse as jsparse
from jax import vmap
from jaxtyping import Array, Float, Int, PRNGKeyArray


class GraphConv(eqx.Module):
    linear: nn.Linear

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
        self.linear = nn.Linear(hidden_dim, hidden_dim, key=key)

    def __call__(
        self,
        nodes: Float[Array, "n_nodes hidden_dim"],
        adjacency: Int[jsparse.BCOO, "n_nodes n_nodes"]
    ) -> Float[Array, "n_nodes hidden_dim"]:
        messages = vmap(self.linear)(nodes)
        return adjacency @ messages
```

The only thing a bit special here is the usage of JAX's [sparse module][jax-sparse-module]. Sparse
computations makes graph convolutions highly efficient. You can even extend the layer to the `mean`
aggregation operation by dividing the result by `adjacency.sum(axis=1)`.

> Note that the sparse module still in experimental.

The code is pretty straighforward. It does the following update:

$$
n_i \leftarrow \sum_{j \in \mathcal{N(i)}} W n_j
$$

The only thing to take care about is how you define your adjacency matrix. In the code above I considered
that `A[i, j] = 1` if an edge $j \rightarrow i$ exists.

Implementing the updates for the `min` or `max` aggregation schemes looks a bit trickier. It is much
easier to use the edge list data structure.

## Using the Edge List

This one took me a while for the first time. Everything clicked once I found out about
[`jax.ops.segment_max`][segment-ops]:

```python
import equinox as eqx
import equinox.nn as nn
from jax import vmap
from jaxtyping import Array, Float, Int, PRNGKeyArray

class GraphConv(eqx.Module):
    linear: nn.Linear

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray):
        self.linear = nn.Linear(hidden_dim, hidden_dim, key=key)

    def __call__(
        self,
        nodes: Float[Array, "n_nodes hidden_dim"],
        edges: Int[Array, "n_edges 2"],
    ) -> Float[Array, "n_nodes hidden_dim"]:
        messages = vmap(self.linear)(nodes)
        messages = messages[edges[:, 0]]  # Shape of [n_edges hidden_dim].
        messages = jax.ops.segment_max(
            data=messages,
            segment_ids=edges[:, 1],
            num_segments=len(nodes),
        )  # Shape of [n_nodes hidden_dim].
        return messages
```

I consider here that `edges[e] = [j, i]` means that the $e$-th edge goes from node $j$ to node $i$.
This layer apply a basic GNN update using the max aggregation operation:

$$
n_i \leftarrow \max_{j \in \mathcal{N(i)}} W n_j
$$

Using [`jax.ops`][jax-ops] is much harder to read when encountered for the first time but it
basically does exactly what we want. You should be able to tweak this implementation to suit your
needs.

**Take care of default values!** If a node id is not present in `segment_ids`, it will be filled
with some default value that depends on the `jax.ops` used. For example, `jax.ops.segment_max` will
fill missing values with `-inf`. This is probably not what you want!

Here's a general aggregation implementation that covers everything:

```python
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

@partial(jax.jit, static_argnums=[2, 3, 4])
def aggregate(
    messages: Float[Array, "n_edges hidden_dim"],
    destination_index: Int[Array, " n_edges"],
    n_nodes: int,
    aggregation_type: str,
    default_value: float = 0.0,
) -> Float[Array, "n_nodes hidden_dim"]:
    """Aggregate the edge features accross the destination index.

    ---
    Args:
        messages: The features to aggregate.
        destination_index: The id of the destination nodes.
        n_nodes: Total number of nodes.
        aggregation_type: The aggregation to apply, either "sum", "mean", "min" or "max".
        default_value: The default value to use for missing destination nodes.

    ---
    Returns:
        The aggregated features.
    """
    match aggregation_type:
        case "sum":
            segment_fn = jax.ops.segment_sum
        case "mean":
            segment_fn = jax.ops.segment_sum
        case "min":
            segment_fn = jax.ops.segment_min
        case "max":
            segment_fn = jax.ops.segment_max
        case _:
            raise ValueError(f"Unknown aggregation type {aggregation_type}")

    # Do the aggregation.
    messages = segment_fn(messages, segment_ids=destination_index, num_segments=n_nodes)

    # Count the degree of each destination.
    ones = jnp.ones((len(destination_index), 1), dtype=jnp.int32)
    degrees = jax.ops.segment_sum(
        ones, segment_ids=destination_index, num_segments=n_nodes
    )

    if aggregation_type == "mean":
        messages = messages / degrees

    # Replace with the default value where degree is 0.
    messages = jnp.where(degrees == 0, default_value, messages)

    return messages
```

This code does not use `equinox` so you should be able to use it with any JAX framework.

## JIT Tips

When using JAX, it is crucial to JIT your computations. The way JIT works is that the first time it
encounters your function, it will compile it and keep a cache of the compilation so that later on
when you call the same function again it can just use the cached compilation directly.

But the cached version of your function is shape-dependent, meaning that if you pass an argument
with a different shape, it will need to recompile everything and cache the new result again. This is
an issue for our graphs because we typically have a variable number of nodes and edges for different
graphs of our datasets.

It means that in order to avoid recompilation, we need to pad our graphs before feeding them into
the model. For the adjacency matrix, we can simply fill it with more 0s. For the edge list, we can
create fictive self-loops to a padded fictive node.

You can have a look at this [explanation][jit-shape-discussion] from a JAX dev for a more in-depth
understanding of how JIT works under-the-hood.

## Final Thoughts

While the adjacency representation makes it easy to define the classical GCN, it is less
customizable. The second implementation, using the edge list, is flexible and allows for more
complex GNNs. Nervertheless, keep in mind that the adjacency representation remains more
computationally efficient and requires less memory.

You can have a look at the whole code used to train the models here: [gnn-tuto][github-impl].


[equinox-docs]:         https://docs.kidger.site/equinox/
[gat-paper]:            https://arxiv.org/abs/1710.10903
[github-impl]:          https://github.com/pierrot-lc/gnn-tuto
[jax-docs]:             https://jax.readthedocs.io/en/latest/quickstart.html
[jax-ops]:              https://jax.readthedocs.io/en/latest/jax.ops.html
[jax-sparse-module]:    https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html
[jit-shape-discussion]: https://github.com/google/jax/issues/2521#issuecomment-604759386
[kendall-rank]:         https://en.wikipedia.org/wiki/kendall_rank_correlation_coefficient
[long-live-jax]:        https://neel04.github.io/my-website/blog/pytorch_rant/
[networkx-clustering]:  https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html
[networkx]:             https://networkx.org/
[reddit-binary]:        https://paperswithcode.com/dataset/reddit-binary
[segment-sum]:          https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.segment_sum.html
[segment-ops]:          https://docs.jax.dev/en/latest/jax.ops.html
