---
title: How to Build a Graph Neural Network with JAX and Equinox!
description: >-
  JAX implementation of graph neural networks. Uses Equinox and shows two different ways of
  implementing GNNs.
illustration: illustration.png

tags: "Created: 2024-09-02 | Updated: 2025-06-10"
---

Because I really love using JAX I had to use it for my latest project involving GNNs. In PyTorch,
you have many options to build your own GNNs, most notably [PyTorch Geometric][PyG] and [Deep Graph
Library][DGL]. But the graph ecosystem is not as developed in JAX, which means that I had to
implement my own GNNs.

I ended up with two approaches, either with the **adjacency matrix** or with the **edge list**.

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

The code should be easy to read. It does the following update:

$$
n_i \leftarrow \sum_{j \in \mathcal{N(i)}} W n_j
$$

I used JAX'[sparse module][jax-sparse-module] for efficiency, which is still experimental. The
only thing to take care about is how you define your adjacency matrix. In the code above I consider
that `A[i, j]` is 1 if an edge $j \rightarrow i$ exists. Implementing the updates for the `min` or
`max` aggregation schemes looks a bit trickier. It is much easier to use the edge list data
structure.

## Using the Edge List

This one took me a while for the first time. Everything clicked once I found out about
[`jax.ops`][jax-ops]:

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

I consider here the edges as a list of tuples `[j, i]` for every edge $j \rightarrow i$. This layer
applies a basic GNN update using the max aggregation operation:

$$
n_i \leftarrow \max_{j \in \mathcal{N(i)}} W n_j
$$

Using [`jax.ops`][jax-ops] is much harder to read when encountered for the first time but it
basically does exactly what we want. We give an array of values to which we want to apply the
aggregation operation and an array of indices. The aggregation is then computed over the grouped
values (segments) defined by the indices.

**Take care of default values!** If a node index is not present in `segment_ids`, it will be filled
with some default value that depends on the aggregation used. For example, `jax.ops.segment_max`
will fill missing values with `-inf`, this is probably not what you want!

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

    messages = segment_fn(messages, segment_ids=destination_index, num_segments=n_nodes)

    ones = jnp.ones((len(destination_index), 1), dtype=jnp.int32)
    degrees = jax.ops.segment_sum(
        ones, segment_ids=destination_index, num_segments=n_nodes
    )

    if aggregation_type == "mean":
        messages = messages / degrees

    messages = jnp.where(degrees == 0, default_value, messages)
    return messages
```

This code does not use `equinox` so you should be able to use it with any JAX framework.

## JIT'ing Graphs

Of couse you will need to JIT your computations. JAX's JIT is shape-dependent so you have to make
sure all of your graphs have the same shape to avoid frequents recompilations. Padding on graphs can
be done with a fake node and fake edges going to that node. Don't forget to pad both the nodes and
edges. *You will need to add fake edges to the sparse matrix as well!* JIT's cache relies on the
number of elements in the sparse matrix.

It would be too verbose to show the implementation here but feel free to have a look at [my
repo][github-impl].

## Final Thoughts

I was quite surprised to see that no simple JAX GNN implementation can be found online. The main
library for manipulating GNNs seems to be [jraph][jraph] but it is unmaintained and pretty hard to
understand.

I really like the control that JAX gives to the developer and I feel that it will always be more fun
to implement my own models myself. I know I can rely on JAX's core operations that will be
efficiently compiled.

[DGL]:                  https://www.dgl.ai/
[PyG]:                  https://pytorch-geometric.readthedocs.io/en/latest/index.html
[equinox-docs]:         https://docs.kidger.site/equinox/
[github-impl]:          https://github.com/pierrot-lc/gnn-tuto
[jax-ops]:              https://jax.readthedocs.io/en/latest/jax.ops.html
[jax-sparse-module]:    https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html
[jraph]:                https://github.com/google-deepmind/jraph
