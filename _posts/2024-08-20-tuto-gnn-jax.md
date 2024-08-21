---
title:  "How to Build a Graph Convolutional Network with Jax and Equinox!"
layout: post
date:   2024-08-20
---

I've been learning [Jax][jax-docs] for the past few months and for my last
project I needed to train a basic Graph Neural Network. Surprisingly, I did not
find good libraries to easily build GNNs. Most of them are
[outdated](https://github.com/google-deepmind/jraph) and are not using my DL
framework of choice ([Equinox!][equinox-docs]). This was the perfect excuse to
implement my own GNNs from scratch.

> I actually really like to implement my models from the basic components given
> by my DL framework. It becomes much easier to try some experimental ideas.

> The Jax/Equinox combinaison is perfect for DL research in my opinion. It
> gives the user an approachable low-level API that it can play with and it
> scales well thanks to its powerful JIT compiler. If you're not already using
> Jax, have a look at [this blogpost][long-live-jax].

So let's dive in. We'll see the simplest implementation first using the
adjacency matrix of the graph. We'll then continue with the more complex but
modular approach using the list of edges representation.

## Level 1: Using the Adjacency

Let's first recap how we usually represent a graph. For a list of nodes $$N$$,
we represent the relation between the nodes by some *adjacency matrix* $$A \in
\mathbb{N}^{N \times N}$$ where $$a_{i, j} \in \{0, 1\}$$ represents the edge
from the node $$j$$ to node $$i$$ ($$1$$ if the edge actually exists, $$0$$
otherwise). We will use this adjacency matrix to summarize how the nodes are
connected within each other.

Each node is represented by some vector of size $$H$$, so we can represent the
set of nodes as a matrix in $$\mathbb{R}^{N \times H}$$. This let us write our
first graph convolutional layer:

```py
class GraphConv(eqx.Module):
    linear: nn.Linear

    def __init__(self, hidden_dim: int, *, key: PRNGKey):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, key=key)

    def __call__(
        self,
        nodes: Float[Array, "n_nodes hidden_dim"],
        adjacency: Int[SparseArray, "n_nodes n_nodes"]
    ) -> Float[Array, "n_nodes hidden_dim"]:
        messages = vmap(self.linear)(nodes)
        return adjacency @ messages
```

Doing this matrix multiplication between the nodes representation and the adjacency matrix
is equivalent to the following computation:

$$
n_i = \sum_{j \in \mathcal{N(i)}} W n_j
$$

Where $$n_i \in \mathbb{R}^H$$ is the hidden representation of the node $$i$$
and $$\mathcal{N(i)}$$ is the set of neighbours of the node $$i$$.

Essentially, each node's representation is updated by taking the sum of its
neighbours' after having applied a linear transformation. You can see how
using the adjacency matrix is really simple here.

> To be computationally efficient, you need to use the
> [sparse][jax-sparse-module] matrix multiplication of Jax. Note that at the
> time of writing this article, this module is still experimental.


[equinox-docs]:         https://docs.kidger.site/equinox/
[jax-docs]:             https://jax.readthedocs.io/en/latest/quickstart.html
[jax-sparse-module]:    https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html
[long-live-jax]:        https://neel04.github.io/my-website/blog/pytorch_rant/
[reddit-binary]:        https://paperswithcode.com/dataset/reddit-binary
