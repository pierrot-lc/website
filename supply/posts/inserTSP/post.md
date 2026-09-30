---
title: InserTSP
date: 2026-09-30
tags:
  - Research blogpost
  - TSP & NCO
  # - <a href="https://codeberg.org/pierrot-lc/circular-tsp">code</a>

# illustration: solution-example.png
bibliography: biblio.bib
---

In this post I want to explore the idea of neural insertion solvers for TSP. We can motivate this
work by the idea that such solvers are more flexible than the classical autoregressive next-visit
prediction. Or we can justify it by [comparing][tsp-utils] how the classical random insertion
heuristic compare to the classical nearest neighbor heuristic. But my original motivation was to
better use the GPU during training by always using a full self-attention over all cities of the
training instances, as I'll discuss later.

As a reminder, the world of *autoregressive constructive neural TSP solvers* is dominated by
next-visit predictions. Starting from an initial city, the model predicts one-by-one the next cities
in the sequence. This constructs the final tour and we make sure at every step that only the
remaining unvisited cities can be selected, such that the sequence is a valid TSP solution. In
comparison, a neural insertion model maintains a tour over all currently visited cities and inserts
unvisited cities in between edges of that tour.

<div class=figure-container>
  <figure>
    <img src="solve-next-visit.gif" alt="Next-visit prediction">
    <figcaption>Next-visit prediction</figcaption>
  </figure>

  <figure>
    <img src="solve-insertion.gif" alt="Insertion prediction">
    <figcaption>Insertion prediction</figcaption>
  </figure>
</div>

<style>
  .figure-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    justify-content: center;
  }
  .figure-container figure {
    flex: 1 1 300px;
    margin: 0;
    max-width: 100%;
  }
  .figure-container img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 8px;
  }
</style>

So far, only one neural insertion model has been proposed by [@Luo2025]. In their work, they select
the next city to insert as the one being the closest unvisited city from the current tour. This
limits inference flexibility and that's why in this work we instead randomly select the cities to
insert.

Highlights:
- We propose a new neural insertion TSP solver. Compared to its predecessor, it randomly selects the
  cities to insert in the tour.
- We exploit its flexibility to expose a time-quality trade-off. We can insert multiple cities at
  once to cut the total inference time, at the cost of solution quality.
- We design a dedicated neural architecture that is simple and effective.
- We reach new state-of-the-art performances compared to other constructive neural solvers here and
  there.

## Training algorithm

Training is done using supervised learning from a set of 1M random TSP instances solved to
optimality with `Concorde` ([@Applegate2006]).

![Training steps](./training-steps.png)

Starting from the optimal tour, some random nodes are removed such that we keep track to which edge
each of them should be inserted to. The model is asked to retrieve that edge for any of random node
we ask for.

The node to insert is marked (see model architecture) and the loss is computed only for this
specific node. Loss is a cross-entropy over the set of edges in the tour.

## Model architecture

The neural architecture needs four things:
1. Represent the current tour.
2. Knows which city is asked to be inserted.
3. Let information flow between cities easily.
4. Output a probability over each edges of the tour.

Each city $i$ is represented by a vector $x_i \in \mathbb{R}^h$ of some hidden dimension $h$. They
share information using a standard self-attention layer. Before each self-attention layer, their
representation is enriched by the current tour: each node receives the embeddings of its neighbors
by concatenation and reduce them back to the original hidden dimension.

<figure class="image">
  <img src="tour-embedding.png" alt="Tour embedding" width="500">
  <figcaption>Each city in the tour receives the embedding of its neighbors.</figcaption>
</figure>

When a city has no neighbors (it is not in the tour), we consider that it is its own neighbor.
Here's maybe a more detailed implementation in JAX:

```python
n = vmap(self.norms[0])(x)
n = jnp.concat((n, (n[prev] + n[next]) / 2), axis=1)  # [n_cities, 2 * hidden_dim]
n = vmap(self.reduce)(n)  # [n_cities, hidden_dim]
n = jax.nn.gelu(n)
x += n

n = vmap(self.norms[1])(x)
x += self.mha(n)

n = vmap(self.norms[2])(x)
return x + vmap(self.ffn)(n)
```

Combined with the self-attention operation, our model efficiently represents the current tour while
letting information flowing freely with a single self-attention operation (this avoids mixing GNNs
and self-attention layers and makes our model arguably simpler).

To specify which city is to be inserted, we mark such city with a dedicated learned embedding (easy).

Finally, we represent edges by the combination of their nodes embeddings and compute the
probabilities using a dot product between the marked city and the edges.

TODO: Comparison with L2C.

## Experiment 1: Baseline

What would such model do when trained on 1M TSP-100 instances?

| Model    | TSP-100 | TSP-250 | TSP-500 | TSP-1000 |
|:---------|:-------:|:-------:|:-------:|:--------:|
| BQ-NCO   | 0.31    | 0.67    | 1.17    | 2.19     |
| PENS-AR  | 0.28    | 0.47    | 0.74    | 1.17     |
| L2C      | 0.55    | 1.21    | 2.29    | 4.65     |
| InserTSP | 0.50    | 0.90    | 1.47    | 2.44     |

Not bad! We already beat L2C ([@Luo2025]), probably thanks to our more efficient architecture.
BQ-NCO ([@Drakulic2023]) and PENS-AR are classical neural autoregressive solvers that predicts the
next city to visit. All models here are used in a greedy fashion without any particular decoding
scheme. At every step of the solution construction, they take the most probable next action.

## Experiment 2: Exploiting random insertion flexibility

Since any random node can be inserted, we can mark multiple nodes at the same time and insert them
in parallel. As long as the predicted insertions do not share the same edges, we can safely insert
those nodes in one go.

So we trained a model with multiple cities to insert at the same time. For each element in the
mini-batch, we sample between 1 and 10 unvisited cities to insert and compute the loss over those
cities. Each of those cities are marked such that the model knows which are to be inserted.

Here's the results when the model gets evaluated with a variable number of insertions in parallel:

| Parallel | TSP-100 |        | TSP-250 |        | TSP-500 |        | TSP-1000 |        |
|:---------|:-------:|:------:|:-------:|:------:|:-------:|:------:|:--------:|:------:|
|          | _Gap_   | _Time_ | _Gap_   | _Time_ | _Gap_   | _Time_ | _Gap_    | _Time_ |
| 1 mark   | 0.21    | 0.4m   | 0.59    | 0.3m   | 1.14    | 0.7m   | 2.49     | 1.6m   |
| 2 marks  | 0.33    | 0.3m   | 0.76    | 0.2m   | 1.50    | 0.4m   | 3.78     | 0.9m   |
| 5 marks  | 0.59    | 0.3m   | 0.89    | 0.1m   | 2.08    | 0.2m   | 7.39     | 0.4m   |

Here, "5 marks" means that during solution generation we always mark 5 random unvisited cities and
insert as many as possible. You can see that performance degrades but the time taken to solve the
instances also drop: **there's a time-quality trade-off!**

![Comparing 5-2-1 parallel marks](./parallel-comparison.gif)

This trade-off has two possible explanations:
1. The model has to split its focus over multiple elements at the same time, making its prediction
   less sharp.
2. The model builds incompatible solutions by predicting insertions that do not align with each
   other.

**Splitting focus hypothesis.**
We test this hypothesis by comparing how good solutions are when we mark multiple cities at the same
time against marking them one by one and inserting only after the last city (parallel vs sequential
marks).

|                    | TSP-100 | TSP-250 | TSP-500 | TSP-1000 |
|:-------------------|:-------:|:-------:|:-------:|:--------:|
| 1 mark             | 0.21    | 0.59    | 1.14    | 2.49     |
| 5 marks parallel   | 0.59    | 0.89    | 2.08    | 7.39     |
| 5 marks sequential | 0.52    | 0.84    | 1.50    | 2.73     |

The difference is more pronounced when the instances gets harder, which makes sense as it is where
the model needs more brain juice.

**Incompatible solution hypothesis.**
This is harder to test so I'll only present some complaisant arguments.

First, we can see that even the 5 sequential marks didn't completely recover. If compute focus was
the only issue, the model would have predicted similar solutions as the model inserting a single
city at once.

Second, this explanation is actually easy to show in diffusion language models. You can actually see
that some sentences are a mixed of two different valid sentences (see [@Zhao2026]). Such models predict
multiple token distributions in parallel and sampling the tokens can generate incompatible results.

Yet, I believe this hypothesis to only minimally impact the model. If two cities are geometrically
far from each other, there is a small chance that inserting one greatly impacts the insertion choice
of the second.

## Experiment 3: Insertion uncertainty

pass

## Appendix

**Tour encoding variants.**
We initially played with multiple attention ideas:
- Dedicated MHA layer with attention bias relative to the tour distance (only the visited nodes
would participate).
- Dedicated MHA layer with attention bias relative to the insertion cost (only unvisited nodes would
attend to visited nodes).
- Mixing both attentions at the same time.

Overall it added a lot of complexity for little gain so we dropped the idea and went with our
simpler tour encoding. The most promising results were obtained with three dedicated MHAs: one for
unvisited nodes over insertion edges, one for visited nodes over themselves, and one global MHA
without any bias. But this requires three different attention operations for each layer which was
too heavy.

**Architecture details.**
Coordinates are encoded using 2D RoPE, similar to PENS-R. The marks (insertion, visited and
unvisited) are added at the beginning of every layers, during the tour encoding.

[tsp-utils]: https://tangled.org/pierrot-lc.bsky.social/tsp-utils
