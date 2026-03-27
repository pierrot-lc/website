---
title: Encoding the TSP Solution on a Circle
date: 2026-03-23
tags:
  - Research blogpost
  - TSP & NCO

illustration: solution-example.png
---

Neural Combinatorial Optimization (NCO) aims at simplifying the design of combinatorial optimization
solvers using neural networks. Its an exciting area of research as we expect learning-driven
heuristics to outperform manually designed ones. We focus in this work on the Traveling Salesman
Problem (TSP), one of the most famous NCO problem, and propose to encode the solution onto a circle.
Our motivation is to provide a flexible and straightforward way to represent the solution. As
example, we use flow matching to generate the solutions by moving the points on the circle. We show
how a simple modification to RoPE, called CircularRoPE, makes our neural solver invariant to
rotations of the circle. While our results are not on par with the state-of-the-art, we hope this
post will motivate future similar research directions.

## Motivation
**About autoregressive solvers.** State-of-the-art solvers like BQ-NCO or INViT predict the solution
step by step. Starting from an initial city, the solver is repeatedly called to select the next city
to visit. This starting point is arbitrary and not an inner part of the TSP definition. This can
bias the solver which further carry that bias along the whole solving process. Furthermore, such
solvers can't generate a solution with less iterations than the number of cities in the instance to
solve, an important characteristic if we want to tradeoff solution quality for solving time.

**About heatmap solvers.** Heatmap solvers like DIFUSCO consider the whole solution at once by
predicting the probability for each edge to belong to the optimal solution. They require a decoding
search algorithm to generate the final solution to respect the cycle constraint (ex: MCTS). Search
space is large as all $N \times N$ edges can be considered. In practice some additional assumptions
are made to sparsify the graph, for example by considering only the edges that belong to the k
nearest neighbours only, but this further adds a bias to the solving process.

Ideally, only a strict minimum set of inductive biases are embedded within the solver and the rest
should be handled by the neural network. In this work we propose to embed the cycle constraint
within a circle: the tour is given by a sequence of angles. The neural solver predict $N$ angles and
the solution is directly read by sorting the nodes following their order on the circle. This
continuous representation additionally unlocks generative frameworks such as the ones used by image
generators. We illustrate how to train solvers on this particular solution representation using flow
matching. Our solvers consider the TSP solution as a whole, remove the choice of a starting point
and the necessity to tie the iterations with the number of cities of the instance. Furthermore,
generating solutions do not require a complex search algorithm as we only require a ODE solver.

In summary, our contributions are the following:
1. We propose a new TSP solution representation that characterize a TSP solution that is continuous
   and easy to decode.
2. We design a dedicated neural architecture using CircularRoPE and use an adequate flow matching
   objective that takes into account this specific representation.
3. We show that flow matching allows us to generate solutions in less NFEs than the autoregressive
   neural solvers while being fully end-to-end.

## Solution Representation
TSP is defined by a set of $N$ cities to visit, where each city has 2D euclidean coordinates
$(x_i)_1^N \in \mathbb{R}^2$. The goal is to find the shortest hamiltonian cycle, a tour that visits
every cities such that the total euclidean distance is minimized. We denote the tour by a
permutation $(p_i)_1^N$ and the cost of a solution is computed by:

$$
  \sum_{i = 1}^{N - 1} d_{p_i, p_{i+1}} + d_{p_N, p_1},
$$

where $d_{i, j} = ||x_i - x_j||_2$ is the euclidean distance between city $i$ and $j$.

**We consider the cyclic nature of the solution and places the cities on a circle.** Each city is
assigned to an angle $a_i$ and the solution is given by following the order of the nodes along this
circle. Given an optimal tour $p^*$, we place the nodes uniformly around the circle:

$$
a_{p_i^*} = \frac{2 p_i^* \pi}{N}.
$$

The neural solver is asked to predict the angles based on the euclidean position of the cities.

Compared to autoregressive solvers, we do not define any starting or ending node. The solution is
considered altogether and the model modify the current solution simply by moving the nodes around
the circle. Compared to heatmap solvers, our representation always defines a valid tour. The neural
network do not have to rely on an additional search algorithm to generate a solution. In general, we
found this representation to be more aligned with the TSP constraints. Finally, our representation
is continuous which makes it compatible with many generative methods. In this work, we use flow
matching to progressively move the nodes around the circle.

## Circular Flow Matching

<figure class="image">
  <img src="tsp-20_solving.gif" alt="Circular flow matching">
  <figcaption>Example of a TSP-20 instance being solved using flow matching.</figcaption>
</figure>

We cast the generative process of predicting the angles using flow matching. The neural network
progressively moves the nodes towards their optimal angles, from $t = 0$ to $t = 1$. At $t = 0$, the
angles are randomly initialized following the uniform distribution $U[0, 2\pi]$. We define the flow
to be the shortest movement between the initial angles $a(t = 0)$ and the final angles $a(t = 1) =
a^*$, thus using the optimal transport formulation of the flow matching framework. The neural
network is involved in the following ODE:

$$
  \frac{da(t)}{dt} = f_\theta(a(t), t, X),
$$

where $a(t) \in \mathbb{R^N}$ is the vector of cities angles at time $t$, $X \in \mathbb{ R^{N
\times 2} }$ is the matrix of cities coordinates in the euclidean space, and $\theta$ are the
learnable parameters of the neural network.

The optimal solution on the circle is invariant to angular shifts, so it would be inefficient to ask
for the model to predict a particular angle absolute configuration. Instead, we characterize the
solution predicted by the model compared to the optimal solution. Precisely, we apply the flow
predicted by the model and compute the resulting pairwise relative distances of the nodes on the
circle. Those distances are compared to the ones of the optimal solutions:

$$
  \mathcal{L_{\text{cycle}}}(\theta) = || D(\hat{a}) - D(a^*) ||_2, \quad \hat{a} = a(t) + (1 - t) f_\theta(a(t), t, X)
$$

where $D \in \mathbb{R}^{N \times N \times 2}$ is the matrix of pairwise signed distances, such that
$D(a)_{i, j} = (\text{cos}(a_i) - \text{cos}(a_j), \text{sin}(a_i) - \text{sin}(a_j))$. This loss is
circular invariant, respecting the invariances of the TSP solutions.

## Neural Network Architecture
TSP can be modeled by a complete graph, we thus use a transformer where each city is represented by
a token. A token is initialized by concatenating the current timestep $t$ with its corresponding
city euclidean coordinates.

To properly encode the solution, we want our model to be invariant to circular shifts of the angles.
To do so, our model must perceive the angles relatively to each other. RoPE is a powerful choice to
embed relative relationships between our tokens and so we use it where the token's position is
defined by its angle on the circle:

$$
  \hat{q} = \text{RoPE}(q, a) = \text{Re}[q e^{i a \theta}] \\
  \hat{k} = \text{RoPE}(k, a) = \text{Re}[k e^{i a \theta}] \\
  \hat{q}_m \hat{k}_n^T \propto \text{cos}((a_m - a_n) \theta).
$$

This ensure that only relative differences in the angles are taken into account. Finally, we set the
basis $\theta$ to make the interaction rotation invariant:

$$
  \text{cos}[(a_m - a_n + 2k\pi) \theta] = \text{cos}[(a_m - a_n) \theta] \quad \forall k \in \mathbb{Z} \\
  \implies \theta \in \mathbb{Z}.
$$

Hence, we make RoPE invariant to circular shifts of the angles by fixing its basis to integer
values. Following the usual exponential decay of the bases, we set our bases to increase
exponentially:

$$
  (\theta_i)_1^N = \text{round}[ e^{ \text{log}(K_{ \text{max} }) \frac{ i }{N - 1} } ],
$$

with $N$ the number of bases and $K_{\text{max}}$ set to $5$ in our experiments. We name such
positional encoding **CircularRoPE** to denote its circular invariance.

## Weighted Loss
We noticed that uniformly sampling $t \sim U[0, 1]$ during training is not the most efficient
strategy as it puts a lot of weights to the earliest timesteps, where the task is the hardest. This
makes the model trade some of its capabilities later for a better flow estimate at the beginning. To
bias the sampling, we use a beta distribution $\text{Beta}(\alpha, \beta)$ and find experimentally
that $\alpha$ and $\beta$ fixed to 5 and 1 respectively improves training efficiency and the final
solver quality.

## Experiments
All instances are randomly generated by sampling points uniformly on the unit square and using
`Concorde` to generate optimal solutions. Training datasets contains 1M random instances.
Performance is measured with the optimality gap

$$
  \text{gap}(\%) = 100 * \frac{\text{cost}_{\text{pred}}}{\text{cost}_{\text{opt}}}.
$$

**Initial experiment.** We first train three models for three different TSP sizes: 20, 50 and 100.
Models are trained for 100k iterations with a batch size of 256. They have 700k, 6M and 25M
parameters for the model TSP-20, TSP-50 and TSP-100 respectively. Once trained, we use the Euler ODE
solver and specify the number of solver steps.

| Instance | 1 step  |        | 10 steps |        | 100 steps |        | 1000 steps |        |
|:---------|:-------:|:------:|:--------:|:------:|:---------:|:------:|:----------:|:------:|
|          | _Gap_   | _Time_ | _Gap_    | _Time_ | _Gap_     | _Time_ | _Gap_      | _Time_ |
| TSP-20   | 20.92   | 0.9m   | 0.35     | 0.9m   | 0.26      | 1.0m   | 0.26       | 1.6m   |
| TSP-50   | 74.21   | 1.2m   | 3.77     | 1.2m   | 2.97      | 1.4m   | 2.89       | 1.8m   |
| TSP-100  | 167.21  | 1.1m   | 7.73     | 1.2m   | 5.05      | 1.2m   | 4.83       | 1.8m   |

As expected, increasing the number of steps both increase solution quality and solving time. Flow
matching naturally handle the tradeoff between computation time and solution quality. We can see for
example that TSP-50 can reach a pretty good solution in only 10 ODE steps (10 NFEs), but that going
up to 1000 steps further improve the solution.

**Problem-size Generalization.** In a second example, we train a 25M parameters neural solver on
training instances of sizes between 128 and 256. The variable training TSP sizes enhance the solver
ability to generalize to larger instances.

| Model      | TSP-100 |        |TSP-250 |        | TSP-500 |        |
|:-----------|:-------:|:------:|:------:|:------:|:-------:|:------:|
|            | _Gap_   | _Time_ | _Gap_  | _Time_ | Gap     | _Time_ |
| Our (100)  | 5.67    | 1.4m   | 11.01  | 1.8m   | 29.00   | 1.3m   |
| Our (1000) | 5.28    | 2.4m   | 9.76   | 3.1m   | 27.95   | 3.6m   |
| BQ-NCO     | 0.31    | 0.6m   | 0.67   | 1.5m   | 1.17    | 3.8m   |
| INViT-3V   | 4.95    | 1.0m   | 5.92   | 2.8m   | 6.30    | 5.9m   |

We report the performance of our neural solver when using 100 and 1000 ODE solver steps. We compare
against two well-known baselines, BQ-NCO and INViT. Sadly, we can see that our model is not
competitive with the state-of-the-art.

**Ablations.** We now evaluate the pertinence of CircularRoPE and our flow matching loss. For
CircularRoPE, we compare the TSP-100 model against a baseline that does not round the basis $\theta$
(RoPE) and a third baseline that simply initialize the transformer tokens with their corresponding
angles.

<figure class="image">
  <img src="ablations_circular-rope-loss.png" alt="CircularRoPE training dynamic comparison against a RoPE baseline and direct input baseline.">
  <figcaption>CircularRoPE and RoPE have very similar training dynamics, and are faster to train comparably to the input baseline.</figcaption>
</figure>

We can see that CircularRoPE and RoPE are slightly faster to train compared to the third baseline,
but all of them plateau at the same spot. It appears that CircularRoPE do not bring much difference
against the classical RoPE, probably because RoPE is already almost circular invariant (up to some
rounding error).

We now evaluate our circular invariant flow loss. We compare against the usual flow matching loss:
$\mathcal{L}(\theta) = || f_\theta(a(t), t, X) - \text{flow}(a(0), a(1)) ||_2$. We also validate the
beta distribution sampling of our timesteps by replacing it by the usual uniform sampling: $t \sim
U[0, 1]$.

| Model                                  | TSP-100 |
|:---------------------------------------|:-------:|
| Baseline                               | 10.45   |
| + $\mathcal{L}_{\text{cycle}}(\theta)$ |         |
| + $\text{Beta}(5, 1)$                  | 5.05    |

The original flow loss imposes for the model to predict exactly where the final angles must be. This
imposes a non-necessary constraint and is also impossible for the model to guess. Adding our
specific loss improves from $17.36\%$ to $10.00\%$. Finally, biasing the training sampling of $t$ to
follow the $\text{Beta}(5, 1)$ distribution forces the solver to focus on later stages of the
solving process and further boost performances to $5\%$.

## Analysis
We experimentally saw a performance ceiling on TSP-100 instances. Even when training larger models
our performance would not improve. This suggests a deeper issue, maybe linked to our specific
training procedure. Improvements like our biased sampling and the invariant flow loss greatly helped
our models (going from 10% to 5% of optimality gap on TSP-100). There might be a similar issue that
prevent our models from going further.

<figure class="image">
  <img src="tsp-100_best.gif" alt="Circular flow matching">
  <figcaption>Good example.</figcaption>
</figure>

<figure class="image">
  <img src="tsp-100_worst.gif" alt="Circular flow matching">
  <figcaption>Bad example.</figcaption>
</figure>
