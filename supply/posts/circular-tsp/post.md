---
title: Encoding the TSP Solution on a Circle
date: 2026-03-23
tags:
  - Research blogpost
  - TSP & NCO

illustration: solution-example.png
---

Neural Combinatorial Optimization (NCO) aims at simplifying the design of combinatorial optimization
solvers using neural networks. Its an exciting area of research with many real applications, as we
expect learning-driven heuristics to outperform manually designed ones. We focus in this work on the
Traveling Salesman Problem (TSP), one of the most famous NCO problem, and propose to encode the
solution onto a circle. This representation is continuous and non-ambiguously decodes to a single
solution, allowing us to use flow matching to train our solver. We show how a simple modification to
RoPE, which we called CircularRoPE, makes our neural solver invariant to rotations of the circle.
Crucially, we can generate solutions in less NFEs than the number of nodes in the instance.

## Motivation
**About autoregressive solvers.** State-of-the-art solvers like BQ-NCO or INViT predict the solution
step by step, by choosing an starting point and unroll the model as many times as there are nodes in
the instance. This starting point is arbitrary and not an inner part of the TSP definition. The
neural solver is biased by this initial starting city and further carry that bias along the whole
solving process. Furthermore, such solvers can't solve an instance with less iterations than the
number of cities in the instance to solve, an important characteristic if we want to tradeoff
solution quality with solving time.

**About heatmap solvers.** Heatmap solvers like DIFUSCO consider the whole solution at once by
predicting the probability for each edge to belong to the optimal solution. They require a decoding
search algorithm to generate the final solution to respect the cycle constraint (ex: MCTS). Search
space is large as all $N \times N$ edges can be considered. In practice some additional assumptions
are made to sparsify the graph, for example by considering only the edges that belong to the k
nearest neighbours only, but this further adds a bias to the solving process.

Ideally, only a strict minimum set of inductive biases are embedded within the solver and the rest
should be handled by the neural network. That's why in this work we propose to embed the cycle
constraint within a circle: the tour is given by a sequence of angles. The neural solver predict $N$
angles and the solution directly read by sorting the nodes following their order on the circle. This
continuous representation additionally unlocks generative frameworks such as the ones used by image
generators. We take as example flow matching and apply it to this specific solution representation.
Our resulting solver considers the solution as a whole, removes the choice of a starting point and
the necessity to tie the iterations with the number of cities in the instance. Furthermore,
generating solutions do not require a complex search algorithm to be generated as we only require a
ODE solver.

In summary, our contributions are the following:
1. We propose a new TSP solution representation that characterize a TSP solution in a continuous and
   non-ambiguous fashion.
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
considered as a whole and the model is be able to perceive and modify the current solution simply by
moving the nodes around the circle.

Compared to heatmap solvers, our representation always defines a valid tour. The neural network do
not have to rely on an additional search algorithm to generate a solution.

In general, we found this representation to be more aligned with the TSP constraints. Finally, our
representation is continuous which makes it compatible with many generative methods. In this work,
we use flow matching to progressively move the nodes around the circle.

## Circular Flow Matching

<figure class="image">
  <img src="tsp-20_solving.gif" alt="Circular flow matching">
  <figcaption>Example of a TSP-20 instance being solved using flow matching.</figcaption>
</figure>

We cast the generative process of predicting the angles using flow matching. The neural network
progressively moves the nodes towards their optimal angles, from $t = 0$ to $t = 1$. At $t = 0$, the
angles are randomly initialized following the uniform distribution $U[0, 2\pi]$. We define the flow
to be the shortest movement between the initial angles $a(t = 0)$ and the final angles $a(t = 1)$,
thus using the optimal transport formulation of the flow matching framework. The neural network
is involved in the following ODE:

$$
  \frac{da(t)}{dt} = f(a(t), t, X; \theta),
$$

where $a(t) \in \mathbb{R^N}$ is the vector of cities angles at time $t$, $X \in \mathbb{ R^{N \times 2} }$
is the matrix of cities coordinates in the euclidean space, and $\theta$ are the learnable
parameters of the neural network.

The optimal solution on the circle is invariant to angular shifts, so it would be inefficient to ask
for the model to predict a particular angle absolute configuration. Instead, we characterize the
solution predicted by the model compared to the optimal solution. Precisely, we apply the flow
predicted by the model and compute the resulting pairwise relative distances of the nodes on the
circle. Those distances are compared to the ones of the optimal solutions:

$$
  \hat{a} = a(t) + (1 - t) f(a(t), t, X; \theta), \\
  loss(\hat{a}, a^*; \theta) = || D(\hat{a}) - D(a^*) ||_2,
$$

Where $D \in \mathbb{R}^{N \times N \times 2}$ is the matrix of pairwise signed distances, such that
$D(a)_{i, j} = (cos(a_i) - cos(a_j), sin(a_i) - sin(a_j))$. This loss is circular invariant,
respecting the invariances of the TSP solutions.

## Neural Network Architecture
TSP can be modeled by a complete graph, we thus use a transformer where each city is represented by
a token. A token is initialized by concatenating the current timestep $t$ with its corresponding
coordinates.

To properly encode the solution, we want our model to be invariant to circular shifts of the angles.
To do so, our model must perceive the angles relatively to each other. RoPE is a powerful choice to
embed relative relationships between our tokens and so we use it where the token's position is
defined by its angle on the circle:

$$
  \hat{q} = RoPE(q, a) = Re[q e^{i a \theta}] \\
  \hat{k} = RoPE(k, a) = Re[k e^{i a \theta}] \\
  \hat{q}_m \hat{k}_n^T \propto cos((a_m - a_n) \theta).
$$

This ensure that only relative differences in the angles are taken into account. Finally, we set the
basis $\theta$ to make the interaction rotation invariant:

$$
  cos[(a_m - a_n + 2k\pi) \theta] = cos[(a_m - a_n) \theta] \forall k \in \mathbb{Z}, \\
  \implies \theta \in \mathbb{Z}.
$$

Hence, we make RoPE invariant to circular shifts of the angles by fixing its basis to integer
values. Following the usual exponential decay of the bases, we set our bases to increase
exponentially:

$$
  (\theta_n)_1^N = \text{round}[ \text{exp}( log(K_{ \text{max} }) \frac{ n }{N - 1} ) ],
$$

with $N$ the number of bases ($d_{\text{head}} / 2$) and $K_{\text{max}}$ set to $5$ in our
experiments.

We name such positional encoding **CircularRoPE** to denote its circular invariance.

## Experiments
We generate random instances by sampling points uniformly on the unit square $x_i \in [0, 1]^2$ and
generate the optimal solution using `Concorde`. For each training dataset, we generate 1M training
instances. Final performance is always measured on 128 new instances of the same size.

### Main Results

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-baqh" colspan="2">10 steps</th>
    <th class="tg-baqh" colspan="2">100 steps</th>
    <th class="tg-baqh" colspan="2">1000 steps</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Model</td>
    <td class="tg-0pky">euler</td>
    <td class="tg-0lax">dopri5</td>
    <td class="tg-0pky">euler</td>
    <td class="tg-0lax">dopri5</td>
    <td class="tg-0pky">euler</td>
    <td class="tg-0lax">dopri5</td>
  </tr>
  <tr>
    <td class="tg-0pky">TSP-20</td>
    <td class="tg-0pky">4.84</td>
    <td class="tg-0lax">2.09</td>
    <td class="tg-0pky">2.17</td>
    <td class="tg-0lax">2.01</td>
    <td class="tg-0pky">2.02</td>
    <td class="tg-0lax">2.01</td>
  </tr>
  <tr>
    <td class="tg-0pky">TSP-50</td>
    <td class="tg-0pky">12.62</td>
    <td class="tg-0lax">5.20</td>
    <td class="tg-0pky">4.59</td>
    <td class="tg-0lax">3.41</td>
    <td class="tg-0pky">3.74</td>
    <td class="tg-0lax">3.41</td>
  </tr>
  <tr>
    <td class="tg-0pky">TSP-100</td>
    <td class="tg-0pky">70.13</td>
    <td class="tg-0lax">20.78</td>
    <td class="tg-0pky">12.21</td>
    <td class="tg-0lax">10.43</td>
    <td class="tg-0pky">10.59</td>
    <td class="tg-0lax">10.20</td>
  </tr>
</tbody></table>

We first show our mains results in Table 1. For each specific TSP size, we trained a dedicated
model. Flow matching allow us to natively trade compute for precision by varying the number of ODE
solver steps. We further compare the results between two solvers: Euler and Dopri5. Euler is a basic
ODE solver that evaluates the flow only once per steps, whereas Dopri5 evaluates at most 7 times per
steps to have a better flow estimate.

Let's first have a look at how the number of steps impact the solution quality.

Now let's focus on the two evaluated solvers.

As expected, the specific ODE solver is important to get both better solutions and to produce them
faster.


### Ablations
We evaluate in this section the importance of using CircularRoPE and our circular invariant loss.

We compare models trained on TSP-50 using CircularRoPE, RoPE and absolute angles.

We now compare our model trained with our invariant loss to a model with the standard flow matching
objective: $L(\theta) = || f(a(t), t, X; \theta) - \text{flow}(a(0), a(1)) ||_2$.

We can conclude that it is important to respect the invariances induced by representing the TSP
solution on a circle.

### Weighted Loss
We noticed that uniformly sampling $t \sim U[0, 1]$ during training is not the most efficient
strategy as it puts a lot of weights to the earliest timesteps, where the task is the hardest. This
makes the model trade some of its capabilities later for a better flow estimate at the beginning. We
noticed a decrease in solution quality when sampling uniformly compared to biased sampling where
later timesteps are sampled more often. To bias the sampling, we use a beta distribution
parameterized by $\alpha$ and $\beta$. We performed a random searched on TSP-20 instances where both
values where sampled between 0.5 and 9. Our results suggest that taking a value of $5.3$ and $1.5$
for $\alpha$ and $\beta$ respectively improves both training efficiency and final solution quality.

![!Alpha-beta sweep](alpha-beta_parallel-coordinates.png)

### Problem-size Generalization

We train a model on a varying size of instances, from 128 to 256, and evaluate how it generalizes on
larger instances. We compare this model to other state-of-the-art NCO solvers.

Sadly, our model does not outperform other solvers, specifically the autoregressive ones.

## Conclusion

