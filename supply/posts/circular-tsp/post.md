# Encoding the TSP Solution on a Circle
Neural Combinatorial Optimization (NCO) aims at simplifying the design of combinatorial optimization
solvers by training neural solvers. Its an exciting area of research with many real applications, as
we can expect learning-driven heuristics to outperform manually designed ones. We focus in this work
on the Traveling Salesman Problem (TSP), one of the most famous NCO problem, and propose to encode
the solution onto a circle. This representation is continuous and non-ambiguously decodes to a
single solution. We show how a simple modification to RoPE, which we called CircularRoPE, makes our
neural solver invariant to rotations of the circle. Finally we experiment with flow matching and
compare the solution qualities when a variable amount of NFEs are used. Crucially, we can generate
solutions in less NFEs than the number of nodes in the instance.

## Motivation
NCO allows us to think at a high level of the solver design. Here are the questions one might ask
when designing a neural solver:
1. What invariances should my neural network have?
2. How is the information processed?
3. What is the 

One of the promise of NCO is to replace all handmade heuristics with neural ones, putting the
practitioner at a higher level of choosing the neural solver characteristics. Many neural solvers
are autoregressives, tying their generation to all previous predictions and requiring as many
iterations as the size of the instance to solve. Another type of neural solvers predict a
probability heatmap over candidate variables. They consider the whole solution at once but they
require an additional search algorithm to generate a solution. One of the most famous and simple NCO
problem is the TSP, for which the best neural solvers are either based on a step-by-step
constructive approach or on a heatmap probability. We find those two approaches unsatisfying for the
following reason:
1. The constructive approaches are autoregressives, which means they must decide on an unatural
   starting point, which might unnecessary bias the final solution. Moreover, the solver requires at
   least $O(N)$ NFEs to generate a single solution.
2. Solvers that generate a heatmap are end-to-end and require a good search algorithm on top of the
   predicted edge probabilities to generate a good solution. This blurs the quality of the neural
solver and makes it hard to compare against the other neural solvers. It also rely on some other
sets of searching heuristics (ex: MCTS), making it harder to use.

We propose in this work a neural solver that predicts the TSP solution as a whole. We represent the
solution with nodes being laid out on a circle. A solution is obtained by taking the order of the
nodes on the circle. We argue this representation allow for a more flexible generative framework and
train a solver using flow matching as an example.

Our contributions are as follow:
1. We propose a new TSP solution representation and design compatible neural network architecture.
2. Using this new representation, we train a simple neural solver using flow matching.
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

## Neural Network Architecture
We now present our neural network model. The TSP can be modeled by a complete graph, we thus use a
transformer where each city is represented by a token. A token is initialized by concatenating the
current timestep $t$ with its corresponding coordinates.

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
generate the optimal solution using `Concorde`. For each training dataset, we generate and solve 1M
instances. Final performance is measured on 128 new instances of the same size.

## Conclusion

