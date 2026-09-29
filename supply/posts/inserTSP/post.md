In this post I want to explore the idea of neural insertion solvers for TSP. We can motivate this
work by the idea that such solvers are more flexible than the classical autoregressive next-visit
prediction. Or we can justify it by [comparing][tsp-utils] how the classical random insertion
heuristic compare to the classical nearest neighbor heuristic. But my original motivation was to
better use the GPU during training by always using a full self-attention over all cities of the
training instances, as I'll discuss later.

As a reminder, the world of *autoregressive constructive neural TSP solvers* is dominated by
next-visit predictions. Starting from an initial city, the model predicts one-by-one the next cities
in the sequence. This constructs the final tour and we make sure at every step that only the
remaining unvisited cities can be selected, such that the sequence is a valid TSP solution.

[tsp-utils]: https://tangled.org/pierrot-lc.bsky.social/tsp-utils
