# #4961: the training-domain overlap check runs before Test, and picks its own reference

- `trust-stats` — **Compare against** now starts on the dataset holding most of the detector's labelled examples, and a *likely domain shift* warning toast can be up when a detector meets a dataset unlike its training one
