# Approximate Bayes Computation: an application on the SIR model

In this project, we apply a likelihood-free inference technique called **Approximate Bayes Computation*** (ABC). The goal is to infer two hidden parameters of the SIR model by simulating synthetic data. For more details look at the notebook in this repo. Also, a comparison between two ABC algorithms and the standard Metropolis-Hastings is made.

## How to use
`ABC_SIR_results.ipynb`: a summary of the project. It contains an explanation of the context of ABC and SIR, plus our implementation and results.

`multi_alg3.py`: it implements a Likelihood-free rejection sampler with a MCMC-informed optimization. It runs in a multiprocessing fashion.


## Likelihood-free rejection sampler
References: [Approximate Bayesian computational methods](https://link.springer.com/article/10.1007/s11222-011-9288-2#preview)
[Bayesian SIR model with change points with application to the Omicron wave in Singapore](https://www.nature.com/articles/s41598-022-25473-y)

We explore different rejection criteria (i.e. eucledian distance and autocorrelation distance) and evaluate their performance over the hyperparameters space.
