# Understanding Black-box Predictions via Influence Functions

This code replicates the experiments from the following paper:

> Pang Wei Koh and Percy Liang
>
> [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)
>
> International Conference on Machine Learning (ICML), 2017.

We have a reproducible, executable, and Dockerized version of these scripts on [Codalab](https://worksheets.codalab.org/worksheets/0x2b314dc3536b482dbba02783a24719fd/).

Dependencies:
- Numpy/Scipy/Scikit-learn/Pandas
- Tensorflow/Keras
- Spacy
- Matplotlib/Seaborn (for visualizations)

---

In this paper, we use influence functions --- a classic technique from robust statistics --- 
to trace a model's prediction through the learning algorithm and back to its training data, 
thereby identifying training points most responsible for a given prediction.
To scale up influence functions to modern machine learning settings,
we develop a simple, efficient implementation that requires only oracle access to gradients 
and Hessian-vector products.
We show that even on non-convex and non-differentiable models
where the theory breaks down,
approximations to influence functions can still provide valuable information.
On linear models and convolutional neural networks,
we demonstrate that influence functions are useful for multiple purposes:
understanding model behavior, debugging models, detecting dataset errors,
and even creating visually-indistinguishable training-set attacks.

If you have questions, please contact Pang Wei Koh (<pangwei@cs.stanford.edu>).
