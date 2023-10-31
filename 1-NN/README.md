
---
katex: true
---
This is an example of inline \\(\LaTeX\\). The following is Stokes' theorem in a
`displaymath` environment:  $$\int_{\partial \Omega} \omega = \int_{\Omega} d\omega $$ 
 $\int_{\partial \Omega} \omega = \int_{\Omega} d\omega $

 ```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```
```math
\sqrt{\frac{1}{2}}
```

$$
\begin{pmatrix}a & b\\ c & d\end{pmatrix} $$
$\begin{pmatrix}a & b\\ c & d\end{pmatrix}$
# Introduction to Neural Networks for beginners
(Expected prior knowledge: basics of graphs, meaning of mapping and partial derivatives)

Neural networks are a machine learning technique with which we intend to approximate a (possibly very complicated) function (or mapping) using some kind of graph constructed out of simple neurons. 

Lets take a look at such a neuron, one which has n inputs $x_1, ..., x_n$, with corresponding weights $w_1, ..., w_n$.
![0-NN/neuron.png](https://github.com/TU-e-Honors-Academy-AI-Track/DeepL_Intro/blob/00introtoNN/0-NN/neuron.png?raw=true)

This neuron would take its inputs, multiply them with their weights and sum them up, together with a bias $b$. Finally, it will run them through a non-linear activation function $g(x)$, as to allow the network to approximate non-linear functions (this can be done with the inaccuracy of floats, but thats less effective). So the output $y$ of the neuron would be $y=g(b +\sum$ $_{i=1}^{n} x_i*w_i)$.

A neural network would consists of those neurons, commonly divided in an input layer where the neurons get input from the outside, some amount of hidden layers and an output layer, in these neurons commonly get their inputs from the neurons in the previous layer. Here we have an example with a 3 neuron input layer (left), hidden layer (middle) and a 1 neuron output layer (right).

![0-NN/nn.png](https://github.com/TU-e-Honors-Academy-AI-Track/DeepL_Intro/blob/00introtoNN/0-NN/nn.png?raw=true)

We can express parts of this network using linear algebra, the input layer is just given a vector of all inputs, lets call it **x** $= \[x_1, x_2, x_3\]$. We can then express the outcome of the second, hidden, layer as **y**. If all weights of the second layer are stored in a matrix **A**, where $\bm{A}_{i,j}$ is the weigth between input neuron j and hidden layer neuron i (such that all weights for neuron i are in the same row), then:
$`**y**=g(A**x**+b)`$
 $`\sqrt{3x-1}+(1+x)^2`$

By doing a "forward pass" through the network, sequentially computing the values of each layer, the neural network transforms its input into an output. With the right network and weights, it should be possible to approximate any function/mapping, but the problem is, while we can easily make a sufficiently large network, how do we get the weights?

The currently most common solution to this is the combination of gradient descent and backpropogation. We take an (input, output) sample from a dataset and using its input we do a forward pass through our network, then we compare the expected output with the computed output and call the distance between them the "Loss".

Before we do something with the loss, lets consider gradient descent. 

![0-NN/gradient.png](https://github.com/TU-e-Honors-Academy-AI-Track/DeepL_Intro/blob/00introtoNN/0-NN/gradient.png?raw=true)

Imagine you are at the red dot in the graph and you want to reach the minimum, then one can simply follow the gradient (the derivative) to go to a minimum. While this strategy has the risk of getting stuck in a local minimum (left) instead of ending up at the global minimum (right), it works quite well, even in higher dimensions, which is often the case for neural networks. 

We can use this with the partial derivative of the effect each weight/bias has on the loss to adjust them closer towards a minimum in the loss, so a neural network which performs better. The amount we move is decided by the learning rate, too little and it will take forever or too much and you will overshoot the minimums (possibly even the global). This process would be done starting at the last layer and working towards the first, as this allows reusing computations.

..... WORK IN PROGRESS
